"""
fp16_baseline.py

Target-only (non-speculative) decoding baseline for speculative_scout.py and
benchmark_ablation.py.

============================================================================
WHY THIS WAS REWRITTEN (2026-09-03)
============================================================================
Symptom: avg_power_watts and joules_per_token climbed run over run, regardless
of prompt order or whether prompts were run separately.

Diagnosis: it is not a measurement bug and not a prompt effect. Sorting the
N=10 summary by timestamp instead of by category shows power rising
~465 -> ~476 W over the first ~6 measurements (~40 s of wall clock) and then
sitting flat at 476.7 +/- 1.0 W for the remaining two minutes. Throughput is
constant across the whole run (38.4-38.7 tok/s), so J/tok is just a rescaled
copy of the power trace. That is a settling transient: leakage current rising
with die temperature at fixed clocks.

Supporting evidence: the N=5 run (no throwaway warmup round) starts at
457.18 W; the N=10 run (with the 3-prompt throwaway round) starts at 464.60 W.
Both plateau at ~476-477 W. The 20 s warmup recovered ~7 W of a ~19 W deficit,
i.e. ~37%, implying a time constant near 40 s. Reaching within ~1 W of steady
state therefore needs on the order of 120 s of sustained load.

So: the throwaway warmup round was the right idea, roughly 3x too short.

Fixes in this version:
  1. Closed-loop warmup. Instead of a fixed 3-prompt round, keep generating
     until measured power AND temperature stop moving (or a hard time cap is
     hit). The warmup trace is written into the summary so convergence is
     auditable rather than assumed.
  2. Energy from nvmlDeviceGetTotalEnergyConsumption (monotonic mJ counter,
     Volta+) instead of mean(samples) * duration. The old method is biased:
     NVML samples are not evenly spaced and the window filter discarded the
     partial intervals at both edges. Sampled energy is still computed, by
     proper trapezoidal integration with boundary interpolation, and both are
     recorded so they can be cross-checked.
  3. Per-trial thermal telemetry: temperature at window start/end, mean SM and
     memory clock, min/max power, enforced power limit, throttle reasons,
     sample count. The confound that took three sessions to find is now
     visible directly in the CSV.
  4. Optional SM/memory clock locking via NVML (needs elevated privileges;
     falls back gracefully with an explanatory message).
  5. Drift diagnostics in the summary: least-squares slope of power and J/tok
     against chronological trial index, with R^2, plus an explicit warning if
     residual drift exceeds a threshold. If the warmup was insufficient on a
     given machine, the script says so instead of quietly reporting a mean.
  6. KV-cache decoding (USE_KV_CACHE). The old loop re-ran the full forward
     over the whole growing sequence every step, which is why an 8B in bf16 on
     a ~480 W card was decoding at 38 tok/s. It also made throughput depend on
     prompt token length -- which is exactly the cross-category difference this
     harness is trying to measure. See the note on USE_KV_CACHE below before
     changing it.

Retained from the previous version (these were correct):
  - tokenizer.apply_chat_template(...), matching benchmark_ablation.py.
  - Interleaved (round-robin) trial order rather than contiguous blocks.
  - Shared model / tokenizer / monitor across all trials.

============================================================================
PARITY WARNING
============================================================================
Every setting in the CONFIG block below must match speculative_scout.py and
benchmark_ablation.py or the comparison is not apples-to-apples. In particular
USE_KV_CACHE and STOP_ON_EOS change the amount of work done per trial. If the
comparison scripts decode without a cache, set USE_KV_CACHE = False here and
accept the slow throughput; a matched-but-slow harness is a valid baseline, a
mismatched one is not.
"""

import argparse
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pynvml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CONFIG -- must match the comparison scripts
# ============================================================================

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# NOTE: the filename says fp16 but the original loaded bfloat16. Kept as
# bfloat16 to preserve numerical parity with the recorded runs. Change only if
# you change it in the comparison scripts too.
DTYPE = torch.bfloat16

# True  = real incremental decoding with past_key_values (correct, ~3-5x faster)
# False = full re-forward over the growing sequence every step (the old
#         behaviour). Use False only if the comparison scripts also do this.
USE_KV_CACHE = False

# False = always generate exactly max_new_tokens, so every trial does identical
#         work. Recommended for a power/energy baseline. The recorded runs all
#         hit 250/250 anyway, so this changes nothing in practice but removes a
#         possible source of variance.
STOP_ON_EOS = False

# Per-call warmup forward passes inside each timed trial (kept from original).
WARMUP_STEPS = 5

POLL_RATE_HZ = 100          # NVML power poll rate
TEMP_POLL_EVERY = 10        # sample temp/clocks every Nth power poll

# Closed-loop warmup parameters. Defaults are sized from the ~40 s time
# constant measured on the recorded runs; raise MIN_WARMUP_SEC on a hotter or
# more heavily power-limited card.
MIN_WARMUP_SEC = 120.0
MAX_WARMUP_SEC = 420.0
WARMUP_WINDOW = 4           # consecutive warmup generations that must agree
WARMUP_POWER_TOL_W = 1.5    # max-min power across that window, watts
WARMUP_TEMP_TOL_C = 1.0     # max-min temperature across that window, celsius

# Post-hoc drift check: warn if fitted power drift across the trial sequence
# exceeds this fraction of the mean.
DRIFT_WARN_FRACTION = 0.005  # 0.5%

PROMPTS = [
    ("Poem", "Write an original Chant Royal poem in English celebrating mathematics."),
    ("Physics", "Explain the physics of semiconductor memory bandwidth and the memory wall."),
    ("Code", "Write a Python implementation of a binary search tree with type annotations."),
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
MASTER_CSV = os.path.join(TELEMETRY_DIR, "telemetry_fp16_baseline.csv")


# ============================================================================
# NVML monitor
# ============================================================================

class NVMLPowerMonitor:
    """Background NVML sampler.

    Records power at POLL_RATE_HZ and temperature / clocks at a lower rate.
    Energy is taken from the hardware energy counter when available, since
    integrating sparse instantaneous power samples is biased.
    """

    def __init__(self, device_index=0, poll_rate_hz=POLL_RATE_HZ):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device_index = device_index
        self.interval = 1.0 / poll_rate_hz
        self.running = False
        self.lock = threading.Lock()
        self.samples = deque()  # (t, watts, temp_c_or_None, sm_mhz_or_None, mem_mhz_or_None)
        self.thread = None
        self.poll_errors = 0
        self._clocks_locked = False
        self._energy_supported = self._probe_energy_counter()

    # -- capability probes ---------------------------------------------------

    def _probe_energy_counter(self):
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return True
        except Exception:
            return False

    def read_energy_j(self):
        """Cumulative board energy in joules, or None if unsupported."""
        if not self._energy_supported:
            return None
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0
        except Exception:
            return None

    def device_info(self):
        info = {"device_index": self.device_index,
                "energy_counter_supported": self._energy_supported}
        for key, fn in (
            ("name", lambda: pynvml.nvmlDeviceGetName(self.handle)),
            ("driver_version", pynvml.nvmlSystemGetDriverVersion),
            ("power_limit_w", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0),
            ("max_sm_clock_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_SM)),
            ("max_mem_clock_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)),
        ):
            try:
                val = fn()
                info[key] = val.decode() if isinstance(val, bytes) else val
            except Exception as exc:
                info[key] = f"unavailable ({type(exc).__name__})"
        return info

    def throttle_reasons(self):
        for name in ("nvmlDeviceGetCurrentClocksEventReasons",
                     "nvmlDeviceGetCurrentClocksThrottleReasons"):
            fn = getattr(pynvml, name, None)
            if fn is None:
                continue
            try:
                return int(fn(self.handle))
            except Exception:
                continue
        return None

    # -- clock locking -------------------------------------------------------

    def lock_clocks(self):
        """Pin SM and memory clocks to their max. Needs elevated privileges."""
        try:
            sm = pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, sm, sm)
            self._clocks_locked = True
        except Exception as exc:
            print(f"[!] Could not lock SM clocks ({type(exc).__name__}): {exc}")
            print("    Run as root, or pin manually:  sudo nvidia-smi -lgc <mhz>")
            print("    Continuing without locked clocks -- expect more variance.")
            return False
        try:
            mem = pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            pynvml.nvmlDeviceSetMemoryLockedClocks(self.handle, mem, mem)
        except Exception:
            pass  # memory clock locking is unsupported on many parts
        print(f"[*] SM clocks locked to {sm} MHz.")
        return True

    def unlock_clocks(self):
        if not self._clocks_locked:
            return
        for fn_name in ("nvmlDeviceResetGpuLockedClocks", "nvmlDeviceResetMemoryLockedClocks"):
            fn = getattr(pynvml, fn_name, None)
            if fn is None:
                continue
            try:
                fn(self.handle)
            except Exception:
                pass
        self._clocks_locked = False

    # -- sampling ------------------------------------------------------------

    def start(self):
        self.running = True
        with self.lock:
            self.samples.clear()
        self.poll_errors = 0
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        next_deadline = time.perf_counter()
        counter = 0
        while self.running:
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                temp = sm = mem = None
                if counter % TEMP_POLL_EVERY == 0:
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                        sm = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
                        mem = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                    except Exception:
                        pass
                with self.lock:
                    self.samples.append((time.perf_counter(), pwr, temp, sm, mem))
            except Exception:
                self.poll_errors += 1
            counter += 1

            next_deadline += self.interval
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_deadline = time.perf_counter()  # fell behind; resync

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def close(self):
        self.unlock_clocks()
        self.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    # -- window statistics ---------------------------------------------------

    def window_stats(self, t0, t1):
        """Statistics over [t0, t1] from the sampled trace."""
        with self.lock:
            snap = list(self.samples)

        power_pts = [(t, p) for t, p, _, _, _ in snap]
        in_win = [(t, p) for t, p in power_pts if t0 <= t <= t1]
        duration = max(t1 - t0, 1e-9)

        stats = {
            "window_sec": duration,
            "n_power_samples": len(in_win),
            "effective_sample_hz": len(in_win) / duration if duration > 0 else 0.0,
        }

        if in_win:
            vals = [p for _, p in in_win]
            stats["mean_w"] = float(np.mean(vals))
            stats["median_w"] = float(np.median(vals))
            stats["min_w"] = float(np.min(vals))
            stats["max_w"] = float(np.max(vals))
            stats["energy_j_sampled"] = _integrate(power_pts, t0, t1)
        else:
            # Window shorter than the sample interval, or the poller died.
            stats["mean_w"] = None
            stats["median_w"] = None
            stats["min_w"] = None
            stats["max_w"] = None
            stats["energy_j_sampled"] = None
            stats["warning"] = "no power samples inside window"

        temps = [(t, v) for t, _, v, _, _ in snap if v is not None and t0 <= t <= t1]
        sms = [v for t, _, _, v, _ in snap if v is not None and t0 <= t <= t1]
        mems = [v for t, _, _, _, v in snap if v is not None and t0 <= t <= t1]
        stats["temp_start_c"] = temps[0][1] if temps else None
        stats["temp_end_c"] = temps[-1][1] if temps else None
        stats["temp_max_c"] = max(v for _, v in temps) if temps else None
        stats["sm_clock_mean_mhz"] = float(np.mean(sms)) if sms else None
        stats["sm_clock_min_mhz"] = int(np.min(sms)) if sms else None
        stats["mem_clock_mean_mhz"] = float(np.mean(mems)) if mems else None
        return stats


def _integrate(points, t0, t1):
    """Trapezoidal integral of a (time, value) series over [t0, t1].

    Interpolates at the boundaries so the partial intervals at each edge are
    included, which naive in-window averaging drops.
    """
    if len(points) < 2:
        return None
    pts = sorted(points)
    series = []

    def interp(ta, va, tb, vb, t):
        if tb == ta:
            return va
        return va + (vb - va) * (t - ta) / (tb - ta)

    for i in range(len(pts) - 1):
        (ta, va), (tb, vb) = pts[i], pts[i + 1]
        if tb < t0 or ta > t1:
            continue
        if ta < t0 <= tb:
            series.append((t0, interp(ta, va, tb, vb, t0)))
        if t0 <= ta <= t1:
            series.append((ta, va))
        if ta <= t1 < tb:
            series.append((t1, interp(ta, va, tb, vb, t1)))
    if t0 <= pts[-1][0] <= t1:
        series.append(pts[-1])

    series = sorted(set(series))
    if len(series) < 2:
        return None
    ts = np.array([t for t, _ in series])
    vs = np.array([v for _, v in series])
    return float(np.trapezoid(vs, ts)) if hasattr(np, "trapezoid") else float(np.trapz(vs, ts))


# ============================================================================
# Decoding
# ============================================================================

def _eos_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    for tok in ("<|eot_id|>", "<|end_of_text|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.add(tid)
        except Exception:
            pass
    return ids


def encode_prompt(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt").to("cuda")["input_ids"]


def generate(model, tokenizer, input_ids, max_new_tokens,
             use_cache=USE_KV_CACHE, stop_on_eos=STOP_ON_EOS):
    """Greedy decode. Returns (tokens_generated, hit_eos)."""
    eos = _eos_ids(tokenizer)
    tokens = 0
    hit_eos = False

    with torch.inference_mode():
        if use_cache:
            out = model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = 1
            if stop_on_eos and nxt.item() in eos:
                return tokens, True
            while tokens < max_new_tokens:
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens += 1
                if stop_on_eos and nxt.item() in eos:
                    hit_eos = True
                    break
        else:
            cur = input_ids
            while tokens < max_new_tokens:
                out = model(cur)
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=-1)
                tokens += 1
                if stop_on_eos and nxt.item() in eos:
                    hit_eos = True
                    break

    return tokens, hit_eos


# ============================================================================
# Single timed trial
# ============================================================================

def run_trial(model, tokenizer, monitor, input_ids, prompt_label,
              max_new_tokens, warmup_steps=WARMUP_STEPS, record=True,
              global_index=None):
    # Short in-trial warmup so the timed window does not start from an idle SM.
    with torch.inference_mode():
        warm = input_ids.clone()
        for _ in range(warmup_steps):
            o = model(warm)
            warm = torch.cat([warm, o.logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=-1)
    torch.cuda.synchronize()

    e_start = monitor.read_energy_j()
    t_start = time.perf_counter()

    tokens, hit_eos = generate(model, tokenizer, input_ids, max_new_tokens)

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    e_end = monitor.read_energy_j()

    stats = monitor.window_stats(t_start, t_end)
    latency = t_end - t_start

    energy_counter = (e_end - e_start) if (e_start is not None and e_end is not None) else None
    energy_sampled = stats["energy_j_sampled"]
    energy = energy_counter if energy_counter is not None else energy_sampled
    energy_source = "nvml_counter" if energy_counter is not None else "sampled_trapezoid"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "category": "FP16 Baseline (matched harness)",
        "prompt_label": prompt_label,
        "global_index": global_index,
        "tokens": tokens,
        "hit_eos": hit_eos,
        "prompt_tokens": int(input_ids.shape[-1]),
        "latency_sec": round(latency, 6),
        "throughput_tok_sec": round(tokens / latency, 2) if latency > 0 else 0.0,
        "avg_power_watts": round(stats["mean_w"], 2) if stats["mean_w"] is not None else None,
        "median_power_watts": round(stats["median_w"], 2) if stats["median_w"] is not None else None,
        "min_power_watts": round(stats["min_w"], 2) if stats["min_w"] is not None else None,
        "max_power_watts": round(stats["max_w"], 2) if stats["max_w"] is not None else None,
        "total_energy_joules": round(energy, 4) if energy is not None else None,
        "energy_source": energy_source,
        "energy_j_counter": round(energy_counter, 4) if energy_counter is not None else None,
        "energy_j_sampled": round(energy_sampled, 4) if energy_sampled is not None else None,
        "joules_per_token": round(energy / tokens, 6) if (energy is not None and tokens) else None,
        "temp_start_c": stats["temp_start_c"],
        "temp_end_c": stats["temp_end_c"],
        "temp_max_c": stats["temp_max_c"],
        "sm_clock_mean_mhz": round(stats["sm_clock_mean_mhz"], 1) if stats["sm_clock_mean_mhz"] else None,
        "sm_clock_min_mhz": stats["sm_clock_min_mhz"],
        "mem_clock_mean_mhz": round(stats["mem_clock_mean_mhz"], 1) if stats["mem_clock_mean_mhz"] else None,
        "throttle_reasons": monitor.throttle_reasons(),
        "n_power_samples": stats["n_power_samples"],
        "effective_sample_hz": round(stats["effective_sample_hz"], 1),
    }

    if record:
        exists = os.path.exists(MASTER_CSV)
        with open(MASTER_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if not exists:
                w.writeheader()
            w.writerow(entry)

    return entry


# ============================================================================
# Closed-loop thermal warmup
# ============================================================================

def warm_to_steady_state(model, tokenizer, monitor, encoded, max_new_tokens,
                         min_sec=MIN_WARMUP_SEC, max_sec=MAX_WARMUP_SEC,
                         window=WARMUP_WINDOW, power_tol=WARMUP_POWER_TOL_W,
                         temp_tol=WARMUP_TEMP_TOL_C):
    """Generate continuously until power and temperature stop moving.

    Replaces the fixed 3-prompt throwaway round, which recovered only about a
    third of the cold-start power deficit on the recorded runs.
    """
    print("\n" + "#" * 95)
    print("# CLOSED-LOOP THERMAL WARMUP (discarded, not recorded)")
    print(f"# target: {window} consecutive rounds within {power_tol} W and {temp_tol} C")
    print(f"# floor: {min_sec:.0f} s   cap: {max_sec:.0f} s")
    print("#" * 95)

    history = deque(maxlen=window)
    trace = []
    t_origin = time.perf_counter()
    i = 0

    while True:
        label, ids = encoded[i % len(encoded)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate(model, tokenizer, ids, max_new_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        st = monitor.window_stats(t0, t1)
        elapsed = t1 - t_origin
        pw = st["mean_w"]
        tc = st["temp_end_c"]
        trace.append({
            "iteration": i + 1,
            "prompt_label": label,
            "elapsed_sec": round(elapsed, 2),
            "mean_power_w": round(pw, 2) if pw is not None else None,
            "temp_c": tc,
            "sm_clock_mean_mhz": round(st["sm_clock_mean_mhz"], 1) if st["sm_clock_mean_mhz"] else None,
        })
        print(f"  warmup {i + 1:>3}  {label:<8}  t={elapsed:6.1f}s  "
              f"P={pw:7.2f} W  " if pw is not None else f"  warmup {i + 1:>3}  {label:<8}  P=n/a  ",
              end="")
        print(f"T={tc} C" if tc is not None else "T=n/a")

        if pw is not None and tc is not None:
            history.append((pw, tc))
        i += 1

        if elapsed >= min_sec and len(history) == window:
            ps = [h[0] for h in history]
            ts = [h[1] for h in history]
            if (max(ps) - min(ps)) <= power_tol and (max(ts) - min(ts)) <= temp_tol:
                print(f"[*] Steady state reached after {elapsed:.1f} s "
                      f"({i} warmup generations). Power spread "
                      f"{max(ps) - min(ps):.2f} W, temp spread {max(ts) - min(ts)} C.")
                return {"converged": True, "elapsed_sec": round(elapsed, 2),
                        "iterations": i, "trace": trace}

        if elapsed >= max_sec:
            print(f"[!] Warmup cap of {max_sec:.0f} s hit without convergence.")
            print("    Trials will still run, but check drift_diagnostics in the")
            print("    summary before trusting the means.")
            return {"converged": False, "elapsed_sec": round(elapsed, 2),
                    "iterations": i, "trace": trace}


# ============================================================================
# Drift diagnostics
# ============================================================================

def fit_drift(indices, values):
    """Least-squares slope of values against chronological index."""
    pairs = [(x, y) for x, y in zip(indices, values) if y is not None]
    if len(pairs) < 3:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    span = float(slope) * (x.max() - x.min())
    mean = float(y.mean())
    return {
        "slope_per_trial": float(slope),
        "r2": r2,
        "total_change_over_run": span,
        "fraction_of_mean": span / mean if mean else 0.0,
        "n": len(pairs),
    }


# ============================================================================
# Suite
# ============================================================================

def run_n_trial_suite(num_trials=5, max_new_tokens=250, lock_clocks=False,
                      min_warmup_sec=MIN_WARMUP_SEC, max_warmup_sec=MAX_WARMUP_SEC):
    print("=" * 95)
    print("MATCHED FP16/BF16 BASELINE")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading target model into VRAM (shared across all trials)...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, dtype=DTYPE, device_map="cuda"
    )
    model.eval()

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=POLL_RATE_HZ)
    dev = monitor.device_info()
    print(f"[*] Device: {dev.get('name')}  driver {dev.get('driver_version')}")
    print(f"[*] Power limit: {dev.get('power_limit_w')} W   "
          f"energy counter: {'yes' if dev['energy_counter_supported'] else 'NO (falling back to sampling)'}")
    if lock_clocks:
        monitor.lock_clocks()
    monitor.start()

    encoded = [(label, encode_prompt(tokenizer, text)) for label, text in PROMPTS]
    for label, ids in encoded:
        print(f"[*] {label}: {ids.shape[-1]} prompt tokens")

    summary_path = os.path.join(TELEMETRY_DIR, f"fp16_baseline_n{num_trials}_summary.json")

    try:
        warmup = warm_to_steady_state(
            model, tokenizer, monitor, encoded, max_new_tokens,
            min_sec=min_warmup_sec, max_sec=max_warmup_sec,
        )

        raw = {label: [] for label, _ in PROMPTS}
        chronological = []
        gidx = 0

        for trial_idx in range(1, num_trials + 1):
            print(f"\n{'#' * 95}\n# TRIAL ROUND {trial_idx}/{num_trials}\n{'#' * 95}")
            for label, ids in encoded:
                gidx += 1
                entry = run_trial(
                    model, tokenizer, monitor, ids,
                    prompt_label=f"{label}_trial{trial_idx}",
                    max_new_tokens=max_new_tokens,
                    global_index=gidx,
                )
                print(f"  {label:<8} trial {trial_idx}/{num_trials}  "
                      f"tok/s={entry['throughput_tok_sec']:<7} "
                      f"J/tok={entry['joules_per_token']:.3f}  "
                      f"P={entry['avg_power_watts']} W  T={entry['temp_end_c']} C")
                raw[label].append(entry)
                chronological.append(entry)
    finally:
        monitor.stop()
        monitor.unlock_clocks()

    # -- aggregate ----------------------------------------------------------
    aggregated = {}
    for label, trials in raw.items():
        tps = [t["throughput_tok_sec"] for t in trials]
        jtok = [t["joules_per_token"] for t in trials if t["joules_per_token"] is not None]
        pwr = [t["avg_power_watts"] for t in trials if t["avg_power_watts"] is not None]
        aggregated[label] = {
            "n_trials": len(trials),
            "tps_mean": float(np.mean(tps)),
            "tps_std": float(np.std(tps, ddof=1)) if len(tps) > 1 else 0.0,
            "j_tok_mean": float(np.mean(jtok)) if jtok else None,
            "j_tok_std": float(np.std(jtok, ddof=1)) if len(jtok) > 1 else 0.0,
            "power_mean": float(np.mean(pwr)) if pwr else None,
            "power_std": float(np.std(pwr, ddof=1)) if len(pwr) > 1 else 0.0,
            "prompt_tokens": trials[0]["prompt_tokens"] if trials else None,
            "raw_trials": trials,
        }

    idx = [e["global_index"] for e in chronological]
    drift = {
        "power_w": fit_drift(idx, [e["avg_power_watts"] for e in chronological]),
        "joules_per_token": fit_drift(idx, [e["joules_per_token"] for e in chronological]),
        "throughput_tok_sec": fit_drift(idx, [e["throughput_tok_sec"] for e in chronological]),
        "temp_c": fit_drift(idx, [e["temp_end_c"] for e in chronological]),
    }
    pdrift = drift["power_w"]
    residual_ok = bool(pdrift is not None and abs(pdrift["fraction_of_mean"]) <= DRIFT_WARN_FRACTION)
    drift["residual_drift_acceptable"] = residual_ok
    drift["threshold_fraction"] = DRIFT_WARN_FRACTION

    out = {
        "_meta": {
            "timestamp": datetime.now().isoformat(),
            "device": dev,
            "config": {
                "model": TARGET_MODEL_ID,
                "dtype": str(DTYPE),
                "use_kv_cache": USE_KV_CACHE,
                "stop_on_eos": STOP_ON_EOS,
                "max_new_tokens": max_new_tokens,
                "num_trials": num_trials,
                "warmup_steps_per_trial": WARMUP_STEPS,
                "poll_rate_hz": POLL_RATE_HZ,
                "clocks_locked": monitor._clocks_locked,
                "trial_order": "interleaved round-robin",
            },
            "warmup": warmup,
            "drift_diagnostics": drift,
            "poll_errors": monitor.poll_errors,
        },
        **aggregated,
    }
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)

    monitor.close()

    # -- report -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"N={num_trials} BASELINE COMPLETE -- MEAN +/- STD")
    print(f"{'Prompt':<10} | {'prompt_tok':<10} | {'tok/s':<18} | {'J/tok':<18} | {'W':<10}")
    print("-" * 95)
    for label, a in aggregated.items():
        print(f"{label:<10} | {a['prompt_tokens']:<10} | "
              f"{a['tps_mean']:.2f} +/- {a['tps_std']:.2f}".ljust(33) + " | "
              f"{a['j_tok_mean']:.3f} +/- {a['j_tok_std']:.3f}".ljust(20) + " | "
              f"{a['power_mean']:.1f}")
    print("=" * 95)

    print("\nDRIFT CHECK (chronological, all categories pooled)")
    if pdrift:
        print(f"  power slope       : {pdrift['slope_per_trial']:+.4f} W/trial "
              f"(R^2={pdrift['r2']:.3f}, total {pdrift['total_change_over_run']:+.2f} W = "
              f"{pdrift['fraction_of_mean'] * 100:+.2f}%)")
    jd = drift["joules_per_token"]
    if jd:
        print(f"  J/tok slope       : {jd['slope_per_trial']:+.5f} /trial "
              f"({jd['fraction_of_mean'] * 100:+.2f}% over run)")
    td = drift["temp_c"]
    if td:
        print(f"  temperature slope : {td['slope_per_trial']:+.3f} C/trial")

    if not warmup["converged"]:
        print("\n[!] Warmup did not converge. Raise --max-warmup.")
    if not residual_ok:
        print(f"\n[!] Residual drift exceeds {DRIFT_WARN_FRACTION * 100:.1f}% of mean power.")
        print("    The trial means are contaminated by the settling transient.")
        print("    Raise --min-warmup (try 240) and/or lock clocks, then re-run.")
    else:
        print("\n[ok] Residual drift within threshold. Means are usable.")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Per-trial CSV:    {MASTER_CSV}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Target-only decoding baseline with thermal-aware warmup.")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--min-warmup", type=float, default=MIN_WARMUP_SEC,
                    help="Minimum sustained-load seconds before timed trials start.")
    ap.add_argument("--max-warmup", type=float, default=MAX_WARMUP_SEC,
                    help="Hard cap on warmup; proceeds with a warning if not converged.")
    ap.add_argument("--lock-clocks", action="store_true",
                    help="Attempt NVML clock locking (needs root/elevated privileges -- "
                         "never succeeds under WSL2 without it, which is why this defaults "
                         "to off).")
    args = ap.parse_args()

    run_n_trial_suite(
        num_trials=args.trials,
        max_new_tokens=args.tokens,
        lock_clocks=args.lock_clocks,
        min_warmup_sec=args.min_warmup,
        max_warmup_sec=args.max_warmup,
    )