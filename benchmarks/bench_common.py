"""
bench_common.py

Shared measurement infrastructure for the fixed-k5 benchmark scripts, factored
out of new_fp16_baseline.py (2026-09-03 rewrite) so speculative_scout.py and
benchmark_ablation.py get the same fixes instead of a fourth/fifth hand-copied
version of the same NVML code drifting out of sync with each other. See
new_fp16_baseline.py's module docstring for the full diagnosis of the problem
this solves (GPU thermal/clock settling transient, ~40s time constant).

Provides:
  - NVMLPowerMonitor: background power/temp/clock sampler, energy from the
    hardware counter (nvmlDeviceGetTotalEnergyConsumption) with sampled
    trapezoidal integration as a cross-check/fallback.
  - encode_prompt: chat-template-based prompt encoding (matches
    benchmark_ablation.py / new_fp16_baseline.py; fixes the raw-untemplated-
    prompt issue in the old speculative_scout.py).
  - warm_to_steady_state: closed-loop thermal warmup. Generic over the
    workload -- pass in a zero-arg callable that does one unit of GPU work
    and internally calls torch.cuda.synchronize() before returning; this
    function handles timing, power/temp sampling, convergence checking, and
    the discarded warmup trace.
  - fit_drift: least-squares drift diagnostics (slope, R^2, % of mean) for a
    chronological sequence of measurements.
  - json_safe: default= handler for json.dump that fixes the numpy.bool_ /
    numpy.integer / numpy.floating serialization crash found and fixed in
    new_fp16_baseline.py on 2026-09-03 (the fraction_of_mean comparison in
    fit_drift produces numpy scalars despite the float() cast on slope).
  - safe_append_csv: appends a dict row to a CSV, but if the file already
    exists with a *different* column set (e.g. an older, narrower schema),
    renames the old file to "<name>_legacy.csv" first instead of silently
    writing misaligned rows under a stale header. This is the exact bug
    (csv.DictWriter only writes a header for brand-new files) that was found
    and fixed by hand in new_fp16_baseline.py on 2026-09-03; this helper
    makes the fix apply automatically to every script that uses it.
"""

import csv
import os
import threading
import time
from collections import deque

import numpy as np
import pynvml
import torch

# ============================================================================
# Closed-loop warmup defaults (from new_fp16_baseline.py's measured ~40s
# thermal time constant -- see that file's docstring for the derivation)
# ============================================================================
MIN_WARMUP_SEC = 120.0
MAX_WARMUP_SEC = 420.0
WARMUP_WINDOW = 4
WARMUP_POWER_TOL_W = 1.5
WARMUP_TEMP_TOL_C = 1.0

DRIFT_WARN_FRACTION = 0.005  # 0.5%

POLL_RATE_HZ = 100
TEMP_POLL_EVERY = 10


# ============================================================================
# Reference run settings -- single source of truth for cross-script parity
# ============================================================================
# The whole point of comparing fp16_baseline / benchmark_ablation /
# speculative_scout numbers against each other is that they're the same
# experiment run under different conditions. That only holds if K, token
# count, and the three reference prompts are literally the same values in
# each script, not independently-typed copies that can drift apart (this is
# the same duplication risk flagged elsewhere in this file, applied to
# experiment parameters instead of monitoring code). Every script should
# import these rather than hardcoding its own copies.

REFERENCE_MAX_TOKENS = 250
REFERENCE_K = 5
REFERENCE_WARMUP_STEPS = 5  # per-trial in-trial warmup, where applicable

REFERENCE_PROMPT_LABELS = ["Poem", "Physics", "Code"]
REFERENCE_PROMPTS = [
    "Write an original Chant Royal poem in English celebrating mathematics.",
    "Explain the physics of semiconductor memory bandwidth and the memory wall.",
    "Write a Python implementation of a binary search tree with type annotations.",
]
REFERENCE_PROMPTS_BY_LABEL = dict(zip(REFERENCE_PROMPT_LABELS, REFERENCE_PROMPTS))


# ============================================================================
# NVML monitor
# ============================================================================

class NVMLPowerMonitor:
    """Background NVML sampler: power at POLL_RATE_HZ, temp/clocks at a lower
    rate. Energy comes from the hardware energy counter when available (see
    read_energy_j); integrating sparse instantaneous power samples is biased,
    so that is retained only as a cross-check/fallback (window_stats).
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
            pass
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
                next_deadline = time.perf_counter()

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
    """Trapezoidal integral of a (time, value) series over [t0, t1], with
    boundary interpolation so partial edge intervals aren't dropped."""
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
# Prompt encoding (chat template)
# ============================================================================

def encode_prompt(tokenizer, prompt, device="cuda"):
    """Encode a user-turn prompt via the chat template. Matches
    benchmark_ablation.py / new_fp16_baseline.py. Using the raw prompt string
    directly (no chat template) understates the real prompt-token count and
    was a confirmed mismatch in the old speculative_scout.py."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt").to(device)["input_ids"]


# ============================================================================
# Closed-loop thermal warmup (generic over workload)
# ============================================================================

def warm_to_steady_state(monitor, step_fn, min_sec=MIN_WARMUP_SEC, max_sec=MAX_WARMUP_SEC,
                          window=WARMUP_WINDOW, power_tol=WARMUP_POWER_TOL_W,
                          temp_tol=WARMUP_TEMP_TOL_C):
    """Run step_fn() repeatedly (discarded, not recorded) until power AND
    temperature stop moving, or max_sec is hit.

    step_fn(iteration_index) must do one unit of GPU work and return a short
    label string for logging. It must call torch.cuda.synchronize() itself
    before returning if the work is async (this function synchronizes again
    around the call regardless, so it's safe either way).

    Replaces a fixed-length throwaway round, which on this project's hardware
    (see new_fp16_baseline.py) recovered only ~1/3 of the cold-start power
    deficit. Generic across workloads (target-only decode, speculative
    scout+target cycles, etc.) since the thermal transient is a GPU-wide
    property of sustained load, not specific to any one kernel mix.

    Convergence check design (fixed 2026-09-0X): temperature is tracked as a
    single rolling window across ALL iterations, since it's a slow,
    workload-independent signal -- the physical driver of the settling
    transient is die temperature, not which specific kernel mix is running.
    Power is tracked per DISTINCT LABEL instead, in separate rolling windows
    keyed by whatever string step_fn returns. This matters when step_fn
    alternates between workloads with genuinely different steady-state power
    draw by design -- e.g. benchmark_ablation.py's warmup alternates baseline
    and speculative decode, which differ in power by tens of watts as a real
    effect, not noise. The original design checked whether N *consecutive*
    readings (regardless of label) agreed with each other; with two
    interleaved workloads that differ by design, consecutive readings almost
    never agree, so that check could run indefinitely without ever
    converging. Checking whether each label's own readings have stopped
    moving, relative to its own recent history, is correct for both the
    homogeneous case (fp16_baseline / speculative_scout, one workload type,
    only the prompt varies) and the heterogeneous case (benchmark_ablation,
    two workload types by design).
    """
    print("\n" + "#" * 95)
    print("# CLOSED-LOOP THERMAL WARMUP (discarded, not recorded)")
    print(f"# target: temp stable within {temp_tol} C over last {window} iterations (any label);")
    print(f"# each distinct label's own power stable within {power_tol} W over its last {window} readings")
    print(f"# floor: {min_sec:.0f} s   cap: {max_sec:.0f} s")
    print("#" * 95)

    temp_history = deque(maxlen=window)
    power_history_by_label = {}
    trace = []
    t_origin = time.perf_counter()
    i = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        label = step_fn(i)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        st = monitor.window_stats(t0, t1)
        elapsed = t1 - t_origin
        pw = st["mean_w"]
        tc = st["temp_end_c"]
        trace.append({
            "iteration": i + 1,
            "label": label,
            "elapsed_sec": round(elapsed, 2),
            "mean_power_w": round(pw, 2) if pw is not None else None,
            "temp_c": tc,
            "sm_clock_mean_mhz": round(st["sm_clock_mean_mhz"], 1) if st["sm_clock_mean_mhz"] else None,
        })
        pw_str = f"P={pw:7.2f} W  " if pw is not None else "P=n/a         "
        tc_str = f"T={tc} C" if tc is not None else "T=n/a"
        print(f"  warmup {i + 1:>3}  {str(label):<12}  t={elapsed:6.1f}s  {pw_str}{tc_str}")

        if tc is not None:
            temp_history.append(tc)
        if pw is not None:
            power_history_by_label.setdefault(label, deque(maxlen=window)).append(pw)
        i += 1

        if elapsed >= min_sec:
            temp_ok = (len(temp_history) == window
                       and (max(temp_history) - min(temp_history)) <= temp_tol)
            label_spreads = {
                lbl: (max(h) - min(h)) for lbl, h in power_history_by_label.items() if len(h) == window
            }
            power_ok = bool(power_history_by_label) and len(label_spreads) == len(power_history_by_label) \
                and all(spread <= power_tol for spread in label_spreads.values())

            if temp_ok and power_ok:
                temp_spread = max(temp_history) - min(temp_history)
                spreads_str = ", ".join(f"{lbl}={spread:.2f}W" for lbl, spread in label_spreads.items())
                print(f"[*] Steady state reached after {elapsed:.1f} s "
                      f"({i} warmup iterations). Temp spread {temp_spread} C. "
                      f"Per-label power spreads: {spreads_str}")
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
    span = float(slope) * float(x.max() - x.min())
    mean = float(y.mean())
    return {
        "slope_per_trial": float(slope),
        "r2": float(r2),
        "total_change_over_run": float(span),
        "fraction_of_mean": float(span / mean) if mean else 0.0,
        "n": len(pairs),
    }


def drift_verdict(power_drift, warn_fraction=DRIFT_WARN_FRACTION):
    """bool(...)-wrapped verdict -- see json_safe below for why this matters:
    numpy comparisons here produce numpy.bool_, which crashes json.dump
    without either this wrapper or the json_safe default handler."""
    return bool(power_drift is not None and abs(power_drift["fraction_of_mean"]) <= warn_fraction)


# ============================================================================
# JSON-safe numpy handling
# ============================================================================

def json_safe(obj):
    """default= handler for json.dump. Fixes the crash found in
    new_fp16_baseline.py on 2026-09-03: fit_drift's span = float(slope) *
    (x.max() - x.min()) still produces a numpy.float64 (x.max()/x.min() are
    numpy scalars regardless of the float() cast on slope), and that
    numpy-ness propagates through comparisons into numpy.bool_, which the
    stdlib json module cannot serialize. Using this as json.dump(..., 
    default=json_safe) is a second line of defense on top of explicit
    bool()/float() wrapping at the call site -- keep both, since the call-site
    wrapping is what actually caught this the first time."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ============================================================================
# Speculative scout->target decode (shared accept/reject loop)
# ============================================================================
#
# Both benchmark_ablation.py's generate_speculative_exact() and the old
# speculative_scout.py implemented this loop by hand, identically. That's the
# same "N independent hand-copied implementations drift apart" pattern
# already documented elsewhere in this project (see the status file's note on
# four separate SchmittTrigger*Clutch classes) -- factored here once so a fix
# in the accept/reject logic can't happen in one copy and not the other.

def eos_ids(tokenizer):
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


def speculative_generate(target_model, scout_model, input_ids, K, max_tokens, eos=None):
    """Lossless scout(draft)->target(verify) speculative decoding, fixed K.

    Greedy on both sides: scout proposes K candidate tokens one at a time,
    target verifies all K in a single forward pass, tokens are accepted while
    target's argmax matches the scout's draft and rejected (replaced by
    target's own choice) at the first mismatch; a fully-accepted cycle gets a
    bonus token from target's next-position logit "for free" in the same
    verify pass. This is exact/lossless: the accepted token sequence is
    always identical to what target alone would have produced greedily.

    Returns a dict with generated_ids (list[int], length <= max_tokens),
    current_ids (the full running sequence tensor, prompt + generated),
    total_drafted, total_accepted, cycles.
    """
    current_ids = input_ids
    generated = []
    total_drafted = 0
    total_accepted = 0
    cycle_count = 0

    with torch.inference_mode():
        while len(generated) < max_tokens:
            cycle_count += 1
            draft_ids = current_ids.clone()
            draft_tokens = []

            for _ in range(K):
                scout_out = scout_model(draft_ids)
                next_draft_tok = torch.argmax(scout_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_draft_tok)
                draft_ids = torch.cat([draft_ids, next_draft_tok], dim=-1)

            total_drafted += K
            target_out = target_model(draft_ids)
            target_logits = target_out.logits
            prefix_len = current_ids.shape[1]

            accepted_in_cycle = 0
            new_accepted_ids = []

            for i in range(K):
                expected_token = torch.argmax(target_logits[:, prefix_len - 1 + i, :], dim=-1, keepdim=True)
                drafted_token = draft_tokens[i]
                if expected_token.item() == drafted_token.item():
                    accepted_in_cycle += 1
                    new_accepted_ids.append(drafted_token)
                else:
                    new_accepted_ids.append(expected_token)
                    break
            else:
                bonus_token = torch.argmax(target_logits[:, prefix_len - 1 + K, :], dim=-1, keepdim=True)
                new_accepted_ids.append(bonus_token)

            total_accepted += accepted_in_cycle
            accepted_tensor = torch.cat(new_accepted_ids, dim=-1)
            current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
            new_ids_list = accepted_tensor.squeeze(0).tolist()
            if isinstance(new_ids_list, int):
                new_ids_list = [new_ids_list]
            generated.extend(new_ids_list)

            if eos and any(t in eos for t in new_ids_list):
                break

    return {
        "generated_ids": generated[:max_tokens],
        "current_ids": current_ids,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "cycles": cycle_count,
    }


def target_only_generate(model, input_ids, max_tokens, eos=None):
    """Plain greedy target-only decode (the FP16/BF16 baseline). Same
    full-reforward-every-step method as benchmark_ablation.py's old
    generate_baseline() -- no KV cache, matching the rest of this repo's
    scripts (see new_fp16_baseline.py's PARITY WARNING for why that matters).

    Returns dict with generated_ids and current_ids, for symmetry with
    speculative_generate().
    """
    current_ids = input_ids
    generated = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            out = model(current_ids)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = next_token.item()
            generated.append(token_id)
            if eos and token_id in eos:
                break
            current_ids = torch.cat([current_ids, next_token], dim=-1)
    return {"generated_ids": generated, "current_ids": current_ids}


# ============================================================================
# CSV schema-collision guard
# ============================================================================

def safe_append_csv(path, entry):
    """Append entry (a flat dict) to the CSV at path as a new row.

    If the file already exists but its header doesn't match entry's keys
    (an older/narrower schema, e.g. from a script version predating some
    fields), the old file is renamed to '<name>_legacy<ext>' first and a
    fresh file is started with the new header. This is exactly the bug found
    by hand in new_fp16_baseline.py on 2026-09-03: csv.DictWriter only writes
    a header row for brand-new files, so appending a wider schema onto an
    old-headered file silently misaligns every subsequent row under the
    wrong column names (a real example: the old header's 'avg_power_watts'
    column ended up containing the string "False", from a new schema's
    'hit_eos' field, for every affected row).
    """
    new_fields = list(entry.keys())
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = None
        if existing_header is not None and existing_header != new_fields:
            root, ext = os.path.splitext(path)
            legacy_path = f"{root}_legacy{ext}"
            if os.path.exists(legacy_path):
                # Don't clobber a previous legacy file; make it uniquely named.
                i = 2
                while os.path.exists(f"{root}_legacy{i}{ext}"):
                    i += 1
                legacy_path = f"{root}_legacy{i}{ext}"
            os.rename(path, legacy_path)
            print(f"[!] CSV schema at {path} didn't match this script's columns.")
            print(f"    Renamed old file to {legacy_path} and started a fresh one.")

    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fields)
        if not exists:
            w.writeheader()
        w.writerow(entry)
