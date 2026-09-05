"""
benchmark_ablation.py

N-trial baseline-vs-speculative ablation, fixed K=5, three reference prompts.
This is the source of the paper-worthy numbers: up to 1.82x speedup and
32.6-60.3% energy reduction vs. FP16 baseline, 100% output fidelity,
N=10 trials/prompt. See README.md for the exact reported results.

============================================================================
CHANGES (2026-09-0X): ported the new_fp16_baseline.py fixes
============================================================================
Prior versions of this script (see the 2026-09-02 warmup fix noted below)
still had a structural confound the status file flagged as a priority: for
each prompt, all NUM_TRIALS baseline trials ran before any speculative
trials did. That means baseline was systematically measured during the
"cooler/earlier" part of each prompt's block and speculative during the
"warmer/later" part -- exactly the kind of GPU thermal settling transient
diagnosed and fixed in new_fp16_baseline.py (~40s time constant, ~120s to
steady state). A speed/power difference produced partly by that transient
would show up as though it were a property of the two decoding methods.

Fixed here by pulling in bench_common.py (shared with speculative_scout.py,
factored out of new_fp16_baseline.py):
  1. Closed-loop thermal warmup (bench_common.warm_to_steady_state) before
     any timed trial, alternating baseline/speculative warmup work so both
     code paths are hot (kernels compiled, caches primed) before the timed
     region starts. This replaces relying on the per-trial 5-step warmup
     alone to compensate for the multi-minute settling transient.
  2. Fully interleaved trial order: each trial round runs Poem, Physics,
     Code, and for each prompt, baseline and speculative back-to-back --
     alternating which of the pair goes first each round, so neither
     condition is systematically first (and therefore systematically
     cooler) across the run. This is the direct fix for the confound above.
  3. Energy from the NVML hardware energy counter, with the old
     avg_power_watts * elapsed method kept as a recorded cross-check.
  4. Per-trial thermal telemetry (temperature, clocks, throttle reasons)
     written to a new per-trial CSV (telemetry_ablation.csv), and drift
     diagnostics (least-squares slope of power/J-per-token/throughput
     against chronological trial index) computed per condition and pooled,
     the same way new_fp16_baseline.py does -- so a bad run is visible in
     the numbers themselves instead of assumed clean.
  5. bench_common.safe_append_csv / json_safe guard against the CSV
     schema-collision and numpy.bool_ JSON-serialization bugs found and
     fixed in new_fp16_baseline.py on 2026-09-03.

ablation_results.json's top-level shape (fidelity_by_prompt + ablation dict
keyed "FP16 Baseline" / "Speculative (1B->8B)") is unchanged, so
plot_ablation_results.py keeps working without modification. The new
drift/warmup/device info is added under a new "_meta" key that the plot
script doesn't read.

Methodology note (2026-09-02, retained from the previous version): both
condition functions run WARMUP_STEPS untimed forward passes immediately
before each timed measurement, matching fp16_baseline.py's per-trial warmup
procedure. This is on top of, not instead of, the closed-loop warmup added
above -- the per-trial warmup avoids each individual timed window starting
from an idle SM; the closed-loop warmup handles the much slower multi-minute
thermal settling transient.
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import bench_common

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SCOUT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda:0"
NUM_TRIALS = 10
MAX_TOKENS = bench_common.REFERENCE_MAX_TOKENS
K_DRAFT = bench_common.REFERENCE_K
WARMUP_STEPS = bench_common.REFERENCE_WARMUP_STEPS  # per-trial in-trial warmup
# Closed-loop warmup burst length. Matches MAX_TOKENS (not a shorter
# throwaway value) -- see the 2026-09-0X thermal-fix note added to
# speculative_scout.py: warmup bursts shorter than the real trial length can
# report "converged" at a power/temp level the real, longer trial then
# exceeds, because the GPU takes real time to ramp to sustained power once a
# burst starts. Confirmed empirically on this hardware (WSL2, RTX 5090):
# 15-token warmup bursts converged at ~220W, but the first real 250-token
# trial still rose 8C mid-measurement and averaged ~311W.
WARMUP_UNIT_TOKENS = MAX_TOKENS

PROMPT_LABELS = bench_common.REFERENCE_PROMPT_LABELS
PROMPTS = bench_common.REFERENCE_PROMPTS

BASELINE_KEY = "FP16 Baseline"
SPEC_KEY = "Speculative (1B->8B)"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(TELEMETRY_DIR, "ablation_results.json")
TRIAL_CSV = os.path.join(TELEMETRY_DIR, "telemetry_ablation.csv")


# ============================================================================
# Timed trial (either condition)
# ============================================================================

def run_condition_trial(condition, target_model, scout_model, tokenizer, monitor,
                         input_ids, prompt_label, round_idx, max_tokens, eos,
                         warmup_steps=WARMUP_STEPS, K=K_DRAFT, global_index=None,
                         record=True):
    """Run one timed trial of either 'baseline' or 'speculative'.

    Mirrors new_fp16_baseline.py's run_trial(): short in-trial warmup (not
    timed), then a timed window with energy read from the NVML counter and
    full thermal telemetry from monitor.window_stats.
    """
    total_drafted = total_accepted = 0

    with torch.inference_mode():
        if condition == "baseline":
            warm = input_ids.clone()
            for _ in range(warmup_steps):
                o = target_model(warm)
                warm = torch.cat([warm, torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)], dim=-1)
        else:
            warm = input_ids.clone()
            for _ in range(warmup_steps):
                o = scout_model(warm)
                warm = torch.cat([warm, torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)], dim=-1)
            _ = target_model(warm)  # also warm target on a multi-token suffix, like a real verify pass
    torch.cuda.synchronize()

    e_start = monitor.read_energy_j()
    t_start = time.perf_counter()

    if condition == "baseline":
        result = bench_common.target_only_generate(target_model, input_ids, max_tokens, eos=eos)
    else:
        result = bench_common.speculative_generate(target_model, scout_model, input_ids, K, max_tokens, eos=eos)
        total_drafted = result["total_drafted"]
        total_accepted = result["total_accepted"]

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    e_end = monitor.read_energy_j()

    stats = monitor.window_stats(t_start, t_end)
    latency = t_end - t_start
    tokens = len(result["generated_ids"])

    energy_counter = (e_end - e_start) if (e_start is not None and e_end is not None) else None
    energy_sampled = stats["energy_j_sampled"]
    energy = energy_counter if energy_counter is not None else energy_sampled
    energy_source = "nvml_counter" if energy_counter is not None else "sampled_trapezoid"

    accept_rate = (total_accepted / total_drafted * 100.0) if total_drafted > 0 else None

    entry = {
        "timestamp": datetime.now().isoformat(),
        "condition": BASELINE_KEY if condition == "baseline" else SPEC_KEY,
        "prompt_label": prompt_label,
        "round": round_idx,
        "global_index": global_index,
        "tokens": tokens,
        "prompt_tokens": int(input_ids.shape[-1]),
        "latency_sec": round(latency, 6),
        "throughput_tok_sec": round(tokens / latency, 2) if latency > 0 else 0.0,
        "avg_power_watts": round(stats["mean_w"], 2) if stats["mean_w"] is not None else None,
        "min_power_watts": round(stats["min_w"], 2) if stats["min_w"] is not None else None,
        "max_power_watts": round(stats["max_w"], 2) if stats["max_w"] is not None else None,
        "total_energy_joules": round(energy, 4) if energy is not None else None,
        "energy_source": energy_source,
        "energy_j_counter": round(energy_counter, 4) if energy_counter is not None else None,
        "energy_j_sampled": round(energy_sampled, 4) if energy_sampled is not None else None,
        "joules_per_token": round(energy / tokens, 6) if (energy is not None and tokens) else None,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accept_rate_pct": round(accept_rate, 2) if accept_rate is not None else None,
        "temp_start_c": stats["temp_start_c"],
        "temp_end_c": stats["temp_end_c"],
        "temp_max_c": stats["temp_max_c"],
        "sm_clock_mean_mhz": round(stats["sm_clock_mean_mhz"], 1) if stats["sm_clock_mean_mhz"] else None,
        "mem_clock_mean_mhz": round(stats["mem_clock_mean_mhz"], 1) if stats["mem_clock_mean_mhz"] else None,
        "throttle_reasons": monitor.throttle_reasons(),
        "n_power_samples": stats["n_power_samples"],
        "effective_sample_hz": round(stats["effective_sample_hz"], 1),
        "generated_ids": result["generated_ids"],  # only used for fidelity check; not written to CSV
    }

    if record:
        csv_entry = {k: v for k, v in entry.items() if k != "generated_ids"}
        bench_common.safe_append_csv(TRIAL_CSV, csv_entry)

    return entry


# ============================================================================
# Suite
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=NUM_TRIALS)
    ap.add_argument("--tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--min-warmup", type=float, default=bench_common.MIN_WARMUP_SEC)
    ap.add_argument("--max-warmup", type=float, default=bench_common.MAX_WARMUP_SEC)
    ap.add_argument("--lock-clocks", action="store_true",
                     help="Attempt NVML clock locking (needs root/elevated privileges -- "
                          "never succeeds under WSL2 without it, which is why this defaults "
                          "to off).")
    args = ap.parse_args()

    num_trials = args.trials
    max_tokens = args.tokens

    monitor = bench_common.NVMLPowerMonitor(device_index=0)
    dev = monitor.device_info()
    print("=" * 85)
    print(f"[*] SPECULATIVE DECODING ACADEMIC BENCHMARK ({dev.get('name')})")
    print(f"[*] Target: {TARGET_MODEL_ID} | Scout: {SCOUT_MODEL_ID} | Trials: N={num_trials}")
    print(f"[*] Power limit: {dev.get('power_limit_w')} W   "
          f"energy counter: {'yes' if dev['energy_counter_supported'] else 'NO (falling back to sampling)'}")
    print("=" * 85)

    if args.lock_clocks:
        monitor.lock_clocks()
    monitor.start()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = bench_common.eos_ids(tokenizer)

    print("\n[*] Loading Target Model (8B)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE
    )
    target_model.eval()

    print("[*] Loading Scout Model (1B)...")
    scout_model = AutoModelForCausalLM.from_pretrained(
        SCOUT_MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE
    )
    scout_model.eval()

    encoded = [(label, bench_common.encode_prompt(tokenizer, text, device=DEVICE))
               for label, text in zip(PROMPT_LABELS, PROMPTS)]
    for label, ids in encoded:
        print(f"[*] {label}: {ids.shape[-1]} prompt tokens")

    try:
        # -- Closed-loop thermal warmup, alternating baseline/speculative work --
        def warmup_step(i):
            label, ids = encoded[i % len(encoded)]
            if i % 2 == 0:
                bench_common.target_only_generate(target_model, ids.clone(), WARMUP_UNIT_TOKENS, eos=eos)
                return f"{label}/base"
            else:
                bench_common.speculative_generate(target_model, scout_model, ids.clone(),
                                                    K_DRAFT, WARMUP_UNIT_TOKENS, eos=eos)
                return f"{label}/spec"

        warmup = bench_common.warm_to_steady_state(
            monitor, warmup_step, min_sec=args.min_warmup, max_sec=args.max_warmup,
        )

        # -- TEST 1: Exact Lossless Fidelity ---------------------------------
        print("\n[1/2] Verifying Exact Lossless Fidelity (Baseline vs. Speculative)...")
        fidelity_by_prompt = {}
        for p_idx, (label, ids) in enumerate(encoded, 1):
            base = run_condition_trial("baseline", target_model, scout_model, tokenizer, monitor,
                                        ids, f"{label}_fidelity", 0, max_tokens, eos,
                                        global_index=None, record=False)
            spec = run_condition_trial("speculative", target_model, scout_model, tokenizer, monitor,
                                        ids, f"{label}_fidelity", 0, max_tokens, eos,
                                        global_index=None, record=False)
            base_tok, spec_tok = base["generated_ids"], spec["generated_ids"]
            match_count = sum(1 for a, b in zip(base_tok, spec_tok) if a == b)
            total_tok = max(len(base_tok), len(spec_tok))
            match_pct = (match_count / total_tok) * 100.0 if total_tok > 0 else 0.0
            fidelity_by_prompt[f"prompt_{p_idx}"] = match_pct
            print(f"  Prompt {p_idx} ({label}): {match_pct:.1f}% exact token match ({len(base_tok)} tokens)")

        agg_fidelity = float(np.mean(list(fidelity_by_prompt.values())))
        print(f"[*] Aggregate Fidelity Match: {agg_fidelity:.2f}%")

        # -- TEST 2: Multi-trial, fully interleaved --------------------------
        print(f"\n[2/2] Running Multi-Trial Performance Benchmark (interleaved order)...")
        by_prompt_condition = {label: {"baseline": [], "speculative": []} for label, _ in encoded}
        chronological = []
        gidx = 0

        for round_idx in range(1, num_trials + 1):
            print(f"\n{'#' * 85}\n# TRIAL ROUND {round_idx}/{num_trials}\n{'#' * 85}")
            # Alternate which condition goes first each round so neither one
            # is systematically first (and therefore systematically cooler).
            order = ["baseline", "speculative"] if round_idx % 2 == 1 else ["speculative", "baseline"]
            for label, ids in encoded:
                for condition in order:
                    gidx += 1
                    entry = run_condition_trial(
                        condition, target_model, scout_model, tokenizer, monitor,
                        ids, label, round_idx, max_tokens, eos, global_index=gidx,
                    )
                    acc_str = f"  accept={entry['accept_rate_pct']:.1f}%" if entry["accept_rate_pct"] is not None else ""
                    print(f"  {label:<8} {entry['condition']:<22} round {round_idx}/{num_trials}  "
                          f"tok/s={entry['throughput_tok_sec']:<7} "
                          f"J/tok={entry['joules_per_token']}  "
                          f"P={entry['avg_power_watts']} W  T={entry['temp_end_c']} C{acc_str}")
                    by_prompt_condition[label]["baseline" if condition == "baseline" else "speculative"].append(entry)
                    chronological.append(entry)
    finally:
        monitor.stop()
        monitor.unlock_clocks()

    # -- aggregate, matching the existing ablation_results.json schema ------
    ablation = {}
    for p_idx, (label, _) in enumerate(encoded, 1):
        p_key = f"prompt_{p_idx}"
        ablation[p_key] = {}

        base_trials = by_prompt_condition[label]["baseline"]
        tps_b = [t["throughput_tok_sec"] for t in base_trials]
        j_b = [t["joules_per_token"] for t in base_trials if t["joules_per_token"] is not None]
        pwr_b = [t["avg_power_watts"] for t in base_trials if t["avg_power_watts"] is not None]
        ablation[p_key][BASELINE_KEY] = {
            "tps_mean": float(np.mean(tps_b)),
            "tps_std": float(np.std(tps_b)),
            "j_tok_mean": float(np.mean(j_b)) if j_b else None,
            "j_tok_std": float(np.std(j_b)) if j_b else 0.0,
            "power_mean": float(np.mean(pwr_b)) if pwr_b else None,
        }

        spec_trials = by_prompt_condition[label]["speculative"]
        tps_s = [t["throughput_tok_sec"] for t in spec_trials]
        j_s = [t["joules_per_token"] for t in spec_trials if t["joules_per_token"] is not None]
        pwr_s = [t["avg_power_watts"] for t in spec_trials if t["avg_power_watts"] is not None]
        acc_s = [t["accept_rate_pct"] for t in spec_trials if t["accept_rate_pct"] is not None]
        ablation[p_key][SPEC_KEY] = {
            "tps_mean": float(np.mean(tps_s)),
            "tps_std": float(np.std(tps_s)),
            "j_tok_mean": float(np.mean(j_s)) if j_s else None,
            "j_tok_std": float(np.std(j_s)) if j_s else 0.0,
            "power_mean": float(np.mean(pwr_s)) if pwr_s else None,
            "accept_rate_pct_mean": float(np.mean(acc_s)) if acc_s else None,
        }

    # -- drift diagnostics: pooled and per condition -------------------------
    def drift_block(entries):
        idx = [e["global_index"] for e in entries]
        d = {
            "power_w": bench_common.fit_drift(idx, [e["avg_power_watts"] for e in entries]),
            "joules_per_token": bench_common.fit_drift(idx, [e["joules_per_token"] for e in entries]),
            "throughput_tok_sec": bench_common.fit_drift(idx, [e["throughput_tok_sec"] for e in entries]),
            "temp_c": bench_common.fit_drift(idx, [e["temp_end_c"] for e in entries]),
        }
        d["residual_drift_acceptable"] = bench_common.drift_verdict(d["power_w"])
        d["threshold_fraction"] = bench_common.DRIFT_WARN_FRACTION
        return d

    baseline_entries = [e for e in chronological if e["condition"] == BASELINE_KEY]
    spec_entries = [e for e in chronological if e["condition"] == SPEC_KEY]
    drift = {
        "pooled": drift_block(chronological),
        "baseline_only": drift_block(baseline_entries),
        "speculative_only": drift_block(spec_entries),
    }

    print("\n" + "=" * 85)
    for p_key, m_dict in ablation.items():
        print(f"\n-- {p_key} --")
        print(f"{'Mode':<26} | {'Throughput (tok/s)':<20} | {'Energy (J/token)':<18} | {'Accept %':<10}")
        print("-" * 80)
        for m, st in m_dict.items():
            tps_str = f"{st['tps_mean']:.2f} ± {st['tps_std']:.2f}"
            j_str = f"{st['j_tok_mean']:.3f} ± {st['j_tok_std']:.3f}" if st["j_tok_mean"] is not None else "n/a"
            acc_str = f"{st['accept_rate_pct_mean']:.1f}%" if st.get("accept_rate_pct_mean") is not None else "n/a"
            print(f"{m:<26} | {tps_str:<20} | {j_str:<18} | {acc_str:<10}")
    print("=" * 85)

    pdrift = drift["pooled"]["power_w"]
    print("\nDRIFT CHECK (chronological, pooled across both conditions)")
    if pdrift:
        print(f"  power slope       : {pdrift['slope_per_trial']:+.4f} W/trial "
              f"(R^2={pdrift['r2']:.3f}, total {pdrift['total_change_over_run']:+.2f} W = "
              f"{pdrift['fraction_of_mean'] * 100:+.2f}%)")
    if drift["pooled"]["residual_drift_acceptable"]:
        print("[ok] Residual drift within threshold. Means are usable.")
    else:
        print(f"[!] Residual drift exceeds {bench_common.DRIFT_WARN_FRACTION * 100:.1f}% of mean power.")
        print("    Raise --min-warmup and/or lock clocks, then re-run.")
    if not warmup["converged"]:
        print("[!] Warmup did not converge before the cap. Raise --max-warmup.")

    out = {
        "fidelity_by_prompt": fidelity_by_prompt,
        "ablation": ablation,
        "_meta": {
            "timestamp": datetime.now().isoformat(),
            "device": dev,
            "config": {
                "target_model": TARGET_MODEL_ID,
                "scout_model": SCOUT_MODEL_ID,
                "K": K_DRAFT,
                "max_tokens": max_tokens,
                "num_trials": num_trials,
                "warmup_steps_per_trial": WARMUP_STEPS,
                "clocks_locked": monitor._clocks_locked,
                "trial_order": "interleaved: category round-robin, condition alternated each round",
            },
            "warmup": warmup,
            "drift_diagnostics": drift,
            "poll_errors": monitor.poll_errors,
        },
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=bench_common.json_safe)

    monitor.close()
    print(f"\n[*] Results saved to: {RESULTS_JSON}")
    print(f"[*] Per-trial CSV:    {TRIAL_CSV}")


if __name__ == "__main__":
    main()
