"""
cognitive_benchmark.py

Entropy-gated dynamic INT4/FP16 "Resolution Gear" benchmark: the first real
end-to-end wiring of SDSIE's core gating concept to actual computation,
rather than a logged-but-unused decision. The previous version of this file
computed a gear value every step and counted it, but nothing downstream
ever branched on it -- the exact "logged, not acted on" pattern the status
doc already documents across several other files (sdsie_server.py,
harness_step1.py, four root-folder speculative engines). This version
actually branches.

What this does:
  - Loads Llama-3.1-8B-Instruct once, in bfloat16.
  - Calibrates real per-channel INT4 weights (vllm_sdsie/quantization/
    calibrate.py -- min-max, no calibration dataset needed) for each
    decoder layer's mlp.down_proj, and wraps each one in a GatedLinear
    (vllm_sdsie/quantization/gated_linear.py) that executes either the
    INT4 or the original FP16 path depending on a shared gear flag.
  - Drives that gear flag with the project's real
    SchmittTriggerEntropyClutch -- not a hand-rolled duplicate. The
    previous version of this file had its own independent clutch
    implementation (rolling-average + separate hysteresis-margin design,
    theta_low=1.0/theta_high=2.5), a fifth copy of the same
    already-documented duplication problem.
  - Compares against a genuine FP16-only baseline: the SAME model, same
    GatedLinear layers, gear permanently forced off. Verified in
    gated_linear.py's own self-test to be bit-exact to an unpatched layer,
    so this needs no second model copy in VRAM.

HONESTY NOTE: this is NOT lossless, unlike this project's speculative
decoding work. INT4-quantizing live weights changes numerical output
whenever that path is used, which can and does sometimes change which
token gets emitted. This script measures and reports a real
fidelity/divergence number against the FP16 baseline -- it does not claim
exact match the way the speculative-decoding benchmarks correctly do.

Also fixes the previous version's measurement-methodology gaps against the
rest of this project: different (non-reference) prompts, max_tokens=75
instead of 250, no warmup at all, and energy via avg_power*duration instead
of the NVML hardware counter. All now handled via bench_common.py, shared
with the rest of this repo's benchmarks.
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
from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch
from vllm_sdsie.quantization.gated_linear import (
    GearState, install_gated_layers, collect_gate_usage, reset_gate_counters,
)

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0"

# Smaller than the speculative-decoding suite's N=10: this is a first
# working version of a genuinely new benchmark category (gated INT4/FP16
# execution + real fidelity measurement), not yet run enough times to
# justify a larger N. Raise once this has been run and reviewed a few times.
NUM_TRIALS = 5
MAX_TOKENS = bench_common.REFERENCE_MAX_TOKENS
WARMUP_STEPS = bench_common.REFERENCE_WARMUP_STEPS
# Full-length warmup bursts, not a shorter throwaway value -- see this
# session's fix to speculative_scout.py / benchmark_ablation.py for why a
# warmup burst shorter than the real trial length is not safe.
WARMUP_UNIT_TOKENS = MAX_TOKENS

PROMPT_LABELS = bench_common.REFERENCE_PROMPT_LABELS
PROMPTS = bench_common.REFERENCE_PROMPTS

BASELINE_KEY = "FP16 Baseline"
GATED_KEY = "Entropy-Gated (INT4/FP16)"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(TELEMETRY_DIR, "cognitive_benchmark_results.json")
TRIAL_CSV = os.path.join(TELEMETRY_DIR, "telemetry_cognitive.csv")


def gated_generate(model, input_ids, clutch, gear_state, max_tokens, eos=None):
    """Greedy decode where GatedLinear layers switch between INT4 and FP16
    based on gear_state, updated once per token from the model's own output
    entropy via the real SchmittTriggerEntropyClutch.

    One-token lag is inherent and intentional: the gear used to PRODUCE a
    given token's logits was decided from the PREVIOUS token's entropy,
    since a token's entropy can't be known before it's been computed. This
    matches how a real deployed system would have to work.
    """
    current_ids = input_ids
    generated = []
    gear_history = []  # gear active for the forward call that produced each token

    with torch.inference_mode():
        for _ in range(max_tokens):
            gear_history.append(gear_state.active)
            out = model(current_ids)
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            generated.append(token_id)

            active, _h_step, _h_ema = clutch.update_and_decide(logits)
            gear_state.active = active

            if eos and token_id in eos:
                break
            current_ids = torch.cat([current_ids, next_token], dim=-1)

    return {
        "generated_ids": generated,
        "current_ids": current_ids,
        "gear_history": gear_history,
    }


def run_condition_trial(condition, model, monitor, gear_state, clutch,
                         input_ids, prompt_label, round_idx, max_tokens, eos,
                         warmup_steps=WARMUP_STEPS, global_index=None, record=True):
    """Run one timed trial of either 'baseline' (gear forced off, plain
    FP16 decode) or 'gated' (real entropy-driven INT4/FP16 switching).
    """
    # Set gear state for the (untimed) in-trial warmup too, so Triton's
    # INT4 kernel gets its autotuning/compilation exercised before timing
    # starts on a gated trial, matching the same intent as the per-trial
    # warmup elsewhere in this project.
    if condition == "baseline":
        gear_state.active = False
    else:
        gear_state.active = False  # start safe (FP16) -- see GearState docstring
        clutch.reset()
    reset_gate_counters(model)

    with torch.inference_mode():
        warm = input_ids.clone()
        for _ in range(warmup_steps):
            o = model(warm)
            warm = torch.cat([warm, torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)], dim=-1)
    torch.cuda.synchronize()

    # Reset again so the TIMED region's own gate-usage counts and entropy
    # EMA start clean, not polluted by the warmup steps just run above.
    reset_gate_counters(model)
    if condition != "baseline":
        clutch.reset()
        gear_state.active = False  # start safe (FP16) -- see GearState docstring

    e_start = monitor.read_energy_j()
    t_start = time.perf_counter()

    if condition == "baseline":
        result = bench_common.target_only_generate(model, input_ids, max_tokens, eos=eos)
    else:
        result = gated_generate(model, input_ids, clutch, gear_state, max_tokens, eos=eos)

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

    int4_calls, fp16_calls = collect_gate_usage(model)
    total_calls = int4_calls + fp16_calls
    int4_fraction = (int4_calls / total_calls) if total_calls > 0 else None

    entry = {
        "timestamp": datetime.now().isoformat(),
        "condition": BASELINE_KEY if condition == "baseline" else GATED_KEY,
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
        "joules_per_token": round(energy / tokens, 6) if (energy is not None and tokens) else None,
        "int4_layer_calls": int4_calls,
        "fp16_layer_calls": fp16_calls,
        "int4_fraction": round(int4_fraction, 4) if int4_fraction is not None else None,
        "temp_start_c": stats["temp_start_c"],
        "temp_end_c": stats["temp_end_c"],
        "temp_max_c": stats["temp_max_c"],
        "sm_clock_mean_mhz": round(stats["sm_clock_mean_mhz"], 1) if stats["sm_clock_mean_mhz"] else None,
        "throttle_reasons": monitor.throttle_reasons(),
        "n_power_samples": stats["n_power_samples"],
        "generated_ids": result["generated_ids"],  # only used for fidelity check; not written to CSV
    }

    if record:
        csv_entry = {k: v for k, v in entry.items() if k != "generated_ids"}
        bench_common.safe_append_csv(TRIAL_CSV, csv_entry)

    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=NUM_TRIALS)
    ap.add_argument("--tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--min-warmup", type=float, default=bench_common.MIN_WARMUP_SEC)
    ap.add_argument("--max-warmup", type=float, default=bench_common.MAX_WARMUP_SEC)
    ap.add_argument("--lock-clocks", action="store_true",
                     help="Attempt NVML clock locking (needs root/elevated privileges -- "
                          "never succeeds under WSL2 without it, which is why this defaults "
                          "to off rather than the usual --no-lock-clocks opt-out pattern "
                          "used elsewhere in this repo).")
    ap.add_argument("--target-attr", default="down_proj",
                     help="Which mlp projection to gate (default: down_proj).")
    args = ap.parse_args()

    num_trials = args.trials
    max_tokens = args.tokens

    monitor = bench_common.NVMLPowerMonitor(device_index=0)
    dev = monitor.device_info()
    print("=" * 85)
    print(f"[*] ENTROPY-GATED INT4/FP16 COGNITIVE BENCHMARK ({dev.get('name')})")
    print(f"[*] Target: {TARGET_MODEL_ID} | Trials: N={num_trials} | Tokens: {max_tokens}")
    print(f"[*] Gating: mlp.{args.target_attr}")
    print("=" * 85)

    if args.lock_clocks:
        monitor.lock_clocks()
    monitor.start()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = bench_common.eos_ids(tokenizer)

    print("\n[*] Loading model (8B)...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    print(f"[*] Calibrating real per-channel INT4 weights and installing GatedLinear...")
    shared_gear_state = GearState(active=False)
    n_patched = install_gated_layers(model, shared_gear_state, target_attr=args.target_attr)
    print(f"[*] Patched {n_patched} layers (mlp.{args.target_attr}) with real INT4 calibration.")

    clutch = SchmittTriggerEntropyClutch()  # inherits the project's canonical thresholds
    print(f"[*] Clutch: theta_low={clutch.theta_low}, theta_high={clutch.theta_high}, alpha={clutch.alpha}")

    encoded = [(label, bench_common.encode_prompt(tokenizer, text, device=DEVICE))
               for label, text in zip(PROMPT_LABELS, PROMPTS)]
    for label, ids in encoded:
        print(f"[*] {label}: {ids.shape[-1]} prompt tokens")

    try:
        # -- Closed-loop thermal warmup, alternating baseline/gated work ----
        def warmup_step(i):
            label, ids = encoded[i % len(encoded)]
            if i % 2 == 0:
                shared_gear_state.active = False
                bench_common.target_only_generate(model, ids.clone(), WARMUP_UNIT_TOKENS, eos=eos)
                return f"{label}/base"
            else:
                shared_gear_state.active = False  # start safe (FP16) -- see GearState docstring
                clutch.reset()
                gated_generate(model, ids.clone(), clutch, shared_gear_state, WARMUP_UNIT_TOKENS, eos=eos)
                return f"{label}/gated"

        warmup = bench_common.warm_to_steady_state(
            monitor, warmup_step, min_sec=args.min_warmup, max_sec=args.max_warmup,
        )

        # -- Fidelity check: real divergence, not assumed lossless ----------
        print("\n[1/2] Measuring fidelity (FP16 baseline vs. entropy-gated)...")
        fidelity_by_prompt = {}
        divergence_by_prompt = {}
        for p_idx, (label, ids) in enumerate(encoded, 1):
            base = run_condition_trial("baseline", model, monitor, shared_gear_state, clutch,
                                        ids, f"{label}_fidelity", 0, max_tokens, eos,
                                        global_index=None, record=False)
            gated = run_condition_trial("gated", model, monitor, shared_gear_state, clutch,
                                         ids, f"{label}_fidelity", 0, max_tokens, eos,
                                         global_index=None, record=False)
            base_tok, gated_tok = base["generated_ids"], gated["generated_ids"]
            common_len = min(len(base_tok), len(gated_tok))
            match_count = sum(1 for a, b in zip(base_tok[:common_len], gated_tok[:common_len]) if a == b)
            match_pct = (match_count / common_len) * 100.0 if common_len > 0 else 0.0
            first_divergence = next((i for i in range(common_len) if base_tok[i] != gated_tok[i]), None)
            fidelity_by_prompt[f"prompt_{p_idx}"] = match_pct
            divergence_by_prompt[f"prompt_{p_idx}"] = first_divergence
            div_str = f"first divergence at token {first_divergence}" if first_divergence is not None else "no divergence"
            print(f"  Prompt {p_idx} ({label}): {match_pct:.1f}% token match, {div_str}")

        # -- N-trial timed suite, fully interleaved --------------------------
        print(f"\n[2/2] Running N={num_trials} trial suite (interleaved order)...")
        by_prompt_condition = {label: {"baseline": [], "gated": []} for label, _ in encoded}
        chronological = []
        gidx = 0

        for round_idx in range(1, num_trials + 1):
            print(f"\n{'#' * 85}\n# TRIAL ROUND {round_idx}/{num_trials}\n{'#' * 85}")
            order = ["baseline", "gated"] if round_idx % 2 == 1 else ["gated", "baseline"]
            for label, ids in encoded:
                for condition in order:
                    gidx += 1
                    entry = run_condition_trial(
                        condition, model, monitor, shared_gear_state, clutch,
                        ids, label, round_idx, max_tokens, eos, global_index=gidx,
                    )
                    gate_str = f"  int4%={entry['int4_fraction'] * 100:.1f}" if entry["int4_fraction"] is not None else ""
                    print(f"  {label:<8} {entry['condition']:<26} round {round_idx}/{num_trials}  "
                          f"tok/s={entry['throughput_tok_sec']:<7} "
                          f"J/tok={entry['joules_per_token']}  "
                          f"P={entry['avg_power_watts']} W{gate_str}")
                    by_prompt_condition[label]["baseline" if condition == "baseline" else "gated"].append(entry)
                    chronological.append(entry)
    finally:
        monitor.stop()
        monitor.unlock_clocks()

    # -- aggregate --------------------------------------------------------
    results = {}
    for p_idx, (label, _) in enumerate(encoded, 1):
        p_key = f"prompt_{p_idx}"
        results[p_key] = {}

        base_trials = by_prompt_condition[label]["baseline"]
        tps_b = [t["throughput_tok_sec"] for t in base_trials]
        j_b = [t["joules_per_token"] for t in base_trials if t["joules_per_token"] is not None]
        pwr_b = [t["avg_power_watts"] for t in base_trials if t["avg_power_watts"] is not None]
        results[p_key][BASELINE_KEY] = {
            "tps_mean": float(np.mean(tps_b)), "tps_std": float(np.std(tps_b)),
            "j_tok_mean": float(np.mean(j_b)) if j_b else None,
            "j_tok_std": float(np.std(j_b)) if j_b else 0.0,
            "power_mean": float(np.mean(pwr_b)) if pwr_b else None,
        }

        gated_trials = by_prompt_condition[label]["gated"]
        tps_g = [t["throughput_tok_sec"] for t in gated_trials]
        j_g = [t["joules_per_token"] for t in gated_trials if t["joules_per_token"] is not None]
        pwr_g = [t["avg_power_watts"] for t in gated_trials if t["avg_power_watts"] is not None]
        int4_frac = [t["int4_fraction"] for t in gated_trials if t["int4_fraction"] is not None]
        results[p_key][GATED_KEY] = {
            "tps_mean": float(np.mean(tps_g)), "tps_std": float(np.std(tps_g)),
            "j_tok_mean": float(np.mean(j_g)) if j_g else None,
            "j_tok_std": float(np.std(j_g)) if j_g else 0.0,
            "power_mean": float(np.mean(pwr_g)) if pwr_g else None,
            "int4_fraction_mean": float(np.mean(int4_frac)) if int4_frac else None,
            "fidelity_pct_vs_baseline": fidelity_by_prompt[p_key],
            "first_divergence_token": divergence_by_prompt[p_key],
        }

    def drift_block(entries):
        idx = [e["global_index"] for e in entries]
        d = {
            "power_w": bench_common.fit_drift(idx, [e["avg_power_watts"] for e in entries]),
            "joules_per_token": bench_common.fit_drift(idx, [e["joules_per_token"] for e in entries]),
            "throughput_tok_sec": bench_common.fit_drift(idx, [e["throughput_tok_sec"] for e in entries]),
        }
        d["residual_drift_acceptable"] = bench_common.drift_verdict(d["power_w"])
        d["threshold_fraction"] = bench_common.DRIFT_WARN_FRACTION
        return d

    baseline_entries = [e for e in chronological if e["condition"] == BASELINE_KEY]
    gated_entries = [e for e in chronological if e["condition"] == GATED_KEY]
    drift = {
        "pooled": drift_block(chronological),
        "baseline_only": drift_block(baseline_entries),
        "gated_only": drift_block(gated_entries),
    }

    print("\n" + "=" * 85)
    for p_key, m_dict in results.items():
        print(f"\n-- {p_key} --")
        for m, st in m_dict.items():
            tps_str = f"{st['tps_mean']:.2f} ± {st['tps_std']:.2f}"
            j_str = f"{st['j_tok_mean']:.3f} ± {st['j_tok_std']:.3f}" if st["j_tok_mean"] is not None else "n/a"
            extra = ""
            if "fidelity_pct_vs_baseline" in st:
                int4_pct = f"{st['int4_fraction_mean'] * 100:.1f}%" if st["int4_fraction_mean"] is not None else "n/a"
                extra = f"  fidelity={st['fidelity_pct_vs_baseline']:.1f}%  int4_usage={int4_pct}"
            print(f"{m:<26} | tok/s {tps_str:<20} | J/tok {j_str:<18}{extra}")
    print("=" * 85)

    if drift["pooled"]["residual_drift_acceptable"]:
        print("[ok] Residual drift within threshold. Means are usable.")
    else:
        print(f"[!] Residual drift exceeds {bench_common.DRIFT_WARN_FRACTION * 100:.1f}% of mean power.")
    if not warmup["converged"]:
        print("[!] Warmup did not converge before the cap. Raise --max-warmup.")

    out = {
        "fidelity_by_prompt": fidelity_by_prompt,
        "first_divergence_by_prompt": divergence_by_prompt,
        "results": results,
        "_meta": {
            "timestamp": datetime.now().isoformat(),
            "device": dev,
            "config": {
                "target_model": TARGET_MODEL_ID,
                "gated_projection": args.target_attr,
                "num_gated_layers": n_patched,
                "calibration": "per-channel min-max INT4 (vllm_sdsie/quantization/calibrate.py)",
                "clutch_theta_low": clutch.theta_low,
                "clutch_theta_high": clutch.theta_high,
                "clutch_alpha": clutch.alpha,
                "max_tokens": max_tokens,
                "num_trials": num_trials,
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
