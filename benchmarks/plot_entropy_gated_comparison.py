"""
plot_entropy_gated_comparison.py

Plots the entropy-gated speculative decoding controller (step4_entropy_gated_scout.py)
against the matched FP16 baseline (step4_fp16_baseline_matched.py), reading directly
from their JSON summaries in tools/telemetry/.

Usage:
    python3 plot_entropy_gated_comparison.py
    python3 plot_entropy_gated_comparison.py --gated-summary tools/telemetry/entropy_gated_scout_n5_summary.json

The output filename embeds the actual theta/alpha values used, read from the summary
JSON itself (not hardcoded), so a tuned-threshold run and a production-threshold run
never silently overwrite each other.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_DIR = REPO_ROOT / "telemetry"
ASSETS_DIR = REPO_ROOT / "assets"

PROMPT_ORDER = ["Poem", "Physics", "Code"]


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_thresholds_from_gated_summary(gated: dict) -> tuple:
    """Pull theta_low/theta_high/alpha from the first raw trial found, so the
    output filename/title always reflects what was ACTUALLY run, not an assumption."""
    for prompt_data in gated.values():
        raw_trials = prompt_data.get("raw_trials", [])
        if raw_trials:
            t0 = raw_trials[0]
            theta_low = t0.get("theta_low")
            theta_high = t0.get("theta_high")
            return theta_low, theta_high
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gated-summary", type=Path,
                         default=TELEMETRY_DIR / "entropy_gated_scout_n5_summary.json")
    parser.add_argument("--baseline-summary", type=Path,
                     default=TELEMETRY_DIR / "fp16_baseline_n5_summary.json")
    parser.add_argument("--out-dir", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gated = load_json(args.gated_summary)
    baseline = load_json(args.baseline_summary)

    theta_low, theta_high = get_thresholds_from_gated_summary(gated)
    if theta_low is None:
        print("[!] Could not read theta_low/theta_high from raw_trials in the gated summary "
              "-- filename will omit them.")
        threshold_tag = "unknown_thresholds"
        threshold_title = ""
    else:
        threshold_tag = f"theta_{theta_low}_{theta_high}"
        threshold_title = f" (\u03b8_low={theta_low}, \u03b8_high={theta_high})"

    prompts = [p for p in PROMPT_ORDER if p in gated and p in baseline]
    missing = [p for p in PROMPT_ORDER if p not in gated or p not in baseline]
    if missing:
        print(f"[!] Skipping prompts not found in both summaries: {missing}")

    base_tps = [baseline[p]["tps_mean"] for p in prompts]
    base_tps_std = [baseline[p]["tps_std"] for p in prompts]
    gated_tps = [gated[p]["tps_mean"] for p in prompts]
    gated_tps_std = [gated[p]["tps_std"] for p in prompts]

    base_j = [baseline[p]["j_tok_mean"] for p in prompts]
    base_j_std = [baseline[p]["j_tok_std"] for p in prompts]
    gated_j = [gated[p]["j_tok_mean"] for p in prompts]
    gated_j_std = [gated[p]["j_tok_std"] for p in prompts]

    n_trials = gated[prompts[0]].get("n_trials", "?")

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Entropy-Gated Speculative Decoding vs. Matched FP16 Baseline "
                 f"(N={n_trials} trials/prompt, RTX 5090){threshold_title}", fontsize=13)

    x = np.arange(len(prompts))
    width = 0.35

    # --- Throughput ---
    b1 = ax1.bar(x - width / 2, base_tps, width, yerr=base_tps_std, capsize=4,
                 label="FP16 Baseline (matched)", color="#94a3b8")
    b2 = ax1.bar(x + width / 2, gated_tps, width, yerr=gated_tps_std, capsize=4,
                 label="Entropy-Gated (real branching)", color="#16db65")
    ax1.set_title("Throughput: Baseline vs. Entropy-Gated", fontweight="bold")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompts)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # --- Energy ---
    b3 = ax2.bar(x - width / 2, base_j, width, yerr=base_j_std, capsize=4,
                 label="FP16 Baseline (matched)", color="#94a3b8")
    b4 = ax2.bar(x + width / 2, gated_j, width, yerr=gated_j_std, capsize=4,
                 label="Entropy-Gated (real branching)", color="#16db65")
    ax2.set_title("Energy: Baseline vs. Entropy-Gated", fontweight="bold")
    ax2.set_ylabel("Joules / token")
    ax2.set_xticks(x)
    ax2.set_xticklabels(prompts)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = args.out_dir / f"step4_vs_baseline_comparison_{threshold_tag}.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    # Print the delta table too, for a quick sanity check against the plot
    print(f"\n{'Prompt':<10} | {'Speedup':<10} | {'Energy \u0394':<10}")
    print("-" * 36)
    for p, bt, gt, bj, gj in zip(prompts, base_tps, gated_tps, base_j, gated_j):
        speedup = (gt / bt - 1) * 100
        energy_delta = (gj / bj - 1) * 100
        print(f"{p:<10} | {speedup:+.1f}%    | {energy_delta:+.1f}%")


if __name__ == "__main__":
    main()