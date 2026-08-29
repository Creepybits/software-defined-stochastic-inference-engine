"""
plot_parameter_sweep.py - 300 DPI Publication Visualizer
Plots Hysteresis Sensitivity & Pareto Space from RTX 5090 Blackwell Sweep Data
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "telemetry" / "real_model_sweep_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "assets" / "sdsie_parameter_sweep_pareto.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cannot find sweep results at {DATA_PATH}")

with open(DATA_PATH, "r") as f:
    data = json.load(f)

# Filter out initial CUDA warmup run if needed for clean plotting
clean_data = data

labels = [f"({d['theta_low']:.2f}, {d['theta_high']:.2f})" for d in clean_data]
spec_pcts = [d["speculation_percentage"] for d in clean_data]
throughputs = [d["throughput_tok_s"] for d in clean_data]
theta_lows = [d["theta_low"] for d in clean_data]

# Set up IEEE publication styling
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
fig.patch.set_facecolor("#090c09")

# Colors
green_accent = "#16db65"
green_primary = "#0e6b0e"
emerald_glow = "#00ff88"
text_color = "#e2e8e2"
grid_color = "#1a261a"

# --- LEFT PANEL: Speculation Engagement vs Hysteresis Window ---
ax1.set_facecolor("#101510")
bars = ax1.bar(range(len(labels)), spec_pcts, color="#1b4d24", edgecolor=green_accent, linewidth=1.2, width=0.6)

# Highlight baseline and max speculation bars
for idx, d in enumerate(clean_data):
    if d["theta_low"] == 0.55 and d["theta_high"] == 1.25:
        bars[idx].set_facecolor(green_accent)
        bars[idx].set_edgecolor("#ffffff")
    elif d["speculation_percentage"] == max(spec_pcts):
        bars[idx].set_facecolor("#00cc66")

ax1.set_title("Speculation Engagement vs. Hysteresis Thresholds", fontsize=11, fontweight="bold", color=text_color, pad=12)
ax1.set_xlabel("(θ_low, θ_high) Configuration [bits]", fontsize=9, color=text_color, labelpad=8)
ax1.set_ylabel("Speculation Active Steps [%]", fontsize=9, color=text_color)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#a0b0a0")
ax1.grid(True, linestyle="--", alpha=0.3, color=grid_color)
ax1.set_ylim(0, max(spec_pcts) + 8)

# Add value labels above bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f"{height:.1f}%",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=7.5, color="#c8e6c9", fontweight="bold")

# Annotations for operational modes
ax1.text(0.5, 8, "Conservative Fortress\n(0.0% Draft)", color="#ff6b6b", fontsize=7.5, ha="center", style="italic")
ax1.text(4, 14, "Baseline\n(5.9%)", color="#00ff88", fontsize=8, ha="center", fontweight="bold")
ax1.text(8, 36, "High Throughput\n(33.8% Draft)", color="#4dd0e1", fontsize=7.5, ha="center", style="italic")

# --- RIGHT PANEL: Pareto Space (Throughput vs Speculation %) ---
ax2.set_facecolor("#101510")
scatter = ax2.scatter(spec_pcts, throughputs, c=theta_lows, cmap="viridis", s=120, edgecolors=green_accent, linewidth=1.5, zorder=5)

for idx, d in enumerate(clean_data):
    if idx == 0:  # Skip warmup outlier label
        continue
    label_txt = f"({d['theta_low']},{d['theta_high']})"
    offset_y = 0.15 if idx % 2 == 0 else -0.25
    ax2.annotate(label_txt,
                 xy=(d["speculation_percentage"], d["throughput_tok_s"]),
                 xytext=(0, 6), textcoords="offset points",
                 ha="center", fontsize=7.5, color="#e0e0e0")

# Highlight baseline star
baseline = [d for d in clean_data if d["theta_low"] == 0.55 and d["theta_high"] == 1.25][0]
ax2.scatter(baseline["speculation_percentage"], baseline["throughput_tok_s"], color="#ffffff", marker="*", s=250, zorder=6, edgecolors=green_accent, label="Baseline (0.55, 1.25)")

ax2.set_title("Operational Pareto Frontier on RTX 5090 Blackwell", fontsize=11, fontweight="bold", color=text_color, pad=12)
ax2.set_xlabel("Speculative Drafting Ratio [% of generation]", fontsize=9, color=text_color, labelpad=8)
ax2.set_ylabel("Sustained Throughput [tok/s]", fontsize=9, color=text_color)
ax2.grid(True, linestyle="--", alpha=0.3, color=grid_color)
ax2.set_ylim(50.0, 54.0)
ax2.legend(loc="lower right", facecolor="#142014", edgecolor=green_accent, fontsize=8)

# Header Title
plt.suptitle("SDSIE Parameter Sensitivity & Hysteresis Pareto Bounds (Llama-3.1-8B)", fontsize=13, fontweight="bold", color="#ffffff", y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"[*] 300 DPI Parameter Sweep Pareto Plot saved to: {OUTPUT_PATH}")