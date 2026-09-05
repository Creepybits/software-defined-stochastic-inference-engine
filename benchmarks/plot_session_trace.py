# plot_session_trace.py
"""
SDSIE Session Telemetry Plotter
Visualizes real-time Shannon entropy vs. Schmitt-trigger speculative clutch transitions.
"""

import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Locate latest telemetry JSON
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_files = glob.glob(os.path.join(REPO_ROOT, "telemetry", "sessions", "*.json"))
if not log_files:
    print("[Error] No session logs found in ./telemetry/sessions/")
    exit(1)

latest_file = max(log_files, key=os.path.getctime)
print(f"[*] Reading telemetry trace from: {latest_file}")

with open(latest_file, "r", encoding="utf-8") as f:
    data = json.load(f)

trace = data["telemetry_trace"]
steps = [t["step"] for t in trace]
entropy = [t["entropy_bits"] for t in trace]
k_draft = [t["k_draft"] for t in trace]

# 2. Configure Matplotlib Dark Aesthetic
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
fig.patch.set_facecolor("#0a0c0a")
ax1.set_facecolor("#0f140f")
ax2.set_facecolor("#0f140f")

# --- PANEL 1: SHANNON ENTROPY & HYSTERESIS THRESHOLDS ---
ax1.plot(steps, entropy, color="#16db65", linewidth=1.8, label="Token Entropy $H(t)$ (Shannon Bits)", zorder=4)
ax1.fill_between(steps, entropy, color="#16db65", alpha=0.12, zorder=3)

# Threshold lines
ax1.axhline(1.25, color="#ff4d4d", linestyle="--", linewidth=1.2, alpha=0.85, label=r"$\theta_{high} = 1.25\text{ bits}$ (Disengage $\to$ Fallback)", zorder=5)
ax1.axhline(0.55, color="#00e5ff", linestyle="--", linewidth=1.2, alpha=0.85, label=r"$\theta_{low} = 0.55\text{ bits}$ (Re-engage $\to$ Speculate)", zorder=5)

# Annotations
max_idx = np.argmax(entropy)
min_idx = np.argmin(entropy)

ax1.annotate(
    f"Cognitive Rhyme Search Fork\n$H_{{max}} = {entropy[max_idx]:.2f}$ bits (Step {steps[max_idx]})",
    xy=(steps[max_idx], entropy[max_idx]),
    xytext=(steps[max_idx] - 70, entropy[max_idx] + 0.35),
    arrowprops=dict(facecolor="#ff4d4d", edgecolor="#ff4d4d", arrowstyle="->", lw=1.2),
    fontsize=9, color="#ffb3b3", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e0f0f", edgecolor="#ff4d4d", alpha=0.9)
)

ax1.annotate(
    f"Deterministic Meter Cadence\n$H_{{min}} = {entropy[min_idx]:.4f}$ bits (Step {steps[min_idx]})",
    xy=(steps[min_idx], entropy[min_idx]),
    xytext=(steps[min_idx] - 40, entropy[min_idx] + 0.75),
    arrowprops=dict(facecolor="#16db65", edgecolor="#16db65", arrowstyle="->", lw=1.2),
    fontsize=9, color="#b3ffcc", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f1e14", edgecolor="#16db65", alpha=0.9)
)

ax1.set_ylabel("Entropy (Bits / Token)", fontsize=11, fontweight="bold", color="#e0e0e0")
ax1.set_title(f"SDSIE Stochastic Speculation Telemetry — Llama-3.1-8B ({len(steps)} Steps @ {data['throughput_tok_s']} tok/s)", fontsize=13, fontweight="bold", pad=12, color="#ffffff")
ax1.grid(True, linestyle=":", alpha=0.25, color="#557755")
ax1.legend(loc="upper right", framealpha=0.85, facecolor="#141a14", edgecolor="#223322", fontsize=9)
ax1.set_ylim(-0.2, max(entropy) + 0.9)

# --- PANEL 2: SCHMITT-TRIGGER CLUTCH STATE ---
k_arr = np.array(k_draft)
ax2.step(steps, k_arr, where="post", color="#00e5ff", linewidth=1.5, label="Clutch State (k proposed)", zorder=4)

# Color shading for states
ax2.fill_between(steps, 0, k_arr, step="post", color="#16db65", alpha=0.25, label="🚀 Speculative (k=5)")
ax2.fill_between(steps, 0, np.where(k_arr == 0, 5, 0), step="post", color="#ff4d4d", alpha=0.15, label="🛡️ Single-Step Fallback (k=0)")

ax2.set_ylabel("Draft Window (k)", fontsize=10, fontweight="bold", color="#e0e0e0")
ax2.set_xlabel("Autoregressive Token Generation Step", fontsize=11, fontweight="bold", color="#e0e0e0")
ax2.set_yticks([0, 5])
ax2.set_yticklabels(["k=0\n(Fallback)", "k=5\n(Speculate)"], fontsize=8.5)
ax2.grid(True, linestyle=":", alpha=0.25, color="#557755")
ax2.legend(loc="center right", framealpha=0.85, facecolor="#141a14", edgecolor="#223322", fontsize=8.5)
ax2.set_ylim(-0.5, 6.0)

plt.tight_layout()

# 3. Save publication-grade asset
os.makedirs("./assets", exist_ok=True)
out_path = "./assets/sdsie_chant_royal_trace.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"✓ Publication-grade plot generated and saved to: {out_path}")