import csv
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(REPO_ROOT, "telemetry", "telemetry_entropy_gear.csv")
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

categories, throughputs, energies, high_gears = [], [], [], []
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("category") or row.get("model_id", "Run")
            categories.append(cat)
            throughputs.append(float(row["throughput_tok_sec"]))
            energies.append(float(row["joules_per_token"]))
            high_gears.append(float(row["high_gear_pct"]))

if not categories:
    print("❌ No data found in telemetry/telemetry_entropy_gear.csv. Run cognitive_benchmark.py first.")
    exit(1)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Energy per Token
colors = ['#ff4b4b', '#00d26a', '#00d26a', '#00d26a', '#3b82f6'][:len(categories)]
bars1 = ax1.bar(categories, energies, color=colors, edgecolor='white', alpha=0.85, width=0.5)
ax1.set_title("Energy Efficiency (Joules / Token) on RTX 5090", fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel("Joules per Token (Lower is Better)", fontsize=11)
ax1.set_xticklabels(categories, rotation=20, ha='right', fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} J", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: High Gear Utilization
bars2 = ax2.bar(categories, high_gears, color='#00d26a', edgecolor='white', alpha=0.85, width=0.5)
ax2.set_title("SDSIE High Gear (Sub-Byte) Utilization Rate (%)", fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel("High Gear % (Higher is Better)", fontsize=11)
ax2.set_xticklabels(categories, rotation=20, ha='right', fontsize=9)
ax2.set_ylim(0, 110)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_file = os.path.join(ASSETS_DIR, "sdsie_empirical_telemetry.png")
plt.savefig(plot_file, dpi=300)
print(f"📊 Publication-grade chart generated and saved to: {plot_file}")