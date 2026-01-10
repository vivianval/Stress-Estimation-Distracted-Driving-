
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ====== PATHS ======
ROOT = Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/phased_csvs")
CSV = ROOT / "MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN.csv"
OUT =  Path("/home/vivib/emoca/emoca/dataset/emoca_stress_correlation_pastel.png")

LABEL_COL = "GTlabel"      # your stress label column
N_TOP = 10               # number of features to show

print(f"Loading {CSV} ...")
df = pd.read_csv(CSV, low_memory=False)

# Keep rows with known labels
df = df.dropna(subset=[LABEL_COL])

# ====== SELECT EMOCA FEATURES ======
exp_cols  = [c for c in df.columns if c.startswith("exp_")]
pose_cols = [c for c in df.columns if c.startswith("pose_")]
feature_cols = exp_cols + pose_cols

print(f"Using {len(exp_cols)} expression + {len(pose_cols)} pose = {len(feature_cols)} EMOCA features.")

# ====== CORRELATION WITH STRESS ======
corr_series = df[feature_cols + [LABEL_COL]].corr()[LABEL_COL].drop(LABEL_COL)
abs_corr = corr_series.abs().sort_values(ascending=False)
top = abs_corr.head(N_TOP)

print("\nTop correlated features:")
print(top)

# ====== PLOT ======
plt.figure(figsize=(9, 6), dpi=300)

y = np.arange(len(top))
colors = plt.cm.Pastel1(np.linspace(0, 1, len(top)))  # pastel palette

bars = plt.barh(y, top.values, color=colors, edgecolor='gray')

plt.yticks(y, top.index, fontsize=9)
plt.xlabel("Absolute correlation with stress label", fontsize=12)
plt.title("Top EMOCA PCA Components Correlated with Stress", fontsize=14)

plt.gca().invert_yaxis()  # highest at top
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Annotate values
for bar, value in zip(bars, top.values):
    plt.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height()/2,
        f"{value:.3f}",
        va='center',
        fontsize=8,
        color='black'
    )

plt.tight_layout()
plt.savefig(OUT, dpi=300)
print(f"\nSaved figure:\n{OUT.resolve()}")
