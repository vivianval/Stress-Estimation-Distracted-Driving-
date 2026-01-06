#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = Path("/home/vivib/emoca/emoca/dataset/FINAL_PERFRAME")
OUT_DIR  = Path("/home/vivib/emoca/emoca/dataset")
OUT_STATS      = OUT_DIR / "gaze_dynamics_per_subject.csv"
OUT_CORR       = OUT_DIR / "gaze_dynamics_correlations.csv"
OUT_HEATMAP    = OUT_DIR / "gaze_dynamics_corr_heatmap.png"
OUT_MISSINGCSV = OUT_DIR / "missing_biosignals_report.csv"
FPS = 30.0  # change if different

BIOSIGNALS = ["Heart.Rate", "Perinasal.Perspiration", "Breathing.Rate",
              "Gaze.X.Pos", "Gaze.Y.Pos"]

# --------------- HELPERS ----------------
def find_label_column(df: pd.DataFrame):
    """Find a binary label column in df (0/1). Prefer common names."""
    candidates_by_name = [c for c in df.columns
                          if c.lower() in {"label", "stress", "stress_label",
                                           "stresslabel", "y"}]
    # try name-first
    for c in candidates_by_name:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if set(np.unique(vals)).issubset({0, 1}):
            return c
    # otherwise scan all columns
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if len(vals) > 0 and set(np.unique(vals)).issubset({0, 1}):
            return c
    return None

def safe_diff(a, dt=1.0):
    a = np.asarray(pd.to_numeric(a, errors="coerce"), dtype=float)
    a = np.nan_to_num(a)
    if a.size < 2:
        return np.zeros_like(a)
    return np.diff(a, prepend=a[0]) / dt

def compute_entropy2d(x, y, bins=30):
    """Normalized 2D entropy in [0,1]; safe for degenerate cases."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
    H, _, _ = np.histogram2d(x, y, bins=bins)
    s = H.sum()
    if s <= 0:
        return np.nan
    p = (H.flatten() / s)
    p = p[p > 0]
    # if all mass in 1 bin → len(p)==1 → return 0 (no spatial diversity)
    if p.size <= 1:
        return 0.0
    denom = np.log2(p.size)
    if denom == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum() / denom)

# --------------- MAIN -------------------
missing_rows = []
stats_rows = []

dt = 1.0 / FPS
csv_files = sorted(DATA_DIR.glob("T*_MD*_FINAL_PERFRAME.csv"))

for csv_path in csv_files:
    subj = csv_path.stem.split("_")[0]  # e.g., "T003"
    # be robust to mixed dtypes
    df = pd.read_csv(csv_path, low_memory=False)

    # -------- 1) Missing biosignals report --------
    for col in BIOSIGNALS:
        if col not in df.columns:
            print(f"{col} is missing in subject {subj}")
            missing_rows.append((subj, col, "missing_column"))
        else:
            present_frac = pd.to_numeric(df[col], errors="coerce").notna().mean()
            if present_frac < 0.01:
                print(f"{col} is missing in subject {subj} (only {present_frac*100:.1f}% present)")
                missing_rows.append((subj, col, f"{present_frac*100:.1f}% present"))

    # If gaze absent, skip dynamics
    if not {"Gaze.X.Pos", "Gaze.Y.Pos"}.issubset(df.columns):
        continue

    # -------- 2) Stress ratio from label inside CSV --------
    label_col = find_label_column(df)
    if label_col is None:
        # cannot compute correlations later without stress ratio
        stress_ratio = np.nan
    else:
        lab = pd.to_numeric(df[label_col], errors="coerce")
        stress_ratio = float((lab == 1).mean())

    # -------- 3) Gaze dynamics --------
    gx = pd.to_numeric(df["Gaze.X.Pos"], errors="coerce").fillna(0.0).to_numpy()
    gy = pd.to_numeric(df["Gaze.Y.Pos"], errors="coerce").fillna(0.0).to_numpy()

    vx, vy = safe_diff(gx, dt=dt), safe_diff(gy, dt=dt)
    vmag = np.sqrt(vx**2 + vy**2)

    ax, ay = safe_diff(vx, dt=dt), safe_diff(vy, dt=dt)
    amag = np.sqrt(ax**2 + ay**2)

    path_len = float(np.sqrt(np.diff(gx)**2 + np.diff(gy)**2).sum())
    disp_x, disp_y = float(np.std(gx)), float(np.std(gy))
    disp_total = float(np.hypot(disp_x, disp_y))
    gaze_entropy = compute_entropy2d(gx, gy, bins=30)

    stats_rows.append(dict(
        Subject=subj,
        StressRatio=stress_ratio,
        GazeVel_mean=float(np.mean(vmag)),
        GazeVel_std=float(np.std(vmag)),
        GazeVel_max=float(np.max(vmag)),
        GazeAcc_mean=float(np.mean(amag)),
        GazeAcc_std=float(np.std(amag)),
        GazeAcc_max=float(np.max(amag)),
        GazePathLength=path_len,
        GazeDispersion=disp_total,
        GazeEntropy=gaze_entropy
    ))

# ---- Save missing biosignals report
miss_df = pd.DataFrame(missing_rows, columns=["Subject", "Signal", "Comment"])
miss_df.to_csv(OUT_MISSINGCSV, index=False)

# ---- Save per-subject gaze dynamics + stress ratio
gaze_df = pd.DataFrame(stats_rows)
gaze_df.to_csv(OUT_STATS, index=False)
print(f"✅ Saved gaze dynamics stats → {OUT_STATS}")
print(f"✅ Saved missing-biosignals report → {OUT_MISSINGCSV}")

# ---- Correlations (subject-level features vs stress ratio computed above)
if not gaze_df.empty and "StressRatio" in gaze_df.columns:
    corr_rows = []
    features = [c for c in gaze_df.columns if c not in ["Subject", "StressRatio"]]
    for feat in features:
        x = pd.to_numeric(gaze_df[feat], errors="coerce").to_numpy()
        y = pd.to_numeric(gaze_df["StressRatio"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2:
            pear, _ = pearsonr(x[mask], y[mask])
            spear, _ = spearmanr(x[mask], y[mask])
            corr_rows.append(dict(Feature=feat, Pearson=pear, Spearman=spear))

    corr_df = pd.DataFrame(corr_rows).sort_values("Pearson", ascending=False)
    corr_df.to_csv(OUT_CORR, index=False)
    print(f"✅ Saved correlation results → {OUT_CORR}")

    # Heatmap saved only (no display)
    if not corr_df.empty:
        plt.figure(figsize=(max(6, 0.5*len(corr_df)), 4))
        plt.imshow(corr_df[["Pearson", "Spearman"]].T, cmap="coolwarm",
                   aspect="auto", vmin=-1, vmax=1)
        plt.xticks(range(len(corr_df)), corr_df["Feature"], rotation=90)
        plt.yticks([0, 1], ["Pearson", "Spearman"])
        plt.colorbar(label="Correlation")
        plt.title("Gaze-derived feature correlations with StressRatio")
        plt.tight_layout()
        plt.savefig(OUT_HEATMAP, dpi=200)
        plt.close()
        print(f"✅ Saved heatmap → {OUT_HEATMAP}")
    else:
        print("⚠️ No valid correlation rows to plot.")
else:
    print("⚠️ No StressRatio available to compute correlations.")


#!/usr/bin/env python3
# dataset/enrich_gaze_features.py
import numpy as np
import pandas as pd
from pathlib import Path

# -------- CONFIG ----------
DATA_DIR = Path("/home/vivib/emoca/emoca/dataset/FINAL_PERFRAME")
FPS = 30.0  # change if your per-frame csvs use another fps
BIOSIGNALS = ["Heart.Rate", "Perinasal.Perspiration", "Breathing.Rate", "Gaze.X.Pos", "Gaze.Y.Pos"]

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def diff_per_sec(a, dt):
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a)
    if a.size < 2:
        return np.zeros_like(a)
    return np.diff(a, prepend=a[0]) / dt

def process_file(csv_path: Path, fps: float):
    subj = csv_path.stem.split("_")[0]
    df = pd.read_csv(csv_path, low_memory=False)

    # ---- 1) Missing biosignals report (print only) ----
    for col in BIOSIGNALS:
        if col not in df.columns:
            print(f"{col} is missing in subject {subj}")
        else:
            present = safe_num(df[col]).notna().mean()
            if present < 0.01:
                print(f"{col} is missing in subject {subj} (only {present*100:.1f}% present)")

    # If gaze missing, just copy file with no changes
    if not {"Gaze.X.Pos", "Gaze.Y.Pos"}.issubset(df.columns):
        out_path = csv_path.with_name(csv_path.stem.replace("_FINAL_PERFRAME", "_FINAL_PERFRAME_ENRICHED") + ".csv")
        df.to_csv(out_path, index=False)
        print(f"[{subj}] Gaze columns missing → saved unchanged → {out_path}")
        return

    # ---- 2) Add per-frame gaze dynamics ----
    gx = safe_num(df["Gaze.X.Pos"]).fillna(0.0).to_numpy()
    gy = safe_num(df["Gaze.Y.Pos"]).fillna(0.0).to_numpy()
    dt = 1.0 / fps

    vx = diff_per_sec(gx, dt); vy = diff_per_sec(gy, dt)
    vmag = np.hypot(vx, vy)

    ax = diff_per_sec(vx, dt); ay = diff_per_sec(vy, dt)
    amag = np.hypot(ax, ay)

    df["GazeVel_X"] = vx
    df["GazeVel_Y"] = vy
    df["GazeVel"]   = vmag
    df["GazeAcc_X"] = ax
    df["GazeAcc_Y"] = ay
    df["GazeAcc"]   = amag

    # Rolling stats that were most predictive at subject-level
    win1 = int(round(1.0 * fps))
    win3 = int(round(3.0 * fps))
    win2 = int(round(2.0 * fps))

    # rolling mean/std of speed
    df["GazeVel_mean_1s"] = pd.Series(vmag).rolling(win1, min_periods=1).mean()
    df["GazeVel_std_1s"]  = pd.Series(vmag).rolling(win1, min_periods=1).std().fillna(0.0)
    df["GazeVel_mean_3s"] = pd.Series(vmag).rolling(win3, min_periods=1).mean()
    df["GazeVel_std_3s"]  = pd.Series(vmag).rolling(win3, min_periods=1).std().fillna(0.0)

    # rolling dispersion √(std_x^2 + std_y^2) over 2s
    rx = pd.Series(gx).rolling(win2, min_periods=1).std().fillna(0.0)
    ry = pd.Series(gy).rolling(win2, min_periods=1).std().fillna(0.0)
    df["GazeDispersion_2s"] = np.hypot(rx, ry)

    # ---- 3) Save enriched CSV ----
    out_path = csv_path.with_name(csv_path.stem.replace("_FINAL_PERFRAME", "_FINAL_PERFRAME_ENRICHED") + ".csv")
    df.to_csv(out_path, index=False)
    print(f"[{subj}] enriched → {out_path}")

def main():
    files = sorted(DATA_DIR.glob("T*_MD*_FINAL_PERFRAME.csv"))
    if not files:
        print("No files found.")
        return
    for p in files:
        process_file(p, FPS)

if __name__ == "__main__":
    main()
