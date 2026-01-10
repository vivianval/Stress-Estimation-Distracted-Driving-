############################### FILE FOR REPLICATING FIGURE 8 OF THE PAPER #####
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
import re

CSV = Path("/home/vivib/emoca/emoca/phasewise_md_nd_stats_convk1_finite.csv")
OUT_DIR = Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASES = ["P1", "P2", "P3", "P4", "P5"]  # enforce order

# --- appearance controls ---
FIG_W, FIG_H = 22, 4.2
DPI = 300
X_LABEL_ROT = 90
X_FONTSIZE = 11
Y_FONTSIZE = 13
TITLE_FONTSIZE = 16

# --- highlighting options ---
HIGHLIGHT_P2P4_BOTH = True
HIGHLIGHT_ONLY_P2P4 = True  # significant in P2 & P4 and NOT significant in P1,P3,P5

# visualize raw p or corrected p
PVAL_COL = "p"      # or "p_bonf"

def sig_level(p: float) -> int:
    if not np.isfinite(p):
        return 0
    if p < 1e-3:
        return 3
    if p < 1e-2:
        return 2
    if p < 5e-2:
        return 1
    return 0

def feature_sort_key(name: str):
    s = str(name)
    m = re.match(r"^(exp|pose|delta_pose)_(\d+)$", s, flags=re.IGNORECASE)
    if m:
        kind = m.group(1).lower()
        idx = int(m.group(2))
        kind_rank = {"exp": 0, "pose": 1, "delta_pose": 2}.get(kind, 9)
        return (kind_rank, idx)
    return (99, s)

def plot_metric(df_all: pd.DataFrame, metric: str, out_path: Path):
    df = df_all.copy()

    need = {"phase", "feature", "metric", PVAL_COL}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV missing required columns {need}. Found: {list(df.columns)}")

    df = df[df["metric"] == metric]
    df[PVAL_COL] = pd.to_numeric(df[PVAL_COL], errors="coerce")

    df = df[df["phase"].isin(PHASES)]
    df["phase"] = pd.Categorical(df["phase"], categories=PHASES, ordered=True)

    # pivot
    pmat = df.pivot_table(index="phase", columns="feature", values=PVAL_COL, aggfunc="min")

    # sort features
    feats = sorted(list(pmat.columns), key=feature_sort_key)
    pmat = pmat[feats]

    # map to categorical significance levels
    level = pmat.applymap(sig_level).astype(int)

    # colormap
    cmap = ListedColormap([
        (0.85, 0.85, 0.85, 1.0),  # ns
        (0.80, 0.90, 1.00, 1.0),  # p<0.05
        (0.50, 0.75, 1.00, 1.0),  # p<0.01
        (0.10, 0.35, 0.80, 1.0),  # p<0.001
    ])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.imshow(level.values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    # title
    tag = "Bonferroni" if PVAL_COL == "p_bonf" else "raw p"
    ax.set_title(f"Significance map of MD–ND differences with cubic splines ({metric})",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")

    # ticks / labels
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=X_LABEL_ROT, ha="right",
                       fontsize=X_FONTSIZE, fontweight="bold")

    phase_labels = ["P1", "P2 (Stressor)", "P3", "P4 (Stressor)", "P5"]
    ax.set_yticks(np.arange(len(PHASES)))
    ax.set_yticklabels(phase_labels, fontsize=Y_FONTSIZE, fontweight="bold")

    # gridlines
    ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(PHASES), 1), minor=True)
    ax.grid(which="minor", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---- highlighting ----
    HIGHLIGHT_KINDS = ("exp_", "pose_")   # add "delta_pose_" if you want
    RECT_COLOR = "green"
    RECT_LW = 3.0

    if HIGHLIGHT_P2P4_BOTH and ("P2" in pmat.index) and ("P4" in pmat.index):
        is_p2 = (pmat.loc["P2"] < 0.05)
        is_p4 = (pmat.loc["P4"] < 0.05)
        both = is_p2 & is_p4

        if HIGHLIGHT_ONLY_P2P4:
            nonstress_masks = []
            for ph in ["P1", "P3", "P5"]:
                if ph in pmat.index:
                    nonstress_masks.append(pmat.loc[ph] >= 0.05)
            both = both & (np.logical_and.reduce(nonstress_masks) if nonstress_masks else True)

        kind_mask = pd.Series([any(str(c).startswith(k) for k in HIGHLIGHT_KINDS) for c in pmat.columns],
                              index=pmat.columns)
        mask = both & kind_mask

        # highlight tick labels
        for tick, feat in zip(ax.get_xticklabels(), feats):
            if bool(mask.get(feat, False)):
                tick.set_bbox(dict(
                    facecolor=(1.0, 0.95, 0.25, 0.85),
                    edgecolor="none",
                    boxstyle="round,pad=0.15"
                ))
                tick.set_fontweight("bold")

        # rectangles around P2 and P4 cells
        row_idx = {ph: i for i, ph in enumerate(PHASES)}
        r2, r4 = row_idx["P2"], row_idx["P4"]

        for j, feat in enumerate(feats):
            if not bool(mask.get(feat, False)):
                continue
            for r in [r2, r4]:
                rect = patches.Rectangle(
                    (j - 0.5, r - 0.5), 1.0, 1.0,
                    fill=False, linewidth=RECT_LW, edgecolor=RECT_COLOR, zorder=5
                )
                ax.add_patch(rect)

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")

def main():
    df = pd.read_csv(CSV)

    plot_metric(df, "mean_level", OUT_DIR / "emoca1dconv_phase_significance_mean_level.png")
    plot_metric(df, "velocity",   OUT_DIR / "emoca1dconv_phase_significance_velocitysplines.png")

if __name__ == "__main__":
    main()
