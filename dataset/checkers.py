# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import numpy as np
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm
# import matplotlib.patches as patches
# import re
# from matplotlib.colorbar import ColorbarBase

# CSV = Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/emoca_phase_ttests.csv")
# OUT = Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/emoca_phase_significance_heatmap.png")

# PHASES = ["P1", "P2", "P3", "P4", "P5"]  # enforce order

# # --- appearance controls ---
# FIG_W, FIG_H = 22, 4.2
# DPI = 300
# X_LABEL_ROT = 90
# X_FONTSIZE = 11
# Y_FONTSIZE = 13
# TITLE_FONTSIZE = 16

# # --- highlighting options ---
# HIGHLIGHT_P2P4_BOTH = True
# HIGHLIGHT_ONLY_P2P4 = True  # if True: significant in P2 & P4 AND NOT significant (p<0.05) in P1,P3,P5

# # ------------------------------------------------------------

# def sig_level(p: float) -> int:
#     """Map p-value to categorical levels: 0=ns, 1=p<0.05, 2=p<0.01, 3=p<0.001."""
#     if not np.isfinite(p):
#         return 0
#     if p < 1e-3:
#         return 3
#     if p < 1e-2:
#         return 2
#     if p < 5e-2:
#         return 1
#     return 0

# def feature_sort_key(name: str):
#     """
#     Sort exp_00..exp_49, then pose_00..pose_05, then delta_pose_00.. etc.
#     Falls back to lexical.
#     """
#     s = str(name)
#     m = re.match(r"^(exp|pose|delta_pose)_(\d+)$", s, flags=re.IGNORECASE)
#     if m:
#         kind = m.group(1).lower()
#         idx = int(m.group(2))
#         kind_rank = {"exp": 0, "pose": 1, "delta_pose": 2}.get(kind, 9)
#         return (kind_rank, idx)
#     return (99, s)

# def main():
#     df = pd.read_csv(CSV)
#     # expected columns: Phase, Feature, p (and maybe others)
#     if not {"Phase", "Feature", "p"}.issubset(df.columns):
#         raise SystemExit(f"CSV missing required columns. Found: {list(df.columns)}")

#     df = df.copy()
#     df["p"] = pd.to_numeric(df["p"], errors="coerce")

#     # keep phases we care about + enforce order
#     df = df[df["Phase"].isin(PHASES)]
#     df["Phase"] = pd.Categorical(df["Phase"], categories=PHASES, ordered=True)

#     # pivot: rows=Phase, cols=Feature, values=p
#     pmat = df.pivot_table(index="Phase", columns="Feature", values="p", aggfunc="min")

#     # sort features nicely
#     feats = sorted(list(pmat.columns), key=feature_sort_key)
#     pmat = pmat[feats]

#     # map to categorical significance levels
#     level = pmat.applymap(sig_level).astype(int)

#     # --- colormap: 0 gray, 1 very light blue, 2 light blue, 3 dark blue ---
#     # (Using explicit RGBA values so the plot is deterministic)
#     cmap = ListedColormap([
#         (0.85, 0.85, 0.85, 1.0),  # ns: gray
#         (0.80, 0.90, 1.00, 1.0),  # p<0.05: very light blue
#         (0.50, 0.75, 1.00, 1.0),  # p<0.01: light blue
#         (0.10, 0.35, 0.80, 1.0),  # p<0.001: dark blue
#     ])
#     norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)

#     fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

#     im = ax.imshow(level.values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
   

#     # cbar = fig.colorbar(
#     #     im,
#     #     ax=ax,
#     #     fraction=0.035,   # width of the colorbar
#     #     pad=0.02          # distance from the heatmap
#     # )

#     # cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
#     # cbar.set_ticklabels([
#     #     "n.s.\n($p \\geq 0.05$)",
#     #     "$p < 0.05$",
#     #     "$p < 0.01$",
#     #     "$p < 0.001$"
#     # ])
#     # cbar.ax.tick_params(labelsize=13)
#     # cbar.outline.set_linewidth(1.5)



#     ax.set_title("Significance map of MD–ND differences (raw EMOCA coefficients)", fontsize=TITLE_FONTSIZE, fontweight="bold")
#     # ax.set_xlabel("EMOCA coefficient", fontweight="bold")
#     # ax.set_ylabel("Phase", fontweight="bold")

#     # ticks and labels
#     ax.set_xticks(np.arange(len(feats)))
#     ax.set_xticklabels(feats, rotation=X_LABEL_ROT, ha="right", fontsize=X_FONTSIZE, fontweight="bold")


#     phase_labels = ["P1", "P2 (Stressor)", "P3", "P4 (Stressor)", "P5"]

#     ax.set_yticks(np.arange(len(phase_labels)))
#     ax.set_yticklabels(phase_labels, fontsize=Y_FONTSIZE, fontweight="bold")


#     # ax.set_yticks(np.arange(len(PHASES)))
#     # ax.set_yticklabels(PHASES, fontsize=Y_FONTSIZE, fontweight="bold")

#     # gridlines (checkerboard feel)
#     ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
#     ax.set_yticks(np.arange(-.5, len(PHASES), 1), minor=True)
#     ax.grid(which="minor", linewidth=0.6)
#     ax.tick_params(which="minor", bottom=False, left=False)

#     # ---- OPTIONAL HIGHLIGHTING ----
#     # highlight coefficients that are significant in BOTH P2 and P4.
#     # optionally "only P2 & P4": non-sig (p>=0.05) in P1,P3,P5.

#         # ---- OPTIONAL HIGHLIGHTING ----
#     # 1) highlight x-axis labels for features significant in BOTH P2 and P4 (yellow "highlighter")
#     # 2) draw rectangles around the P2 and P4 cells for those same features
#     # Applies to exp_* and pose_* (optionally delta_pose_*)

#     HIGHLIGHT_KINDS = ("exp_", "pose_")   # add "delta_pose_" if you want it too
#     RECT_COLOR = "green"                 # or e.g. "#00aa00"
#     RECT_LW = 3.0

#     if HIGHLIGHT_P2P4_BOTH:
#         # boolean masks per feature
#         is_p2 = (pmat.loc["P2"] < 0.05) if "P2" in pmat.index else pd.Series(False, index=pmat.columns)
#         is_p4 = (pmat.loc["P4"] < 0.05) if "P4" in pmat.index else pd.Series(False, index=pmat.columns)
#         both = is_p2 & is_p4

#         if HIGHLIGHT_ONLY_P2P4:
#             nonstress_masks = []
#             for ph in ["P1", "P3", "P5"]:
#                 if ph in pmat.index:
#                     nonstress_masks.append(pmat.loc[ph] >= 0.05)
#             mask = both & (np.logical_and.reduce(nonstress_masks) if nonstress_masks else True)
#         else:
#             mask = both

#         # restrict to selected coefficient kinds (exp_ and pose_)
#         kind_mask = pd.Series(
#             [any(str(c).startswith(k) for k in HIGHLIGHT_KINDS) for c in pmat.columns],
#             index=pmat.columns
#         )
#         mask = mask & kind_mask

#         # --- (A) highlight ONLY the x tick labels (yellow) ---
#         for tick, feat in zip(ax.get_xticklabels(), feats):
#             if bool(mask.get(feat, False)):
#                 tick.set_bbox(dict(
#                     facecolor=(1.0, 0.95, 0.25, 0.85),  # yellow highlighter
#                     edgecolor="none",
#                     boxstyle="round,pad=0.15"
#                 ))
#                 tick.set_fontweight("bold")

#         # --- (B) draw rectangles around the P2 and P4 cells for each selected feature ---
#         row_idx = {ph: i for i, ph in enumerate(PHASES)}
#         r2, r4 = row_idx.get("P2"), row_idx.get("P4")

#         for j, feat in enumerate(feats):
#             if not bool(mask.get(feat, False)):
#                 continue

#             for r in [r2, r4]:
#                 if r is None:
#                     continue
#                 rect = patches.Rectangle(
#                     (j - 0.5, r - 0.5), 1.0, 1.0,
#                     fill=False, linewidth=RECT_LW, edgecolor=RECT_COLOR, zorder=5
#                 )
#                 ax.add_patch(rect)

        

#     # if HIGHLIGHT_P2P4_BOTH:
#     #     # boolean masks per feature
#     #     is_p2 = (pmat.loc["P2"] < 0.05) if "P2" in pmat.index else pd.Series(False, index=pmat.columns)
#     #     is_p4 = (pmat.loc["P4"] < 0.05) if "P4" in pmat.index else pd.Series(False, index=pmat.columns)
#     #     both = is_p2 & is_p4

#     #     if HIGHLIGHT_ONLY_P2P4:
#     #         nonstress = []
#     #         for ph in ["P1", "P3", "P5"]:
#     #             if ph in pmat.index:
#     #                 nonstress.append(pmat.loc[ph] >= 0.05)
#     #         if nonstress:
#     #             only = both & np.logical_and.reduce(nonstress)
#     #         else:
#     #             only = both
#     #         mask = only
#     #     else:
#     #         mask = both

#     #     # draw a rectangle around the P2 and P4 cells for each selected feature
#     #     # row indices for P2, P4 in the heatmap:
#     #     row_idx = {ph: i for i, ph in enumerate(PHASES)}
#     #     r2, r4 = row_idx.get("P2"), row_idx.get("P4")

#     #     for j, feat in enumerate(feats):
#     #         if not bool(mask.get(feat, False)):
#     #             continue
#     #         # outline the two cells (P2, P4) in that column
#     #         for r in [r2, r4]:
#     #             if r is None:
#     #                 continue
#     #             rect = patches.Rectangle(
#     #                 (j - 0.5, r - 0.5), 1.0, 1.0,
#     #                 fill=False, linewidth=3.0, edgecolor="green"
#     #             )
#     #             ax.add_patch(rect)

#     # simple legend (manual)
#     # (avoid colorbar to keep it compact)
#     # legend_text = "ns  (p≥0.05)   |   *  (p<0.05)   |   ** (p<0.01)   |   *** (p<0.001)"
#     # ax.text(0.0, -0.20, legend_text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

#     # fig.tight_layout()
#     fig.tight_layout(rect=[0, 0, 0.95, 1])

#     fig.savefig(OUT, bbox_inches="tight")
#     plt.close(fig)
#     print(f"[SAVE] {OUT}")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
