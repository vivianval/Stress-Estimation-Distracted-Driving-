# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.decomposition import PCA
# # from sklearn.manifold import TSNE
# # import matplotlib.pyplot as plt

# # CSV = Path("/home/vivib/emoca/emoca/dataset/all_subjects_merged_fixed_with_gaze.csv")

# # # -----------------------------
# # # LOAD
# # # -----------------------------
# # df = pd.read_csv(CSV)

# # # Choose features (example: expression only)
# # EXP_COLS = [c for c in df.columns if c.startswith("exp_")]
# # FEATURES = EXP_COLS    # change here for multimodal sets

# # # Drop rows with missing
# # df_clean = df.dropna(subset=FEATURES + ["Subject", "label"])

# # X = df_clean[FEATURES].to_numpy()
# # subjects = df_clean["Subject"].astype(str).to_numpy()
# # labels   = df_clean["label"].astype(int).to_numpy()

# # # Optional phase, if your CSV includes it
# # phase = df_clean["phase"].astype(str).to_numpy() if "phase" in df_clean.columns else None

# # # -----------------------------
# # # PCA → speed up t-SNE
# # # -----------------------------
# # X = StandardScaler().fit_transform(X)
# # pca = PCA(n_components=30)  # safe for 50D expressions
# # X_pca = pca.fit_transform(X)

# # # -----------------------------
# # # Multicore t-SNE
# # # -----------------------------
# # tsne = TSNE(
# #     n_components=2,
# #     perplexity=40,
# #     learning_rate=200,
# #     n_iter=2000,
# #     init="pca",
# #     random_state=42
# # )
# # Z = tsne.fit_transform(X_pca)

# # np.save("tsne_Z_exp.npy", Z)
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import matplotlib.patheffects as pe

# -----------------------------
# PATHS
# -----------------------------
CSV = Path("/home/vivib/emoca/emoca/dataset/all_subjects_merged_fixed_with_gaze.csv")
TSNE_NPY = Path("tsne_Z_exp.npy")  # change name if you used another
OUT_SUBJECT_PNG = Path("tsne_colored_bysubject.png")
OUT_LABEL_PNG   = Path("tsne_colored_bylabel.png")
OUT_PHASE_PNG   = Path("tsne_colored_byphase.png")

# -----------------------------
# LOAD TSNE + DATA
# -----------------------------
print("Loading t-SNE embedding from:", TSNE_NPY)
Z = np.load(TSNE_NPY)   # shape (N, 2)

print("Loading CSV:", CSV)
df = pd.read_csv(CSV, low_memory=False)

# same feature selection as in your t-SNE script
EXP_COLS = [c for c in df.columns if c.startswith("exp_")]
FEATURES = EXP_COLS

# replicate the same filtering: drop rows with missing in features + Subject + label
needed_cols = FEATURES + ["Subject", "label"]
df_clean = df.dropna(subset=needed_cols).reset_index(drop=True)

if Z.shape[0] != len(df_clean):
    raise RuntimeError(f"Shape mismatch: Z has {Z.shape[0]} rows, "
                       f"df_clean has {len(df_clean)} rows.")

subjects = df_clean["Subject"].astype(str).to_numpy()
labels   = df_clean["label"].astype(int).to_numpy()
phase    = df_clean["phase"].astype(str).to_numpy() if "phase" in df_clean.columns else None

# # -----------------------------
# # PLOT 1: Color by Subject
# # -----------------------------
# plt.figure(figsize=(10, 8))
# subjects_unique = np.unique(subjects)
# cmap = plt.cm.get_cmap("tab20", len(subjects_unique))

# for i, sid in enumerate(subjects_unique):
#     idx = np.where(subjects == sid)[0]
#     plt.scatter(Z[idx, 0], Z[idx, 1], s=2, color=cmap(i), label=sid)

# plt.title("t-SNE of EMOCA Expressions — Colored by Subject")
# plt.legend(markerscale=4, fontsize=6, ncol=2)
# plt.tight_layout()
# plt.savefig(OUT_SUBJECT_PNG, dpi=300)
# plt.close()
# print("Saved:", OUT_SUBJECT_PNG)

# # -----------------------------
# # PLOT 2: Color by Stress Label (0/1)
# # -----------------------------
# plt.figure(figsize=(8, 6))
# sc = plt.scatter(Z[:, 0], Z[:, 1], s=2, c=labels, cmap="coolwarm", alpha=0.9)
# cb = plt.colorbar(sc, ticks=[0, 1])
# cb.set_label("Stress label (0=no, 1=yes)")
# plt.title("t-SNE of EMOCA Expressions — Colored by Stress Label")
# plt.tight_layout()
# plt.savefig(OUT_LABEL_PNG, dpi=300)
# plt.close()
# print("Saved:", OUT_LABEL_PNG)

# # -----------------------------
# # PLOT 3 (optional): Color by Phase, if exists
# # -----------------------------
# if phase is not None:
#     plt.figure(figsize=(10, 8))
#     phase_unique = np.unique(phase)
#     for ph in phase_unique:
#         idx = np.where(phase == ph)[0]
#         plt.scatter(Z[idx, 0], Z[idx, 1], s=2, label=f"Phase {ph}")
#     plt.title("t-SNE of EMOCA Expressions — Colored by Phase")
#     plt.legend(markerscale=4, fontsize=8)
#     plt.tight_layout()
#     plt.savefig(OUT_PHASE_PNG, dpi=300)
#     plt.close()
#     print("Saved:", OUT_PHASE_PNG)
# else:
#     print("No 'phase' column in CSV → skipping phase plot.")



# Z:      (N, 2) t-SNE coordinates
# subjects: array of subject IDs as strings
# labels:   0 / 1


plt.figure(figsize=(10, 8))

# ------------------------------------
# 1) BASE LAYER: color = subject
# ------------------------------------
subjects_unique = np.unique(subjects)
cmap = plt.cm.get_cmap("tab20", len(subjects_unique))

# assign each subject a stable color
sub_idx_map = {sid: i for i, sid in enumerate(subjects_unique)}
colors = np.array([cmap(sub_idx_map[sid]) for sid in subjects])

plt.scatter(
    Z[:, 0], Z[:, 1],
    s=2,
    c=colors,
    alpha=0.80,
    linewidths=0
)

# ------------------------------------
# 2) STRESS OVERLAY (label = 1)
# ------------------------------------
stress_mask = labels == 1

plt.scatter(
    Z[stress_mask, 0],
    Z[stress_mask, 1],
    s=6,
    facecolors="white",   # white center
    edgecolors="black",   # thin contour
    linewidths=0.25,
    alpha=0.65,
    label="Stress (label=1)"
)

plt.title("t-SNE of EMOCA Expressions — Subject Islands with Stress Overlay")
plt.legend(loc="upper right", markerscale=2)
plt.tight_layout()
plt.savefig("tsne_subject_plus_stress_soft.png", dpi=300)
plt.close()


# plt.figure(figsize=(10, 8))

# subjects_unique = np.unique(subjects)
# cmap = plt.cm.get_cmap("tab20", len(subjects_unique))

# # 1) base layer: all points colored by subject, no label info yet
# sub_idx_map = {sid: i for i, sid in enumerate(subjects_unique)}
# colors = [cmap(sub_idx_map[sid]) for sid in subjects]
# colors = np.array(colors)

# plt.scatter(Z[:, 0], Z[:, 1], s=2, c=colors, alpha=0.6)

# # 2) overlay: stressed points (label==1) with black edge so they stand out
# stress_mask = labels == 1
# plt.scatter(
#     Z[stress_mask, 0],
#     Z[stress_mask, 1],
#     s=8,
#     facecolors='none',   # keep subject color visible underneath
#     edgecolors='k',      # black outline to mark stress
#     linewidths=0.4,
#     label="Stress (label=1)",
# )

# plt.title("t-SNE of EMOCA Expressions — Subject Islands with Stress Overlay")
# # small legend only for stress marker
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig("tsne_subject_plus_stress.png", dpi=300)
# plt.close()
