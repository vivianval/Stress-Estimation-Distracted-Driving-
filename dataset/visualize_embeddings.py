# # # #!/usr/bin/env python3
# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.decomposition import PCA
# # from sklearn.manifold import TSNE
# # import matplotlib.pyplot as plt

# # # ---------- paths ----------
# # DATA_FILE = Path("/home/vivib/emoca/emoca/dataset/all_subjects_merged.csv")
# # OUT_PCA   = Path("/home/vivib/emoca/emoca/dataset/pca_stress_filtered.png")
# # OUT_TSNE  = Path("/home/vivib/emoca/emoca/dataset/tsne_stress_filtered.png")

# # # ---------- 1) load data ----------
# # df = pd.read_csv(DATA_FILE, low_memory=False)
# # print(f"Loaded merged dataset: {df.shape}")

# # # ---------- 2) define exclusions ----------
# # drop_cols = [
# #     "Subject", "subject_id", "frame_idx", "t_sec", "label", "Label",
# #     "dt", 'delta_pose0', 'delta_pose1', 'delta_pose2', 'delta_pose3',
# #        'delta_pose4', 'delta_pose5',  'Breathing.Rate', 'Heart.Rate', 'Perinasal.Perspiration', 'exp_00', 'exp_01', 'exp_02', 'exp_03', 'exp_04', 'exp_05', 'exp_06',
# #        'exp_07', 'exp_08', 'exp_09', 'exp_10', 'exp_11', 'exp_12', 'exp_13',
# #        'exp_14', 'exp_15', 'exp_16', 'exp_17', 'exp_18', 'exp_19', 'exp_20',
# #        'exp_21', 'exp_22', 'exp_23', 'exp_24', 'exp_25', 'exp_26', 'exp_27',
# #        'exp_28', 'exp_29', 'exp_30', 'exp_31', 'exp_32', 'exp_33', 'exp_34',
# #        'exp_35', 'exp_36', 'exp_37', 'exp_38', 'exp_39', 'exp_40', 'exp_41',
# #        'exp_42', 'exp_43', 'exp_44', 'exp_45', 'exp_46', 'exp_47', 'exp_48',
# #        'exp_49', 'pose_00', 'pose_01', 'pose_02', 'pose_03', 'pose_04', 'pose_05'
# # ]
# # present_drops = [c for c in drop_cols if c in df.columns]
# # print(f"Dropping {len(present_drops)} non-feature columns: {present_drops}")

# # # find label column (case-insensitive)
# # label_col = None
# # for c in df.columns:
# #     if c.lower() in {"label", "stress", "stress_label", "y"}:
# #         label_col = c
# #         break
# # if label_col is None:
# #     raise RuntimeError("Could not find label column!")
# # y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

# # # ---------- 3) feature matrix ----------
# # X = df.drop(columns=present_drops, errors="ignore")
# # print(X.columns)
# # X = X.select_dtypes(include=[np.number]).fillna(0)
# # print(f"Remaining feature dim: {X.shape[1]}")

# # # ---------- 4) scale ----------
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # # ---------- 5) PCA ----------
# # # pca = PCA(n_components=2, random_state=0)
# # # X_pca = pca.fit_transform(X_scaled)
# # # plt.figure(figsize=(6,5))
# # # plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", s=1)
# # # plt.title("PCA (filtered multimodal features, color = stress label)")
# # # plt.xlabel("PC1")
# # # plt.ylabel("PC2")
# # # plt.tight_layout()
# # # plt.savefig(OUT_PCA, dpi=200)
# # # plt.close()
# # # print(f"✅ saved PCA → {OUT_PCA}")

# # # ---------- 6) t-SNE ----------
# # tsne = TSNE(n_components=2, random_state=0, perplexity=50,
# #             learning_rate="auto", init="pca")
# # limit = 600000  # subsample if huge
# # subset = slice(None) if len(X_scaled) <= limit else slice(0, limit)
# # X_tsne = tsne.fit_transform(X_scaled[subset])
# # plt.figure(figsize=(6,5))
# # plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y[subset], cmap="coolwarm", s=1)
# # plt.title("t-SNE (filtered multimodal features, color = stress label)")
# # plt.tight_layout()
# # plt.savefig(OUT_TSNE, dpi=200)
# # plt.close()
# # print(f"✅ saved t-SNE → {OUT_TSNE}")

# #!/usr/bin/env python3
# """
# Visualize the entire MD dataset (no subsampling) using:
#   • Multicore t-SNE (fast parallel)
#   • UMAP (large-scale 2D embedding)

# Each point = one frame (row in CSV)
# Color = stress label (0 = non-stress, 1 = stress)
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from openTSNE import TSNE as MTSNE
# import umap 
# # ========== CONFIG ==========
# DATA_FILE = Path("/home/vivib/emoca/emoca/dataset/all_subjects_merged.csv")
# OUT_DIR   = Path("/home/vivib/emoca/emoca/dataset/embeddings_all")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # ========== LOAD DATA ==========
# df = pd.read_csv(DATA_FILE, low_memory=False)
# print(f"Loaded merged dataset: {df.shape}")

# # ========== LABEL ==========
# label_col = next((c for c in df.columns if str(c).lower() in {"label", "stress", "stress_label", "y"}), None)
# if label_col is None:
#     raise RuntimeError("Could not find label column!")
# y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).values

# #####========== FEATURE SELECTION ==========
# drop_cols = [
#     "Subject", "subject_id", "frame_idx", "t_sec", "label", "Label",
#     "dt", 'Stimulus', 'Drive'
# ]
# present_drops = [c for c in drop_cols if c in df.columns]
# X = df.drop(columns=present_drops, errors="ignore").select_dtypes(include=[np.number]).fillna(0)
# print(f"Remaining feature dim: {X.shape[1]}")
# print(f"FEATURES VISUALIZED: {X.columns}")

# # # ========== SCALE ==========
# print("Scaling features...")
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# # pca = PCA(n_components=50, random_state=0)
# # X_pca = pca.fit_transform(X_scaled).astype(np.float32)

# # # 2) stratified/balanced subset for fitting UMAP
# # def balanced_indices(y, per_class=150000, seed=0):
# #     rng = np.random.default_rng(seed)
# #     keep = []
# #     for lab in np.unique(y):
# #         idx = np.where(y == lab)[0]
# #         if len(idx) > per_class:
# #             keep.append(rng.choice(idx, per_class, replace=False))
# #         else:
# #             keep.append(idx)
# #     return np.concatenate(keep)

# # fit_idx = balanced_indices(y, per_class=300_000)   # ~300k total if 2 classes
# # X_fit, y_fit = X_pca[fit_idx], y[fit_idx]

# # # 3) UMAP fit on subset, then transform all points in batches
# # reducer = umap.UMAP(
# #     n_neighbors=50,
# #     min_dist=0.1,
# #     metric="euclidean",
# #     n_components=2,
# #     random_state=0,
# #     low_memory=True,
# #     verbose=True,              # show progress
# # )

# # print("Fitting UMAP on subset…")
# # reducer.fit(X_fit)

# # print("Transforming full dataset in chunks…")
# # def umap_transform_in_chunks(model, X, chunk=200_000):
# #     zs = []
# #     for i in range(0, len(X), chunk):
# #         zs.append(model.transform(X[i:i+chunk]))
# #     return np.vstack(zs)

# # Z_umap = umap_transform_in_chunks(reducer, X_pca, chunk=200_000)

# # plt.figure(figsize=(7,6))
# # sc = plt.scatter(Z_umap[:,0], Z_umap[:,1], c=y, s=1, cmap="coolwarm", alpha=0.8)
# # cb = plt.colorbar(sc, ticks=[0,1]); cb.set_label("label")
# # plt.title("UMAP • All Frames (fit on balanced subset, transform all)")
# # plt.tight_layout()
# # plt.savefig(OUT_DIR/"umap_all_fit_subset_transform_all.png", dpi=300)
# # plt.close()

# # ============================================================
# # 1️⃣ MULTICORE t-SNE (for large datasets)
# # ============================================================
# try:
    
#     tsne_model = MTSNE(
#         n_components=2,
#         perplexity=50,
#         n_jobs=-1,               # use all CPU cores
#         random_state=0,
#         initialization="pca",
#         metric="euclidean"
#     )
#     print("Running Multicore t-SNE on full dataset...")
#     Z_tsne = tsne_model.fit(X_scaled)
#     plt.figure(figsize=(7,6))
#     sc = plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=y, s=1, cmap="coolwarm", alpha=0.8)
#     cb = plt.colorbar(sc, ticks=[0,1]); cb.set_label("label")
#     plt.title("Multicore t-SNE • Biosignals only (color = stress label)")
#     plt.tight_layout()
#     out_tsne = OUT_DIR/"tsne_multicore_all.png"
#     plt.savefig(out_tsne, dpi=300); plt.close()
#     print(f"✅ Saved t-SNE → {out_tsne}")
# except Exception as e:
#     print(f"⚠️ Multicore t-SNE failed ({e}); falling back to sklearn TSNE (may be slow)")
#     from sklearn.manifold import TSNE
#     Z_tsne = TSNE(
#         n_components=2, random_state=0,
#         perplexity=50, learning_rate="auto", init="pca"
#     ).fit_transform(X_scaled)
#     plt.figure(figsize=(7,6))
#     sc = plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=y, s=1, cmap="coolwarm", alpha=0.8)
#     cb = plt.colorbar(sc, ticks=[0,1]); cb.set_label("label")
#     plt.title("Multicore t-SNE • Biosignals  only (color = stress label)")
#     plt.tight_layout()
#     out_tsne = OUT_DIR/"tsne_sklearn_all.png"
#     plt.savefig(out_tsne, dpi=300); plt.close()
#     print(f"✅ Saved fallback t-SNE → {out_tsne}")

# # ============================================================
# # 2️⃣ UMAP (faster & scalable)
# # ============================================================
# # try:
# #     import umap
# #     print("Running UMAP...")
# #     reducer = umap.UMAP(
# #         n_neighbors=50,
# #         n_jobs = -1,
# #         min_dist=0.1,
# #         metric="euclidean",
# #         random_state=0,
# #         n_components=2
# #     )
# #     Z_umap = reducer.fit_transform(X_scaled)
# #     plt.figure(figsize=(7,6))
# #     sc = plt.scatter(Z_umap[:,0], Z_umap[:,1], c=y, s=1, cmap="coolwarm", alpha=0.8)
# #     cb = plt.colorbar(sc, ticks=[0,1]); cb.set_label("label")
# #     plt.title("UMAP • Expression only (color = stress label)")
# #     plt.tight_layout()
# #     out_umap = OUT_DIR/"umap_all.png"
# #     plt.savefig(out_umap, dpi=300); plt.close()
# #     print(f"✅ Saved UMAP → {out_umap}")
# # except Exception as e:
# #     print(f"⚠️ UMAP failed: {e}")

# print("✅ Done — all embeddings computed and saved.")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-subject embeddings (t-SNE and optional UMAP) for EMOCA/physio features.

Each dot = one frame for that subject.
Color = stress label (0/1).

Choose which feature family to visualize with MODES below.
One PNG per subject is saved in OUT_DIR/<mode>/.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----------------- CONFIG -----------------
CSV     = Path("/home/vivib/emoca/emoca/dataset/all_subjects_merged.csv")
OUT_DIR = Path("/home/vivib/emoca/emoca/dataset/emb_per_subject")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose one or more modes to render
MODES = [
 #   "expressions",   # exp_00..exp_49
    "pose",          # pose_00..pose_05
    "biosignals",    # Heart.Rate, Perinasal.Perspiration, Breathing.Rate
    # "all",         # uncomment to include expressions+pose+biosignals
]

# Per-subject limits (keeps plots readable & speeds up)
MAX_PER_CLASS = 15000     # cap points per label per subject (adjust as you like)
SEED          = 0         # reproducibility
DO_UMAP       = True      # set False if you only want t-SNE

# ------------------------------------------

rng = np.random.default_rng(SEED)

# try fast multicore t-SNE; fallback to sklearn if not available
def run_tsne(Xz, perplexity):
    Z = None
    try:
        from openTSNE import TSNE as MTSNE
        Z = MTSNE(
            n_components=2,
            perplexity=float(perplexity),
            n_jobs=-1,
            random_state=SEED,
            initialization="pca",
            metric="euclidean",
        ).fit(Xz)
    except Exception:
        from sklearn.manifold import TSNE
        Z = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=float(perplexity),
            learning_rate="auto",
            init="pca",
        ).fit_transform(Xz)
    return Z

def run_umap(Xz):
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=40, min_dist=0.1, metric="euclidean",
            n_components=2, random_state=SEED, low_memory=True, verbose=False
        )
        return reducer.fit_transform(Xz)
    except Exception:
        return None

def balanced_idx_per_subject(y, max_per_class):
    """stratified cap per label for a single subject array y"""
    idxs = []
    for lab in np.unique(y):
        lab_idx = np.where(y == lab)[0]
        if len(lab_idx) > max_per_class:
            idxs.append(rng.choice(lab_idx, size=max_per_class, replace=False))
        else:
            idxs.append(lab_idx)
    return np.concatenate(idxs)

def choose_perplexity(n):
    # rule-of-thumb: perplexity in [5, 50], and < n/3
    return max(5, min(50, int(0.25 * max(10, n),)))

# ---------- load ----------
df = pd.read_csv(CSV, low_memory=False)
print("Loaded:", df.shape)

# subject column guess
subj_col = None
for c in ["Subject", "subject", "subject_id"]:
    if c in df.columns:
        subj_col = c
        break
if subj_col is None:
    raise RuntimeError("Could not find a subject column (Subject/subject/subject_id).")

# label column guess
label_col = next((c for c in df.columns if str(c).lower() in {"label", "stress", "stress_label", "y"}), None)
if label_col is None:
    raise RuntimeError("Could not find label column (label/stress/stress_label/y).")

# feature groups
exp_cols  = [c for c in df.columns if re.fullmatch(r"exp_\d{2}", str(c))]
pose_cols = [c for c in df.columns if re.fullmatch(r"pose_\d{2}", str(c))]
bio_cols  = [c for c in df.columns if c in {"Heart.Rate","Perinasal.Perspiration","Breathing.Rate"}]

def cols_for_mode(mode: str):
    if mode == "expressions":
        return exp_cols
    if mode == "pose":
        return pose_cols
    if mode == "biosignals":
        return bio_cols
    if mode == "all":
        return exp_cols + pose_cols + bio_cols
    raise ValueError(f"Unknown mode: {mode}")

subjects = sorted(df[subj_col].dropna().unique().tolist())
print(f"Found {len(subjects)} subjects.")

for mode in MODES:
    cols = cols_for_mode(mode)
    if not cols:
        print(f"[SKIP {mode}] No columns found for this mode.")
        continue

    out_mode_dir = OUT_DIR / mode
    out_mode_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Mode: {mode} | features: {len(cols)} ===")
    for s in subjects:
        df_s = df[df[subj_col] == s]

        # X, y for this subject
        X = df_s[cols].select_dtypes(include=[np.number]).to_numpy()
        y = pd.to_numeric(df_s[label_col], errors="coerce").fillna(0).astype(int).to_numpy()

        # clean
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        if X.shape[0] < 10:
            print(f"[{mode}] {s}: too few rows after cleaning, skip.")
            continue

        # balance per label within subject
        keep = balanced_idx_per_subject(y, MAX_PER_CLASS)
        X, y = X[keep], y[keep]

        # scale within subject (subject-specific z-scoring)
        Xz = StandardScaler().fit_transform(X).astype(np.float32)

        # pick perplexity appropriate for this subject
        perp = min(choose_perplexity(len(Xz)), max(5, (len(Xz) // 3) - 1))
        if perp < 5: perp = 5

        # --- t-SNE ---
        Z = run_tsne(Xz, perplexity=perp)

        plt.figure(figsize=(6,5))
        sc = plt.scatter(Z[:,0], Z[:,1], c=y, s=2, cmap="coolwarm", alpha=0.9)
        cb = plt.colorbar(sc, ticks=[0,1]); cb.set_label("label")
        plt.title(f"{s} • t-SNE • {mode}  (n={len(Xz)}, perp={perp})")
        plt.tight_layout()
        png_tsne = out_mode_dir / f"{s}_{mode}_tsne.png"
        plt.savefig(png_tsne, dpi=260)
        plt.close()
        print(f"[SAVE] {png_tsne}")

        # --- UMAP (optional) ---
        if DO_UMAP:
            U = run_umap(Xz)
            if U is not None:
                plt.figure(figsize=(6,5))
                sc = plt.scatter(U[:,0], U[:,1], c=y, s=2, cmap="coolwarm", alpha=0.9)
                cb = plt.colorbar(sc, ticks=[0,1]); cb.set_label("label")
                plt.title(f"{s} • UMAP • {mode}  (n={len(Xz)})")
                plt.tight_layout()
                png_umap = out_mode_dir / f"{s}_{mode}_umap.png"
                plt.savefig(png_umap, dpi=260)
                plt.close()
                print(f"[SAVE] {png_umap}")
            else:
                print(f"[INFO] UMAP not available; skipped for {s} / {mode}.")

