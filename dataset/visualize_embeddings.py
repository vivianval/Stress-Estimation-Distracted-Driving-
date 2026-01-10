


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-subject embeddings (t-SNE and optional UMAP) for 3D expression and pose /physio features.

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

