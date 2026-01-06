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
# #        'delta_pose4', 'delta_pose5',  'Breathing.Rate', 'Heart.Rate', 'Perinasal.Perspiration', 'Gaze.X.Pos', 'Gaze.Y.Pos', 'Stimulus', 'Drive', 'GazeVel_X',
# #        'GazeVel_Y', 'GazeVel', 'GazeAcc_X', 'GazeAcc_Y', 'GazeAcc',
# #        'GazeVel_mean_1s', 'GazeVel_std_1s', 'GazeVel_mean_3s',
# #        'GazeVel_std_3s', 'GazeDispersion_2s' ]
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
# # print(f"Remaining features:", X.columns)
# # X = X.select_dtypes(include=[np.number]).fillna(0)
# # print(f"Remaining feature dim: {X.shape[1]}")

# # # ---------- 4) scale ----------
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # # ---------- 5) PCA ----------
# # pca = PCA(n_components=2, random_state=0)
# # X_pca = pca.fit_transform(X_scaled)
# # plt.figure(figsize=(6,5))
# # plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", s=1)
# # plt.title("PCA (filtered multimodal features, color = stress label)")
# # plt.xlabel("PC1")
# # plt.ylabel("PC2")
# # plt.tight_layout()
# # plt.savefig(OUT_PCA, dpi=200)
# # plt.close()
# # print(f"✅ saved PCA → {OUT_PCA}")


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Append missing MD subjects to all_subjects_merged.csv.

# Assumptions:
# - Run from repo root: /home/vivib/emoca/emoca
# - Data layout:
#     dataset/all_subjects_merged.csv
#     dataset/FINAL_PERFRAME_CLEAN_PHASED/Txxx_MD*_FINAL_PERFRAME_CLEAN_PHASED.csv
# - Subject id is 'Txxx' (e.g., 'T001') and appears as 'subject_id' in merged.
# """

# from pathlib import Path
# import re
# import numpy as np
# import pandas as pd

# # -------- paths --------
# ROOT = Path("dataset")
# MERGED_PATH = ROOT / "all_subjects_merged.csv"
# MD_DIR = ROOT / "FINAL_PERFRAME_CLEAN_PHASED"
# OUT_PATH = ROOT / "all_subjects_merged_fixed.csv"

# print("Merged CSV :", MERGED_PATH.resolve())
# print("MD DIR     :", MD_DIR.resolve())

# # -------- load merged --------
# df_merged = pd.read_csv(MERGED_PATH, low_memory=False)
# print("\n=== ORIGINAL MERGED SHAPE ===")
# print(df_merged.shape)

# if "subject_id" not in df_merged.columns:
#     raise RuntimeError("Column 'subject_id' not found in merged CSV.")

# merged_subjects = set(df_merged["subject_id"].astype(str).unique())
# print(f"Subjects in merged: {len(merged_subjects)}")

# # -------- subjects from MD folder --------
# md_subjects = set()
# for f in MD_DIR.glob("T*_MD*_FINAL_PERFRAME_CLEAN_PHASED.csv"):
#     m = re.search(r"(T\d+)_MD", f.name)
#     if m:
#         md_subjects.add(m.group(1))

# print(f"Subjects in MD folder: {len(md_subjects)}")

# missing_subjects = sorted(md_subjects - merged_subjects)
# print("\n=== MISSING SUBJECTS TO ADD ===")
# print(missing_subjects)

# if not missing_subjects:
#     print("No missing subjects, nothing to do.")
#     raise SystemExit

# # -------- helper to load & align one subject file --------
# frames_to_add = []

# for subj in missing_subjects:
#     pattern = f"{subj}_MD*_FINAL_PERFRAME_CLEAN_PHASED.csv"
#     files = sorted(MD_DIR.glob(pattern))
#     if not files:
#         print(f"[WARNING] No MD files found for subject {subj} with pattern {pattern}")
#         continue

#     for f in files:
#         print(f"\nLoading {f.name} for subject {subj} ...")
#         df_md = pd.read_csv(f, low_memory=False)

#         # Ensure we have 'subject_id' and 'Subject'
#         if "subject_id" not in df_md.columns:
#             df_md["subject_id"] = subj
#         else:
#             df_md["subject_id"] = df_md["subject_id"].astype(str).fillna(subj)

#         if "Subject" not in df_md.columns:
#             df_md["Subject"] = subj
#         else:
#             df_md["Subject"] = df_md["Subject"].astype(str).fillna(subj)

#         # Ensure frame_idx exists
#         if "frame_idx" not in df_md.columns:
#             df_md["frame_idx"] = np.arange(len(df_md), dtype=int)

#         # Add any columns that exist in merged but not in this MD csv as NaN
#         for col in df_merged.columns:
#             if col not in df_md.columns:
#                 df_md[col] = np.nan

#         # Drop extra columns that are not in merged, and reorder
#         df_md = df_md[df_merged.columns]

#         print(f"  -> rows to add from {f.name}: {len(df_md)}")
#         frames_to_add.append(df_md)

# if not frames_to_add:
#     print("\nNo rows collected for missing subjects. Nothing written.")
#     raise SystemExit

# df_added = pd.concat(frames_to_add, ignore_index=True)
# print("\n=== ROWS ADDED (ALL MISSING SUBJECTS) ===")
# print(df_added.shape)

# # -------- concatenate and save --------
# df_new = pd.concat([df_merged, df_added], ignore_index=True)
# print("\n=== NEW MERGED SHAPE ===")
# print(df_new.shape)

# df_new.to_csv(OUT_PATH, index=False)
# print(f"\nSaved updated merged CSV to:\n  {OUT_PATH.resolve()}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
