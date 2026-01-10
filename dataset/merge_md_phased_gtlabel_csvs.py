#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ROOT = Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/phased_csvs")
OUT_CSV = ROOT / "MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN.csv"

SUFFIX = "FINAL_PERFRAME_ENRICHED_PHASED.csv"

REQUIRED_COLS = ["subject_id", "Phase", "GTlabel"]

# --------------------------------------------------
# LOAD + FILTER
# --------------------------------------------------
all_dfs = []
subject_stats = []

files = sorted(ROOT.glob(f"*{SUFFIX}"))

print(f"\nFound {len(files)} MD phased CSV files\n")

for f in files:
    df = pd.read_csv(f)

    print(f"Processing {f.name}")

    # --- sanity check ---
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{f.name} is missing columns: {missing_cols}")

    n_total = len(df)

    # --- keep only frames with valid Phase and GTlabel ---
    df_clean = df[
        df["Phase"].notna() &
        df["GTlabel"].isin([0, 1])
    ].copy()

    n_kept = len(df_clean)

    # --- subject name ---
    subject = (
        df_clean["subject_id"].iloc[0]
        if "subject_id" in df_clean.columns and len(df_clean) > 0
        else f.stem
    )

    # --- label distribution ---
    n_stress = (df_clean["GTlabel"] == 1).sum()
    n_nostress = (df_clean["GTlabel"] == 0).sum()

    p_stress = 100.0 * n_stress / max(n_kept, 1)
    p_nostress = 100.0 * n_nostress / max(n_kept, 1)

    # --- NaN check ---
    nan_counts = df_clean.isna().sum()
    n_nan_total = int(nan_counts.sum())

    print(
        f"  subject_id: {subject}\n"
        f"  Frames kept: {n_kept} / {n_total}\n"
        f"  Stress: {n_stress} ({p_stress:.1f}%) | "
        f"Non-stress: {n_nostress} ({p_nostress:.1f}%)\n"
        f"  Total NaNs in subject DF: {n_nan_total}\n"
    )

    subject_stats.append({
        "subject_id": subject,
        "Frames_kept": n_kept,
        "Stress_%": p_stress,
        "NonStress_%": p_nostress,
        "Total_NaNs": n_nan_total
    })

    all_dfs.append(df_clean)

# --------------------------------------------------
# MERGE
# --------------------------------------------------
merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)

print("\n================ MERGED DATASET ================\n")
print(f"Total frames after merge: {len(merged_df)}")
print(f"Unique subjects: {merged_df['subject_id'].nunique()}")

print("\nFrames per subject:")
print(merged_df["subject_id"].value_counts())

# --------------------------------------------------
# NaN REPORT (merged)
# --------------------------------------------------
nan_report = merged_df.isna().sum()
nan_cols = nan_report[nan_report > 0]

print("\nNaN report (merged CSV):")
if len(nan_cols) == 0:
    print("  ✔ No NaN values found")
else:
    print(nan_cols.sort_values(ascending=False))

# --------------------------------------------------
# SAVE
# --------------------------------------------------
merged_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved merged CSV to:\n{OUT_CSV}\n")
