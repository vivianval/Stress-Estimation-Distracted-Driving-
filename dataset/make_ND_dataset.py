#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge ND phased EMOCA CSVs, add biosignals from RAW R-friendly CSVs (ND drive),
compute per-phase GTlabel (P2/P4=1 else 0), and compute delta1 velocities
(consecutive-frame differences) within each phase per subject_id.

Input:
  ND phased EMOCA CSVs:
    /home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/phased_csvs/T047_exp_pose_PHASED.csv
    and all *_exp_pose_PHASED.csv

  RAW R-friendly per-subject_id CSVs:
    /media/storage/vivib/Structured Study Data/R-Friendly Study Data/T047.csv

Output:
  One merged CSV with:
    - subject_id, t_sec, Phase, GTlabel
    - exp_*, pose_*
    - Perinasal.Perspiration, Heart.Rate, Breathing.Rate (interpolated)
    - delta1_exp_*, delta1_pose_* (+ delta1_ for bios too if you want)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------
# Config / constants
# ---------------------------

MODALITIES = ["Breathing.Rate", "Heart.Rate", "Perinasal.Perspiration"]

# default ND drive id (your Pavlidis script)
DEFAULT_ND_DRIVE_ID = 4

SUBJ_RX = re.compile(r"(T\d{3,})", re.IGNORECASE)


# ---------------------------
# Helpers
# ---------------------------

def infer_subject_id_id_from_path(p: Path) -> str:
    m = SUBJ_RX.search(p.stem)
    if not m:
        raise ValueError(f"Cannot infer subject_id id from filename: {p.name}")
    return m.group(1).upper()


def find_col(df, names):
    for c in df.columns:
        if c.lower() in [n.lower() for n in names]:
            return c
    return None


def detect_phase_col(df: pd.DataFrame) -> str:
    # common candidates
    for cand in ["Phase", "phase", "PHASE", "phase_id"]:
        if cand in df.columns:
            return cand
    # fallback: any column containing "phase"
    for c in df.columns:
        if "phase" in c.lower():
            return c
    raise ValueError("No Phase column found in ND phased CSV.")


def detect_time_col(df: pd.DataFrame) -> str:
    # prefer t_sec if available
    for cand in ["t_sec", "time_sec", "TimeSec", "timestamp_s", "time"]:
        if cand in df.columns:
            return cand
    # if none, try 'Time' (sometimes used)
    if "Time" in df.columns:
        return "Time"
    raise ValueError("No time column found. Need t_sec (recommended) or Time.")


def pick_emoca_cols(df: pd.DataFrame):
    exp_cols = [c for c in df.columns if c.startswith("exp_")]
    pose_cols = [c for c in df.columns if c.startswith("pose_")]
    if not exp_cols or not pose_cols:
        raise ValueError("Missing EMOCA columns: expected exp_* and pose_*.")
    # sort by numeric suffix if possible
    def key_num(c):
        m = re.search(r"(\d+)$", c)
        return int(m.group(1)) if m else 10**9
    exp_cols = sorted(exp_cols, key=key_num)
    pose_cols = sorted(pose_cols, key=key_num)
    return exp_cols, pose_cols


def phase_to_gtlabel(phase_value) -> int:
    if phase_value is None or (isinstance(phase_value, float) and not np.isfinite(phase_value)):
        return 0
    s = str(phase_value).strip().upper()
    # accept "P2", "2", 2, etc.
    if s in {"P2", "2"}:
        return 1
    if s in {"P4", "4"}:
        return 1
    return 0


def interp_signal(raw_t, raw_v, query_t):
    """
    raw_t, raw_v: 1D arrays (finite)
    query_t: 1D float array
    returns interpolated float array (same length as query_t)
    """
    raw_t = np.asarray(raw_t, dtype=float)
    raw_v = np.asarray(raw_v, dtype=float)
    query_t = np.asarray(query_t, dtype=float)

    m = np.isfinite(raw_t) & np.isfinite(raw_v)
    if m.sum() < 2:
        return np.full_like(query_t, np.nan, dtype=float)

    t = raw_t[m]
    v = raw_v[m]
    # ensure monotonic increasing for np.interp
    if np.any(np.diff(t) < 0):
        idx = np.argsort(t)
        t = t[idx]
        v = v[idx]

    return np.interp(query_t, t, v, left=v[0], right=v[-1])


def compute_delta1_within_groups(df: pd.DataFrame, cols, group_cols):
    """
    delta1 for cols: difference between consecutive rows INSIDE each group.
    First row of each group gets 0.
    group_cols: e.g. ["subject_id","Phase"]
    """
    out = df.copy()
    for c in cols:
        dc = "delta1_" + c
        out[dc] = out.groupby(group_cols)[c].diff().fillna(0.0)
    return out


def read_raw_subject_id(raw_path: Path, nd_drive_id: int):
    """
    Returns raw ND slice with Time and modalities.
    """
    rdf = pd.read_csv(raw_path, low_memory=False)
    req = ["Time", "Drive"] + MODALITIES
    miss = [c for c in req if c not in rdf.columns]
    if miss:
        raise ValueError(f"{raw_path.name}: missing columns {miss}")

    rdf = rdf.copy()
    rdf["Time"] = pd.to_numeric(rdf["Time"], errors="coerce")
    rdf["Drive"] = pd.to_numeric(rdf["Drive"], errors="coerce")
    for m in MODALITIES:
        rdf[m] = pd.to_numeric(rdf[m], errors="coerce")

    rdf = rdf[rdf["Drive"] == nd_drive_id].copy()
    rdf = rdf[np.isfinite(rdf["Time"].to_numpy(dtype=float))].sort_values("Time")
    return rdf


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phased_dir", type=Path,
                    default=Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/phased_csvs"),
                    help="Directory containing *_exp_pose_PHASED.csv files")
    ap.add_argument("--raw_dir", type=Path,
                    default=Path("/media/storage/vivib/Structured Study Data/R-Friendly Study Data"),
                    help="Directory containing per-subject_id RAW T###.csv")
    ap.add_argument("--out_csv", type=Path,
                    default=Path("/home/vivib/emoca/emoca/training/comparisons/ND_ALL_subject_idS_EXPPOSE_BIO_DELTA1_PHASED_GTLABEL.csv"))
    ap.add_argument("--nd_drive_id", type=int, default=DEFAULT_ND_DRIVE_ID,
                    help="ND drive id in RAW files (default 4)")
    ap.add_argument("--add_bio_delta1", action="store_true",
                    help="Also compute delta1 for biosignals (optional)")
    args = ap.parse_args()

    phased_files = sorted(args.phased_dir.glob("*_exp_pose_PHASED.csv"))
    if not phased_files:
        raise SystemExit(f"No phased files found in {args.phased_dir} matching *_exp_pose_PHASED.csv")

    merged_rows = []
    report = []

    for f in phased_files:
        sid = infer_subject_id_id_from_path(f)
        raw_path = args.raw_dir / f"{sid}.csv"

        try:
            df = pd.read_csv(f, low_memory=False)
            phase_col = detect_phase_col(df)
            time_col = detect_time_col(df)
            exp_cols, pose_cols = pick_emoca_cols(df)

            # basic required columns
            df = df.copy()
            df["subject_id"] = sid
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
            df = df[np.isfinite(df[time_col].to_numpy(dtype=float))].copy()
            df = df.sort_values(time_col).reset_index(drop=True)

            # keep/standardize columns
            df.rename(columns={time_col: "t_sec", phase_col: "Phase"}, inplace=True)

            # GTlabel from Phase
            df["GTlabel"] = df["Phase"].apply(phase_to_gtlabel).astype(int)

            # read RAW ND and interpolate biosignals to frame times
            if raw_path.exists():
                rdf = read_raw_subject_id(raw_path, nd_drive_id=args.nd_drive_id)
                t = df["t_sec"].to_numpy(dtype=float)

                for mod in MODALITIES:
                    df[mod] = interp_signal(
                        rdf["Time"].to_numpy(dtype=float),
                        rdf[mod].to_numpy(dtype=float),
                        t
                    )
                bio_ok = True
            else:
                # keep bios columns as NaN if raw missing
                for mod in MODALITIES:
                    df[mod] = np.nan
                bio_ok = False

            # compute delta1 for EMOCA within (subject_id, Phase)
            cols_for_delta = exp_cols + pose_cols
            df = compute_delta1_within_groups(df, cols_for_delta, group_cols=["subject_id", "Phase"])

            # optional: delta1 bios too
            if args.add_bio_delta1:
                df = compute_delta1_within_groups(df, MODALITIES, group_cols=["subject_id", "Phase"])

            # order columns nicely
            keep_first = ["subject_id", "frame_id", "t_sec", "Phase", "GTlabel"]
            emoca_cols = exp_cols + pose_cols
            delta_cols = ["delta1_" + c for c in emoca_cols]
            bio_cols = MODALITIES
            bio_delta_cols = (["delta1_" + c for c in bio_cols] if args.add_bio_delta1 else [])

            # only include columns that exist
            final_cols = []
            for group in [keep_first, emoca_cols, bio_cols, delta_cols, bio_delta_cols]:
                for c in group:
                    if c in df.columns:
                        final_cols.append(c)

            df = df[final_cols].copy()

            merged_rows.append(df)

            report.append({
                "subject_id": sid,
                "phased_csv": str(f),
                "raw_csv": str(raw_path),
                "raw_found": bool(raw_path.exists()),
                "bio_interpolated": bool(bio_ok),
                "n_frames": int(len(df)),
                "t_min": float(df["t_sec"].min()),
                "t_max": float(df["t_sec"].max()),
                "pos_frac(GTlabel)": float(df["GTlabel"].mean()),
            })

            print(f"[OK] {sid}: frames={len(df)} bio={'yes' if bio_ok else 'no'} pos_frac={df['GTlabel'].mean():.3f}")

        except Exception as e:
            print(f"[SKIP] {sid} ({f.name}): {e}")
            report.append({
                "subject_id": sid,
                "phased_csv": str(f),
                "status": "skip_error",
                "error": str(e),
            })
            continue

    if not merged_rows:
        raise SystemExit("No subject_ids processed successfully.")

    out_df = pd.concat(merged_rows, axis=0, ignore_index=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[SAVE] merged ND -> {args.out_csv}")
    print(f"       rows={len(out_df)} subject_ids={out_df['subject_id'].nunique()}")

    rep_path = args.out_csv.with_suffix(".report.csv")
    pd.DataFrame(report).to_csv(rep_path, index=False)
    print(f"[SAVE] report -> {rep_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-






# IN_CSV  = Path("/home/vivib/emoca/emoca/training/comparisons/ND_ALL_subject_idS_EXPPOSE_BIO_DELTA1_PHASED_GTLABEL.csv")
# OUT_CSV = IN_CSV.with_name(IN_CSV.stem + "_PHASEONLY.csv")

# df = pd.read_csv(IN_CSV, low_memory=False)

# # Treat empty strings / whitespace as missing
# phase = df["Phase"].astype(str).str.strip()
# phase = phase.replace({"": np.nan, "nan": np.nan, "None": np.nan})

# before = len(df)
# df = df[phase.notna()].copy()
# after = len(df)

# df.to_csv(OUT_CSV, index=False)

# print(f"[CLEAN] Dropped rows without Phase: {before-after} / {before}")
# print(f"[SAVE] {OUT_CSV}")
