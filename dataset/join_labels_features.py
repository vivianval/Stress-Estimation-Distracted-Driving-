#!/usr/bin/env python3
import argparse, re, sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

# ================== PATHS (edit if needed) ==================
DATASET_DIR         = Path("/home/vivib/emoca/emoca/dataset")
BIOSIGNALS_DIR      = DATASET_DIR / "BIOSIGNALS_CSVS"     # *_biosignals_PERFRAME.csv
LABELS_DIR          = DATASET_DIR / "LABELS_PERFRAME"     # *_labels_PERFRAME.csv
FEATURES_DIR        = Path("/home/vivib/emoca/emoca/emoca_results/CSV_STREAM")  # T0XX_exp_pose.csv

JOINED_BL_DIR       = DATASET_DIR / "JOINED_BIO+LABELS"   # per-frame biosignals+labels
FINAL_PERFRAME_DIR  = DATASET_DIR / "FINAL_PERFRAME"      # features (+ biosignals if found) + labels + deltas
MASTER_OUT          = DATASET_DIR / "MASTER_all_subjects.csv"  # concatenated final (optional)
# ============================================================

SID_MD_RE = re.compile(r"^(T\d{3,})_MD(\d+)", re.IGNORECASE)

def parse_sid_md_from_file(stem: str) -> Optional[Tuple[str, int]]:
    """
    Expect names like T031_MD7_labels_PERFRAME / T031_MD7_biosignals_PERFRAME
    """
    m = SID_MD_RE.match(stem)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))

def find_biosignals_file(sid: str, md: int) -> Optional[Path]:
    cand = BIOSIGNALS_DIR.glob(f"{sid}_MD{md}_biosignals_PERFRAME.csv")
    return next(cand, None)

def find_labels_file(sid: str, md: int) -> Optional[Path]:
    cand = LABELS_DIR.glob(f"{sid}_MD{md}_labels_PERFRAME.csv")
    return next(cand, None)

def find_features_file(sid: str) -> Optional[Path]:
    """
    Look for 'T0XX_exp_pose.csv' or any file that starts with sid and contains 'exp_pose'
    """
    preferred = FEATURES_DIR / f"{sid}_exp_pose.csv"
    if preferred.exists():
        return preferred
    # fallback: any file containing sid and exp_pose
    for p in FEATURES_DIR.glob(f"*{sid}*exp_pose*.csv"):
        return p
    return None

def pick_frame_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return name of the column representing frame index in features/biosignals/labels.
    Tries several common variants.
    """
    candidates = ["frame_idx", "frame", "frame_id", "Frame", "FrameID"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def add_delta_pose_cols(df: pd.DataFrame, prefix: str = "pose_", n: int = 6) -> pd.DataFrame:
    """
    Adds delta_pose0..delta_pose5 = diff of pose_00..pose_05, per subject.
    Assumes df is already sorted by frame index.
    """
    # find pose columns robustly: pose_00 ... pose_05 (or pose_0 .. pose_5)
    pose_cols = []
    for i in range(n):
        cands = [f"{prefix}{i:02d}", f"{prefix}{i:01d}"]
        c = next((c for c in cands if c in df.columns), None)
        if c is None:
            raise ValueError(f"Missing pose column for index {i}: tried {cands}")
        pose_cols.append(c)

    for i, col in enumerate(pose_cols):
        dcol = f"delta_pose{i}"
        df[dcol] = df[col].astype(float).diff().fillna(0.0)

    return df

def safe_merge_on_frames(left: pd.DataFrame, right: pd.DataFrame, how="inner",
                         left_name="left", right_name="right") -> pd.DataFrame:
    """
    Merge two per-frame DataFrames by auto-detecting frame column names.
    """
    lc = pick_frame_column(left)
    rc = pick_frame_column(right)
    if not lc or not rc:
        raise ValueError(f"Cannot detect frame columns for merge ({left_name}:{lc}, {right_name}:{rc}).")
    L = left.rename(columns={lc: "frame_idx"})
    R = right.rename(columns={rc: "frame_idx"})
    merged = pd.merge(L, R, on="frame_idx", how=how, suffixes=("", f"_{right_name}"))
    return merged

def build_joined_bio_labels(sid: str, md: int) -> Optional[Path]:
    """
    Create per-frame biosignals+labels for a subject/MD.
    Returns output path or None if skipped/failed.
    """
    JOINED_BL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = JOINED_BL_DIR / f"{sid}_MD{md}_bio+labels_PERFRAME.csv"
    if out_path.exists():
        print(f"[SKIP] {sid} MD{md}: bio+labels already exists -> {out_path.name}")
        return out_path

    bio_p = find_biosignals_file(sid, md)
    lab_p = find_labels_file(sid, md)
    if not lab_p:
        print(f"[WARN] {sid} MD{md}: labels file missing; skip bio+labels join.")
        return None

    try:
        labels = pd.read_csv(lab_p)
    except Exception as e:
        print(f"[WARN] {sid} MD{md}: cannot read labels: {e}")
        return None

    # If biosignals missing, write labels-only file to keep pipeline moving
    if not bio_p:
        out = labels.copy()
        out.to_csv(out_path, index=False)
        print(f"[OK]  {sid} MD{md}: wrote labels-only -> {out_path.name}")
        return out_path

    try:
        bios = pd.read_csv(bio_p)
    except Exception as e:
        print(f"[WARN] {sid} MD{md}: cannot read biosignals: {e}; writing labels-only.")
        labels.to_csv(out_path, index=False)
        return out_path

    # Merge by frame_idx
    try:
        joined = safe_merge_on_frames(bios, labels, how="left", left_name="bio", right_name="lab")
    except Exception as e:
        print(f"[WARN] {sid} MD{md}: merge bio+labels failed: {e}; writing labels-only.")
        labels.to_csv(out_path, index=False)
        return out_path

    # If both had t_sec columns, keep the one from biosignals as primary
    # and drop duplicate from labels side if created by suffix.
    dup_t = [c for c in joined.columns if c.startswith("t_sec_")]
    if "t_sec" in joined.columns and dup_t:
        joined = joined.drop(columns=dup_t)

    joined.to_csv(out_path, index=False)
    print(f"[OK]  {sid} MD{md}: wrote bio+labels -> {out_path.name}")
    return out_path

def build_final_perframe(sid: str, md: int) -> Optional[Path]:
    """
    Merge features with labels (and biosignals if present), add delta_pose*, write final CSV.
    Returns path or None.
    """
    FINAL_PERFRAME_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_PERFRAME_DIR / f"{sid}_MD{md}_FINAL_PERFRAME.csv"
    if out_path.exists():
        print(f"[SKIP] {sid} MD{md}: FINAL already exists -> {out_path.name}")
        return out_path

    lab_p = find_labels_file(sid, md)
    if not lab_p:
        print(f"[WARN] {sid} MD{md}: labels file missing; cannot build FINAL.")
        return None

    feat_p = find_features_file(sid)
    if not feat_p:
        print(f"[WARN] {sid} MD{md}: features file missing in {FEATURES_DIR}; cannot build FINAL.")
        return None

    try:
        labels = pd.read_csv(lab_p)
        feats  = pd.read_csv(feat_p)
    except Exception as e:
        print(f"[WARN] {sid} MD{md}: read error (labels/features): {e}")
        return None

    # First merge features + labels on frame index
    try:
        f_l = safe_merge_on_frames(feats, labels, how="left", left_name="feat", right_name="lab")
    except Exception as e:
        print(f"[WARN] {sid} MD{md}: merge features+labels failed: {e}")
        return None

    # Optionally merge biosignals if exist (via joined bio+labels or directly)
    bio_p = find_biosignals_file(sid, md)
    if bio_p and Path(bio_p).exists():
        try:
            bios = pd.read_csv(bio_p)
            f_l_b = safe_merge_on_frames(f_l, bios, how="left", left_name="fl", right_name="bio")
            # Drop duplicate t_sec columns
            t_dups = [c for c in f_l_b.columns if c.startswith("t_sec_")]
            if "t_sec" in f_l_b.columns and t_dups:
                f_l_b = f_l_b.drop(columns=t_dups)
            f_l = f_l_b
        except Exception as e:
            print(f"[WARN] {sid} MD{md}: could not merge biosignals into FINAL: {e} (continuing without bios)")

    # Ensure sorted by frame_idx before deltas
    fr_col = pick_frame_column(f_l) or "frame_idx"
    f_l = f_l.sort_values(fr_col).reset_index(drop=True)

    # Add delta pose
    try:
        f_l = add_delta_pose_cols(f_l, prefix="pose_", n=6)
    except Exception as e:
        print(f"[WARN] {sid} MD{md}: delta pose creation failed: {e}")

    # Write
    f_l.to_csv(out_path, index=False)
    print(f"[OK]  {sid} MD{md}: wrote FINAL -> {out_path.name}")
    return out_path

def subjects_from_labels() -> Dict[str, List[int]]:
    """
    Discover available (sid, md) from LABELS_DIR filenames.
    Returns dict: sid -> [md1, md2, ...]
    """
    found: Dict[str, List[int]] = {}
    for p in LABELS_DIR.glob("T*_MD*_labels_PERFRAME.csv"):
        parsed = parse_sid_md_from_file(p.stem)
        if not parsed: continue
        sid, md = parsed
        found.setdefault(sid, []).append(md)
    return found

def main():
    ap = argparse.ArgumentParser(description="Join biosignals+labels and merge features; add delta poses.")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated subject IDs (e.g., T032,T033). Defaults to all discovered from labels.")
    ap.add_argument("--concat-master", action="store_true",
                    help=f"Also write a concatenated master CSV at {MASTER_OUT}")
    args = ap.parse_args()

    # Discover subjects/MDs from labels already produced
    sid_to_mds = subjects_from_labels()
    if not sid_to_mds:
        print(f"[ERR] No labels found in {LABELS_DIR}. Nothing to do.")
        sys.exit(1)

    # Filter subjects
    sids = sorted(sid_to_mds.keys())
    if args.only:
        wanted = [s.strip().upper() for s in args.only.split(",") if s.strip()]
        sids = [s for s in sids if s in wanted]
        if not sids:
            print(f"[ERR] None of the requested subjects have labels: {wanted}")
            sys.exit(1)

    produced = []
    for sid in sids:
        for md in sorted(sid_to_mds[sid]):
            # 1) bio+labels (not strictly required for FINAL, but useful to keep)
            build_joined_bio_labels(sid, md)
            # 2) FINAL = features (+bios if found) + labels + delta poses
            outp = build_final_perframe(sid, md)
            if outp:
                produced.append(outp)

    if args.concat_master and produced:
        # Concatenate all FINAL files
        print(f"[INFO] Concatenating {len(produced)} FINAL files -> {MASTER_OUT.name}")
        dfs = []
        for p in produced:
            try:
                df = pd.read_csv(p)
                # add subject & md columns to each row for clarity
                parsed = parse_sid_md_from_file(Path(p).stem)
                if parsed:
                    sid, md = parsed
                    df.insert(0, "md", md)
                    df.insert(0, "subject_id", sid)
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Skipping {p.name} during concat: {e}")
        if dfs:
            master = pd.concat(dfs, axis=0, ignore_index=True)
            MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
            master.to_csv(MASTER_OUT, index=False)
            print(f"[OK] Wrote {MASTER_OUT}")

if __name__ == "__main__":
    main()
