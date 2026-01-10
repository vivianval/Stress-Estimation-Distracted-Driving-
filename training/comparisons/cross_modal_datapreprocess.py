

"""
Build NPZ with per-window *per-frame* tensors + per-window velocity-difference features.

Inputs:
  MD per-frame CSV (has GTlabel + delta1_* already computed):
    /home/vivib/emoca/emoca/training/comparisons/MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN_with_delta1.csv

  ND per-frame CSV (baseline):
    /home/vivib/emoca/emoca/training/comparisons/ND_ALL_subject_idS_EXPPOSE_BIO_DELTA1_PHASED_GTLABEL_PHASEONLY.csv

Windowing:
  - Within each (subject, phase) sort by frameid (or frame_idx) and slice into windows:
      window_frames = fps * window_sec
      stride_frames = fps * stride_sec  (default = window_sec -> non-overlapping)

We store:
  X_frame: [Nw, T, F_frame]  where F_frame = exp(50)+pose(6)+bio(3)+delta1_exp(50)+delta1_pose(6)=115
  X_vel  : [Nw, F_vel]       where F_vel = exp(50)+pose(6)+bio(3)=59
  y_frame: [Nw, T]           per-frame GTlabel from MD
  y_window: [Nw]             aggregated label (mean>=0.4)

Velocity differences:
  For each window (subject, phase, window_index):
    veldiff_feat = mean(diff(MD_feat)) - mean(diff(ND_feat))   1ST ORDER NORMALIZATION
  where diff is per-frame delta within the window.

Mismatch (ND shorter than MD):
  --mismatch drop       : drop MD windows without a matching ND window
  --mismatch nearest    : use nearest ND window_index within same (subject, phase)
  --mismatch phase_mean : use mean ND delta over the phase for missing windows
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BIOSIGNALS = ["Perinasal.Perspiration", "Breathing.Rate", "Heart.Rate"]
EXCLUDE_GAZE = {"Gaze.X.Pos", "Gaze.Y.Pos"}

def _gaze_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("Gaze")]
    cols = [c for c in cols if c not in EXCLUDE_GAZE and not c.startswith("Gaze.X") and not c.startswith("Gaze.Y")]
    return sorted(cols)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found: {candidates}")

def _sorted_cols_by_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    def key_fn(c: str):
        tail = c[len(prefix):]
        tail = tail.replace("_", "")
        try:
            return int(tail)
        except Exception:
            return c
    return sorted(cols, key=key_fn)

def _make_windows(df: pd.DataFrame,
                  subject_col: str,
                  phase_col: str,
                  frameid_col: str,
                  window_frames: int,
                  stride_frames: int) -> pd.DataFrame:
    """
    Adds:
      - pos_in_phase (0..)
      - window_index
      Keeps only rows that fall inside their window span (important for overlap).
    """
    df = df.sort_values([subject_col, phase_col, frameid_col]).reset_index(drop=True)
    df["__pos_in_phase"] = df.groupby([subject_col, phase_col], sort=False).cumcount()
    df["window_index"] = (df["__pos_in_phase"] // stride_frames).astype(int)

    # keep only rows within each window's [start, start+window_frames)
    start = df["window_index"] * stride_frames
    end = start + window_frames
    inside = (df["__pos_in_phase"] >= start) & (df["__pos_in_phase"] < end)
    return df.loc[inside].copy()

def _group_complete_windows(df: pd.DataFrame,
                            subject_col: str,
                            phase_col: str,
                            window_frames: int) -> pd.DataFrame:
    gkeys = [subject_col, phase_col, "window_index"]
    counts = df.groupby(gkeys, sort=False).size().rename("__n").reset_index()
    complete = counts[counts["__n"] == window_frames][gkeys]
    return df.merge(complete, on=gkeys, how="inner")

def _window_arrays_md(md: pd.DataFrame,
                      subject_col: str,
                      phase_col: str,
                      frameid_col: str,
                      gtlabel_col: str,
                      exp_cols: list[str],
                      pose_cols: list[str],
                      delta1_exp_cols: list[str],
                      delta1_pose_cols: list[str],
                      gaze_cols: list[str],          # <-- ADD gaze!
                      window_frames: int):

    """
    Returns per-window tensors for MD:
      X_frame: [Nw, T, F_frame]
      y_frame: [Nw, T]
      meta    : subject_id, phase, window_index, frameid_start/end
    Also returns window-level mean deltas for exp/pose/bio for velocity diffs.
    """
    gkeys = [subject_col, phase_col, "window_index"]
    md = _group_complete_windows(md, subject_col, phase_col, window_frames)

    # ensure numeric
    frame_feat_cols = exp_cols + pose_cols + BIOSIGNALS + gaze_cols + delta1_exp_cols + delta1_pose_cols

    for c in frame_feat_cols + [gtlabel_col]:
        md[c] = pd.to_numeric(md[c], errors="coerce")

    # per-frame deltas for velocity features (exp+pose+bio)
    vel_cols = exp_cols + pose_cols + BIOSIGNALS
    md = md.sort_values(gkeys + [frameid_col])
    for c in vel_cols:
        md[f"delta__{c}"] = md.groupby(gkeys, sort=False)[c].diff()

    # Build window tensors
    groups = list(md.groupby(gkeys, sort=False))
    Nw = len(groups)
    T = window_frames
    F_frame = len(frame_feat_cols)

    X_frame = np.zeros((Nw, T, F_frame), dtype=np.float32)
    y_frame = np.zeros((Nw, T), dtype=np.float32)

    subj = np.empty((Nw,), dtype=object)
    phase = np.empty((Nw,), dtype=object)
    widx = np.zeros((Nw,), dtype=np.int64)
    fstart = np.zeros((Nw,), dtype=np.int64)
    fend = np.zeros((Nw,), dtype=np.int64)

    # window mean deltas (MD)
    md_delta_mean = np.zeros((Nw, len(vel_cols)), dtype=np.float32)

    for i, ((sid, ph, wi), g) in enumerate(groups):
        g = g.sort_values(frameid_col)
        X_frame[i] = g[frame_feat_cols].to_numpy(dtype=np.float32)
        y_frame[i] = g[gtlabel_col].to_numpy(dtype=np.float32)

        subj[i] = sid
        phase[i] = ph
        widx[i] = int(wi)
        fstart[i] = int(g[frameid_col].iloc[0])
        fend[i] = int(g[frameid_col].iloc[-1])


        dm = np.zeros((len(vel_cols),), dtype=np.float32)
        bad = 0
        for j, c in enumerate(vel_cols):
            arr = g[f"delta__{c}"].to_numpy(dtype=np.float32)
            arr = arr[np.isfinite(arr)]  # drop NaN/inf
            if arr.size == 0:
                dm[j] = 0.0               # neutral velocity if we have no delta info
                bad += 1
            else:
                dm[j] = float(arr.mean())
        md_delta_mean[i] = dm

        

# OPTIONAL: track how often this happens
# (declare outside loop: empty_delta_count = 0; empty_delta_total = 0)
# empty_delta_count += (bad > 0)
# empty_delta_total += bad


        # # mean delta within window (ignore first NaN)
        # dm = []
        # for c in vel_cols:
        #     dm.append(np.nanmean(g[f"delta__{c}"].to_numpy(dtype=np.float32)))
        # md_delta_mean[i] = np.array(dm, dtype=np.float32)

    return X_frame, y_frame, subj, phase, widx, fstart, fend, md_delta_mean, vel_cols, frame_feat_cols

def _nd_delta_means(nd: pd.DataFrame,
                    subject_col: str,
                    phase_col: str,
                    frameid_col: str,
                    exp_cols: list[str],
                    pose_cols: list[str],
                    window_frames: int,
                    stride_frames: int):
    """
    Returns ND baseline mean deltas per (subject, phase, window_index) for vel_cols.
    """
    gkeys = [subject_col, phase_col, "window_index"]
    vel_cols = exp_cols + pose_cols + BIOSIGNALS

    nd = _make_windows(nd, subject_col, phase_col, frameid_col, window_frames, stride_frames)
    nd = _group_complete_windows(nd, subject_col, phase_col, window_frames)

    for c in vel_cols:
        nd[c] = pd.to_numeric(nd[c], errors="coerce")

    nd = nd.sort_values(gkeys + [frameid_col])
    for c in vel_cols:
        nd[f"delta__{c}"] = nd.groupby(gkeys, sort=False)[c].diff()

    # aggregate mean delta per window
    out = nd.groupby(gkeys, sort=False)[[f"delta__{c}" for c in vel_cols]].mean().reset_index()
    return out, vel_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_csv", type=str,
                    default="/home/vivib/emoca/emoca/training/comparisons/MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN_with_delta1.csv")
    ap.add_argument("--nd_csv", type=str,
                    default="/home/vivib/emoca/emoca/training/comparisons/ND_ALL_subject_idS_EXPPOSE_BIO_DELTA1_PHASED_GTLABEL_PHASEONLY.csv")
    ap.add_argument("--out_npz", type=str,
                    default="/home/vivib/emoca/emoca/training/comparisons/FINAL_WINDOWSEQ_BASELINEv1.npz")

    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--window_sec", type=float, default=9.0)
    ap.add_argument("--stride_sec", type=float, default=9.0)  # non-overlapping by default

    ap.add_argument("--mismatch", type=str, default="drop", choices=["drop", "nearest", "phase_mean"])
    ap.add_argument("--y_thresh", type=float, default=0.4)
    args = ap.parse_args()

    window_frames = int(round(args.fps * args.window_sec))
    stride_frames = int(round(args.fps * args.stride_sec))
    if window_frames <= 2:
        raise ValueError("window_frames too small")
    if stride_frames <= 0:
        raise ValueError("stride_frames must be >=1")

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = pd.read_csv(args.md_csv)
    nd = pd.read_csv(args.nd_csv)

    # columns
    subject_md = _find_col(md, ["subject_id", "Subject", "subject", "subj"])
    subject_nd = _find_col(nd, ["subject_id", "Subject", "subject", "subj"])
    phase_md   = _find_col(md, ["phase", "Phase", "PHASE"])
    phase_nd   = _find_col(nd, ["phase", "Phase", "PHASE"])
    frame_md   = _find_col(md, ["frame_id", "frame_idx", "FrameID", "frame", "Frame"])
    frame_nd   = _find_col(nd, ["frame_id", "frame_idx", "FrameID", "frame", "Frame"])
    gtlabel    = _find_col(md, ["label"])             #GTlabel", "GTLabel", "gtlabel", 

    # features (MD must contain all; ND must contain exp/pose/bio for baseline deltas)
    exp_cols = _sorted_cols_by_prefix(md, "exp_")
    pose_cols = _sorted_cols_by_prefix(md, "pose_")

    # IMPORTANT: you said pose is 6 dims. If your CSV has more pose_* cols, we keep ONLY first 6.
    if len(pose_cols) > 6:
        pose_cols = pose_cols[:6]

    delta1_exp_cols = _sorted_cols_by_prefix(md, "delta1_exp_")
    delta1_pose_cols = _sorted_cols_by_prefix(md, "delta1_pose_")
    if len(delta1_pose_cols) > 6:
        delta1_pose_cols = delta1_pose_cols[:6]


    gaze_cols = _gaze_feature_cols(md)

    print(f"[INFO] Using {len(gaze_cols)} gaze features in X_frame only (excluding Gaze.X.Pos/Y.Pos).")


    # sanity
    if len(exp_cols) != 50:
        print(f"[WARN] exp_* count is {len(exp_cols)} (expected 50). Proceeding anyway.")
    if len(pose_cols) != 6:
        print(f"[WARN] pose_* count is {len(pose_cols)} (expected 6). Proceeding anyway.")
    for b in BIOSIGNALS:
        if b not in md.columns:
            raise KeyError(f"Missing {b} in MD")
        if b not in nd.columns:
            raise KeyError(f"Missing {b} in ND (needed for baseline deltas)")

    # window assignment (MD & ND) within (subject, phase)
    md = _make_windows(md, subject_md, phase_md, frame_md, window_frames, stride_frames)
    nd = _make_windows(nd, subject_nd, phase_nd, frame_nd, window_frames, stride_frames)

    # Build MD per-window tensors
    X_frame, y_frame, subj, ph, widx, fstart, fend, md_delta_mean, vel_cols, frame_feat_cols = _window_arrays_md(
        md, subject_md, phase_md, frame_md, gtlabel,
        exp_cols, pose_cols, delta1_exp_cols, delta1_pose_cols,
        gaze_cols,
        window_frames
    )

    # ND baseline delta means per window
    nd_delta_tbl, vel_cols2 = _nd_delta_means(
        nd, subject_nd, phase_nd, frame_nd,
        exp_cols, pose_cols,
        window_frames, stride_frames
    )
    assert vel_cols2 == vel_cols

    # merge baseline onto MD windows
    md_keys = pd.DataFrame({
        "subject_id": subj,
        "phase": ph,
        "window_index": widx
    })

    nd_tbl = nd_delta_tbl.rename(columns={
        subject_nd: "subject_id",
        phase_nd: "phase"
    })
    # columns in nd_tbl: subject_id, phase, window_index, delta__feat...

    merged = md_keys.merge(nd_tbl, on=["subject_id", "phase", "window_index"], how="left")

    # handle mismatch
    delta_cols = [f"delta__{c}" for c in vel_cols]

    if args.mismatch == "drop":
        ok = np.ones(len(merged), dtype=bool)
        for c in delta_cols:
            ok &= merged[c].notna().to_numpy()
        keep_idx = np.where(ok)[0]

    elif args.mismatch == "phase_mean":
        phase_mean = nd_tbl.groupby(["subject_id", "phase"], sort=False)[delta_cols].mean().reset_index()
        merged2 = merged.merge(phase_mean, on=["subject_id", "phase"], how="left", suffixes=("", "__PHASEMEAN"))
        for c in delta_cols:
            merged2[c] = merged2[c].fillna(merged2[f"{c}__PHASEMEAN"])
        merged = merged2.drop(columns=[f"{c}__PHASEMEAN" for c in delta_cols])
        ok = np.ones(len(merged), dtype=bool)
        for c in delta_cols:
            ok &= merged[c].notna().to_numpy()
        keep_idx = np.where(ok)[0]

    else:  # nearest
        # build index for ND windows per (subject, phase)
        nd_groups = {
            (sid, phx): g.sort_values("window_index")
            for (sid, phx), g in nd_tbl.groupby(["subject_id", "phase"], sort=False)
        }
        merged = merged.copy()
        for i in range(len(merged)):
            if all(pd.notna(merged.loc[i, delta_cols])):
                continue
            key = (merged.loc[i, "subject_id"], merged.loc[i, "phase"])
            if key not in nd_groups or len(nd_groups[key]) == 0:
                continue
            wi = int(merged.loc[i, "window_index"])
            g = nd_groups[key]
            wi_vals = g["window_index"].to_numpy()
            j = int(np.argmin(np.abs(wi_vals - wi)))
            row = g.iloc[j]
            for c in delta_cols:
                merged.loc[i, c] = row[c]
        ok = np.ones(len(merged), dtype=bool)
        for c in delta_cols:
            ok &= merged[c].notna().to_numpy()
        keep_idx = np.where(ok)[0]

    # apply keep
    X_frame = X_frame[keep_idx]
    y_frame = y_frame[keep_idx]
    subj = subj[keep_idx]
    ph = ph[keep_idx]
    widx = widx[keep_idx]
    fstart = fstart[keep_idx]
    fend = fend[keep_idx]
    md_delta_mean = md_delta_mean[keep_idx]
    merged = merged.iloc[keep_idx].reset_index(drop=True)

    # compute velocity differences (MD mean delta - ND mean delta)
    nd_delta_mean = merged[delta_cols].to_numpy(dtype=np.float32)  # [Nw, 59]
    X_vel = md_delta_mean.astype(np.float32) - nd_delta_mean

    # window label
    y_window = (np.nanmean(y_frame, axis=1) >= args.y_thresh).astype(np.int64)

    # feature names
    feature_names_frame = np.array(frame_feat_cols, dtype=object)  # len 115
    feature_names_vel = np.array([f"veldiff__{c}" for c in vel_cols], dtype=object)  # len 59

    # print shapes
    print("\n[NPZ SUMMARY]")
    print(f"  window_frames (T): {window_frames}")
    print(f"  X_frame: {X_frame.shape}  (Nw, T, F_frame)")
    print(f"  X_vel  : {X_vel.shape}    (Nw, F_vel)")
    print(f"  y_frame: {y_frame.shape}  (Nw, T)")
    print(f"  y_window: {y_window.shape} pos_rate={y_window.mean():.3f}")
    print(f"  subjects: {len(np.unique(subj))} | windows kept: {len(subj)} | mismatch={args.mismatch}")

    print("\n[FEATURE COUNTS]")
    print(f"  exp: {len(exp_cols)} | pose: {len(pose_cols)} | bio: {len(BIOSIGNALS)}")
    print(f"  delta1_exp: {len(delta1_exp_cols)} | delta1_pose: {len(delta1_pose_cols)}")
    print(f"  F_frame={len(frame_feat_cols)} | F_vel={len(feature_names_vel)}")

    print("\n[EXAMPLE FEATURE NAMES]")
    print("  frame feats head:", list(feature_names_frame[:8]))
    print("  frame feats tail:", list(feature_names_frame[-8:]))
    print("  vel feats head  :", list(feature_names_vel[:8]))
    print("  vel feats tail  :", list(feature_names_vel[-8:]))

    # save
    np.savez_compressed(
        out_path,
        X_frame=X_frame,
        X_vel=X_vel,
        y_frame=y_frame.astype(np.float32),
        y_window=y_window,
        feature_names_frame=feature_names_frame,
        feature_names_vel=feature_names_vel,
        subject_id=np.array(subj, dtype=object),
        phase=np.array(ph, dtype=object),
        window_index=widx.astype(np.int64),
        frameid_start=fstart.astype(np.int64),
        frameid_end=fend.astype(np.int64),
        fps=np.float32(args.fps),
        window_sec=np.float32(args.window_sec),
        stride_sec=np.float32(args.stride_sec),
        mismatch=np.array([args.mismatch], dtype=object),
        y_thresh=np.float32(args.y_thresh),
    )

    print(f"\n[DONE] Saved NPZ → {out_path}")

if __name__ == "__main__":
    main()
