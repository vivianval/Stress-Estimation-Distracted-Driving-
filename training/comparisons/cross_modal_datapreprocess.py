# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Build cross-modal NPZ with:
# - MD window means: exp_*, pose_*, biosignals
# - MD delta1 window means: delta1_exp_*, delta1_pose_*
# - Velocity differences (MD - ND) per window for exp/pose/biosignals:
#     veldiff_<feat> = mean(delta_MD_<feat> in window) - mean(delta_ND_<feat> in window)

# Critical constraint:
# - Subtractions are done ONLY within the same (subject, phase, window_index).
# - If ND doesn't have that window, we handle via mismatch_strategy (default drop).

# Outputs .npz:
#   X, y, feature_names,
#   subject_id, phase, window_index, frameid_start, frameid_end
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd


# BIOSIGNALS = ["Perinasal.Perspiration", "Breathing.Rate", "Heart.Rate"]


# def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
#     for c in candidates:
#         if c in df.columns:
#             return c
#     raise KeyError(f"None of the candidate columns found: {candidates}")


# def _sorted_cols_by_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
#     cols = [c for c in df.columns if c.startswith(prefix)]
#     # try to sort by trailing integer, fallback to lexicographic
#     def key_fn(c: str):
#         tail = c[len(prefix):]
#         try:
#             return int(tail)
#         except Exception:
#             # sometimes names like delta1_exp_00, exp_00 etc; strip underscores and retry
#             tail2 = tail.replace("_", "")
#             try:
#                 return int(tail2)
#             except Exception:
#                 return c
#     return sorted(cols, key=key_fn)


# def _compute_group_deltas(df: pd.DataFrame, group_keys: list[str], feat_cols: list[str]) -> pd.DataFrame:
#     """
#     Compute per-frame deltas within each group.
#     delta[t] = x[t] - x[t-1] ; first row in group becomes NaN and will be ignored in window mean.
#     """
#     out = df.copy()
#     out[feat_cols] = out[feat_cols].astype(float)
#     out_delta = out.groupby(group_keys, sort=False)[feat_cols].diff()
#     out_delta.columns = [f"delta__{c}" for c in feat_cols]
#     return pd.concat([out, out_delta], axis=1)


# def _assign_windows_in_phase(df: pd.DataFrame,
#                             subject_col: str,
#                             phase_col: str,
#                             frameid_col: str,
#                             window_frames: int,
#                             stride_frames: int) -> pd.DataFrame:
#     """
#     Within each (subject, phase), sort by frameid and assign window_index based on stride.
#     Only keeps rows that fall into some window slot (window indexing is based on 0..n-1).
#     We later drop incomplete windows at aggregation time.
#     """
#     df = df.copy()
#     df = df.sort_values([subject_col, phase_col, frameid_col]).reset_index(drop=True)

#     # position within (subject, phase)
#     pos = df.groupby([subject_col, phase_col]).cumcount()
#     df["__pos_in_phase"] = pos

#     # assign window index by stride
#     df["window_index"] = (df["__pos_in_phase"] // stride_frames).astype(int)

#     # also compute window-relative start
#     df["__win_start_pos"] = df["window_index"] * stride_frames
#     df["__win_end_pos_excl"] = df["__win_start_pos"] + window_frames

#     # keep only rows that lie inside the window span for their assigned index (important for overlap cases)
#     inside = (df["__pos_in_phase"] >= df["__win_start_pos"]) & (df["__pos_in_phase"] < df["__win_end_pos_excl"])
#     df = df.loc[inside].copy()

#     return df

# def _aggregate_windows(df: pd.DataFrame,
#                        subject_col: str,
#                        phase_col: str,
#                        frameid_col: str,
#                        label_col: str,
#                        mean_cols: list[str],
#                        delta_mean_cols: list[str],
#                        window_frames: int) -> pd.DataFrame:
#     """
#     Aggregate per (subject, phase, window_index) into window-level rows.
#     Keeps ONLY complete windows (exactly window_frames rows).
#     Robust to the case of zero complete windows (returns empty with expected columns).
#     """
#     gkeys = [subject_col, phase_col, "window_index"]

#     if df.empty:
#         # return empty with the expected schema
#         cols = gkeys + ["frameid_start", "frameid_end", "y_window"] + mean_cols + delta_mean_cols
#         return pd.DataFrame(columns=cols)

#     # Count rows per group and keep only complete windows
#     counts = df.groupby(gkeys, sort=False).size().rename("__n").reset_index()
#     complete = counts[counts["__n"] == window_frames][gkeys]

#     if complete.empty:
#         cols = gkeys + ["frameid_start", "frameid_end", "y_window"] + mean_cols + delta_mean_cols
#         return pd.DataFrame(columns=cols)

#     dfc = df.merge(complete, on=gkeys, how="inner")

#     # frame bounds
#     frame_bounds = (
#         dfc.sort_values(gkeys + [frameid_col])
#            .groupby(gkeys, sort=False)[frameid_col]
#            .agg(frameid_start="first", frameid_end="last")
#            .reset_index()
#     )

#     # label aggregation: window positive if mean >= 0.4
#     y_win = (
#         pd.to_numeric(dfc[label_col], errors="coerce")
#           .groupby([dfc[k] for k in gkeys], sort=False)
#           .mean()
#           .reset_index(name="y_mean")
#     )
#     y_win["y_window"] = (y_win["y_mean"] >= 0.4).astype(float)
#     y_win = y_win.drop(columns=["y_mean"])

#     # mean over mean_cols and delta_mean_cols
#     agg_cols = []
#     if mean_cols:
#         agg_cols += mean_cols
#     if delta_mean_cols:
#         agg_cols += delta_mean_cols

#     if agg_cols:
#         feats = (
#             dfc[ gkeys + agg_cols ]
#             .assign(**{c: pd.to_numeric(dfc[c], errors="coerce") for c in agg_cols})
#             .groupby(gkeys, sort=False)[agg_cols]
#             .mean()
#             .reset_index()
#         )
#         out = frame_bounds.merge(y_win, on=gkeys, how="inner").merge(feats, on=gkeys, how="inner")
#     else:
#         out = frame_bounds.merge(y_win, on=gkeys, how="inner")

#     return out


# def _build_nd_baseline_delta_table(nd_win: pd.DataFrame,
#                                   subject_col: str,
#                                   phase_col: str,
#                                   delta_cols: list[str]) -> pd.DataFrame:
#     """
#     ND window table already contains delta means columns (delta__<feat>).
#     We keep only what we need for merging.
#     """
#     keep = [subject_col, phase_col, "window_index"] + delta_cols
#     return nd_win[keep].copy()


# def _merge_with_mismatch(md_win: pd.DataFrame,
#                          nd_base: pd.DataFrame,
#                          subject_col: str,
#                          phase_col: str,
#                          delta_cols: list[str],
#                          mismatch_strategy: str) -> pd.DataFrame:
#     """
#     Merge ND baseline deltas onto MD windows by (subject, phase, window_index).
#     Handles missing ND windows based on mismatch_strategy.
#     """
#     merged = md_win.merge(
#         nd_base,
#         on=[subject_col, phase_col, "window_index"],
#         how="left",
#         suffixes=("", "__ND")
#     )

#     nd_delta_cols = [c for c in merged.columns if c.endswith("__ND")]

#     if mismatch_strategy == "drop":
#         # keep only rows where all ND delta cols exist
#         mask = np.ones(len(merged), dtype=bool)
#         for c in nd_delta_cols:
#             mask &= merged[c].notna().to_numpy()
#         merged = merged.loc[mask].copy()
#         return merged

#     if mismatch_strategy == "phase_mean":
#         # fill missing ND window deltas with per-(subject, phase) mean from available ND windows
#         phase_means = (
#             nd_base.groupby([subject_col, phase_col], sort=False)[delta_cols]
#             .mean()
#             .reset_index()
#         )
#         phase_means = phase_means.rename(columns={c: f"{c}__PHASEMEAN" for c in delta_cols})

#         merged = merged.merge(phase_means, on=[subject_col, phase_col], how="left")
#         for c in delta_cols:
#             nd_c = f"{c}__ND"
#             fill_c = f"{c}__PHASEMEAN"
#             merged[nd_c] = merged[nd_c].fillna(merged[fill_c])
#         merged = merged.drop(columns=[f"{c}__PHASEMEAN" for c in delta_cols])
#         return merged

#     if mismatch_strategy == "nearest":
#         # nearest window index within same (subject, phase)
#         # For each MD row missing ND, find nearest ND window_index and copy its deltas.
#         nd_groups = {
#             (sid, ph): grp.sort_values("window_index")
#             for (sid, ph), grp in nd_base.groupby([subject_col, phase_col], sort=False)
#         }

#         merged = merged.copy()
#         for i in range(len(merged)):
#             # check if missing
#             any_missing = False
#             for c in delta_cols:
#                 if pd.isna(merged.at[i, f"{c}__ND"]):
#                     any_missing = True
#                     break
#             if not any_missing:
#                 continue

#             key = (merged.at[i, subject_col], merged.at[i, phase_col])
#             if key not in nd_groups or len(nd_groups[key]) == 0:
#                 continue

#             target_wi = int(merged.at[i, "window_index"])
#             grp = nd_groups[key]
#             wi_vals = grp["window_index"].to_numpy()
#             j = int(np.argmin(np.abs(wi_vals - target_wi)))
#             src = grp.iloc[j]
#             for c in delta_cols:
#                 merged.at[i, f"{c}__ND"] = src[c]

#         # if still missing (no ND phase at all), drop
#         mask = np.ones(len(merged), dtype=bool)
#         for c in delta_cols:
#             mask &= merged[f"{c}__ND"].notna().to_numpy()
#         merged = merged.loc[mask].copy()
#         return merged

#     raise ValueError(f"Unknown mismatch_strategy: {mismatch_strategy}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--md_csv", type=str, default="/home/vivib/emoca/emoca/training/comparisons/MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN_with_delta1.csv")
#     ap.add_argument("--nd_csv", type=str, default="/home/vivib/emoca/emoca/training/comparisons/ND_ALL_subject_idS_EXPPOSE_BIO_DELTA1_PHASED_GTLABEL_PHASEONLY.csv")
#     ap.add_argument("--out_npz", type=str, default="/home/vivib/emoca/emoca/training/comparisons/FINAL.npz")

#     ap.add_argument("--fps", type=float, default=30.0)
#     ap.add_argument("--window_sec", type=float, default=9.0)
#     ap.add_argument("--stride_sec", type=float, default=9.0)  # non-overlapping by default

#     ap.add_argument("--mismatch_strategy", type=str, default="drop",
#                     choices=["drop", "phase_mean", "nearest"])

#     args = ap.parse_args()

#     md_path = Path(args.md_csv)
#     nd_path = Path(args.nd_csv)
#     out_path = Path(args.out_npz)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     window_frames = int(round(args.window_sec * args.fps))
#     stride_frames = int(round(args.stride_sec * args.fps))
#     if window_frames <= 1:
#         raise ValueError("window_frames must be > 1")
#     if stride_frames <= 0:
#         raise ValueError("stride_frames must be >= 1")

#     print(f"[INFO] window_frames={window_frames}, stride_frames={stride_frames}, mismatch_strategy={args.mismatch_strategy}")

#     # ---------- Load ----------
#     md = pd.read_csv(md_path)
#     nd = pd.read_csv(nd_path)

#     # ---------- Identify key columns robustly ----------
#     subject_col_md = _find_col(md, ["subject_id", "Subject", "subject", "subj", "SubjectID"])
#     subject_col_nd = _find_col(nd, ["subject_id", "Subject", "subject", "subj", "SubjectID"])

#     phase_col_md = _find_col(md, ["Phase", "phase", "PHASE"])
#     phase_col_nd = _find_col(nd, ["Phase", "phase", "PHASE"])

#     frameid_col_md = _find_col(md, ["frameid", "frame_idx", "FrameID", "frame", "Frame"])
#     frameid_col_nd = _find_col(nd, ["frameid", "frame_id", "FrameID", "frame", "Frame"])

#     label_col_md = _find_col(md, ["GTlabel", "gtlabel", "GTLabel", "label", "y"])

#     # ---------- Feature columns ----------
#     exp_cols = _sorted_cols_by_prefix(md, "exp_")
#     pose_cols = _sorted_cols_by_prefix(md, "pose_")
#     if len(exp_cols) == 0:
#         raise KeyError("No exp_* columns found in MD.")
#     if len(pose_cols) == 0:
#         raise KeyError("No pose_* columns found in MD.")

#     # biosignals must exist in MD (and ideally ND)
#     for b in BIOSIGNALS:
#         if b not in md.columns:
#             raise KeyError(f"Missing biosignal column in MD: {b}")
#         if b not in nd.columns:
#             raise KeyError(f"Missing biosignal column in ND: {b}")

#     delta1_exp_cols = _sorted_cols_by_prefix(md, "delta1_exp_")
#     delta1_pose_cols = _sorted_cols_by_prefix(md, "delta1_pose_")
#     if len(delta1_exp_cols) == 0:
#         print("[WARN] No delta1_exp_* columns found in MD. Proceeding without them.")
#     if len(delta1_pose_cols) == 0:
#         print("[WARN] No delta1_pose_* columns found in MD. Proceeding without them.")

#     # The modalities for which we compute per-frame deltas and MD-ND velocity differences:
#     vel_feats = exp_cols + pose_cols + BIOSIGNALS

#     # ---------- Prepare + window assignment ----------
#     # Keep only needed columns to reduce memory
#     md_keep = [subject_col_md, phase_col_md, frameid_col_md, label_col_md] + exp_cols + pose_cols + BIOSIGNALS + delta1_exp_cols + delta1_pose_cols
#     nd_keep = [subject_col_nd, phase_col_nd, frameid_col_nd] + exp_cols + pose_cols + BIOSIGNALS

#     md = md[md_keep].copy()
#     nd = nd[nd_keep].copy()

#     # Window assignment within (subject, phase)
#     md = _assign_windows_in_phase(md, subject_col_md, phase_col_md, frameid_col_md, window_frames, stride_frames)
#     nd = _assign_windows_in_phase(nd, subject_col_nd, phase_col_nd, frameid_col_nd, window_frames, stride_frames)

#     # ---------- Compute per-frame deltas within (subject, phase) ----------
#     md = _compute_group_deltas(md, [subject_col_md, phase_col_md], vel_feats)
#     nd = _compute_group_deltas(nd, [subject_col_nd, phase_col_nd], vel_feats)

#     md_delta_cols = [f"delta__{c}" for c in vel_feats]
#     nd_delta_cols = [f"delta__{c}" for c in vel_feats]

#     # ---------- Aggregate to window level ----------
#     md_mean_cols = exp_cols + pose_cols + BIOSIGNALS + delta1_exp_cols + delta1_pose_cols
#     md_win = _aggregate_windows(
#         md,
#         subject_col=subject_col_md,
#         phase_col=phase_col_md,
#         frameid_col=frameid_col_md,
#         label_col=label_col_md,
#         mean_cols=md_mean_cols,
#         delta_mean_cols=md_delta_cols,
#         window_frames=window_frames
#     )

#     nd_win = _aggregate_windows(
#         nd.assign(dummy_label=0.0),  # label unused
#         subject_col=subject_col_nd,
#         phase_col=phase_col_nd,
#         frameid_col=frameid_col_nd,
#         label_col="dummy_label",
#         mean_cols=[],                 # we only need delta means for ND baseline
#         delta_mean_cols=nd_delta_cols,
#         window_frames=window_frames
#     )

#     # Harmonize key column names (use MD names for merge)
#     nd_win = nd_win.rename(columns={
#         subject_col_nd: subject_col_md,
#         phase_col_nd: phase_col_md
#     })

#     # Build ND baseline delta table (delta__<feat>)
#     nd_base = _build_nd_baseline_delta_table(
#         nd_win,
#         subject_col=subject_col_md,
#         phase_col=phase_col_md,
#         delta_cols=nd_delta_cols
#     )

#     # Merge ND baseline deltas onto MD windows with mismatch handling
#     merged = _merge_with_mismatch(
#         md_win,
#         nd_base,
#         subject_col=subject_col_md,
#         phase_col=phase_col_md,
#         delta_cols=nd_delta_cols,
#         mismatch_strategy=args.mismatch_strategy
#     )

#     # ---------- Build velocity difference features ----------
#     # MD delta mean columns are delta__<feat>
#     # ND delta mean columns after merge are delta__<feat>__ND
#     veldiff_cols = []
#     for feat in vel_feats:
#         md_c = f"delta__{feat}"
#         nd_c = f"delta__{feat}__ND"
#         out_c = f"veldiff__{feat}"
#         merged[out_c] = merged[md_c].astype(float) - merged[nd_c].astype(float)
#         veldiff_cols.append(out_c)

#     # ---------- Assemble final feature matrix ----------
#     # We include:
#     # - MD means: exp_*, pose_*, biosignals
#     # - MD delta1 means: delta1_exp_*, delta1_pose_* (if available)
#     # - velocity differences: veldiff__<feat> for exp/pose/bio
#     feature_cols = []
#     feature_cols += exp_cols
#     feature_cols += pose_cols
#     feature_cols += BIOSIGNALS
#     feature_cols += delta1_exp_cols
#     feature_cols += delta1_pose_cols
#     feature_cols += veldiff_cols

#     # Sanity: ensure all exist
#     missing = [c for c in feature_cols if c not in merged.columns]
#     if missing:
#         raise KeyError(f"Missing expected feature columns after merge: {missing}")

#     X = merged[feature_cols].to_numpy(dtype=np.float32)
#     y = merged["y_window"].to_numpy(dtype=np.int64)

#     subject_id = merged[subject_col_md].to_numpy()
#     phase = merged[phase_col_md].to_numpy()
#     window_index = merged["window_index"].to_numpy(dtype=np.int64)
#     frameid_start = merged["frameid_start"].to_numpy(dtype=np.int64)
#     frameid_end = merged["frameid_end"].to_numpy(dtype=np.int64)

#     # ---------- Save NPZ ----------
#     np.savez_compressed(
#         out_path,
#         X=X,
#         y=y,
#         feature_names=np.array(feature_cols, dtype=object),
#         subject_id=subject_id,
#         phase=phase,
#         window_index=window_index,
#         frameid_start=frameid_start,
#         frameid_end=frameid_end,
#         window_frames=np.int64(window_frames),
#         stride_frames=np.int64(stride_frames),
#         fps=np.float32(args.fps),
#         mismatch_strategy=np.array([args.mismatch_strategy], dtype=object),
#     )

#     print(f"[DONE] Saved: {out_path}")
#     print(f"[INFO] X shape: {X.shape}, y pos rate: {y.mean():.3f}")
#     print(f"[INFO] Features: {len(feature_cols)}")
#     print("[INFO] Example feature name prefixes you can filter later:")
#     print("       exp_, pose_, delta1_exp_, delta1_pose_, veldiff__exp_, veldiff__pose_, veldiff__Perinasal.Perspiration, ...")


# if __name__ == "__main__":
#     main()


# import numpy as np

# npz_path = "/home/vivib/emoca/emoca/training/comparisons/FINAL.npz"

# data = np.load(npz_path, allow_pickle=True)

# print("=== KEYS ===")
# print(list(data.keys()))

# print("\n=== SHAPES ===")
# for k in data.keys():
#     arr = data[k]
#     if hasattr(arr, "shape"):
#         print(f"{k:20s} shape = {arr.shape}, dtype = {arr.dtype}")
#     else:
#         print(f"{k:20s} type = {type(arr)}")

# print("\n=== BASIC STATS ===")
# X = data["X"]
# y = data["y"]
# print("X min/max:", X.min(), X.max())
# print("y distribution:", {0: int((y==0).sum()), 1: int((y==1).sum())})
# print("Positive rate:", y.mean())

# print("\n=== FIRST 10 FEATURE NAMES ===")
# feature_names = data["feature_names"]
# for i, name in enumerate(feature_names[:10]):
#     print(f"{i:3d}: {name}")

# print("\n=== LAST 10 FEATURE NAMES ===")
# for i, name in enumerate(feature_names[-10:], start=len(feature_names)-10):
#     print(f"{i:3d}: {name}")
# import numpy as np
# d = np.load("/home/vivib/emoca/emoca/training/comparisons/FINAL.npz", allow_pickle=True)
# y = d["y"].astype(int)
# ph = d["phase"].astype(str)

# for p in sorted(set(ph)):
#     idx = np.where(ph == p)[0]
#     print(p, "n=", len(idx), "pos_rate=", y[idx].mean())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    veldiff_feat = mean(diff(MD_feat)) - mean(diff(ND_feat))
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
