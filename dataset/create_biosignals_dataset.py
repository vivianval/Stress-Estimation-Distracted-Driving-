#!/usr/bin/env python3
import re
import sys
import json
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import argparse

# ================== CONFIG ==================
ROOT = Path("/media/storage/vivib/Structured Study Data")
R_FRIENDLY = ROOT / "R-Friendly Study Data"
OUT_DIR = Path("/home/vivib/emoca/emoca/dataset/BIOSIGNALS_CSVS")

SUBJ_RE = re.compile(r"^T\d{3,}$", re.IGNORECASE)  # e.g. T001
MD_RE = re.compile(r"(\d+)\s*MD$", re.IGNORECASE)   # e.g. 7MD or 7 MD

# Biosignal columns (numeric, to interpolate)
NUM_COLS = [
    "Breathing.Rate", "Heart.Rate",
    "Perinasal.Perspiration", "Gaze.X.Pos", "Gaze.Y.Pos"
]

# Columns we keep and try to align (Stimulus nearest, Drive constant)
KEEP_COLS = ["Time", "Drive", "Stimulus"] + NUM_COLS

NA_VALUES = ["NA", "NaN", "", " "]
# ============================================


def find_subject_dirs(root: Path) -> List[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir() and SUBJ_RE.match(d.name)],
                  key=lambda p: p.name)


def detect_md_folder(subject_dir: Path) -> Optional[Path]:
    # robust: tolerate spaces/underscores/dashes around "MD"
    for d in subject_dir.iterdir():
        if not d.is_dir():
            continue
        norm = d.name.replace("_", "").replace("-", "")
        if MD_RE.search(norm):
            return d
    return None


def detect_md_drive_num(folder: Path) -> Optional[int]:
    m = MD_RE.search(folder.name.replace("_", "").replace("-", ""))
    if m:
        return int(m.group(1))
    return None


def pick_md_video(md_folder: Path, subject_id: str, md_drive: int) -> Optional[Path]:
    """
    Prefer '<subject>-<drive:03d>...avi1.avi' inside the MD folder.
    If none found, fall back to any '*avi1.avi' in that folder.
    """
    pat_core = f"{subject_id}-{md_drive:03d}".lower()

    # First: strict pattern *avi1.avi
    strict = list(md_folder.glob("*avi1.avi"))

    if strict:
        # Sort with best match first (contains subject-drive), then by name
        strict.sort(key=lambda p: (pat_core not in p.name.lower(), p.name.lower()))
        return strict[0]

    # Fallbacks (if you ever have odd variants)
    cand = []
    for ext in ("*.avi1.avi", "*.avi", "*.mp4", "*.mov", "*.mkv"):
        cand.extend(md_folder.glob(ext))
    if not cand:
        return None

    cand.sort(key=lambda p: (pat_core not in p.name.lower(), p.suffix.lower() != ".avi", p.name.lower()))
    return cand[0]



def ffprobe_frame_times(video_path: Path) -> Optional[np.ndarray]:
    """
    Extract per-frame timestamps (seconds) using ffprobe best_effort_timestamp_time.
    Returns np.ndarray [n_frames], or None if ffprobe not available / fails.
    """
    try:
        cmd = (
            "ffprobe -v error -select_streams v:0 "
            "-show_entries frame=best_effort_timestamp_time "
            "-of json " + shlex.quote(str(video_path))
        )
        out = subprocess.check_output(cmd, shell=True, text=True)
        data = json.loads(out)
        frames = data.get("frames", [])
        ts = []
        for fr in frames:
            t = fr.get("best_effort_timestamp_time")
            if t is not None:
                try:
                    ts.append(float(t))
                except Exception:
                    pass
        if len(ts) == 0:
            return None
        return np.array(ts, dtype=float)
    except Exception:
        return None


def ffprobe_basic_info(video_path: Path) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """
    Returns (fps_nominal, nb_frames, duration_sec) if available.
    """
    try:
        cmd = (
            "ffprobe -v error -print_format json -count_frames "
            "-show_streams -show_format " + shlex.quote(str(video_path))
        )
        out = subprocess.check_output(cmd, shell=True, text=True)
        meta = json.loads(out)
        streams = meta.get("streams", [])
        vstreams = [s for s in streams if s.get("codec_type") == "video"]
        if not vstreams:
            return (None, None, None)
        vs = vstreams[0]
        # nominal fps
        fps_nom = None
        if "r_frame_rate" in vs and vs["r_frame_rate"]:
            num, den = vs["r_frame_rate"].split("/")
            if float(den) != 0.0:
                fps_nom = float(num) / float(den)
        # frames
        nb = None
        if vs.get("nb_read_frames") is not None:
            try:
                nb = int(vs["nb_read_frames"])
            except Exception:
                pass
        # duration
        dur = None
        if vs.get("duration") is not None:
            dur = float(vs["duration"])
        elif meta.get("format", {}).get("duration") is not None:
            dur = float(meta["format"]["duration"])
        return (fps_nom, nb, dur)
    except Exception:
        return (None, None, None)


def guess_subject_csv(r_friendly_dir: Path, subject_id: str) -> Optional[Path]:
    exact = r_friendly_dir / f"{subject_id}.csv"
    if exact.exists():
        return exact
    for p in r_friendly_dir.glob("*.csv"):
        if subject_id.lower() in p.name.lower():
            return p
    return None


def align_to_frame_times(bio_df: pd.DataFrame, frame_times: np.ndarray) -> pd.DataFrame:
    """
    Align biosignals to actual video frame times (seconds) with duplicate-time handling.
    - Aggregates duplicate timestamps before interpolation.
    - Interpolates numeric biosignals on datetime.
    - Attaches meta columns by nearest time.
    """
    bio = bio_df.copy()

    # Ensure numeric Time and sort
    bio["Time"] = pd.to_numeric(bio["Time"], errors="coerce")
    bio = bio.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Build datetimes
    bio["dt"] = pd.to_datetime(bio["Time"], unit="s", origin="unix", utc=True)
    target_dt = pd.to_datetime(frame_times, unit="s", origin="unix", utc=True)
    target_index = pd.DatetimeIndex(target_dt, name="dt")

    out = pd.DataFrame({
        "frame_idx": np.arange(len(frame_times), dtype=int),
        "t_sec": frame_times,
        "dt": target_dt
    })

    # Columns
    num_cols_all = ["Breathing.Rate","Heart.Rate","Perinasal.Perspiration","Gaze.X.Pos","Gaze.Y.Pos"]
    num_cols = [c for c in num_cols_all if c in bio.columns and pd.api.types.is_numeric_dtype(bio[c])]
    meta_cols = [c for c in ["Stimulus","Drive"] if c in bio.columns]

    # ---- 1) Deduplicate by dt ----
    # numeric: average duplicates; meta: take last non-null per dt
    if num_cols:
        bio_num = bio[["dt"] + num_cols].groupby("dt", as_index=True).mean().sort_index()
        # Interpolate on time, then select targets
        interp = (
            bio_num
            .reindex(bio_num.index.union(target_index))
            .interpolate(method="time", limit_direction="both")
            .reindex(target_index)
            .reset_index()
        )
        out = out.merge(interp, on="dt", how="left")

    # ---- 2) Meta (nearest-asof). Dedup first to avoid duplicate-label error
    if meta_cols:
        # Keep last value per dt for meta
        meta_df = (
            bio[["dt"] + meta_cols]
            .sort_values("dt")
            .groupby("dt", as_index=False)
            .agg({col: lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan for col in meta_cols})
            .sort_values("dt")
        )
        out = pd.merge_asof(
            out.sort_values("dt"),
            meta_df.sort_values("dt"),
            on="dt",
            direction="nearest"
        ).sort_values("frame_idx").reset_index(drop=True)

    return out




def process_subject(subject_dir: Path, r_friendly_dir: Path, out_dir: Path):
    sid = subject_dir.name
    md_folder = detect_md_folder(subject_dir)
    if not md_folder:
        print(f"[WARN] {sid}: No MD folder found — skipping.")
        return
    md_drive = detect_md_drive_num(md_folder)
    if md_drive is None:
        print(f"[WARN] {sid}: Could not parse MD drive number — skipping.")
        return

    out_csv = out_dir / f"{sid}_MD{md_drive}_biosignals_PERFRAME.csv"
    meta_json = out_dir / f"{sid}_MD{md_drive}_video_meta.json"
    if out_csv.exists() and meta_json.exists():
        print(f"[SKIP] {sid}: outputs already exist -> {out_csv.name}")
        return

    video = pick_md_video(md_folder, sid, md_drive)
    if not video or not video.exists():
        print(f"[WARN] {sid}: No MD video found in {md_folder} — skipping.")
        return

    # Get per-frame timestamps (VFR-aware)
    frame_times = ffprobe_frame_times(video)
    if frame_times is None:
        print(f"[WARN] {sid}: ffprobe could not extract per-frame times — skipping.")
        return

    # Basic info for logging
    fps_nom, nb_frames_meta, dur_meta = ffprobe_basic_info(video)
    # Compute instantaneous fps stats
    if len(frame_times) >= 2:
        deltas = np.diff(frame_times)
        # Guard against zeros
        inst_fps = 1.0 / np.clip(deltas, 1e-9, None)
        fps_min, fps_med, fps_max = float(np.min(inst_fps)), float(np.median(inst_fps)), float(np.max(inst_fps))
    else:
        fps_min = fps_med = fps_max = float("nan")

    print(f"[INFO] {sid}: MD={md_drive} | video={video.name} | frames={len(frame_times)} | "
          f"fps_nom={fps_nom:.6f} | fps_inst[min/med/max]={fps_min:.3f}/{fps_med:.3f}/{fps_max:.3f}")


    # Load biosignal CSV and filter to this MD drive
    csv_path = guess_subject_csv(r_friendly_dir, sid)
    if not csv_path:
        print(f"[WARN] {sid}: No CSV found in R-Friendly Study Data — skipping.")
        return

    try:
        df = pd.read_csv(csv_path, na_values=NA_VALUES, low_memory=False)
    except Exception as e:
        print(f"[WARN] {sid}: Failed to read {csv_path.name}: {e}")
        return

    df.columns = [c.strip() for c in df.columns]
    if "Drive" not in df.columns or "Time" not in df.columns:
        print(f"[WARN] {sid}: Missing 'Drive' or 'Time' — skipping.")
        return

    md_df = df[df["Drive"].astype(str).str.strip() == str(md_drive)].copy()
    if md_df.empty:
        print(f"[WARN] {sid}: No rows with Drive == {md_drive}")
        return

    # Keep only relevant columns present
    present = [c for c in KEEP_COLS if c in md_df.columns]
    md_df = md_df[present].copy()

    # Align biosignals to the *actual* frame times
    aligned = align_to_frame_times(md_df, frame_times)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{sid}_MD{md_drive}_biosignals_PERFRAME.csv"
    aligned.to_csv(out_csv, index=False)

    # Save a small meta JSON per subject (useful for diagnostics)
    meta = {
        "subject": sid,
        "md_drive": md_drive,
        "video": str(video),
        "n_frames": int(len(frame_times)),
        "fps_nominal": float(fps_nom) if fps_nom is not None else None,
        "n_frames_meta": int(nb_frames_meta) if nb_frames_meta is not None else None,
        "duration_meta": float(dur_meta) if dur_meta is not None else None,
        "inst_fps_min": fps_min, "inst_fps_median": fps_med, "inst_fps_max": fps_max,
        "biosignal_time_min": float(md_df["Time"].min()),
        "biosignal_time_max": float(md_df["Time"].max()),
    }
    (out_dir / f"{sid}_MD{md_drive}_video_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] {sid}: wrote {out_csv} and video_meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_id", type=str, default=None,
                    help="Start from this subject id (e.g., T032) inclusive.")
    ap.add_argument("--only", dest="only", type=str, default=None,
                    help="Comma-separated list of subject ids to process (e.g., T032,T033).")
    args = ap.parse_args()

    subs = find_subject_dirs(ROOT)
    if not subs:
        print("[ERR] No subject folders found.")
        sys.exit(1)

    # Build subject name list
    names = [p.name for p in subs]

    # Filter: --only has priority
    if args.only:
        wanted = [s.strip().upper() for s in args.only.split(",") if s.strip()]
        submap = {p.name.upper(): p for p in subs}
        subs = [submap[s] for s in wanted if s in submap]
        if not subs:
            print(f"[ERR] None of the requested subjects found: {wanted}")
            sys.exit(1)
    elif args.from_id:
        start = args.from_id.strip().upper()
        # subjects are already sorted (T001, T002, …). Find index of start.
        idx = None
        for i, p in enumerate(subs):
            if p.name.upper() == start:
                idx = i
                break
        if idx is None:
            print(f"[ERR] Start subject not found: {start}")
            sys.exit(1)
        subs = subs[idx:]  # slice from start inclusive

    for s in subs:
        process_subject(s, R_FRIENDLY, OUT_DIR)



if __name__ == "__main__":
    main()
