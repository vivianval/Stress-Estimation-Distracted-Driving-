#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

OUT_DIR = Path("/home/vivib/emoca/emoca/dataset/BIOSIGNALS_CSVS")
REQUIRED_COLS = [
    "Time",
    "Breathing.Rate",
    "Heart.Rate",
    "Perinasal.Perspiration",
    "Gaze.X.Pos",
    "Gaze.Y.Pos",
]
OPTIONAL_COLS = ["Drive", "Stimulus", "Frame", "FPS_used"]

def analyze_csv(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"file": path.name, "path": str(path)}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        info["error"] = f"read_error: {e}"
        return info

    info["rows"] = len(df)
    info["cols"] = len(df.columns)
    info["columns"] = list(df.columns)
    info["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    # Basic required columns check
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    info["missing_required"] = ",".join(missing) if missing else ""

    # NaN counts for key columns
    for c in REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]:
        info[f"nan_{c}"] = int(df[c].isna().sum()) if c in df.columns else None

    # Drive info
    if "Drive" in df.columns:
        drives = df["Drive"].dropna().astype(str).str.strip().unique().tolist()
        info["unique_drives"] = ",".join(drives)
        info["drive_count"] = len(drives)
    else:
        info["unique_drives"] = ""
        info["drive_count"] = 0

    # FPS_used (if present)
    if "FPS_used" in df.columns:
        fps_vals = pd.to_numeric(df["FPS_used"], errors="coerce").dropna().unique()
        info["fps_used_unique"] = ",".join([f"{v:.6g}" for v in fps_vals[:10]])
    else:
        info["fps_used_unique"] = ""

    # Time/Frame checks
    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce")
        info["time_min"] = float(np.nanmin(t)) if len(t) else None
        info["time_max"] = float(np.nanmax(t)) if len(t) else None
        # Monotonic (allow equal time for ties)
        info["time_monotonic_nondec"] = bool(t.is_monotonic_increasing) if t.notna().all() else False

    if "Frame" in df.columns:
        f = pd.to_numeric(df["Frame"], errors="coerce")
        info["frame_min"] = float(np.nanmin(f)) if len(f) else None
        info["frame_max"] = float(np.nanmax(f)) if len(f) else None
        # Check all integers (allow pandas Int64 or floats that are whole numbers)
        is_intlike = f.dropna().apply(lambda x: float(x).is_integer()).all()
        info["frame_all_intlike"] = bool(is_intlike)

    # Print a concise console report
    print("=" * 80)
    print(f"File: {path.name}")
    print(f"Shape: {info['rows']} rows x {info['cols']} cols")
    print("Columns:", ", ".join(info["columns"]))
    if info.get("missing_required"):
        print("MISSING required:", info["missing_required"])
    print("dtypes:", info["dtypes"])
    if "Drive" in df.columns:
        print("Unique Drive values:", info["unique_drives"])
    if "FPS_used" in df.columns:
        print("FPS_used unique:", info["fps_used_unique"])
    if "Time" in df.columns:
        print(f"Time range: {info.get('time_min')} → {info.get('time_max')} | monotonic_nondec={info.get('time_monotonic_nondec')}")
    if "Frame" in df.columns:
        print(f"Frame range: {info.get('frame_min')} → {info.get('frame_max')} | intlike={info.get('frame_all_intlike')}")
    # Show heads/tails (safe for console)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\nHEAD(3):")
        print(df.head(3))
        print("\nTAIL(3):")
        print(df.tail(3))

    return info

def main():
    if not OUT_DIR.exists():
        print(f"[ERR] Directory not found: {OUT_DIR}")
        sys.exit(1)

    csvs = sorted(OUT_DIR.glob("T*_MD*_biosignals.csv"))
    if not csvs:
        print(f"[WARN] No per-subject CSVs found in {OUT_DIR}")
        sys.exit(0)

    summary: List[Dict[str, Any]] = []
    for p in csvs:
        info = analyze_csv(p)
        summary.append(info)

    # Save summary CSV
    summary_df = pd.DataFrame(summary)
    summary_path = OUT_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 80)
    print(f"[OK] Wrote summary: {summary_path}")
    print("Columns in summary:", ", ".join(summary_df.columns))

if __name__ == "__main__":
    main()
