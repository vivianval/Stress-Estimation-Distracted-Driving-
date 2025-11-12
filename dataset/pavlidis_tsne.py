
"""
Phase-wise paired differences using phases inferred from .stm Excel files (NO CLEANING).

For each subject Txxx:
  - Find ND drive ID from BIOSIGNALS_CSVS_ND_CLEAN filename.
  - Find MD drive ID from FINAL_PERFRAME_CLEAN_PHASED filename (only to pick the MD drive rows from raw CSV).
  - Find the subject's .stm (readable as Excel: .stm or .stm.xlsx) under SUBJECT_ROOT/Txxx/* MD/.
  - Read StartTime/EndTime rows -> build P1..P5 windows in the MD timeline:
        P1: MD_start -> s1
        P2: [s1, e1]
        P3: (e1, s2)
        P4: [s2, e2]
        P5: (e2, MD_end]
    (If only one stressor exists, you'll get P1, P2, P3; if more than two rows exist, extra baseline/stressor pairs are added accordingly.)
  - Apply SAME durations to the ND drive starting at ND_start.
  - Compute per-phase means (Perinasal.Perspiration, Heart.Rate, Breathing.Rate), Δ=MD−ND.
  - Save boxplots + t-tests; print per-subject phase boundaries and durations.

Outputs in OUT_DIR:
  - paired_deltas.csv
  - ttests.txt
  - paired_boxplot_<modality>.png
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel
import math


# =========================
# CONFIG — EDIT THESE PATHS
# =========================
RAW_DIR       = Path("/media/storage/vivib/Structured Study Data/R-Friendly Study Data")  # Txxx.csv
ND_DIR        = Path("/home/vivib/emoca/emoca/dataset/BIOSIGNALS_CSVS_ND_CLEAN")         # *_ND#_biosignals_PERFRAME_CLEAN.csv
MDPH_DIR      = Path("/home/vivib/emoca/emoca/dataset/FINAL_PERFRAME_CLEAN_PHASED")      # *_MD#_FINAL_PERFRAME_CLEAN_PHASED.csv (for MD id)
SUBJECT_ROOT  = Path("/media/storage/vivib/Structured Study Data")                        # .../Txxx/* MD/*.stm(.xlsx)
OUT_DIR       = Path("/home/vivib/emoca/emoca/dataset/phase_paired_from_stm_no_cleaning")

TIME_COL      = "Time"   # time column in RAW_DIR/Txxx.csv
MODALITIES    = ["Perinasal.Perspiration", "Heart.Rate", "Breathing.Rate"]

# =========================

OUT_DIR.mkdir(parents=True, exist_ok=True)


PHASES = ["P1","P2","P3","P4","P5"]  # keep consistent everywhere

def phases_dict_to_wide_df(subject_id: str, phases_dict: dict, modalities=MODALITIES) -> pd.DataFrame:
    """
    phases_dict looks like:
      {"P1": {"Heart.Rate": 75.2, "Breathing.Rate": 16.3, ...},
       "P2": {...}, ...}
    Returns a wide DF with columns: subject, modality, P1..P5
    """
    rows = []
    for mod in modalities:
        row = {"subject": subject_id, "modality": mod}
        for ph in PHASES:
            row[ph] = phases_dict.get(ph, {}).get(mod, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def melt_phase_means(wide_df: pd.DataFrame, typ: str) -> pd.DataFrame:
    m = wide_df.melt(id_vars=["subject","modality"],
                     value_vars=PHASES,
                     var_name="phase", value_name="value")
    m["type"] = typ
    return m



# ---- add this helper block near the top (after imports) ----
def _first_match(cols, patterns):
    """Return first column name matching any regex in patterns (case-insensitive)."""
    for p in patterns:
        rx = re.compile(p, flags=re.IGNORECASE)
        for c in cols:
            if rx.fullmatch(c) or rx.search(c):
                return c
    return None



def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _pick_col(df: pd.DataFrame, patterns) -> str:
    """
    Return the first column whose canonical name matches any regex in `patterns`.
    """
    cans = {c: _canon(c) for c in df.columns}
    for c, z in cans.items():
        for pat in patterns:
            if re.search(pat, z):
                return c
    return None

def harmonize_raw_columns(df: pd.DataFrame, src_name="(unknown)"):
    """
    Map whatever the site called each signal to canonical names:
      Heart.Rate, Breathing.Rate, Perinasal.Perspiration, Time, Drive
    Returns: (df_copy, mapping_dict)
    """
    dfx = df.copy()

    # --- Time ---
    time_col = _pick_col(dfx, [
        r"^time$", r"^timestamp$", r"^time[s]?$", r"^t$"
    ])
    if time_col is None:
        # some exports use 'Seconds' or 'ElapsedTime'
        time_col = _pick_col(dfx, [r"elapsed", r"second"])
    # --- Drive ---
    drive_col = _pick_col(dfx, [r"^drive$"])

    # --- Heart rate (HR, pulse, bpm) ---
    hr_col = _pick_col(dfx, [
        r"heartrate", r"^hr$", r"pulse", r"bpm(?!.*resp)", r"cardiac"
    ])

    # --- Breathing / Respiration (tons of variants) ---
    br_col = _pick_col(dfx, [
        r"breath.*rate", r"breathingrate", r"resp.*rate", r"^rr$",
        r"respiratoryrate", r"resp", r"brpm", r"ventilation"
    ])

    # --- Perinasal perspiration / thermal metric ---
    pp_col = _pick_col(dfx, [
        r"perinasal", r"nasal.*perspir", r"pp$", r"thermal.*(nose|perinasal)",
        r"delta.*temp", r"nose.*temp"
    ])

    mapping = {}
    if time_col is not None:
        dfx.rename(columns={time_col: "Time"}, inplace=True)
        mapping[time_col] = "Time"
    if drive_col is not None:
        dfx.rename(columns={drive_col: "Drive"}, inplace=True)
        mapping[drive_col] = "Drive"
    if hr_col is not None:
        dfx.rename(columns={hr_col: "Heart.Rate"}, inplace=True)
        mapping[hr_col] = "Heart.Rate"
    if br_col is not None:
        dfx.rename(columns={br_col: "Breathing.Rate"}, inplace=True)
        mapping[br_col] = "Breathing.Rate"
    if pp_col is not None:
        dfx.rename(columns={pp_col: "Perinasal.Perspiration"}, inplace=True)
        mapping[pp_col] = "Perinasal.Perspiration"

    # diagnostics
    missing = [x for x in ["Time","Drive","Heart.Rate","Breathing.Rate","Perinasal.Perspiration"]
               if x not in dfx.columns]
    print(f"[COLUMNS] mapped {src_name}: " + ", ".join([f"{k}→{v}" for k,v in mapping.items()]) +
          ("" if mapping else "(no matches)"))
    if missing:
        print(f"[WARN] {src_name}: missing canonical columns -> {missing}")

    return dfx, mapping

def panel_dodged(ax, t, raw_v, cln_v, title, dodge_frac=0.015):
    raw_v = np.asarray(raw_v, float); cln_v = np.asarray(cln_v, float)

    # robust range (ignore outliers)
    y = np.concatenate([raw_v[np.isfinite(raw_v)], cln_v[np.isfinite(cln_v)]])
    if len(y) == 0: 
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center'); return
    p5, p95 = np.nanpercentile(y, [5, 95])
    yrng = max(1e-12, p95 - p5)
    delta = dodge_frac * yrng  # e.g., 1.5% of robust range

    # plot: cleaned shifted up by delta
    ax.plot(t, cln_v + delta, color="#ff8c00", lw=1.4, alpha=0.9, label=f"cleaned (+{delta:.3g})")
    ax.plot(t, raw_v,          color="black",   lw=1.6, alpha=0.95, linestyle="--", label="raw")
    ax.set_title(title); ax.grid(alpha=0.15)

    # right-axis shows TRUE cleaned values (no offset)
    ax2 = ax.twinx()
    yl = np.array(ax.get_ylim())
    ax2.set_ylim(yl - delta)
    ax2.set_ylabel("cleaned (true scale)", fontsize=8)

    # little Δ inset (difference) for quick glance
    if np.isfinite(raw_v).any() and np.isfinite(cln_v).any():
        res = cln_v - raw_v
        axins = ax.inset_axes([0.67, 0.58, 0.30, 0.34])
        axins.plot(t, res, lw=0.9)
        axins.axhline(0, ls="--", lw=0.8)
        axins.set_title("Δ = cleaned − raw", fontsize=8)
        for spine in axins.spines.values(): spine.set_alpha(0.4)
        axins.tick_params(labelsize=7)

    ax.legend(loc="upper right", fontsize=8)



def add_diff_panel(fig, ax_main, t, raw_v, cln_v):
    res = np.asarray(cln_v,float) - np.asarray(raw_v,float)
    axd = fig.add_axes([ax_main.get_position().x0,
                        ax_main.get_position().y0 - 0.10,   # below
                        ax_main.get_position().width, 0.08])
    axd.plot(t, res, lw=0.9)
    axd.axhline(0, ls="--", lw=0.8)
    axd.set_yticks([]); axd.set_xticks([])
    axd.set_title("Δ", fontsize=8)


DIAG_DIR = OUT_DIR / "diagnostics_plots"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

YLIMS = {
    "Heart.Rate": None,
    "Breathing.Rate": None,
    "Perinasal.Perspiration": None,   # autoscale
}

def _coerce_series(df, col):
    if col not in df.columns:
        return pd.Series([], dtype=float)
    return coerce_numeric_strict(df[col])

def plot_subject_overlays(subj, df_md_raw, df_nd_raw, show_residual=True):
    # time (always coerced from RAW_DIR slices)
    t_md = coerce_numeric_strict(df_md_raw[TIME_COL]) if TIME_COL in df_md_raw else pd.Series([], dtype=float)
    t_nd = coerce_numeric_strict(df_nd_raw[TIME_COL]) if TIME_COL in df_nd_raw else pd.Series([], dtype=float)

    # build cleaned (NO z-score here; true units)
    df_md_cln = df_md_raw.copy()
    df_nd_cln = df_nd_raw.copy()
    for mod in MODALITIES:
        df_md_cln[mod] = clean_signal(_coerce_series(df_md_raw, mod), mod, samp_hz=1)
        df_nd_cln[mod] = clean_signal(_coerce_series(df_nd_raw, mod), mod, samp_hz=1)

    # style
    RAW_STYLE = dict(color="#777777", linewidth=0.8, alpha=0.9, zorder=1)
    CLN_STYLE = dict(color="#ff8c00", linewidth=1.8, alpha=0.95, zorder=2)
    RES_STYLE = dict(linewidth=0.8, alpha=0.6)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), sharex=False)
    axes = np.atleast_2d(axes)

    def panel(ax, t, raw_v, cln_v, title):
        RAW_STYLE = dict(color="black", linewidth=1.6, alpha=0.9, zorder=5, linestyle="--")
        CLN_STYLE = dict(color="#ff8c00", linewidth=1.4, alpha=0.7, zorder=1)

        # 1) cleaned first (goes underneath)
        ax.plot(t, cln_v, **CLN_STYLE, label="cleaned")

        # 2) raw on top so it can’t be hidden
        ax.plot(t, raw_v, **RAW_STYLE, label="raw")
        # markers for raw (on top)
        if len(raw_v) and len(t):
            step = max(1, len(raw_v)//80)
            ax.plot(t[::step], raw_v[::step], "o", ms=2.6, zorder=6, markerfacecolor="white",
                    markeredgecolor="black", alpha=0.9)

        ax.set_title(title); ax.grid(alpha=0.15)

        # stats box
        def safe(v): 
            return (np.nanmin(v), np.nanmedian(v), np.nanmax(v)) if np.isfinite(v).any() else (np.nan, np.nan, np.nan)
        rmin, rmed, rmax = safe(raw_v); cmin, cmed, cmax = safe(cln_v)
        msg = f"raw  min/med/max: {rmin:.2f}/{rmed:.2f}/{rmax:.2f}\n" \
            f"cln  min/med/max: {cmin:.2f}/{cmed:.2f}/{cmax:.2f}"
        ax.text(0.01, 0.97, msg, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, lw=0))

        # residual only if both exist
        if np.isfinite(raw_v).any() and np.isfinite(cln_v).any():
            res = np.asarray(cln_v, float) - np.asarray(raw_v, float)
            ax2 = ax.twinx()
            ax2.plot(t, res, linewidth=0.9, alpha=0.6)
            ax2.set_ylabel("cleaned − raw", fontsize=8)
            # If almost identical, say it explicitly
            if np.nanmax(np.abs(res)) < 1e-6:
                ax.text(0.99, 0.03, "raw ≈ cleaned", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, lw=0))

        ax.legend(loc="upper right", fontsize=8)

    for r, mod in enumerate(MODALITIES):
        # MD left
        panel_dodged(axes[r, 0], t_md.values,
              _coerce_series(df_md_raw, mod).values,
              df_md_cln[mod].values,
              f"{mod} — MD")
        axes[r, 0].set_ylabel(mod.replace(".", " "))

        # ND right
        panel_dodged(axes[r, 1], t_nd.values,
              _coerce_series(df_nd_raw, mod).values,
              df_nd_cln[mod].values,
              f"{mod} — ND")

        if YLIMS.get(mod):
            axes[r, 0].set_ylim(*YLIMS[mod])
            axes[r, 1].set_ylim(*YLIMS[mod])

    for c in range(2):
        axes[-1, c].set_xlabel("Time (s)")

    fig.suptitle(f"{subj}: RAW vs CLEANED overlays (MD & ND)", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_png = DIAG_DIR / f"{subj}_overlay.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[PLOT] -> {out_png}")


def make_all_overlay_plots():
    subjects = subject_ids_from_raw(RAW_DIR)
    for subj in subjects:
        # get MD/ND ids
        nd_id, _ = find_nd_drive_id(ND_DIR, subj)
        md_id, _ = find_md_drive_id(MDPH_DIR, subj)
        raw_csv = RAW_DIR / f"{subj}.csv"
        if not raw_csv.exists() or nd_id is None or md_id is None:
            print(f"[skip] {subj}: missing CSV or ids (MD={md_id}, ND={nd_id})")
            continue

        df_raw = pd.read_csv(raw_csv, low_memory=False)
        df_raw, _map = harmonize_raw_columns(df_raw, src_name=raw_csv.name)


        needed = ["Time", "Drive"] + MODALITIES  # MODALITIES unchanged
        missing = [c for c in needed if c not in df_raw.columns]
        if missing:
            print(f"[SKIP] {subj}: missing required columns after harmonization: {missing}")
            continue


        df_md_raw = df_raw[df_raw["Drive"] == md_id].copy()
        df_nd_raw = df_raw[df_raw["Drive"] == nd_id].copy()
        if df_md_raw.empty or df_nd_raw.empty:
            print(f"[skip] {subj}: empty MD/ND slices")
            continue

        plot_subject_overlays(subj, df_md_raw, df_nd_raw)

def subject_ids_from_raw(raw_dir: Path):
    return sorted([p.stem for p in raw_dir.glob("T*.csv") if p.is_file()])

def find_nd_drive_id(nd_dir: Path, subj: str):
    # e.g., T034_ND7_biosignals_PERFRAME_CLEAN.csv → ND=7
    for p in nd_dir.glob(f"{subj}_ND*_biosignals_PERFRAME_CLEAN.csv"):
        m = re.search(r"_ND(\d+)_", p.name)
        if m:
            return int(m.group(1)), p
    return None, None

def find_md_drive_id(mdph_dir: Path, subj: str):
    # e.g., T034_MD6_FINAL_PERFRAME_CLEAN_PHASED.csv → MD=6
    for p in mdph_dir.glob(f"{subj}_MD*_FINAL_PERFRAME_CLEAN_PHASED.csv"):
        m = re.search(r"_MD(\d+)_", p.name)
        if m:
            return int(m.group(1)), p
    return None, None

def find_stm_excel(subject_root: Path, subj: str):
    """
    Look for .../Txxx/* MD/*.stm or *.stm.xlsx and return a path we can read with pandas.
    If we find a .stm (no extension), we will try to read p.with_suffix(p.suffix + ".xlsx").
    """
    base = subject_root / subj
    candidates = list(base.glob("* MD/*.stm*"))
    if not candidates:
        return None
    # Prefer a file that already endswith .xlsx; else append .xlsx
    p = sorted(candidates)[0]
    if p.suffix.lower() == ".xlsx":
        return p
    # If it's just ".stm", try "<file>.xlsx"
    alt = Path(str(p) + ".xlsx")
    return alt if alt.exists() else p  # pandas can sometimes read non-.xlsx if it's real xlsx

def nd_anchor_normalize(df_md: pd.DataFrame,
                        df_nd: pd.DataFrame,
                        modalities,
                        scale: str = "std"):
    """
    ND-anchored normalization.

    For each modality:
      - compute ND mean (and optionally ND std),
      - express BOTH ND and MD relative to ND.
    scale="std"  -> (x - ND_mean) / ND_std        # dimensionless, preserves MD>ND shift
    scale="none" -> (x - ND_mean)                 # stays in original units (e.g., bpm)

    Returns: (df_md_norm, df_nd_norm)
    """
    df_md = df_md.copy()
    df_nd = df_nd.copy()
    for mod in modalities:
        nd_vals = pd.to_numeric(df_nd[mod], errors="coerce").astype(float)
        nd_mean = float(np.nanmean(nd_vals))
        if scale == "std":
            nd_std = float(np.nanstd(nd_vals, ddof=0))
            if not np.isfinite(nd_std) or nd_std < 1e-6:
                nd_std = 1.0  # avoid blow-ups
            df_nd[mod] = (pd.to_numeric(df_nd[mod], errors="coerce") - nd_mean) / nd_std
            df_md[mod] = (pd.to_numeric(df_md[mod], errors="coerce") - nd_mean) / nd_std
        elif scale == "none":
            df_nd[mod] = pd.to_numeric(df_nd[mod], errors="coerce") - nd_mean
            df_md[mod] = pd.to_numeric(df_md[mod], errors="coerce") - nd_mean
        else:
            raise ValueError("scale must be 'std' or 'none'")
    return df_md, df_nd




def _find_start_end_from_excel(stm_path):
    """
    Return (sheet_name, header_row_idx, start_col_idx, end_col_idx)
    by scanning all sheets with header=None. Handles 'StartTime'/'Start Time'
    and 'EndTime'/'End Time' (case-insensitive).
    """
    xls = pd.ExcelFile(stm_path)
    targets = {
        "start": {"starttime", "start time"},
        "end":   {"endtime", "end time"}
    }

    def norm(x):
        if isinstance(x, str):
            return x.strip().lower()
        return None

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        nrows, ncols = df.shape
        for r in range(nrows):
            # build normalized row tokens
            row_tokens = [norm(v) for v in df.iloc[r].tolist()]
            # find candidate indices for start and end
            start_idx, end_idx = None, None
            for c, tok in enumerate(row_tokens):
                if tok in targets["start"]:
                    start_idx = c
                if tok in targets["end"]:
                    end_idx = c
            if start_idx is not None and end_idx is not None:
                return sheet, r, start_idx, end_idx
    return None, None, None, None


def build_phases_from_stm(df_md: pd.DataFrame, stm_path: Path):
    """
    Robust parser for .stm Excel exports where 'StartTime'/'EndTime'
    are embedded in a row (not as DataFrame headers).
    Generalizes to N stressor intervals:
        baseline, stressor, baseline, stressor, ..., baseline
    Returns:
      phases:   list of (label, start, end)
      durations:list of (label, duration_seconds)
    """
    # MD timeline limits
    df_md = df_md.sort_values(by=TIME_COL).reset_index(drop=True)
    t0 = float(df_md[TIME_COL].iloc[0])
    t1 = float(df_md[TIME_COL].iloc[-1])
    eps = 1e-9

    # Try to locate header row and columns inside the Excel
    sheet, hdr_row, c_start, c_end = _find_start_end_from_excel(stm_path)
    if sheet is None:
        raise ValueError(
            f"Could not find a row with StartTime/EndTime in {stm_path.name}. "
            f"Try opening one file manually to confirm the exact strings."
        )

    # Load the sheet raw (header=None) and slice numeric interval rows
    df_raw = pd.read_excel(stm_path, sheet_name=sheet, header=None)
    # Rows *below* the header row
    df_int = df_raw.iloc[hdr_row+1:, [c_start, c_end]].copy()
    df_int.columns = ["StartTime", "EndTime"]

    # Coerce to floats; keep only rows where both are finite numbers
    def to_float_safe(x):
        try:
            if isinstance(x, str):
                x = x.replace(",", ".")
            v = float(x)
            return v
        except Exception:
            return np.nan

    df_int["StartTime"] = df_int["StartTime"].map(to_float_safe)
    df_int["EndTime"]   = df_int["EndTime"].map(to_float_safe)
    df_int = df_int.dropna(subset=["StartTime", "EndTime"])

    if df_int.empty:
        raise ValueError(f"{stm_path.name}: No numeric StartTime/EndTime rows under header row {hdr_row}.")

    # Clamp & sort
    df_int = df_int.astype(float).sort_values("StartTime").reset_index(drop=True)
    # Filter obviously invalid rows (negative or inverted intervals)
    df_int = df_int[(df_int["EndTime"] >= df_int["StartTime"]) &
                    (df_int["EndTime"] >= 0) &
                    (df_int["StartTime"] <= t1 + 1e6)]  # allow slightly beyond; will clamp

    # Build alternating baseline/stressor phases that cover [t0, t1]
    phases = []
    cursor = t0
    phase_idx = 1

    for _, row in df_int.iterrows():
        s = float(row["StartTime"])
        e = float(row["EndTime"])
        # clamp to MD bounds
        s = max(s, t0)
        e = min(e, t1)
        # Baseline: [cursor, s)
        if s - cursor > 0:
            phases.append((f"P{phase_idx}", cursor, s - eps))
            phase_idx += 1
        # Stressor: [s, e]
        if e > s:
            phases.append((f"P{phase_idx}", s, e))
            phase_idx += 1
        cursor = max(cursor, e) + eps

    # Tail baseline
    if t1 - cursor > 0:
        phases.append((f"P{phase_idx}", cursor, t1))

    # If no intervals found (edge case), fall back to a single phase
    if not phases:
        phases = [("P1", t0, t1)]

    durations = [(lab, float(end - start)) for (lab, start, end) in phases]

    # DEBUG print (so you can verify)
    print(f"       STM sheet: {sheet}  header_row={hdr_row}  cols(start,end)=({c_start},{c_end})")
    print("       Phases inferred from STM:")
    for lab, a, b in phases:
        print(f"         - {lab}: start={a:.3f}s  end={b:.3f}s  dur={b-a:.3f}s")

    return phases, durations


def slice_by_durations(df_drive: pd.DataFrame, start_time: float, durations):
    """
    Apply a sequence of (label, duration) starting at 'start_time' on df_drive[TIME_COL].
    Assumes df_drive is already sorted and reset_index(drop=True).
    Returns list of (label, mask) where mask is a NumPy boolean array aligned to df_drive rows.
    """
    # IMPORTANT: df_drive must already be sorted/reset by caller
    t = df_drive[TIME_COL].to_numpy(dtype=float)
    masks = []
    cursor = start_time
    eps = 1e-9
    for lab, dur in durations:
        a = cursor
        b = cursor + float(dur)
        m = (t >= a) & (t <= b + eps)   # NumPy boolean mask aligned to df_drive
        masks.append((lab, m))
        cursor = b + eps
    return masks


NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


inc_path = Path("/home/vivib/emoca/emoca/dataset/phase_paired_from_stm_no_cleaning/inclusion_log.csv")   # adjust if needed
df = pd.read_csv(inc_path)

# count how many unique modalities are present (non-NaN) per subject
valid_counts = (
    df[df["used_in_delta"] == True]
    .groupby("subject")["modality"]
    .nunique()
    .rename("n_valid_modalities")
)

# mark subjects with <3 as incomplete
incomplete_subjects = valid_counts[valid_counts < 3].index.tolist()

print("Subjects missing ≥1 biosignal:")
print(incomplete_subjects)
print("\nSubjects with all 3 biosignals:")
print(valid_counts[valid_counts == 3].index.tolist())


def coerce_numeric_strict(series: pd.Series) -> pd.Series:
    """
    Convert a messy text/number column to float:
    - Accepts '62', '62.3', '62,3', '62 bpm', '  62  ', etc.
    - Returns NaN if nothing numeric is found in a cell.
    """
    out = []
    for x in series.tolist():
        if x is None or (isinstance(x, float) and np.isnan(x)):
            out.append(np.nan); continue
        if isinstance(x, (int, float, np.integer, np.floating)):
            out.append(float(x)); continue
        s = str(x).strip()
        # swap comma-decimal to dot if present (but keep thousands-style commas out)
        s = s.replace(",", ".")
        m = NUM_RE.search(s)
        if m:
            try:
                out.append(float(m.group(0)))
            except Exception:
                out.append(np.nan)
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="float64")


# def clean_signal(series, modality, samp_hz=1):
#     raw = series.copy()
#     s = coerce_numeric_strict(raw)

#     num_ok = s.notna().mean()*100
#     print(f"[RAW ] {modality:<22} numeric_ok={num_ok:5.1f}% "
#           f"min={np.nanmin(s):.3f} med={np.nanmedian(s):.3f} max={np.nanmax(s):.3f}")

#     total = len(s)
#     before = s.isna().sum()

#     if "Heart.Rate" in modality:
#         # allow physiologically plausible stress spikes
#         s[(s < 35) | (s > 145)] = np.nan
#     elif "Breathing.Rate" in modality:
#         s[(s < 4) | (s > 40)] = np.nan
#     elif "Perinasal" in modality:
#         pass

#     after = s.isna().sum()
#     dropped = max(0, after - before)

#     s_interp = s.interpolate(limit=int(5*max(1, samp_hz)), limit_direction="both")

#     print(f"[CLEAN] {modality:<22} total={total:5d} dropped={dropped:5d} "
#           f"({100*dropped/max(total,1):5.2f}%) "
#           f"post(min/med/max)=({np.nanmin(s_interp):.2f}/{np.nanmedian(s_interp):.2f}/{np.nanmax(s_interp):.2f})")
#     return s_interp

def clean_signal(series, modality, samp_hz=1, interpolate=True):
    raw = series.copy()
    s = coerce_numeric_strict(raw)

    # gentle plausibility clipping only (or comment out entirely if you prefer)
    if "Heart.Rate" in modality:
        s[(s < 35) | (s > 180)] = np.nan  # widen upper bound to avoid chopping stress peaks
    elif "Breathing.Rate" in modality:
        s[(s < 4) | (s > 50)] = np.nan
    # Perinasal: leave as-is

    if interpolate:
        s = s.interpolate(limit=int(5*max(1, samp_hz)), limit_direction="both")

    return s

def zscore_within_subject(df, cols):
    dfz = df.copy()
    for c in cols:
        vals = dfz[c]
        dfz[c] = (vals - vals.mean()) / vals.std(ddof=0)
    return dfz

def phase_mean(series, mask, min_frac=0.5):
    # mean for a phase only if ≥ min_frac of samples are finite
    vals = series[mask]
    frac = np.isfinite(vals).mean()
    return float(np.nanmean(vals)) if frac >= min_frac else np.nan

def mean_by_phase_using_durations(df_drive: pd.DataFrame, durations, min_frac=0.5):
    if df_drive.empty:
        return {}
    df_drive = df_drive.sort_values(by=TIME_COL).reset_index(drop=True).copy()
    start_t = float(df_drive[TIME_COL].iloc[0])
    masks = slice_by_durations(df_drive, start_t, durations)

    out = {}
    for lab, m in masks:
        row = {}
        for mod in MODALITIES:
            row[mod] = phase_mean(df_drive[mod].astype(float), m, min_frac=min_frac)
        out[lab] = row
    return out




def star(p):
    if p < 0.001:
        return "***"
    elif p < 0.0125:  # Bonferroni (n=4 comparisons in the paper)
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"



def main():
    subjects = subject_ids_from_raw(RAW_DIR)
    if not subjects:
        print(f"No Txxx.csv files found under {RAW_DIR}")
        return

    phase_order_ref = None
    all_rows = []
    MD_WIDE_ROWS = []   # collect one wide DF per subject
    ND_WIDE_ROWS = []

    for subj in subjects:
        # Find drives
        nd_id, nd_path = find_nd_drive_id(ND_DIR, subj)
        md_id, mdph_path = find_md_drive_id(MDPH_DIR, subj)
        if nd_id is None or md_id is None:
            print(f"[SKIP] {subj}: ND or MD id not found.")
            continue

        raw_csv = RAW_DIR / f"{subj}.csv"
        if not raw_csv.exists():
            print(f"[SKIP] {subj}: missing raw CSV {raw_csv}")
            continue

        df_raw = pd.read_csv(raw_csv)
        needed = [TIME_COL, "Drive"] + MODALITIES
        if any(c not in df_raw.columns for c in needed):
            print(f"[SKIP] {subj}: required columns missing in {raw_csv}. Need {needed}")
            continue

        df_md = df_raw[df_raw["Drive"] == md_id].copy()
        df_nd = df_raw[df_raw["Drive"] == nd_id].copy()

                # 1) CLEAN on RAW
        for mod in MODALITIES:
            df_nd[mod] = clean_signal(df_nd[mod], mod, samp_hz=1, interpolate = False)
            df_md[mod] = clean_signal(df_md[mod], mod, samp_hz=1, interpolate = False)

        # # 2) ND-anchored normalization (choose scale)
        ND_ANCHOR = False
        ND_ANCHOR_SCALE = "std"   # "none" -> bpm-centered; "std" -> ND-z units

        if ND_ANCHOR:
            df_md, df_nd = nd_anchor_normalize(df_md, df_nd, MODALITIES, scale=ND_ANCHOR_SCALE)


        # # 1) CLEAN on RAW
        # for mod in MODALITIES:
        #     df_nd[mod] = clean_signal(df_nd[mod], mod, samp_hz=1)
        #     df_md[mod] = clean_signal(df_md[mod], mod, samp_hz=1)
           


  

        # 2) (OPTIONAL) z-score AFTER cleaning
        APPLY_ZSCORE = True
        if APPLY_ZSCORE:
            df_nd[MODALITIES] = zscore_within_subject(df_nd[MODALITIES], MODALITIES) #(df_nd[MODALITIES] - df_nd[MODALITIES].median()) / df_nd[MODALITIES].std()                                                                 #df_nd[MODALITIES], MODALITIES)

            df_md[MODALITIES] = zscore_within_subject(df_md[MODALITIES], MODALITIES)
        #     # df_md_centered = df_md[MODALITIES]  - df_md[MODALITIES].mean()
        #     # df_nd_centered = df_nd[MODALITIES] - df_nd[MODALITIES].mean()
        


        if df_md.empty or df_nd.empty:
            print(f"[SKIP] {subj}: empty MD/ND slices (MD rows={len(df_md)}, ND rows={len(df_nd)})")
            continue

        # Find .stm Excel
        stm_path = find_stm_excel(SUBJECT_ROOT, subj)
        if stm_path is None or not stm_path.exists():
            print(f"[SKIP] {subj}: .stm(.xlsx) not found under {SUBJECT_ROOT/subj}")
            continue

        # Build phases from .stm with MD timeline
        try:
            phases, durations = build_phases_from_stm(df_md, stm_path)
        except Exception as e:
            print(f"[SKIP] {subj}: failed to parse {stm_path.name}: {e}")
            continue

        # Print mapping + phases
        print(f"[{subj}] ND={nd_id} (file: {nd_path.name if nd_path else 'n/a'}), "
              f"MD={md_id} (file: {mdph_path.name if mdph_path else 'n/a'})")
        print(f"       STM file: {stm_path.name}")
        print("       Phases inferred from STM:")
        for lab, a, b in phases:
            print(f"         - {lab}: start={a:.3f}s  end={b:.3f}s  dur={b-a:.3f}s")

        if phase_order_ref is None:
            phase_order_ref = [lab for (lab, _, _) in phases]


        # --- OPTION B: phase vs whole ND baseline ---
        # 1) Compute ND baseline mean per modality (entire ND session)
        nd_baseline = {}
        for mod in MODALITIES:
            vals = df_nd[mod].astype(float)
            nd_baseline[mod] = float(vals.mean()) if len(vals) else np.nan
  

        # 2) Compute MD per-phase means
        md_means = mean_by_phase_using_durations(df_md, [(lab, b - a) for (lab, a, b) in phases])
      

        # --- per-phase means on ND using SAME durations starting at ND_start ---
        nd_means = mean_by_phase_using_durations(
            df_nd, [(lab, b - a) for (lab, a, b) in phases]
        )

        included_rows = []
        for lab,_,_ in phases:
            for mod in MODALITIES:
                MDv = md_means.get(lab, {}).get(mod, np.nan)
                NDv = nd_baseline.get(mod, np.nan)
                included = np.isfinite(MDv) and np.isfinite(NDv)
                included_rows.append({
                    "subject": subj, "phase": lab, "modality": mod,
                    "used_in_delta": bool(included),
                    "MD_mean": MDv, "ND_baseline": NDv
                })
        inc_df = pd.DataFrame(included_rows)
        inc_path = OUT_DIR / "inclusion_log.csv"
        (inc_df if not inc_path.exists() else
        pd.concat([pd.read_csv(inc_path), inc_df], ignore_index=True)).to_csv(inc_path, index=False)


        # Convert both dicts -> wide DFs and collect
        MD_WIDE_ROWS.append(phases_dict_to_wide_df(subj, md_means))
        ND_WIDE_ROWS.append(phases_dict_to_wide_df(subj, nd_means))

        

        # 3) Compute Δ = MD_phase_mean − ND_baseline_mean
        for lab in [p[0] for p in phases]:
            for mod in MODALITIES:
                MDv = md_means.get(lab, {}).get(mod, np.nan)
                NDv = nd_baseline.get(mod, np.nan)
                delta = MDv - NDv if (np.isfinite(MDv) and np.isfinite(NDv)) else np.nan
                all_rows.append({
                    "subject": subj,
                    "phase": lab,
                    "modality": mod,
                    "MD_mean": MDv,
                    "ND_baseline": NDv,
                    "delta": delta
                })



    if not all_rows:
        print("No aggregated data. Check paths/columns/.stm files.")
        return

    df_deltas = pd.DataFrame(all_rows)
    df_deltas.to_csv(OUT_DIR / "paired_deltas.csv", index=False)
    # Keep only subjects with all 3 valid signals
    valid_subjects = valid_counts[valid_counts == 3].index.tolist()
    df_deltas = df_deltas[df_deltas["subject"].isin(valid_subjects)]
        
    if not MD_WIDE_ROWS or not ND_WIDE_ROWS:
        print("No per-phase means collected; check data.")
        return

    MD_means_df = pd.concat(MD_WIDE_ROWS, ignore_index=True)
    ND_means_df = pd.concat(ND_WIDE_ROWS, ignore_index=True)

    means_long = pd.concat([
        melt_phase_means(ND_means_df, "ND"),
        melt_phase_means(MD_means_df, "MD"),
    ], ignore_index=True)

    means_long = means_long.dropna(subset=["value"])


        # ====================== MD vs ND per-phase (PAIRED) tests & plots ======================
    # 1) Build paired table: one row per (subject, modality, phase) with MD and ND columns
    paired_tbl = (
        means_long
        .pivot_table(index=["subject","modality","phase"], columns="type", values="value", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Keep rows where both MD and ND are present
    paired_tbl = paired_tbl.dropna(subset=["MD","ND"])

    # Phase order to iterate
    phases_plot = phase_order_ref or [p for p in PHASES if p in paired_tbl["phase"].unique()]

    # 2) Paired t-tests per phase (MD vs ND), with Bonferroni across planned phases (usually P1..P4)
    def bonf_buckets(phs):
        # Paper typically plans P1..P4; adapt if P5 exists
        planned = [p for p in ["P1","P2","P3","P4"] if p in phs]
        return max(1, len(planned)), planned

    m_tests_path = OUT_DIR / "paired_MD_vs_ND_tests.txt"
    with open(m_tests_path, "w", encoding="utf-8") as f:
        for mod in MODALITIES:
            dfm = paired_tbl[paired_tbl["modality"] == mod].copy()
            if dfm.empty:
                continue
            m_mult, planned = bonf_buckets(phases_plot)
            print(f"=== {mod} ==="); f.write(f"=== {mod} ===\n")
            for ph in phases_plot:
                sub = dfm[dfm["phase"] == ph]
                if len(sub) >= 5:
                    
                    t, p = ttest_rel(sub["MD"].values, sub["ND"].values, nan_policy="omit")
                    p_adj = min(1.0, p * m_mult)  # Bonferroni across planned phases
                    sig = star(p_adj)
                    line = f"{ph}: n={len(sub)}  t={t:.3f}  p={p:.4g}  p_adj={p_adj:.4g}  {sig}"
                else:
                    line = f"{ph}: n={len(sub)}  insufficient data"
                print(line); f.write(line + "\n")
    print(f"[SAVE] -> {m_tests_path}")

    # 3) Visualization: for each modality, one row with subplots (per phase) comparing MD vs ND
    
    for mod in MODALITIES:
        dfm = paired_tbl[paired_tbl["modality"] == mod].copy()
        if dfm.empty:
            continue

        cols = len(phases_plot)
        fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4), sharey=True)
        if cols == 1:
            axes = [axes]

        for i, ph in enumerate(phases_plot):
            ax = axes[i]
            sub = dfm[dfm["phase"] == ph]
            if sub.empty:
                ax.set_title(ph); ax.axis("off"); continue

            # Boxplot MD vs ND
            ax.boxplot([sub["MD"].values, sub["ND"].values],
                    labels=["MD","ND"], patch_artist=True,
                    boxprops=dict(facecolor="#ff00ff", alpha=0.6, color="black"),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"))

            # Paired lines (spaghetti)
            # Slight horizontal jitter so lines don't overlap perfectly
            x_md, x_nd = 1-0.05, 2+0.05
            for _, row in sub.iterrows():
                ax.plot([x_md, x_nd], [row["MD"], row["ND"]], alpha=0.35, linewidth=0.8)

            ax.set_title(f"{ph}  (n={len(sub)})")
            if i == 0:
                ax.set_ylabel(f"{mod} (original units)")

        fig.suptitle(f"{mod}: MD vs ND per phase (paired within subject)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"paired_MD_vs_ND_row_{mod.replace('.','_')}.png", dpi=300)
        plt.close(fig)

    # 4) Focus plot for stressor phases (P2, P4) with paired deltas (MD − ND)
    for mod in MODALITIES:
        dfm = paired_tbl[(paired_tbl["modality"] == mod) & (paired_tbl["phase"].isin(["P2","P4"]))].copy()
        if dfm.empty:
            continue
        dfm["paired_delta"] = dfm["MD"] - dfm["ND"]
        fig, ax = plt.subplots(figsize=(5,4))
        data = [dfm[dfm["phase"]=="P2"]["paired_delta"].dropna().values,
                dfm[dfm["phase"]=="P4"]["paired_delta"].dropna().values]
        ax.boxplot(data, labels=["P2 (MD−ND)","P4 (MD−ND)"],
                patch_artist=True,
                boxprops=dict(facecolor="#ff00ff", alpha=0.6, color="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"))
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_ylabel(f"{mod}: MD − ND (original units)")
        ax.set_title(f"{mod}: Paired deltas at stressor phases")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"paired_MD_minus_ND_stress_{mod.replace('.','_')}.png", dpi=300)
        plt.close(fig)
    # ====================== END MD vs ND per-phase (PAIRED) ======================

    means_long.to_csv(OUT_DIR / "means_long.csv", index=False)
    print("[SAVE] ->", OUT_DIR / "means_long.csv")



    # ---- How many subjects contributed per phase & modality ----
    phase_counts = (
        df_deltas.dropna(subset=["delta"])
                .groupby(["modality", "phase"])["subject"]
                .nunique()
                .rename("n_subjects")
                .reset_index()
    )

    # Console summary (wide, phases x modalities)
    print("\nSubjects contributing per phase (rows=phases, cols=modalities):")
    print(
        phase_counts.pivot(index="phase", columns="modality", values="n_subjects")
                    .fillna(0).astype(int).sort_index()
    )

    # Save to CSV
    phase_counts.to_csv(OUT_DIR / "phase_counts.csv", index=False)

   #     -------- 1) One-sample tests vs 0 and collect stars per phase --------
    pvals = {}  # (mod, phase) -> stars

    with open(OUT_DIR / "ttests.txt", "w", encoding="utf-8") as f:
        print()
        for mod in MODALITIES:
            print(f"=== {mod} ==="); f.write(f"=== {mod} ===\n")
            for lab in (phase_order_ref or sorted(df_deltas["phase"].unique())):
                vals = (df_deltas[(df_deltas["modality"] == mod) &
                                (df_deltas["phase"] == lab)]["delta"]
                        .dropna().values)
                if len(vals) >= 2:
                    t, p = ttest_1samp(vals, 0.0)
                    
                    p_adj = min(1.0, p * 4.0)
                    sig = star(p_adj)
                    pvals[(mod, lab)] = sig
                    line = f"{lab}: n={len(vals)}  t={t:.3f}  p={p:.4g}  p_adj={p_adj:.4g}  {sig}"
                else:
                    pvals[(mod, lab)] = "ns"
                    line = f"{lab}: n={len(vals)}  insufficient data"
                print(line); f.write(line + "\n")

    # -------- 2) Paired tests: P2>P1 and P4>P3, per modality, paired by subject --------
    def paired_phase_test(df, mod, a, b):
        A = df[(df["modality"] == mod) & (df["phase"] == a)][["subject","delta"]].rename(columns={"delta":f"{a}"})
        B = df[(df["modality"] == mod) & (df["phase"] == b)][["subject","delta"]].rename(columns={"delta":f"{b}"})
        M = pd.merge(A, B, on="subject").dropna()
        if len(M) >= 5:
            t, p = ttest_rel(M[a].values, M[b].values)
            return len(M), t, p, star(p)
        return 0, np.nan, np.nan, "ns"

    pairs = [("P2","P1"), ("P4","P3")]
    with open(OUT_DIR / "paired_phase_tests.txt", "w", encoding="utf-8") as f:
        for mod in MODALITIES:
            for a, b in pairs:
                n, t, p, sig = paired_phase_test(df_deltas, mod, a, b)
                msg = f"{mod}: {a} vs {b}: n={n}  t={t:.3f}  p={p:.4g}  {sig}" if n else f"{mod}: {a} vs {b}: n<5"
                print(msg); f.write(msg + "\n")



    paired_plot_data = []
    for mod, (a, b) in [(m, p) for m in MODALITIES for p in [("P2","P1"),("P4","P3")]]:
        A = df_deltas[(df_deltas["modality"]==mod) & (df_deltas["phase"]==a)][["subject","delta"]]
        B = df_deltas[(df_deltas["modality"]==mod) & (df_deltas["phase"]==b)][["subject","delta"]]
        M = pd.merge(A,B,on="subject").dropna()
        M["paired_delta"] = M["delta_x"] - M["delta_y"]
        M["pair"] = f"{a}-{b}"
        M["modality"] = mod
        paired_plot_data.append(M[["subject","modality","pair","paired_delta"]])
    paired_plot_df = pd.concat(paired_plot_data, ignore_index=True)

    # Boxplot of paired deltas
    for mod in MODALITIES:
        fig, ax = plt.subplots(figsize=(4,4))
        subset = paired_plot_df[paired_plot_df["modality"]==mod]
        bp = ax.boxplot([subset.loc[subset["pair"]==pair,"paired_delta"].dropna()
                        for pair in ["P2-P1","P4-P3"]],
                        patch_artist=True,
                        labels=["P2−P1","P4−P3"],
                        boxprops=dict(facecolor="#ff00ff",color="black"),
                        medianprops=dict(color="black"),
                        whiskerprops=dict(color="black"))
        ax.axhline(0,color="k",ls="--")
        ax.set_title(f"{mod}: Paired Δ between Baseline–Stressor")
        ax.set_ylabel("Δ (MD z-units)")
        plt.tight_layout()
        plt.savefig(OUT_DIR/f"paired_boxplot_{mod}.png",dpi=300)
        plt.close()


        # # T-tests per phase & modality
    # with open(OUT_DIR / "ttests.txt", "w", encoding="utf-8") as f:
    #     print()
    #     pvals = {} 
    #     for mod in MODALITIES:
    #         print(f"=== {mod} ===")
    #         f.write(f"=== {mod} ===\n")
    #         for lab in (phase_order_ref or sorted(df_deltas["phase"].unique())):
    #             vals = df_deltas[(df_deltas["modality"] == mod) &
    #                              (df_deltas["phase"] == lab)]["delta"].dropna().values
    #             if len(vals) >= 2:
    #                 t, p = ttest_1samp(vals, 0.0)
    #                 sig = star(p)
                    
    #                 pvals[(mod, lab)] = sig
    #                 line = f"{lab}: n={len(vals)}  t={t:.3f}  p={p:.4g}  {sig}"
    #             else:
    #                 line = f"{lab}: n={len(vals)}  insufficient data"

    #             print(line)
    #             f.write(line + "\n")
    # ====================== stylized Pavlidis-style boxplots ======================
    phases_plot = phase_order_ref or sorted(df_deltas["phase"].unique())
    for mod in MODALITIES:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df_deltas[(df_deltas["modality"] == mod) &
                        (df_deltas["phase"] == pl)]["delta"].dropna().values
                for pl in phases_plot]



    


        #     # Look up n per phase for this modality
        # Look up n per phase for this modality (no .query to avoid scope issues)
        n_by_phase = {}
        for pl in phases_plot:
            mask = (phase_counts["modality"] == mod) & (phase_counts["phase"] == pl)
            n_by_phase[pl] = int(phase_counts.loc[mask, "n_subjects"].sum() or 0)

        labels = [f"{pl}\n(n={n_by_phase.get(pl, 0)})" for pl in phases_plot]
        for i, pl in enumerate(phases_plot, start=1):
            sig = pvals.get((mod, pl), "")
            
            if sig:
                y_top = ax.get_ylim()[1]
                ax.text(i, ax.get_ylim()[1]*1.5, sig, ha="center", va="bottom", fontsize=12, fontweight="bold")
                


       


        # n_by_phase = {
        #     pl: int(phase_counts.query("modality == @mod and phase == @pl")["n_subjects"].sum())
        #     for pl in phases_plot
        # }
        # labels = [f"{pl}\n(n={n_by_phase.get(pl,0)})" for pl in phases_plot]
        # Boxplot with custom colors (magenta fill, black edges)
        bp = ax.boxplot(data, patch_artist=True, labels=labels,
                        boxprops=dict(facecolor="#ff00ff", color="black", linewidth=1),
                        medianprops=dict(color="black", linewidth=1),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black"),
                        flierprops=dict(marker='o', markersize=4, markerfacecolor='white', markeredgecolor='black'))

        # # Overlay mean (black horizontal line)
        for i, vals in enumerate(data, start=1):
            if len(vals):
                mean_y = np.mean(vals)
                ax.plot([i-0.2, i+0.2], [mean_y, mean_y], color="black", linewidth=1)

              

        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"Δ = MD − ND baseline — {mod}")
        ax.set_ylabel("Δ (MD − ND)")
        ax.set_xlabel("Phase")
        #ax.set_ylim(-10, 15)
        ax.set_ylim(-1.5, 2)
        # if "Heart.Rate" in mod:
        #     ax.set_ylim(-2, 2)
        # elif "Perinasal" in mod:
        #     ax.set_ylim(-2.5, 2.5)
        # elif "Breathing" in mod:
        #     ax.set_ylim(-2, 2)


        plt.tight_layout()
        plt.savefig(OUT_DIR / f"pavlidis_style_boxplot_{mod.replace('.', '_')}.png", dpi=300)
        plt.close(fig)



if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
    make_all_overlay_plots()

