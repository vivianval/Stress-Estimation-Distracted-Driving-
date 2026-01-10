

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build windowed MD/ND paired dataset with:
  - EMOCA (exp_, pose_, delta1_)
  - BIOSIGNALS (Heart.Rate, Perinasal.Perspiration, Breathing.Rate)
  - GAZE features: all columns starting with "Gaze" EXCEPT raw positions:
        exclude {"Gaze.X.Pos", "Gaze.Y.Pos"} (and some common variants)
  - Conditioning vector (MD–ND velocity stats):
        cond_flame: 6 dims  (exp mean/std | pose mean/std | delta1 mean/std)
        cond_bio  : 6 dims  (HR  mean/std | EDA  mean/std | BR    mean/std)
    total cond_dim = 12

NPZ contains EVERYTHING (facial features  + bio + gaze + cond). In training you choose slices via groups.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def find_col(df, names):
    names_l = [n.lower() for n in names]
    for c in df.columns:
        if c.lower() in names_l:
            return c
    return None


def detect_columns(df):
    subj_col = find_col(df, ["subject", "subject_id", "Subject"])
    t_col    = find_col(df, ["t_sec", "time_sec", "timestamp_s", "time"])
    if "GTlabel" not in df.columns:
        raise ValueError("GTlabel column not found in MD CSV.")
    if subj_col is None or t_col is None:
        raise ValueError(f"Could not detect subject/time columns. subj={subj_col}, t={t_col}")
    return subj_col, t_col, "GTlabel"


def pick_emoca_cols(df):
    cols = [c for c in df.columns if c.startswith("exp_")]
    cols += [c for c in df.columns if c.startswith("pose_")]
    cols += [c for c in df.columns if c.startswith("delta1_")]
    if not cols:
        raise ValueError("No EMOCA columns found (exp_*, pose_*, delta1_*).")
    return cols


def pick_bio_cols(df):
    required = ["Heart.Rate", "Perinasal.Perspiration", "Breathing.Rate"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing biosignal column: {c}")
    return required


def pick_gaze_cols(df):
    """
    Select all columns that start with 'Gaze' (case-insensitive),
    EXCLUDING raw positions (and a few common name variants).
    """
    exclude = {
        "Gaze.X.Pos", "Gaze.Y.Pos",
        "gaze.x.pos", "gaze.y.pos",
        "GazeXPos", "GazeYPos",
        "Gaze.X", "Gaze.Y",
        "gaze.x", "gaze.y",
    }

    gaze_cols = []
    for c in df.columns:
        if c.lower().startswith("gaze"):
            if c in exclude or c.lower() in exclude:
                continue
            gaze_cols.append(c)

    if not gaze_cols:
        # *expect* gaze dynamics but none found, fail loudly.
        # optional, change this to return [].
        raise ValueError(
            "No gaze columns found with prefix 'Gaze*' after excluding raw positions. "
            "Check  CSV headers."
        )
    return gaze_cols


def window_has_enough_valid(df_win, cols, min_valid_frac=0.80):
    """
    Returns True if each column has >= min_valid_frac finite values inside the window.
    """
    for c in cols:
        v = pd.to_numeric(df_win[c], errors="coerce").to_numpy()
        frac = np.isfinite(v).mean() if len(v) else 0.0
        if frac < min_valid_frac:
            return False
    return True


def filter_subjects_with_complete_biosignals(df_md, df_nd, subj_col, bio_cols):
    good_md = df_md.groupby(subj_col)[bio_cols].apply(lambda g: g.notna().all().all())
    good_nd = df_nd.groupby(subj_col)[bio_cols].apply(lambda g: g.notna().all().all())
    keep = (good_md & good_nd)
    keep = keep[keep].index.astype(str).tolist()
    print(f"[BIO FILTER] kept {len(keep)} subjects with complete biosignals in MD+ND")
    return (
        df_md[df_md[subj_col].astype(str).isin(keep)].copy(),
        df_nd[df_nd[subj_col].astype(str).isin(keep)].copy(),
    )


def clean_subject_time(t_raw, target_hz=30.0):
    t = pd.to_numeric(pd.Series(t_raw), errors="coerce")
    t_interp = t.interpolate(method="linear", limit_direction="both")
    vals = t_interp.to_numpy(dtype=float)

    if not np.isfinite(vals).any():
        step = 1.0 / float(target_hz)
        return np.arange(len(vals), dtype=float) * step

    finite = np.isfinite(vals)
    diffs = np.diff(vals[finite])
    if diffs.size == 0 or not np.isfinite(diffs).any():
        dt = 1.0 / float(target_hz)
    else:
        dt = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 1.0 / float(target_hz)

    for i in range(len(vals)):
        if not np.isfinite(vals[i]):
            vals[i] = (vals[i - 1] + dt) if i > 0 and np.isfinite(vals[i - 1]) else (0.0 if i == 0 else vals[i - 1] + dt)

    for i in range(1, len(vals)):
        if not (vals[i] > vals[i - 1]):
            vals[i] = vals[i - 1] + dt

    return vals


def parse_phase(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.lower().startswith("p"):
        s = s[1:]
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def time_window_starts(t, win_sec, stride_sec):
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return []
    t0, t1 = float(t[0]), float(t[-1])
    last_start = t1 - win_sec
    if last_start < t0:
        last_start = t0
    starts = []
    s = t0
    while s <= last_start + 1e-9:
        starts.append(s)
        s += stride_sec
    return starts if starts else [t0]


def linear_resample(times_s, values, grid_s):
    t = np.asarray(pd.to_numeric(times_s, errors="coerce"), dtype=float)
    v = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    if mask.sum() < 2:
        return np.zeros_like(grid_s, dtype=np.float32)
    t = t[mask]
    v = v[mask]
    if np.any(np.diff(t) < 0):
        idx = np.argsort(t)
        t = t[idx]
        v = v[idx]
    y = np.interp(grid_s, t, v, left=v[0], right=v[-1])
    return y.astype(np.float32)


# -------------------------
# Velocity + conditioning
# -------------------------
def window_velocity(X, hz):
    V = np.zeros_like(X, dtype=np.float32)
    V[1:] = (X[1:] - X[:-1]) * float(hz)
    return V


def conditioning_stats_from_velocity_emoca(V, emoca_cols):
    idx_exp  = [i for i,c in enumerate(emoca_cols) if c.startswith("exp_")]
    idx_pose = [i for i,c in enumerate(emoca_cols) if c.startswith("pose_")]
    idx_d1   = [i for i,c in enumerate(emoca_cols) if c.startswith("delta1_")]

    def stats(idxs):
        if len(idxs) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        A = np.abs(V[:, idxs])
        return np.array([float(A.mean()), float(A.std())], dtype=np.float32)

    return np.concatenate([stats(idx_exp), stats(idx_pose), stats(idx_d1)], axis=0)  # [6]


def conditioning_stats_from_velocity_bio(V, bio_cols):
    name2idx = {c:i for i,c in enumerate(bio_cols)}

    def stats_one(colname):
        i = name2idx[colname]
        A = np.abs(V[:, i:i+1])
        return np.array([float(A.mean()), float(A.std())], dtype=np.float32)

    return np.concatenate([
        stats_one("Heart.Rate"),
        stats_one("Perinasal.Perspiration"),
        stats_one("Breathing.Rate"),
    ], axis=0)  # [6]


# -------------------------
# Window builder
# -------------------------
def build_windows(df_md, df_nd, emoca_cols, bio_cols, gaze_cols,
                  subj_col, t_col, label_col,
                  win_sec, stride_sec, hz,
                  label_pos_thr=0.8,
                  require_phase=True,
                  gaze_min_valid_frac=0.80):

    if "Phase" not in df_md.columns or "Phase" not in df_nd.columns:
        raise ValueError("Both MD and ND CSV must contain a 'Phase' column.")

    T = int(round(win_sec * hz))

    # ND lookup by (Subject, Phase)
    nd_map = {}
    for (sid, ph), g in df_nd.groupby([subj_col, "Phase"]):
        ph_i = parse_phase(ph)
        if ph_i is None:
            continue
        nd_map[(str(sid), int(ph_i))] = g

    X_list, y_list, subj_list = [], [], []
    manifest_rows = []

    rep = defaultdict(lambda: dict(
        md_windows_total=0,
        paired_kept=0,
        dropped_no_nd=0,
        dropped_out_of_range=0,
        dropped_empty_md=0,
        dropped_gaze_missing=0,
    ))
    global_rep = defaultdict(int)

    for (sid, ph), gmd in df_md.groupby([subj_col, "Phase"]):
        sid = str(sid)
        ph_i = parse_phase(ph)
        if ph_i is None:
            continue
        ph = int(ph_i)

        gnd = nd_map.get((sid, ph), None)
        if gnd is None:
            rep[(sid, ph)]["dropped_no_nd"] += 1
            global_rep["dropped_no_nd"] += 1
            if require_phase:
                continue
            continue

        t_md = clean_subject_time(gmd[t_col].values, target_hz=hz)
        t_nd = clean_subject_time(gnd[t_col].values, target_hz=hz)

        nd_min, nd_max = float(t_nd[0]), float(t_nd[-1])
        md0, nd0 = float(t_md[0]), float(t_nd[0])

        labels = pd.to_numeric(gmd[label_col], errors="coerce").fillna(0).to_numpy()
        starts = time_window_starts(t_md, win_sec, stride_sec)

        for s in starts:
            rep[(sid, ph)]["md_windows_total"] += 1
            global_rep["md_windows_total"] += 1

            e = s + win_sec
            m_md = (t_md >= s) & (t_md < e)
            if not m_md.any():
                rep[(sid, ph)]["dropped_empty_md"] += 1
                global_rep["dropped_empty_md"] += 1
                continue

            # enforce gaze presence only in MD (ND gaze irrelevant)
            gmd_win = gmd.iloc[np.where(m_md)[0]]
            if not window_has_enough_valid(gmd_win, gaze_cols, min_valid_frac=gaze_min_valid_frac):
                rep[(sid, ph)]["dropped_gaze_missing"] += 1
                global_rep["dropped_gaze_missing"] += 1
                continue

            stress_ratio = float((labels[m_md] == 1).mean())
            y = 1 if stress_ratio >= label_pos_thr else 0

            grid_md = np.linspace(s, e, T, endpoint=False, dtype=np.float64)

            # ND aligned by relative time within phase
            rel_s = float(s - md0)
            s_nd  = nd0 + rel_s
            e_nd  = s_nd + win_sec
            grid_nd = np.linspace(s_nd, e_nd, T, endpoint=False, dtype=np.float64)

            if (s_nd < nd_min) or (e_nd > nd_max):
                rep[(sid, ph)]["dropped_out_of_range"] += 1
                global_rep["dropped_out_of_range"] += 1
                continue

            # Resample EMOCA (MD / ND)
            Xmd = np.stack([linear_resample(t_md, gmd[c].values, grid_md) for c in emoca_cols], axis=-1)
            Xnd = np.stack([linear_resample(t_nd, gnd[c].values, grid_nd) for c in emoca_cols], axis=-1)

            # Resample BIOSIGNALS (MD / ND)
            Bmd = np.stack([linear_resample(t_md, gmd[c].values, grid_md) for c in bio_cols], axis=-1)
            Bnd = np.stack([linear_resample(t_nd, gnd[c].values, grid_nd) for c in bio_cols], axis=-1)

            # Resample GAZE (MD only)
            Gmd = np.stack([linear_resample(t_md, gmd[c].values, grid_md) for c in gaze_cols], axis=-1)

            # Velocities
            Vmd_f = window_velocity(Xmd, hz)
            Vnd_f = window_velocity(Xnd, hz)
            Vmd_b = window_velocity(Bmd, hz)
            Vnd_b = window_velocity(Bnd, hz)

            # Conditioning (MD–ND)
            cond_flame = (
                conditioning_stats_from_velocity_emoca(Vmd_f, emoca_cols)
                - conditioning_stats_from_velocity_emoca(Vnd_f, emoca_cols)
            )
            cond_bio = (
                conditioning_stats_from_velocity_bio(Vmd_b, bio_cols)
                - conditioning_stats_from_velocity_bio(Vnd_b, bio_cols)
            )

            cond = np.concatenate([cond_flame, cond_bio], axis=0).astype(np.float32)  # [12]
            cond_T = np.repeat(cond[None, :], T, axis=0)

            # FINAL WINDOW: emoca + bio + gaze + cond (cond last!)
            X_aug = np.concatenate([
                Xmd.astype(np.float32),      # [T, F_emoca]
                Bmd.astype(np.float32),      # [T, F_bio]
                Gmd.astype(np.float32),      # [T, F_gaze]
                cond_T.astype(np.float32),   # [T, 12]
            ], axis=-1)

            X_list.append(X_aug)
            y_list.append(y)
            subj_list.append(sid)

            rep[(sid, ph)]["paired_kept"] += 1
            global_rep["paired_kept"] += 1

            manifest_rows.append(dict(
                Subject=sid,
                Phase=ph,
                start_sec=float(s),
                end_sec=float(e),
                nd_start_sec=float(s_nd),
                nd_end_sec=float(e_nd),
                stress_ratio=stress_ratio,
                label=int(y),
            ))

    if not X_list:
        raise RuntimeError(f"No windows created. global_rep={dict(global_rep)}")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subj_list, dtype=object)
    manifest = pd.DataFrame(manifest_rows)

    rep_rows = []
    for (sid, ph), d in rep.items():
        md_total = d["md_windows_total"]
        kept = d["paired_kept"]
        rep_rows.append({
            "Subject": sid,
            "Phase": ph,
            **d,
            "dropped_total": (md_total - kept),
            "kept_frac": (kept / md_total) if md_total > 0 else np.nan,
        })
    report_df = pd.DataFrame(rep_rows).sort_values(["Subject", "Phase"])

    global_df = pd.DataFrame([{
        "md_windows_total": int(global_rep["md_windows_total"]),
        "paired_kept": int(global_rep["paired_kept"]),
        "dropped_empty_md": int(global_rep["dropped_empty_md"]),
        "dropped_out_of_range": int(global_rep["dropped_out_of_range"]),
        "dropped_no_nd": int(global_rep["dropped_no_nd"]),
        "dropped_gaze_missing": int(global_rep["dropped_gaze_missing"]),
    }])

    return X, y, subjects, manifest, report_df, global_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_md", type=Path, required=True)
    ap.add_argument("--csv_nd", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--win_sec", type=float, default=9.0)
    ap.add_argument("--stride_sec", type=float, default=9.0)
    ap.add_argument("--target_hz", type=float, default=30.0)
    ap.add_argument("--label_pos_thr", type=float, default=0.8)

    ap.add_argument("--no_bio_filter", action="store_true",
                    help="Do not filter subjects by complete biosignals in MD+ND.")
    ap.add_argument("--gaze_min_valid_frac", type=float, default=0.80,
                    help="Min fraction of finite values per gaze feature inside each MD window.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df_md = pd.read_csv(args.csv_md, low_memory=False)
    df_nd = pd.read_csv(args.csv_nd, low_memory=False)

    subj_col, t_col, label_col = detect_columns(df_md)

    # normalize keys
    df_md[subj_col] = df_md[subj_col].astype(str).str.strip()
    df_nd[subj_col] = df_nd[subj_col].astype(str).str.strip()
    df_md["Phase"] = df_md["Phase"].astype(str).str.strip()
    df_nd["Phase"] = df_nd["Phase"].astype(str).str.strip()

    # columns
    emoca_cols = pick_emoca_cols(df_md)
    bio_cols   = pick_bio_cols(df_md)
    gaze_cols  = pick_gaze_cols(df_md)

    if not args.no_bio_filter:
        df_md, df_nd = filter_subjects_with_complete_biosignals(df_md, df_nd, subj_col, bio_cols)

    # Print summary
    print(
        f"[COLS] EMOCA={len(emoca_cols)} | BIO={len(bio_cols)} | "
        f"GAZE={len(gaze_cols)} (excl X/Y pos) | COND=12 | "
        f"TOTAL F={len(emoca_cols)+len(bio_cols)+len(gaze_cols)+12}"
    )

    X, y, subjects, manifest, report_df, global_df = build_windows(
        df_md, df_nd,
        emoca_cols=emoca_cols,
        bio_cols=bio_cols,
        gaze_cols=gaze_cols,
        subj_col=subj_col,
        t_col=t_col,
        label_col=label_col,
        win_sec=float(args.win_sec),
        stride_sec=float(args.stride_sec),
        hz=float(args.target_hz),
        label_pos_thr=float(args.label_pos_thr),
        require_phase=True,
        gaze_min_valid_frac=float(args.gaze_min_valid_frac),
    )

    print("[X]", X.shape, "y", y.shape, "unique subjects", len(np.unique(subjects)))

    # Verify conditioning is constant over time (cond is last 12 dims by construction)
    cond_dim = 12
    assert np.allclose(X[:, 1:, -cond_dim:], X[:, :-1, -cond_dim:]), "COND dims not constant over time!"

    # Feature names (must match X last dim)
    cond_flame_names = [
        "cond_flame_exp_meanabs","cond_flame_exp_stdabs",
        "cond_flame_pose_meanabs","cond_flame_pose_stdabs",
        "cond_flame_d1_meanabs","cond_flame_d1_stdabs"
    ]
    cond_bio_names = [
        "cond_bio_hr_meanabs","cond_bio_hr_stdabs",
        "cond_bio_eda_meanabs","cond_bio_eda_stdabs",
        "cond_bio_br_meanabs","cond_bio_br_stdabs"
    ]

    feature_cols = list(emoca_cols) + list(bio_cols) + list(gaze_cols) + cond_flame_names + cond_bio_names
    assert len(feature_cols) == X.shape[-1], (len(feature_cols), X.shape[-1])

    # group index ranges for training-time selection
    i0 = 0
    i1 = i0 + len(emoca_cols)
    i2 = i1 + len(bio_cols)
    i3 = i2 + len(gaze_cols)
    i4 = i3 + 12

    groups = {
        "emoca": [i0, i1],
        "bio": [i1, i2],
        "gaze": [i2, i3],
        "cond": [i3, i4],
        "emoca_plus_bio": [i0, i2],
        "emoca_plus_gaze": [i0, i1] + [i2, i3],  # NOTE: non-contiguous; handle manually in trainer if needed
        "all_nocond": [i0, i3],
        "all": [i0, i4],
    }

    # Save NPZ
    npz_path = args.outdir / "windows_mdnd_emoca+bio+gaze+cond.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        subjects=subjects,
        feature_cols=np.array(feature_cols, dtype=object),
        groups=np.array([groups], dtype=object),
        cond_dim=np.array([12], dtype=np.int64),
        win_sec=np.array([args.win_sec], dtype=np.float32),
        stride_sec=np.array([args.stride_sec], dtype=np.float32),
        hz=np.array([args.target_hz], dtype=np.float32),
    )
    print("[SAVED]", npz_path)

    # Save CSVs
    manifest_path = args.outdir / "windows_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("[SAVED]", manifest_path)

    report_path = args.outdir / "pairing_report_by_subject_phase.csv"
    report_df.to_csv(report_path, index=False)
    print("[SAVED]", report_path)

    global_path = args.outdir / "pairing_report_global.csv"
    global_df.to_csv(global_path, index=False)
    print("[SAVED]", global_path)

    phase_agg = report_df.groupby("Phase")[["md_windows_total","paired_kept","dropped_out_of_range","dropped_empty_md","dropped_gaze_missing"]].sum().reset_index()
    phase_agg["kept_frac"] = phase_agg["paired_kept"] / phase_agg["md_windows_total"].replace(0, np.nan)
    phase_agg_path = args.outdir / "pairing_report_by_phase.csv"
    phase_agg.to_csv(phase_agg_path, index=False)
    print("[SAVED]", phase_agg_path)

    print("\n[PAIRING SUMMARY]")
    print(global_df.to_string(index=False))


if __name__ == "__main__":
    main()
