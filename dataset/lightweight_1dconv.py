


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase-aligned MD vs ND analysis using Pavlidis-phased EMOCA CSVs,
with convolution refinements (kernel sweep + better derivatives).

What it does:
- Loads *phased* MD/ND CSVs per subject (Phase ∈ {P1..P5})
- For each (subject, drive, phase) segment:
    * smooths each feature trajectory
    * computes mean level  μ = mean_t Y(t)
    * computes mean velocity ν = mean_t |dY/dt|
- Forms paired deltas per subject:
    Δμ(s,p,f) = μ_MD - μ_ND
    Δν(s,p,f) = ν_MD - ν_ND
- Across subjects:
    one-sample t-test of deltas vs 0 for each (phase, feature, metric)
- Writes one CSV per setting (kernel k etc.)
- Prints a "pattern score" to help you pick the best setting:
    both24      : #features significant in BOTH P2 and P4
    only24      : #features significant in P2 & P4 AND NOT significant in P1,P3,P5
    both24_plus1: #features significant in P2 & P4 plus exactly one of P1/P3/P5
    leak_any_nonstress: total #significant features in non-stress phases (P1/P3/P5)

Recommended:
- Start with mode="conv", deriv_mode="savgol", pad_mode="reflect"
- Sweep conv_kernel over [3,5,7,9,11,15,21]
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from scipy.signal import convolve, savgol_filter

# Optional csaps smoothing (preferred if you use mode="spline")
try:
    from csaps import csaps
    _HAS_CSAPS = True
except Exception:
    _HAS_CSAPS = False

from scipy.interpolate import UnivariateSpline


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    phased_dir: str = "dataset/paired_tests_EMOCA/phased_csvs"
    subjects: Optional[List[str]] = None

    include_delta_pose: bool = False
    phase_col: str = "Phase"       # "P1"..."P5"
    time_hz: float = 10.0          # sampling rate (dt = 1/time_hz)
    min_len: int = 40              # min samples per phase segment

    # Smoothing mode
    mode: str = "conv"             # "conv" or "spline"

    # Convolution smoothing
    conv_kernel: int = 5           # triangular kernel base length (odd recommended)
    pad_mode: str = "reflect"      # "reflect" or "edge" or "none"

    # Derivative method (conv mode)
    deriv_mode: str = "savgol"     # "finite" or "savgol"
    sg_window: int = 11            # odd window length (<= segment length)
    sg_poly: int = 3               # 2 or 3 usually

    # Spline smoothing (if mode="spline")
    spline_p: float = 0.8          # csaps smooth in [0,1]

    # Output
    out_csv: str = "phasewise_md_nd_phased_smoothed_stats.csv"


# -------------------------
# Feature + file helpers
# -------------------------
def list_subjects(phased_dir: Path) -> List[str]:
    subs = set()
    for p in phased_dir.glob("T???_*_PHASED.csv"):
        subs.add(p.name[:4])
    return sorted(subs)

def find_phased_pair(phased_dir: Path, sid: str) -> Tuple[Optional[Path], Optional[Path]]:
    md_cands = sorted(phased_dir.glob(f"{sid}_MD*_*PHASED.csv"))
    nd_cands = sorted(phased_dir.glob(f"{sid}*ND*_*PHASED.csv")) + sorted(phased_dir.glob(f"{sid}_exp_pose*PHASED.csv"))
    md = md_cands[0] if md_cands else None
    nd = nd_cands[0] if nd_cands else None
    return md, nd

def select_feature_cols(df: pd.DataFrame, include_delta_pose: bool) -> List[str]:
    cols = []
    for c in df.columns:
        cs = str(c)
        if cs.startswith("exp_") or cs.startswith("pose_"):
            cols.append(cs)
        elif include_delta_pose and cs.startswith("delta_pose_"):
            cols.append(cs)
    pose = sorted([c for c in cols if c.startswith("pose_")])
    exp  = sorted([c for c in cols if c.startswith("exp_")])
    dpo  = sorted([c for c in cols if c.startswith("delta_pose_")])
    return pose + exp + dpo

def phase_values() -> List[str]:
    return ["P1", "P2", "P3", "P4", "P5"]


# -------------------------
# Smoothing helpers
# -------------------------
def smooth_series_spline(t: np.ndarray, x: np.ndarray, p: float, deriv: int) -> np.ndarray:
    mask = np.isfinite(x)
    if mask.sum() < 4:
        return x.copy()

    if _HAS_CSAPS:
        return csaps(t[mask], x[mask], t, smooth=p, nu=deriv)

    # fallback: UnivariateSpline
    xx = x[mask]
    var = float(np.nanvar(xx)) if np.isfinite(xx).any() else 1.0
    s = max(1e-8, (1.0 - p) * len(xx) * var)
    spl = UnivariateSpline(t[mask], xx, k=3, s=s)
    return spl.derivative(deriv)(t) if deriv else spl(t)

def triangular_smooth_1d(x: np.ndarray, k: int, pad_mode: str = "reflect") -> np.ndarray:
    """
    Symmetric triangular smoothing kernel: box(k) * box(k), normalized.
    Boundary handling is important for phased segments; default is reflect padding.
    """
    if k <= 1:
        return x.copy()

    box = np.ones(k, dtype=float) / k
    tri = convolve(box, box, mode="full")
    tri = tri / tri.sum()

    if pad_mode and pad_mode != "none":
        pad = len(tri) // 2
        xpad = np.pad(x, (pad, pad), mode=pad_mode)
        ypad = np.convolve(xpad, tri, mode="same")
        return ypad[pad:-pad]
    else:
        return np.convolve(x, tri, mode="same")

def smooth_matrix(X: np.ndarray, cfg: Config, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Y  : smoothed signal [T,F]
      dY : time derivative [T,F] (per second)
    """
    T, F = X.shape
    if T < 2:
        return X.copy(), np.zeros_like(X)

    if cfg.mode == "conv":
        Y = np.empty_like(X, dtype=float)
        for f in range(F):
            Y[:, f] = triangular_smooth_1d(X[:, f].astype(float), cfg.conv_kernel, pad_mode=cfg.pad_mode)

        if cfg.deriv_mode == "savgol":
            # SG directly estimates derivative of Y (smoothed) robustly.
            w = int(cfg.sg_window)
            if w % 2 == 0:
                w += 1
            # ensure <= T and odd
            if w > T:
                w = T if (T % 2 == 1) else (T - 1)
            if w < 5 or w <= cfg.sg_poly:
                # fallback if segment too short
                dY = np.gradient(Y, dt, axis=0)
            else:
                dY = savgol_filter(
                    Y, window_length=w, polyorder=int(cfg.sg_poly),
                    deriv=1, delta=dt, axis=0, mode="interp"
                )
        else:
            # finite differences
            dY = np.zeros_like(Y)
            dY[1:-1] = (Y[2:] - Y[:-2]) / (2.0 * dt)
            dY[0]    = (Y[1] - Y[0]) / dt
            dY[-1]   = (Y[-1] - Y[-2]) / dt

        return Y, dY

    if cfg.mode == "spline":
        t = np.arange(T, dtype=float)
        Y  = np.empty_like(X, dtype=float)
        dY = np.empty_like(X, dtype=float)
        for f in range(F):
            col = X[:, f].astype(float)
            Y[:, f]  = smooth_series_spline(t, col, p=cfg.spline_p, deriv=0)
            dY[:, f] = smooth_series_spline(t, col, p=cfg.spline_p, deriv=1) / dt
        return Y, dY

    raise ValueError("cfg.mode must be 'conv' or 'spline'")


# -------------------------
# Per-subject per-phase summaries
# -------------------------
def summarize_phase(df: pd.DataFrame, feature_cols: List[str], ph: str, cfg: Config) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    d = df[df[cfg.phase_col] == ph]
    if len(d) < cfg.min_len:
        return None
    X = d[feature_cols].to_numpy(dtype=float, copy=True)
    dt = 1.0 / cfg.time_hz
    Y, dY = smooth_matrix(X, cfg, dt)

    mu = np.nanmean(Y, axis=0)
    # ν = mean_t |dY/dt|  (matches your paragraph)
    nu = np.nanmean(np.abs(dY), axis=0)
    return mu, nu

def build_subject_summaries(phased_dir: Path, sid: str, cfg: Config) -> Optional[Dict]:
    md_path, nd_path = find_phased_pair(phased_dir, sid)
    if md_path is None or nd_path is None:
        return None

    df_md = pd.read_csv(md_path, low_memory=False)
    df_nd = pd.read_csv(nd_path, low_memory=False)

    if cfg.phase_col not in df_md.columns or cfg.phase_col not in df_nd.columns:
        raise ValueError(f"{sid}: missing {cfg.phase_col} in phased CSVs.")

    feature_cols = select_feature_cols(df_md, cfg.include_delta_pose)
    feature_cols = [c for c in feature_cols if c in df_nd.columns]
    if not feature_cols:
        raise ValueError(f"{sid}: no overlapping feature columns found.")

    out = {"subject": sid, "feature_cols": feature_cols, "MD": {}, "ND": {}}
    for ph in phase_values():
        sm_md = summarize_phase(df_md, feature_cols, ph, cfg)
        sm_nd = summarize_phase(df_nd, feature_cols, ph, cfg)
        if sm_md is None or sm_nd is None:
            continue
        out["MD"][ph] = {"mu": sm_md[0], "nu": sm_md[1]}
        out["ND"][ph] = {"mu": sm_nd[0], "nu": sm_nd[1]}
    return out


# -------------------------
# Stats + scoring
# -------------------------
def cohend_paired(delta: np.ndarray) -> float:
    m = np.nanmean(delta)
    s = np.nanstd(delta, ddof=1)
    return float(m / (s + 1e-12))

def run_tests(all_subj: List[Dict], feature_cols: List[str], alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    phases = phase_values()
    F = len(feature_cols)

    for ph in phases:
        for j in range(F):
            feat = feature_cols[j]

            # Δμ
            deltas = []
            for s in all_subj:
                if ph not in s["MD"] or ph not in s["ND"]:
                    continue
                deltas.append(s["MD"][ph]["mu"][j] - s["ND"][ph]["mu"][j])
            deltas = np.asarray(deltas, dtype=float)
            deltas = deltas[np.isfinite(deltas)]
            if len(deltas) >= 5:
                t, p = ttest_1samp(deltas, 0.0, nan_policy="omit")
                rows.append({
                    "phase": ph, "feature": feat, "metric": "mean_level",
                    "n": int(len(deltas)), "t": float(t), "p": float(p),
                    "d_z": cohend_paired(deltas)
                })

            # Δν
            deltas = []
            for s in all_subj:
                if ph not in s["MD"] or ph not in s["ND"]:
                    continue
                deltas.append(s["MD"][ph]["nu"][j] - s["ND"][ph]["nu"][j])
            deltas = np.asarray(deltas, dtype=float)
            deltas = deltas[np.isfinite(deltas)]
            if len(deltas) >= 5:
                t, p = ttest_1samp(deltas, 0.0, nan_policy="omit")
                rows.append({
                    "phase": ph, "feature": feat, "metric": "velocity",
                    "n": int(len(deltas)), "t": float(t), "p": float(p),
                    "d_z": cohend_paired(deltas)
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid tests produced (too few subjects/phase samples).")

    # Bonferroni over all produced tests
    m = len(df)
    df["p_bonf"] = np.minimum(1.0, df["p"] * m)
    df["sig_0p05"] = df["p"] < 0.05
    df["sig_bonf"] = df["p_bonf"] < alpha
    return df

def pattern_score(df: pd.DataFrame, pcol: str = "p", alpha: float = 0.05) -> dict:
    """
    Score only the velocity metric (since that's where your gains are).
    """
    d = df[df["metric"] == "velocity"].copy()
    if d.empty:
        return {"both24": 0, "only24": 0, "both24_plus1": 0, "leak_any_nonstress": 0}

    pmat = d.pivot_table(index="phase", columns="feature", values=pcol, aggfunc="min")

    # ensure all phases exist
    for ph in ["P1", "P2", "P3", "P4", "P5"]:
        if ph not in pmat.index:
            return {"both24": 0, "only24": 0, "both24_plus1": 0, "leak_any_nonstress": 0}

    sig = (pmat < alpha)

    both24 = (sig.loc["P2"] & sig.loc["P4"])
    only24 = both24 & (~sig.loc["P1"]) & (~sig.loc["P3"]) & (~sig.loc["P5"])

    nonstress_count = (sig.loc["P1"].astype(int) + sig.loc["P3"].astype(int) + sig.loc["P5"].astype(int))
    both24_plus1 = both24 & (nonstress_count == 1)

    leak_any_nonstress = int((sig.loc["P1"] | sig.loc["P3"] | sig.loc["P5"]).sum())

    return {
        "both24": int(both24.sum()),
        "only24": int(only24.sum()),
        "both24_plus1": int(both24_plus1.sum()),
        "leak_any_nonstress": leak_any_nonstress,
    }


# -------------------------
# Main: sweep kernels (and optionally derivative mode)
# -------------------------
def main():
    phased_dir = Path("dataset/paired_tests_EMOCA/phased_csvs")
    if not phased_dir.exists():
        raise SystemExit(f"phased_dir not found: {phased_dir}")

    subs = list_subjects(phased_dir)
    if not subs:
        raise SystemExit(f"No phased CSVs found in {phased_dir}")

    # === WHAT TO SWEEP ===
    kernels = [1]      #, 3, 5, 7, 9]       #, 11, 15, 21]
    deriv_modes = ["finite"]  # try also ["finite", "savgol"] if you want

    # choose whether to score on raw p or corrected p
    SCORE_PCOL = "p"      # or "p_bonf"
    SCORE_ALPHA = 0.05

    for deriv_mode in deriv_modes:
        for k in kernels:
            cfg = Config(
                phased_dir=str(phased_dir),
                include_delta_pose=False,
                mode="conv",
                conv_kernel=k,
                pad_mode="reflect",
                deriv_mode=deriv_mode,
                sg_window=11,
                sg_poly=3,
                time_hz=10.0,
                min_len=40,
                out_csv=f"phasewise_md_nd_stats_convk{k}_{deriv_mode}.csv",
            )

            all_subj = []
            feature_cols_ref = None

            for sid in subs:
                s = build_subject_summaries(phased_dir, sid, cfg)
                if s is None:
                    continue

                if feature_cols_ref is None:
                    feature_cols_ref = s["feature_cols"]
                else:
                    # enforce same order across subjects by intersecting
                    if s["feature_cols"] != feature_cols_ref:
                        inter = [c for c in feature_cols_ref if c in s["feature_cols"]]
                        s["feature_cols"] = inter

                all_subj.append(s)

            if not all_subj or feature_cols_ref is None:
                print(f"[k={k} | {deriv_mode}] no valid subjects.")
                continue

            df = run_tests(all_subj, feature_cols_ref, alpha=0.05)

            out_csv = Path(cfg.out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

            sc = pattern_score(df, pcol=SCORE_PCOL, alpha=SCORE_ALPHA)

            print(
                f"[k={k:2d} | {deriv_mode:6s}] saved={out_csv} | "
                f"score: both24={sc['both24']}, only24={sc['only24']}, "
                f"both24+1={sc['both24_plus1']}, leak_nonstress={sc['leak_any_nonstress']}"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# LR probe for stress-vs-nonstress using phased MD only.

# Samples: one per (subject, phase) from MD drive only.
# Labels:  y=1 for P2,P4 (stressor), y=0 for P1,P3,P5 (non-stress).
# Features: mean_level μ, velocity ν, or concat(μ,ν).
# Operators: conv (triangular kernel k; k=1 means no smoothing) or spline.
# CV: subject-wise GroupKFold.
# """

# from pathlib import Path
# from dataclasses import dataclass
# from typing import List, Optional, Tuple

# import numpy as np
# import pandas as pd
# from scipy.signal import convolve
# from scipy.interpolate import UnivariateSpline

# from sklearn.model_selection import GroupKFold
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score
# )

# # Optional csaps
# try:
#     from csaps import csaps
#     _HAS_CSAPS = True
# except Exception:
#     _HAS_CSAPS = False


# PHASES = ["P1", "P2", "P3", "P4", "P5"]
# STRESS = {"P2", "P4"}
# NONSTRESS = {"P1", "P3", "P5"}


# @dataclass
# class Config:
#     phased_dir: str = "dataset/paired_tests_EMOCA/phased_csvs"
#     phase_col: str = "Phase"
#     time_hz: float = 10.0
#     min_len: int = 40
#     include_delta_pose: bool = False

#     rep: str = "velocity"     # "mean_level" | "velocity" | "concat"

#     mode: str = "conv"        # "conv" | "spline"
#     conv_kernel: int = 1      # 1=no smoothing
#     pad_mode: str = "reflect"

#     spline_p: float = 0.8     # used only if mode="spline"

#     n_splits: int = 5
#     seed: int = 0


# def list_subjects(phased_dir: Path) -> List[str]:
#     return sorted({p.name[:4] for p in phased_dir.glob("T???_*_PHASED.csv")})

# def find_md_path(phased_dir: Path, sid: str) -> Optional[Path]:
#     md_cands = sorted(phased_dir.glob(f"{sid}_MD*_*PHASED.csv"))
#     return md_cands[0] if md_cands else None

# def select_feature_cols(df: pd.DataFrame, include_delta_pose: bool) -> List[str]:
#     cols = []
#     for c in df.columns:
#         cs = str(c)
#         if cs.startswith("exp_") or cs.startswith("pose_"):
#             cols.append(cs)
#         elif include_delta_pose and cs.startswith("delta_pose_"):
#             cols.append(cs)
#     pose = sorted([c for c in cols if c.startswith("pose_")])
#     exp  = sorted([c for c in cols if c.startswith("exp_")])
#     dpo  = sorted([c for c in cols if c.startswith("delta_pose_")])
#     return pose + exp + dpo


# def triangular_smooth_1d(x: np.ndarray, k: int, pad_mode: str = "reflect") -> np.ndarray:
#     if k <= 1:
#         return x.copy()
#     box = np.ones(k, dtype=float) / k
#     tri = convolve(box, box, mode="full")
#     tri = tri / tri.sum()

#     if pad_mode and pad_mode != "none":
#         pad = len(tri) // 2
#         xpad = np.pad(x, (pad, pad), mode=pad_mode)
#         ypad = np.convolve(xpad, tri, mode="same")
#         return ypad[pad:-pad]
#     return np.convolve(x, tri, mode="same")


# def smooth_series_spline(t: np.ndarray, x: np.ndarray, p: float, deriv: int) -> np.ndarray:
#     mask = np.isfinite(x)
#     if mask.sum() < 4:
#         return x.copy()
#     if _HAS_CSAPS:
#         return csaps(t[mask], x[mask], t, smooth=p, nu=deriv)

#     xx = x[mask]
#     var = float(np.nanvar(xx)) if np.isfinite(xx).any() else 1.0
#     s = max(1e-8, (1.0 - p) * len(xx) * var)
#     spl = UnivariateSpline(t[mask], xx, k=3, s=s)
#     return spl.derivative(deriv)(t) if deriv else spl(t)


# def smooth_matrix(X: np.ndarray, cfg: Config, dt: float) -> Tuple[np.ndarray, np.ndarray]:
#     T, F = X.shape
#     if T < 2:
#         return X.copy(), np.zeros_like(X)

#     if cfg.mode == "conv":
#         Y = np.empty_like(X, dtype=float)
#         for f in range(F):
#             Y[:, f] = triangular_smooth_1d(X[:, f].astype(float), cfg.conv_kernel, cfg.pad_mode)

#         dY = np.zeros_like(Y)
#         dY[1:-1] = (Y[2:] - Y[:-2]) / (2.0 * dt)
#         dY[0]    = (Y[1] - Y[0]) / dt
#         dY[-1]   = (Y[-1] - Y[-2]) / dt
#         return Y, dY

#     if cfg.mode == "spline":
#         t = np.arange(T, dtype=float)
#         Y  = np.empty_like(X, dtype=float)
#         dY = np.empty_like(X, dtype=float)
#         for f in range(F):
#             col = X[:, f].astype(float)
#             Y[:, f]  = smooth_series_spline(t, col, p=cfg.spline_p, deriv=0)
#             dY[:, f] = smooth_series_spline(t, col, p=cfg.spline_p, deriv=1) / dt
#         return Y, dY

#     raise ValueError("cfg.mode must be 'conv' or 'spline'")


# def summarize_phase(df: pd.DataFrame, feature_cols: List[str], ph: str, cfg: Config) -> Optional[np.ndarray]:
#     d = df[df[cfg.phase_col] == ph]
#     if len(d) < cfg.min_len:
#         return None

#     X = d[feature_cols].to_numpy(dtype=float, copy=True)
#     dt = 1.0 / cfg.time_hz
#     Y, dY = smooth_matrix(X, cfg, dt)

#     mu = np.nanmean(Y, axis=0)
#     nu = np.nanmean(np.abs(dY), axis=0)

#     if cfg.rep == "mean_level":
#         return mu
#     if cfg.rep == "velocity":
#         return nu
#     if cfg.rep == "concat":
#         return np.concatenate([mu, nu], axis=0)
#     raise ValueError("cfg.rep must be mean_level|velocity|concat")


# def build_dataset(cfg: Config):
#     phased_dir = Path(cfg.phased_dir)
#     subs = list_subjects(phased_dir)

#     X_list, y_list, g_list = [], [], []
#     feature_cols_ref = None

#     for sid in subs:
#         md_path = find_md_path(phased_dir, sid)
#         if md_path is None:
#             continue
#         df = pd.read_csv(md_path, low_memory=False)
#         if cfg.phase_col not in df.columns:
#             continue

#         if feature_cols_ref is None:
#             feature_cols_ref = select_feature_cols(df, cfg.include_delta_pose)
#         feat_cols = [c for c in feature_cols_ref if c in df.columns]
#         if not feat_cols:
#             continue

#         for ph in PHASES:
#             if ph not in STRESS and ph not in NONSTRESS:
#                 continue

#             emb = summarize_phase(df, feat_cols, ph, cfg)
#             if emb is None:
#                 continue

#             y = 1 if ph in STRESS else 0
#             X_list.append(emb.astype(np.float32))
#             y_list.append(y)
#             g_list.append(sid)

#     X = np.vstack(X_list) if X_list else np.zeros((0, 0), dtype=np.float32)
#     y = np.asarray(y_list, dtype=int)
#     groups = np.asarray(g_list, dtype=str)
#     return X, y, groups


# def evaluate_lr(X, y, groups, n_splits=5, seed=0):
#     gkf = GroupKFold(n_splits=n_splits)

#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("lr", LogisticRegression(
#             solver="liblinear", class_weight="balanced",
#             max_iter=2000, random_state=seed
#         ))
#     ])

#     aurocs, auprcs, f1s, accs, baccs = [], [], [], [], []

#     for tr, te in gkf.split(X, y, groups):
#         pipe.fit(X[tr], y[tr])
#         prob = pipe.predict_proba(X[te])[:, 1]
#         pred = (prob >= 0.5).astype(int)

#         aurocs.append(roc_auc_score(y[te], prob))
#         auprcs.append(average_precision_score(y[te], prob))
#         f1s.append(f1_score(y[te], pred))
#         accs.append(accuracy_score(y[te], pred))
#         baccs.append(balanced_accuracy_score(y[te], pred))

#     def ms(a): return float(np.mean(a)), float(np.std(a))
#     return {
#         "AUROC": ms(aurocs),
#         "AUPRC": ms(auprcs),
#         "F1": ms(f1s),
#         "ACC": ms(accs),
#         "BACC": ms(baccs),
#         "N": int(len(y)),
#         "N_pos": int(y.sum()),
#         "N_neg": int((y == 0).sum()),
#     }


# def main():
#     # === Run the three settingS ===
#     settings = [
#         ("raw-vel (k=1 finite)", Config(mode="conv", conv_kernel=1, rep="velocity")),
#         ("tri-conv k=3 (vel)",   Config(mode="conv", conv_kernel=5, rep="velocity")),
#         ("spline p=0.8 (vel)",   Config(mode="spline", spline_p=0.8, rep="velocity")),
#     ]

#     for name, cfg in settings:
#         X, y, groups = build_dataset(cfg)
#         if len(y) == 0:
#             print(f"[{name}] no samples.")
#             continue
#         res = evaluate_lr(X, y, groups, n_splits=cfg.n_splits, seed=cfg.seed)
#         print(f"\n[{name}] N={res['N']} (pos={res['N_pos']} neg={res['N_neg']})")
#         print(f"  AUROC: {res['AUROC'][0]:.3f} ± {res['AUROC'][1]:.3f}")
#         print(f"  AUPRC: {res['AUPRC'][0]:.3f} ± {res['AUPRC'][1]:.3f}")
#         print(f"  F1   : {res['F1'][0]:.3f} ± {res['F1'][1]:.3f}")
#         print(f"  ACC  : {res['ACC'][0]:.3f} ± {res['ACC'][1]:.3f}")
#         print(f"  BACC : {res['BACC'][0]:.3f} ± {res['BACC'][1]:.3f}")


# if __name__ == "__main__":
#     main()
