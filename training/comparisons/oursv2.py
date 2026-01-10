
"""
Visual-only stress classification with a basic Transformer on EMOCA/FLAME features.

Now with 5-fold subject-wise cross-validation.

- Input:   all_subjects_merged.csv
- Uses:    exp_*, pose_*, delta_pose* columns (visual only)
- Labels:  window label = 1 if stress_ratio >= --label_thresh, else 0
- Windowing: sliding windows by time using t_sec (or similar)
- Model:   ConvStem + PositionalEncoding + TransformerEncoder + AttnPool
- Evaluation:
    * 5-fold CV, subject-wise (train/val/test per fold)
    * Per-fold metrics saved to outdir/fold_k/
    * Overall CV summary (mean/std across folds) saved to outdir/metrics_cv_summary.json
"""

import argparse
import math
import random
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from utils.normalization import (
    fit_global_z, fit_subject_z, fit_robust, apply_norm,
    fit_global_then_subject_z
)
# from utils.normalization import fit_global_z, fit_subject_z, fit_robust, apply_norm



from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold





# =========================
# PRETRAIN CONFIG (3-way comparison)
# =========================
PRETRAIN_MODE = "mae_full"   # one of: "scratch", "mae_frozen", "mae_full"
MAE_CKPT_PATH = "/home/vivib/emoca/emoca/runs/pretrain_mae_nd/encoder_only.pt"


# Optional: if MAE was trained with different embed_dim/layers, they MUST match here.
STRICT_LOAD = False  # if False, loads what matches (good for safety)

# =========================
# Utilities
# =========================

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_col(df, names):
    for c in df.columns:
        if c.lower() in [n.lower() for n in names]:
            return c
    return None



def global_then_subjectwise_znorm(X, subjects, m_train, eps=1e-8):
    # (1) global fit on TRAIN only
    mu_g = np.nanmean(X[m_train], axis=(0, 1))
    sig_g = np.nanstd(X[m_train], axis=(0, 1))
    sig_g[sig_g < 1e-6] = 1.0
    Xn = (X - mu_g) / (sig_g + eps)

    # (2) subject override ONLY for subjects that have TRAIN windows
    for s in np.unique(subjects):
        idx_tr = m_train & (subjects == s)
        if idx_tr.sum() < 1:
            continue
        mu_s = np.nanmean(X[idx_tr], axis=(0, 1))
        sig_s = np.nanstd(X[idx_tr], axis=(0, 1))
        sig_s[sig_s < 1e-6] = 1.0
        idx_all = (subjects == s)
        Xn[idx_all] = (X[idx_all] - mu_s) / (sig_s + eps)

    return Xn



def global_then_subjectwise_znorm_tanh(X, subjects, m_train, eps=1e-8):
    """
    1) Fit GLOBAL mu/sigma on TRAIN windows only -> apply to ALL windows
    2) For subjects that exist in TRAIN, override with SUBJECT-specific mu/sigma
       computed from that subject's TRAIN windows only (still no leakage)
    3) Apply tanh to bound values in (-1, 1)
    """
    Xn = X.copy()

    # --- (1) global z-norm fit on TRAIN windows only ---
    mu_g = np.nanmean(X[m_train], axis=(0, 1))          # [F]
    sig_g = np.nanstd(X[m_train], axis=(0, 1))          # [F]
    sig_g[sig_g < 1e-6] = 1.0
    Xn = (Xn - mu_g) / (sig_g + eps)

    # --- (2) subject-wise override (only if subject has TRAIN windows) ---
    for s in np.unique(subjects):
        idx_tr = m_train & (subjects == s)
        if idx_tr.sum() < 1:
            continue  # keep global norm for val/test-only subjects

        mu_s = np.nanmean(X[idx_tr], axis=(0, 1))       # [F]
        sig_s = np.nanstd(X[idx_tr], axis=(0, 1))       # [F]
        sig_s[sig_s < 1e-6] = 1.0

        idx_all = (subjects == s)
        Xn[idx_all] = (X[idx_all] - mu_s) / (sig_s + eps)

    # --- (3) tanh squashing to (-1, 1) ---
    Xn = np.tanh(Xn)
    return Xn

def evaluate_subject_level(model, loader, device, fixed_threshold=None, agg="mean"):
    """
    Subject-level evaluation by aggregating window probabilities per subject.

    agg:
      - "mean"  : mean prob across windows (standard)
      - "median": median prob (robust)
      - "max"   : max prob (detect any stressed window)
      - "p90"   : 90th percentile prob (robust 'max-ish')
    """
    model.eval()
    subj2y = {}
    subj2ps = {}

    with torch.no_grad():
        for x, m, y, subs in loader:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            p = torch.sigmoid(model(x, m)).cpu().numpy()
            y = y.numpy()

            # store per-window probs grouped by subject
            for i, s in enumerate(subs):
                s = str(s)
                subj2ps.setdefault(s, []).append(float(p[i]))
                subj2y.setdefault(s, []).append(int(y[i]))

    # build subject-level arrays
    subjects = sorted(subj2ps.keys())
    y_true_sub = []
    p_sub = []

    for s in subjects:
        # subject label: majority vote over window labels (or max; pick one policy)
        ys = np.array(subj2y[s], dtype=int)
        y_sub = int(np.mean(ys) >= 0.5)   # majority
        y_true_sub.append(y_sub)

        ps = np.array(subj2ps[s], dtype=float)
        if agg == "mean":
            p_hat = float(ps.mean())
        elif agg == "median":
            p_hat = float(np.median(ps))
        elif agg == "max":
            p_hat = float(ps.max())
        elif agg == "p90":
            p_hat = float(np.percentile(ps, 90))
        else:
            raise ValueError(f"Unknown agg={agg}")
        p_sub.append(p_hat)

    y_true_sub = np.array(y_true_sub, dtype=int)
    p_sub = np.array(p_sub, dtype=float)

    # metrics (same style as evaluate())
    metrics = {}
    try:
        metrics["AUROC"] = float(roc_auc_score(y_true_sub, p_sub))
        metrics["AUPRC"] = float(average_precision_score(y_true_sub, p_sub))
    except Exception:
        metrics["AUROC"] = float("nan")
        metrics["AUPRC"] = float("nan")

    prec, rec, thresholds = precision_recall_curve(y_true_sub, p_sub)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1s)) if len(f1s) else 0
    metrics["F1_max"] = float(f1s[best_idx]) if len(f1s) else float("nan")

    if fixed_threshold is not None:
        thr = float(fixed_threshold)
    else:
        thr = 0.5 if (best_idx == 0 or len(thresholds) == 0) else float(thresholds[best_idx - 1])
    metrics["threshold"] = thr

    y_pred_sub = (p_sub >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub, labels=[0, 1]).ravel()
    metrics.update(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))

    tp, fp, tn, fn = map(float, (tp, fp, tn, fn))
    prec_bin = tp / (tp + fp + 1e-9)
    rec_bin  = tp / (tp + fn + 1e-9)
    spec     = tn / (tn + fp + 1e-9)
    acc      = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    bal_acc  = 0.5 * (rec_bin + spec)

    metrics["Precision"] = float(prec_bin)
    metrics["Sensitivity"] = float(rec_bin)
    metrics["Specificity"] = float(spec)
    metrics["Accuracy"] = float(acc)
    metrics["BalancedAccuracy"] = float(bal_acc)
    metrics["F1"] = float(f1_score(y_true_sub, y_pred_sub))
    metrics["n_subjects"] = int(len(subjects))
    metrics["agg"] = agg
    return metrics




def detect_columns(df):
    subj_col  = find_col(df, ["subject", "subject_id", "Subject"])
    frame_col = find_col(df, ["frame_idx", "frame_id", "frame", "t_frame"])
    t_col     = find_col(df, ["t_sec", "time_sec", "timestamp_s", "time"])

    label_col = "GTlabel"
    if label_col not in df.columns:
        raise ValueError(f"GTlabel column not found. Available columns: {list(df.columns)[:50]} ...")

    # sanity: must be binary
    vals = pd.to_numeric(df[label_col], errors="coerce").dropna().unique()
    if not set(np.unique(vals)).issubset({0, 1}):
        raise ValueError(f"GTlabel is not binary. Unique values: {np.unique(vals)[:20]}")

    if subj_col is None or t_col is None:
        raise ValueError(f"Could not detect columns. Found: subj={subj_col}, t_sec={t_col}")

    return subj_col, frame_col, t_col, label_col


def pick_biosignal_cols(df):
    # try exact names first
    preferred = ["Perinasal.Perspiration", "Heart.Rate", "Breathing.Rate"]
    if all(c in df.columns for c in preferred):
        return preferred

    # fallback aliases
    aliases = {
        "Perinasal.Perspiration": ["Perinasal_Perspiration", "perinasal_perspiration", "perinasal"],
        "Heart.Rate": ["Heart_Rate", "heart_rate", "hr"],
        "Breathing.Rate": ["Breathing_Rate", "breathing_rate", "br"],
    }
    cols = []
    for k, alts in aliases.items():
        if k in df.columns:
            cols.append(k)
        else:
            hit = None
            for a in alts:
                if a in df.columns:
                    hit = a
                    break
            if hit is None:
                raise ValueError(f"Missing biosignal column for {k}. Tried: {alts}")
            cols.append(hit)
    return cols


def pick_flame_cols(df):
    # cols = [c for c in df.columns if c.startswith("exp_")]
    # cols += [c for c in df.columns if c.startswith("pose_")]
    cols = [c for c in df.columns if c.startswith("delta1_")]
    # cols += [c for c in df.columns if c.startswith("Perinasal.Perspiration")]
    # cols += [c for c in df.columns if c.startswith("Heart.Rate")]
    # cols += [c for c in df.columns if c.startswith("Breathing.Rate")]


    if not cols:
        raise ValueError("No FLAME/EMOCA columns found (exp_*, pose_*, delta1_*).")
    return cols


def clean_subject_time(t_raw, target_hz=30.0):
    """
    t_raw: 1D array-like of t_sec for one subject (possibly strings/NaNs).
    Returns a strictly increasing float64 array with NaNs interpolated/extrapolated.
    """
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


def time_window_starts(t, win_sec, stride_sec):
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return []
    t0, t1 = float(t[0]), float(t[-1])
    if t1 - t0 < 1e-9:
        return [t0]
    last_start = t1 - win_sec
    if last_start < t0:
        last_start = t0
    starts = []
    s = t0
    while s <= last_start + 1e-9:
        starts.append(s)
        s += stride_sec
    if not starts:
        starts = [t0]
    return starts


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


def make_subject_bins(subjects, y, n_bins=5):
    """
    Build stratification bins at SUBJECT level based on each subject's pos_frac (mean label over windows).
    Returns:
      unique_subs: array of subject IDs
      subj_pos_frac: array of pos_frac per subject
      subj_bins: int bins [0..n_bins-1] per subject
    """
    df_s = pd.DataFrame({"Subject": subjects.astype(str), "y": y.astype(int)})
    subj_pos = df_s.groupby("Subject")["y"].mean().sort_index()
    unique_subs = subj_pos.index.to_numpy()
    subj_pos_frac = subj_pos.to_numpy()

    # Quantile bins; fall back to uniform bins if qcut has duplicate edges
    try:
        subj_bins = pd.qcut(subj_pos_frac, q=n_bins, labels=False, duplicates="drop")
        subj_bins = np.asarray(subj_bins, dtype=int)
        # if duplicates="drop" reduced bins too much, handle below
    except Exception:
        subj_bins = None

    if subj_bins is None or len(np.unique(subj_bins)) < 2:
        # fallback: uniform bins on [min,max]
        edges = np.linspace(subj_pos_frac.min(), subj_pos_frac.max() + 1e-9, n_bins + 1)
        subj_bins = np.digitize(subj_pos_frac, edges[1:-1], right=True).astype(int)

    return unique_subs, subj_pos_frac, subj_bins


class ZNorm:
    def __init__(self, eps=1e-8):
        self.mu = None
        self.sigma = None
        self.eps = eps

    def fit(self, X):
        # X: [N, T, F]
        self.mu = np.nanmean(X, axis=(0, 1))
        self.sigma = np.nanstd(X, axis=(0, 1))
        self.sigma[self.sigma < 1e-6] = 1.0

    def __call__(self, X):
        return (X - self.mu) / (self.sigma + self.eps)


import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    """
    Binary focal loss on logits.
    Supports optional alpha-balancing and optional pos_weight (like BCEWithLogits).
    """
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)  # alpha for positive class
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        # BCE per-element (no reduction), keep logits form for stability
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )

        # pt = P(correct class)
        p = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, p, 1.0 - p)

        focal_factor = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = torch.where(targets >= 0.5, self.alpha, 1.0 - self.alpha)
            loss = alpha_t * focal_factor * bce
        else:
            loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =========================
# Model
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):  # [B,T,E]
        return x + self.pe[:, :x.size(1), :]


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, p=None):
        super().__init__()
        p = k // 2 if p is None else p
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch),
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):  # [B,F,T]
        return self.net(x)


class ConvStem(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        mid = max(64, embed_dim // 2)
        self.proj = nn.Conv1d(in_dim, mid, kernel_size=1)
        self.b1 = DepthwiseConv1d(mid, mid, k=3)
        self.b2 = DepthwiseConv1d(mid, mid, k=5)     ######  k = 5
        self.b3 = DepthwiseConv1d(mid, mid, k=7)     ######  k = 7
        self.out = nn.Conv1d(mid * 3, embed_dim, kernel_size=1)

    def forward(self, x):          # x: [B,T,F]
        x = x.permute(0, 2, 1)     # [B,F,T]
        x = self.proj(x)
        y = [self.b1(x), self.b2(x), self.b3(x)]
        x = torch.cat(y, dim=1)
        x = self.out(x).permute(0, 2, 1)  # [B,T,E]
        return x


class SimpleStem(nn.Module):
    """Colleague-style: 1x1 convs only (per-frame projection)."""
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):          # x: [B,T,F]
        x = x.permute(0, 2, 1)     # [B,F,T]
        x = self.net(x)            # [B,E,T]
        return x.permute(0, 2, 1)  # [B,T,E]



class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layers=2, nhead=4, dropout=0.1, max_len=2000):
        super().__init__()
        # self.stem = ConvStem(in_dim, embed_dim)     
        # self.pos = PositionalEncoding(embed_dim, max_len)

        self.stem = ConvStem(in_dim, embed_dim)
        self.stem_norm = nn.LayerNorm(embed_dim)   # <-- ADD
        self.pos = PositionalEncoding(embed_dim, max_len)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,      # change to 4 if it is worse 
            dropout=dropout,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)


    def forward(self, x, pad_mask=None):
        z = self.stem(x)            # [B,T,E]
        z = self.stem_norm(z)       # <-- IMPORTANT (actually uses the norm)
        z = self.pos(z)
        z = self.enc(z, src_key_padding_mask=pad_mask)
        return z


    # def forward(self, x, pad_mask=None):
    #     z = self.stem(x)
    #     z = self.pos(z)
    #     z = self.enc(z, src_key_padding_mask=pad_mask)
    #     return z


class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d_model, d_model),
                               nn.Tanh(),
                               nn.Linear(d_model, 1))

    def forward(self, x, pad_mask=None):
        # x: [B,T,E]
        w = self.a(x).squeeze(-1)  # [B,T]
        if pad_mask is not None:
            w = w.masked_fill(pad_mask, float("-inf"))
        a = torch.softmax(w, dim=1)  # [B,T]
        return torch.einsum("bte,bt->be", x, a), a



class TemporalDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.rand(x.shape[:2], device=x.device) > self.p
        return x * mask.unsqueeze(-1)



class VisualStressTransformer(nn.Module):
    """
    Single-modality transformer for EMOCA/FLAME streams (MD session).
    """
    def __init__(self, flame_dim, embed_dim=128, enc_layers=2, nhead=4,
                 dropout=0.1, max_len=2000):
        super().__init__()
        self.enc_flame = ModalityEncoder(flame_dim, embed_dim,
                                         num_layers=enc_layers,
                                         nhead=nhead,
                                         dropout=dropout,
                                         max_len=max_len)
        self.pool = AttnPool(embed_dim)
        self.temp_drop = TemporalDropout(p=0.35)


        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),    ############################# // 4
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(embed_dim // 2, 1)                          ############################# // 4
        )

        # self.head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim, 1)
        # )

    def forward_features(self, xf, mask_f=None, return_attn=False):
        zf = self.enc_flame(xf, pad_mask=mask_f)
        zf = self.temp_drop(zf)
        pf, attn = self.pool(zf, pad_mask=mask_f)
        return (pf, attn) if return_attn else pf



    # def forward_features(self, xf, mask_f=None):
    #     zf = self.enc_flame(xf, pad_mask=mask_f)  # [B,T,E]
    #     pf, _ = self.pool(zf, pad_mask=mask_f)    # [B,E]
    #     return pf

    def forward(self, xf, mask_f=None):
        feats = self.forward_features(xf, mask_f)
        return self.head(feats).squeeze(-1)


# =========================
# Dataset
# =========================

class WindowsDataset(Dataset):
    def __init__(self, X, y, subjects):
        """
        X: [N, T, F] float32
        y: [N] float32 or int
        subjects: [N] array of subject IDs (str)
        """
        self.X = X
        self.y = y.astype(np.float32)
        self.subjects = np.array(subjects)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        y = torch.tensor(self.y[i], dtype=torch.float32)
        subj = self.subjects[i]
        mask = torch.zeros((x.shape[0],), dtype=torch.bool)  # no padding
        return x, mask, y, subj


# =========================
# Metrics & evaluation
# =========================
def binary_acc_from_logits(logits, y, thr=0.5):
    """
    logits: torch.Tensor [B]
    y:      torch.Tensor [B] in {0,1}
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate_with_loss(model, loader, device, criterion, thr=0.5):
    model.eval()
    losses = []
    accs = []
    ys, ps = [], []

    for x, m, yb, _sub in loader:
        x = x.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(x, m)
        yb_smooth = yb * 0.95 + 0.025  # moves 0->0.025 and 1->0.975
        loss = criterion(logits, yb_smooth)


        # loss = criterion(logits, yb)
        losses.append(loss.item())
        accs.append(binary_acc_from_logits(logits, yb, thr=thr))

        p = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.detach().cpu().numpy())
        ps.append(p)

  
    y_true = np.concatenate(ys)
    y_scores = np.concatenate(ps)

    out = {}
    try:
        out["AUROC"] = float(roc_auc_score(y_true, y_scores))
        out["AUPRC"] = float(average_precision_score(y_true, y_scores))
    except Exception:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")

    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    out["F1_max"] = float(np.nanmax(f1s)) if len(f1s) else float("nan")

    # fixed-threshold metrics (so acc matches whats  printed)
    y_pred = (y_scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out["Accuracy"] = float((tp + tn) / (tp + tn + fp + fn + 1e-9))
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    out["BalancedAccuracy"] = float(0.5 * (sens + spec))
    out["F1"] = float(f1_score(y_true, y_pred))

    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    out["acc"] = float(np.mean(accs)) if accs else float("nan")
    return out

def evaluate(model, loader, device, fixed_threshold=None):
    """
    Evaluate model on a loader and return a dict with:
      - AUROC, AUPRC
      - F1_max (swept over thresholds from PR curve)
      - threshold (used for hard predictions)
      - F1 (at chosen threshold)
      - Precision, Recall (Sensitivity), Specificity
      - Accuracy, BalancedAccuracy
      - Confusion matrix entries: TP, FP, TN, FN
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, m, y, _sub in loader:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            logit = model(x, m)
            p = torch.sigmoid(logit).cpu().numpy()
            ys.append(y.numpy())
            ps.append(p)

    y_true = np.concatenate(ys)
    y_scores = np.concatenate(ps)
    metrics = {}

    # --- ROC / PR ---
    try:
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
    except Exception:
        auroc, auprc = float("nan"), float("nan")
    metrics["AUROC"] = float(auroc)
    metrics["AUPRC"] = float(auprc)

    # --- F1-max from PR curve ---
    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    if len(f1s) > 0:
        best_idx = int(np.nanargmax(f1s))
        f1_max = float(f1s[best_idx])
    else:
        best_idx = 0
        f1_max = float("nan")
    metrics["F1_max"] = f1_max

    # --- Choose threshold ---
    if fixed_threshold is not None:
        thr = float(fixed_threshold)
    else:
        # thresholds has length len(prec) - 1.
        # Align best_idx>0 to thresholds[best_idx-1]; if best_idx==0, fall back to 0.5
        if best_idx == 0 or len(thresholds) == 0:
            thr = 0.5
        else:
            thr = float(thresholds[best_idx - 1])
    metrics["threshold"] = thr

    # --- Hard predictions at chosen threshold ---
    y_pred = (y_scores >= thr).astype(int)

    # Confusion matrix (labels: 0 = no-stress, 1 = stress)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["TP"] = int(tp)
    metrics["FP"] = int(fp)
    metrics["TN"] = int(tn)
    metrics["FN"] = int(fn)

    tp, fp, tn, fn = float(tp), float(fp), float(tn), float(fn)
    prec_bin = tp / (tp + fp + 1e-9)
    rec_bin = tp / (tp + fn + 1e-9)            # sensitivity / recall
    spec = tn / (tn + fp + 1e-9)               # specificity
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    bal_acc = 0.5 * (rec_bin + spec)

    metrics["Precision"] = float(prec_bin)
    metrics["Recall"] = float(rec_bin)
    metrics["Sensitivity"] = float(rec_bin)
    metrics["Specificity"] = float(spec)
    metrics["Accuracy"] = float(acc)
    metrics["BalancedAccuracy"] = float(bal_acc)
    metrics["F1"] = float(f1_score(y_true, y_pred))

    return metrics

def split_stats(name, y, mask, total_n):
    n = int(mask.sum())
    pos = int((y[mask] == 1).sum())
    neg = int((y[mask] == 0).sum())
    pct = 100.0 * n / max(1, total_n)
    pos_pct = 100.0 * pos / max(1, n)
    neg_pct = 100.0 * neg / max(1, n)
    return dict(
        name=name,
        n=n, pct=pct,
        pos=pos, pos_pct=pos_pct,
        neg=neg, neg_pct=neg_pct
    )

def print_fold_distributions(fold_idx, y, m_tr, m_va, m_te, subjects, train_subs, val_subs, test_subs):
    total_n = len(y)
    total_pos = int((y == 1).sum())
    total_neg = int((y == 0).sum())

    print("\n" + "-" * 70)
    print(f"[Fold {fold_idx}] Dataset totals: windows={total_n}  pos={total_pos}  neg={total_neg}  pos_frac={total_pos/max(1,total_n):.3f}")
    print(f"[Fold {fold_idx}] Subjects: total={len(np.unique(subjects))} | "
          f"train={len(train_subs)} ({100*len(train_subs)/max(1,len(np.unique(subjects))):.1f}%) | "
          f"val={len(val_subs)} ({100*len(val_subs)/max(1,len(np.unique(subjects))):.1f}%) | "
          f"test={len(test_subs)} ({100*len(test_subs)/max(1,len(np.unique(subjects))):.1f}%)")

    for st in [
        split_stats("TRAIN", y, m_tr, total_n),
        split_stats("VAL  ", y, m_va, total_n),
        split_stats("TEST ", y, m_te, total_n),
    ]:
        print(f"[Fold {fold_idx}] {st['name']} windows={st['n']} ({st['pct']:.1f}%) | "
              f"pos={st['pos']} ({st['pos_pct']:.1f}%) | "
              f"neg={st['neg']} ({st['neg_pct']:.1f}%)")
    print("-" * 70)



def fit_subjectwise_znorm(X, subjects, eps=1e-6):
    """
    X: (N, F) training windows
    subjects: (N,) subject id per window
    """
    stats = {}
    for s in np.unique(subjects):
        idx = subjects == s
        mu = X[idx].mean(axis=0)
        std = X[idx].std(axis=0)
        stats[s] = (mu, std + eps)
    return stats


def apply_subjectwise_znorm(X, subjects, stats, clip_c=5.0):
    Z = np.empty_like(X, dtype=np.float32)
    for i, s in enumerate(subjects):
        mu, std = stats[s]
        Z[i] = (X[i] - mu) / std
    if clip_c is not None:
        Z = np.clip(Z, -clip_c, clip_c)
    return Z

def load_mae_encoder_weights_into_visual_transformer(model, mae_path, device, strict=False):
    """
    Loads MAE-pretrained encoder weights into model.enc_flame (ModalityEncoder).
    Supports:
      - encoder_only.pt that is either a raw state_dict or dict with 'state_dict'/'model'
      - best_mae.pt that contains nested keys like 'encoder'/'enc' etc (we try common cases)
    """
    ckpt = torch.load(mae_path, map_location=device)

    # unwrap common checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
            sd = ckpt["encoder"]
        else:
            # maybe it's already the right dict
            # or it's the full MAE dict with "enc_*" keys
            sd = ckpt
    else:
        sd = ckpt

    # 1) If keys already match enc_flame.* directly
    if any(k.startswith("enc_flame.") for k in sd.keys()):
        model.load_state_dict(sd, strict=strict)
        return

    # 2) If checkpoint is encoder-only for the modality encoder, load into model.enc_flame
    # Try to map keys that start with common prefixes:
    prefixes = ["enc_flame.", "encoder.", "enc.", "enc_f.", "flame_enc.", "model.enc_flame.", "module.enc_flame."]
    # If no known prefix exists, we attempt direct load into enc_flame
    # by stripping "module." if present.

    cleaned = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        cleaned[kk] = v

    # Try direct load into enc_flame (most common for encoder_only.pt)
    missing, unexpected = model.enc_flame.load_state_dict(cleaned, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Strict load failed. missing={missing[:20]} unexpected={unexpected[:20]}")
    # If strict=False, it's ok; we at least load matching layers.


def set_encoder_trainable(model, trainable: bool):
    """
    Freeze/unfreeze ONLY the encoder (enc_flame). Head/pool stays trainable.
    """
    for p in model.enc_flame.parameters():
        p.requires_grad = bool(trainable)


def build_optimizer_for_mode(model, lr, weight_decay, mode):
    """
    - scratch, mae_full: optimize all parameters
    - mae_frozen: optimize only pool + head (and any non-encoder parts)
    """
    if mode == "mae_frozen":
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default="/home/vivib/emoca/emoca/training/comparisons/MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN_with_delta1.csv", 
                    help="/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/phased_csvs/MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN.csv")
    ap.add_argument("--outdir", type=Path, default=Path("./runs/visual_only_5fold"))
    ap.add_argument("--win_sec", type=float, default=12.0)
    ap.add_argument("--stride_sec", type=float, default=3.0)
    ap.add_argument("--target_hz", type=float, default=30.0)
    ap.add_argument("--label_thresh", type=float, default=0.30,
                    help="window stress_ratio -> label 1")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--norm", type=str, default="none", choices=["none", "z_global", "z_subject", "robust", "z_clip"], help="Normalization strategy")

    ap.add_argument("--clip_c", type=float, default=5.0,help="Clipping value for z_clip normalization")
    ap.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=-1.0,
                    help="If -1, use balanced alpha from TRAIN pos/neg. Else use given alpha for positive class.")


    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stride_sec < args.win_sec:
        print("[WARN] stride_sec < win_sec → forcing non-overlapping windows")
        args.stride_sec = args.win_sec


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # torch 2.x only, so keep it inside try
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


        print("[CUDA] name:", torch.cuda.get_device_name(0))

    print(f"[Load] {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)
    
    # Put results under outdir/<mode> so 3 runs don't overwrite each other
    args.outdir = Path(args.outdir) / PRETRAIN_MODE
    args.outdir.mkdir(parents=True, exist_ok=True)
    print("[Run Mode]", PRETRAIN_MODE, "| outdir:", args.outdir)

    # 1) compute biosignal columns ONLY for filtering subjects (already used above)
    bio_cols = pick_biosignal_cols(df)
    print(f"[BIO] {len(bio_cols)} cols (used ONLY for subject filtering): {bio_cols}")

    # 2) after df is filtered to valid_subs, pick EMOCA/FLAME columns for training
    flame_cols = pick_flame_cols(df)
    print(f"[FLAME] {len(flame_cols)} cols; first 8: {flame_cols[:8]}")


    subj_col, frame_col, t_col, label_col = detect_columns(df)
    
    # flame_cols = pick_flame_cols(df)
    
    

    print(f"[Cols] subj={subj_col}, t_sec={t_col}, label={label_col}")
   # print(f"[FLAME] {len(flame_cols)} cols; first 8: {flame_cols[:8]}")

    subs_all = sorted(df[subj_col].astype(str).unique())
    print(f"[Subjects] N={len(subs_all)} : {subs_all}")


       ################################## EXCLUDE MISSING BIOSIGNALS################3

    BIO_COLS = pick_biosignal_cols(df)

    # compute missing frac per subject per biosignal
    miss = (
        df.groupby(subj_col)[BIO_COLS]
        .apply(lambda g: g.isna().mean())
    )

    # rule: subject is valid if ALL biosignals <= 0.80 missing
    THR = 0.80
    valid_subs = miss.index[(miss <= THR).all(axis=1)].astype(str).tolist()

    # Print excluded subjects with reason
    excluded = miss.index[~((miss <= THR).all(axis=1))].astype(str).tolist()
    print("\n[Exclusion] dropping subjects due to biosignal missingness:")
    for s in excluded:
        row = miss.loc[s]
        reasons = [f"{c}={row[c]*100:.1f}%" for c in BIO_COLS if row[c] > THR]
        print(f"  - {s}: " + ", ".join(reasons))

    # Filter df
    df = df[df[subj_col].astype(str).isin(valid_subs)].copy()
    print(f"[Exclusion] kept {len(valid_subs)} subjects")

    win_sec = float(args.win_sec)
    stride_sec = float(args.stride_sec)
    hz = float(args.target_hz)
    T = int(round(win_sec * hz))
    print(f"\n[Windowing] win_sec={win_sec} stride_sec={stride_sec} hz={hz} → T={T} steps")

    # -------- Build windows ONCE for all subjects ----------
    X_list, y_list, subj_list = [], [], []
    man_rows = []

    for sid, g in df.groupby(subj_col):
        sid_str = str(sid)
        tt = clean_subject_time(g[t_col].values, target_hz=hz)
        labels = pd.to_numeric(g[label_col], errors="coerce").fillna(0).to_numpy()
        starts = time_window_starts(tt, win_sec, stride_sec)

        if len(starts) == 0:
            continue

        for s in starts:
            e = s + win_sec
            mask = (tt >= s) & (tt <= e)
            if not mask.any():
                continue

            labs = labels[mask]
            # stress_ratio = float((labs == 1).mean()) if labs.size > 0 else 0.0

            stress_ratio = float((labs == 1).mean())

            # # --- NO EXCLUSION ZONE ---
            y = 1 if stress_ratio >= 0.4 else 0        #### best configuration so far 0.3

            # if stress_ratio >= 0.40:
            #     y = 1
            # elif stress_ratio <= 0.10:
            #     y = 0
            # else:
            #     continue   # discard ambiguous window



            # y = 1 if stress_ratio >= args.label_thresh else 0

            grid = np.linspace(s, e, T, endpoint=False, dtype=np.float64)

            ####################################### UNCOMMENT FOR UNIMODAL
            feat_mat = [linear_resample(tt, g[c].values, grid) for c in flame_cols]

            ###################### EARLY FUSION HERE######################################

#             feat_mat = [linear_resample(tt, g[c].values, grid) for c in cols_use]
# if not feat_mat:
#     continue
# f = np.stack(feat_mat, axis=-1).astype(np.float32)  # [T, F_total]
# X_list.append(f)
            # feat_mat = [linear_resample(tt, g[c].values, grid) for c in bio_cols]

            if not feat_mat:
                continue
            f = np.stack(feat_mat, axis=-1)  # [T, F]
            X_list.append(f.astype(np.float32))
            y_list.append(y)
            subj_list.append(sid_str)
            man_rows.append(dict(
                Subject=sid_str,
                start_sec=float(s),
                end_sec=float(e),
                stress_ratio=stress_ratio,
                label=int(y)
            ))

    if not X_list:
        raise RuntimeError("No windows created. Check time/label columns and window parameters.")

    X = np.stack(X_list)   # [N, T, F]
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subj_list)
    manifest = pd.DataFrame(man_rows)
    manifest.to_csv(args.outdir / "manifest_windows.csv", index=False)
    print(f"\n[Windows] N={X.shape[0]}, T={X.shape[1]}, F={X.shape[2]}")
    print(f"[Manifest] → {args.outdir / 'manifest_windows.csv'}")

    # Global per-subject label stats (for sanity)
    print("\n[Per-subject window label stats (ALL windows)]")
    df_stats = pd.DataFrame({
        "Subject": subjects,
        "label": y
    })
    stats = df_stats.groupby("Subject")["label"].agg(
        num_windows="size",
        pos_frac="mean"
    ).reset_index()
    stats["neg_frac"] = 1.0 - stats["pos_frac"]
    print(stats.sort_values("Subject").to_string(index=False))
    stats.to_csv(args.outdir / "per_subject_window_label_stats.csv", index=False)

    flame_dim = X.shape[-1]
    seq_len = X.shape[1]

    # seq_len = X.shape[1]

    # -------- 5-fold subject-wise CV ----------
    # unique_subs = np.array(sorted(np.unique(subjects)))
    # print(f"\n[5-fold CV] Unique subjects = {len(unique_subs)}")

    # kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    cv_metrics = []   # will store per-fold test metrics
    cv_probe_metrics = []  # per-fold probe test metrics
    cv_subject_metrics = []  # define near cv_metrics


################################## UNCOMMENT FOR LOSO EVALUATION#############3

    # # unique_subs = np.array(sorted(np.unique(subjects)))
    # unique_subs, subj_pos_frac, subj_bins = make_subject_bins(subjects, y, n_bins=5)
    # for fold_idx, test_sub in enumerate(unique_subs, start=1):
    #     test_subs = np.array([test_sub])
    #     trainval_subs = unique_subs[unique_subs != test_sub]
    #     fold_dir = args.outdir / f"fold_{fold_idx}"
    #     fold_dir.mkdir(parents=True, exist_ok=True)

#########################################################################################################

######################################### UNCOMMENT FOR 5 CV EVALUATION###############
    unique_subs, subj_pos_frac, subj_bins = make_subject_bins(subjects, y, n_bins=5)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(unique_subs, subj_bins), start=1):
        fold_dir = args.outdir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_subs = unique_subs[test_idx]
        trainval_subs = unique_subs[trainval_idx]


####################################################################################################################

    # for fold_idx, (train_sub_idx, test_sub_idx) in enumerate(kf.split(unique_subs), start=1):
        
    #     print("\n" + "=" * 60)
    #     print(f"[Fold {fold_idx}]")

        # test_subs = unique_subs[test_sub_idx]
        # trainval_subs = unique_subs[train_sub_idx]

        # Inner split of trainval_subs into train + val (subject-wise)
        rng = np.random.default_rng(args.seed + fold_idx)
        tv_shuffled = np.array(trainval_subs)
        rng.shuffle(tv_shuffled)
        n_tv = len(tv_shuffled)
        n_val = max(1, int(round(0.20 * n_tv)))  # ~10% of trainval for val

        val_subs = tv_shuffled[:n_val]
        train_subs_fold = tv_shuffled[n_val:]

        print(f"[Fold {fold_idx}] train_subs={len(train_subs_fold)}, "
              f"val_subs={len(val_subs)}, test_subs={len(test_subs)}")

        # Build masks on window-level data
        m_tr = np.isin(subjects, train_subs_fold)
        m_va = np.isin(subjects, val_subs)
        m_te = np.isin(subjects, test_subs)

        if not m_tr.any() or not m_va.any() or not m_te.any():
            raise RuntimeError(f"Fold {fold_idx}: one of the splits has no windows. "
                               f"train={m_tr.sum()}, val={m_va.sum()}, test={m_te.sum()}")

        print(f"[Fold {fold_idx}] windows: train={m_tr.sum()}  val={m_va.sum()}  test={m_te.sum()}")

        # Z-norm on TRAIN windows of this fold only
        # norm = ZNorm()
        # norm.fit(X[m_tr])
        # X_norm = norm(X)
        
        # X_norm = np.clip(X_norm, -5.0, 5.0).astype(np.float32)

        # X shape: [N, T, F]

   



        # SUBJECT-wise z-norm (fit per subject using TRAIN windows only; applied to all splits)
        # X_norm = global_then_subjectwise_znorm(X, subjects, m_tr)   # <-- THIS is the subject-wise norm
        # X_norm = np.clip(X_norm, -5.0, 5.0).astype(np.float32)   # keep clipping (works well)

                

        # X_norm = subjectwise_znorm_fit_apply(X, subjects, m_tr)

        # X_norm = global_then_subjectwise_znorm_tanh(X, subjects, m_tr)
        print_fold_distributions(
            fold_idx, y,
            m_tr, m_va, m_te,
            subjects,
            train_subs_fold, val_subs, test_subs
        )



        def print_norm_stats(Xn, tag=""):
            flat = Xn.reshape(-1, Xn.shape[-1])  # [(N*T), F]
            mn = np.nanmin(flat, axis=0)
            mx = np.nanmax(flat, axis=0)
            p1 = np.nanpercentile(flat, 1, axis=0)
            p99 = np.nanpercentile(flat, 99, axis=0)
            frac_gt3 = np.nanmean(np.abs(flat) > 3.0)

            print(f"\n[Norm stats]{tag}")
            print(f"  global min={float(np.min(mn)):.3f}, max={float(np.max(mx)):.3f}")
            print(f"  global p01={float(np.min(p1)):.3f}, p99={float(np.max(p99)):.3f}")
            print(f"  frac(|z|>3)={float(frac_gt3)*100:.2f}%")

        # after normalization in each fold:
        # X_norm = subjectwise_znorm_fit_apply(X, subjects, m_tr)
        # print_norm_stats(X_norm[m_tr], tag=f" Fold {fold_idx} TRAIN")
        # print_norm_stats(X_norm[m_va], tag=f" Fold {fold_idx} VAL")
        # print_norm_stats(X_norm[m_te], tag=f" Fold {fold_idx} TEST")

        # --------------------------
# Normalization (fit on TRAIN windows only; apply to all)
        # --------------------------
        X_tr_raw = X[m_tr]
        X_va_raw = X[m_va]
        X_te_raw = X[m_te]

        s_tr = subjects[m_tr]
        s_va = subjects[m_va]
        s_te = subjects[m_te]

        if args.norm == "none":
            X_tr = X_tr_raw
            X_va = X_va_raw
            X_te = X_te_raw

        elif args.norm in ("z_global", "z_clip"):
            stats = fit_global_z(X_tr_raw)  # fit TRAIN only
            X_tr = apply_norm(X_tr_raw, norm=args.norm, stats=stats, clip_c=args.clip_c)
            X_va = apply_norm(X_va_raw, norm=args.norm, stats=stats, clip_c=args.clip_c)
            X_te = apply_norm(X_te_raw, norm=args.norm, stats=stats, clip_c=args.clip_c)

        elif args.norm == "robust":
            stats = fit_robust(X_tr_raw)    # fit TRAIN only
            X_tr = apply_norm(X_tr_raw, norm="robust", stats=stats)
            X_va = apply_norm(X_va_raw, norm="robust", stats=stats)
            X_te = apply_norm(X_te_raw, norm="robust", stats=stats)

        elif args.norm == "z_subject":
            # fit per-subject stats on TRAIN subjects only
            stats = fit_subject_z(X_tr_raw, s_tr)

            # IMPORTANT: only apply subject-wise norm to subjects seen in TRAIN.
            # For VAL/TEST subjects not in TRAIN, fall back to global TRAIN z-norm.
            global_stats = fit_global_z(X_tr_raw)

            def apply_subject_or_global(Xsplit, ssplit):
                Xout = np.empty_like(Xsplit, dtype=np.float32)
                for i, sid in enumerate(ssplit):
                    if sid in stats:
                        mu, std = stats[sid]
                        Xout[i] = (Xsplit[i] - mu) / std
                    else:
                        Xout[i] = apply_norm(Xsplit[i:i+1], norm="z_global", stats=global_stats)[0]
                return Xout

            X_tr = apply_subject_or_global(X_tr_raw, s_tr)
            X_va = apply_subject_or_global(X_va_raw, s_va)
            X_te = apply_subject_or_global(X_te_raw, s_te)

        else:
            raise ValueError(f"Unknown --norm {args.norm}")

        # Ensure float32 (torch likes it)
        X_tr = X_tr.astype(np.float32)
        X_va = X_va.astype(np.float32)
        X_te = X_te.astype(np.float32)


        y_tr, y_va, y_te = y[m_tr], y[m_va], y[m_te]
        s_tr, s_va, s_te = subjects[m_tr], subjects[m_va], subjects[m_te]

        print_norm_stats(X_tr, tag=f" Fold {fold_idx} TRAIN")
        print_norm_stats(X_va, tag=f" Fold {fold_idx} VAL")
        print_norm_stats(X_te, tag=f" Fold {fold_idx} TEST")




        loader_kwargs = dict(
        batch_size=args.batch,
        num_workers=4,                 # δοκίμασε 2/4/8
        pin_memory=(device.type=="cuda"),
        persistent_workers=True,       # only if num_workers>0
        prefetch_factor=4
    )


        # Datasets & loaders
        ds_tr = WindowsDataset(X_tr, y_tr, s_tr)
        ds_va = WindowsDataset(X_va, y_va, s_va)
        ds_te = WindowsDataset(X_te, y_te, s_te)


        dl_tr = DataLoader(ds_tr, shuffle=True,  **loader_kwargs)
        dl_va = DataLoader(ds_va, shuffle=False, **loader_kwargs)
        dl_te = DataLoader(ds_te, shuffle=False, **loader_kwargs)


        # dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
        #                    num_workers=2, pin_memory=True)
        # dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
        #                    num_workers=2, pin_memory=True)
        # dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False,
        #                    num_workers=2, pin_memory=True)

        # # Model, optimizer, scheduler for this fold
        # model = VisualStressTransformer(
        #     flame_dim=flame_dim,
        #     embed_dim=args.embed_dim,
        #     enc_layers=args.layers,
        #     nhead=args.heads,
        #     dropout=args.dropout,
        #     max_len=seq_len + 8
        # ).to(device)

        model = VisualStressTransformer(
            flame_dim=flame_dim,
            embed_dim=args.embed_dim,
            enc_layers=args.layers,
            nhead=args.heads,
            dropout=args.dropout,
            max_len=seq_len + 8
        ).to(device)

        # (2) optionally load MAE into encoder
        if PRETRAIN_MODE in ("mae_frozen", "mae_full"):
            load_mae_encoder_weights_into_visual_transformer(
                model, MAE_CKPT_PATH, device=device, strict=STRICT_LOAD
            )
            print(f"[Fold {fold_idx}] Loaded MAE encoder weights from: {MAE_CKPT_PATH}")

        # (3) freeze if needed
        if PRETRAIN_MODE == "mae_frozen":
            set_encoder_trainable(model, trainable=False)
            print(f"[Fold {fold_idx}] Encoder frozen (training pool+head only).")
        else:
            set_encoder_trainable(model, trainable=True)
            print(f"[Fold {fold_idx}] Encoder trainable (end-to-end).")

                # ---- class balance for TRAIN only ----
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())

        # use_pos_weight = (neg / (pos + 1e-9)) > 1.2 or (pos / (neg + 1e-9)) > 1.2
        # pos_weight = torch.tensor([neg / (pos + 1e-9)], device=device) if use_pos_weight else None

        # if args.loss == "bce":
        #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # else:
        #     # balanced dataset -> no alpha needed (or alpha=0.5)
        #     alpha = None  # or 0.5
        #     criterion = FocalLossWithLogits(
        #         gamma=args.focal_gamma,
        #         alpha=alpha,
        #         pos_weight=pos_weight,   # will be None most of the time now
        #         reduction="mean"
        #     )


        pos_weight = torch.tensor(
            [neg / (pos + 1e-9)],
            device=device
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


        # criterion = nn.BCEWithLogitsLoss()
        # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt = build_optimizer_for_mode(model, lr=args.lr, weight_decay=args.weight_decay, mode=PRETRAIN_MODE)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


        print(f"\n[Fold {fold_idx}] Training")
        best_val = -1.0
        best_path = fold_dir / "best_visual.pt"

        # ---- Early stopping ----
        patience = 5          # try 2 or 3
        min_delta = 1e-4      # require at least this much improvement in AUPRC
        bad_epochs = 0

        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None


        # print(f"\n[Fold {fold_idx}] Training")
        # best_val = -1.0
        # best_path = fold_dir / "best_visual.pt"

        # scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_losses = []
            tr_accs = []

            for x, m, yb, _sub in dl_tr:
                x = x.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        feats, attn = model.forward_features(x, m, return_attn=True)
                        logits = model.head(feats).squeeze(-1)
                        yb_smooth = yb * 0.95 + 0.025  # moves 0->0.025 and 1->0.975
                        loss = criterion(logits, yb_smooth)
                        

                        # loss = criterion(logits, yb)
                        entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=1).mean()
                        loss = loss - 0.001 * entropy

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    feats, attn = model.forward_features(x, m, return_attn=True)
                    logits = model.head(feats).squeeze(-1)

                    yb_smooth = yb * 0.95 + 0.025  # moves 0->0.025 and 1->0.975
                    loss = criterion(logits, yb_smooth)

                    # loss = criterion(logits, yb)
                    entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=1).mean()
                    loss = loss - 0.001 * entropy

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                tr_losses.append(loss.item())
                tr_accs.append(binary_acc_from_logits(logits.detach(), yb.detach(), thr=0.5))

            sched.step()

            train_loss = float(np.mean(tr_losses))
            train_acc  = float(np.mean(tr_accs))

            # # val: loss+acc+auroc/auprc/f1max at same thr
            # val_out = evaluate_with_loss(model, dl_va, device, criterion, thr=0.5)
            val_out = evaluate_with_loss(model, dl_va, device, criterion, thr=0.5)


            lr_now = opt.param_groups[0]["lr"]
            print(
                f"[Fold {fold_idx} | Epoch {epoch:03d}] "
                f"lr={lr_now:.2e} | "
                f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val loss={val_out['loss']:.4f} acc={val_out['acc']:.3f} | "
                f"val AUROC={val_out['AUROC']:.3f} AUPRC={val_out['AUPRC']:.3f} F1max={val_out['F1_max']:.3f}"
            )


            # ---- model selection on VAL AUPRC ----
            improved = (val_out["AUPRC"] > best_val + min_delta)
            if improved:
                best_val = val_out["AUPRC"]
                bad_epochs = 0
                torch.save({
                    "model": model.state_dict(),
                    "config": vars(args),
                    "flame_dim": flame_dim,
                    "seq_len": seq_len,
                    "flame_cols": flame_cols,
                    "subj_col": subj_col,
                    "t_col": t_col,
                    "label_col": label_col,
                    "train_subs": list(train_subs_fold),
                    "val_subs": list(val_subs),
                    "test_subs": list(test_subs),
                }, best_path)
                print(f"  ↳ [Fold {fold_idx}] saved best → {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"  ↳ [Fold {fold_idx}] early stop (no AUPRC improvement for {patience} epochs)")
                    break



        # ---- Test on this fold ----
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])

                        # --- pick threshold on VAL (this is allowed) ---
            val_metrics_final = evaluate(model, dl_va, device)   # this will choose thr from VAL
            # thr_val = float(val_metrics_final["threshold"])
            thr_val = 0.5
            with open(fold_dir / "metrics_val.json", "w") as f:
                json.dump(val_metrics_final, f, indent=2)


        # test_metrics = evaluate(model, dl_te, device)
        # --- evaluate TEST using the VAL-chosen threshold (no leakage) ---
        test_metrics = evaluate(model, dl_te, device, fixed_threshold=thr_val)

        test_subj_metrics = evaluate_subject_level(model, dl_te, device, fixed_threshold=thr_val, agg="mean")

        print(
            f"[Fold {fold_idx} | TEST | SUBJECT] "
            f"AUROC={test_subj_metrics['AUROC']:.3f} "
            f"AUPRC={test_subj_metrics['AUPRC']:.3f} "
            f"F1={test_subj_metrics['F1']:.3f} "
            f"Acc={test_subj_metrics['Accuracy']:.3f} "
            f"BalAcc={test_subj_metrics['BalancedAccuracy']:.3f} "
            f"(n={test_subj_metrics['n_subjects']}, agg={test_subj_metrics['agg']})"
        )

        with open(fold_dir / "metrics_test_subject.json", "w") as f:
            json.dump(test_subj_metrics, f, indent=2)

        test_metrics["threshold_source"] = "val"
        test_metrics["val_threshold_used"] = thr_val

        print(
            f"\n[Fold {fold_idx} | TEST] AUROC={test_metrics['AUROC']:.3f}  "
            f"AUPRC={test_metrics['AUPRC']:.3f}  "
            f"F1_max={test_metrics['F1_max']:.3f}  "
            f"F1={test_metrics['F1']:.3f}  "
            f"Acc={test_metrics['Accuracy']:.3f}  "
            f"Prec={test_metrics['Precision']:.3f}  "
            f"Rec/Sens={test_metrics['Sensitivity']:.3f}  "
            f"Spec={test_metrics['Specificity']:.3f}  "
            f"BalAcc={test_metrics['BalancedAccuracy']:.3f}"
        )
        print(
            f"[Fold {fold_idx} | TEST] Confusion: TP={test_metrics['TP']}, "
            f"FP={test_metrics['FP']}, TN={test_metrics['TN']}, "
            f"FN={test_metrics['FN']} | "
            f"threshold={test_metrics['threshold']:.3f}"
        )

        # Save fold test metrics
        with open(fold_dir / "metrics_test.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        cv_metrics.append(test_metrics)

        cv_subject_metrics.append(test_subj_metrics)


        # ---- Linear probe on pooled embeddings (per fold) ----
        print("\n[Fold {fold_idx}] [Probe] Extracting pooled embeddings for train/val/test …")

        def collect_embeddings(loader):
            model.eval()
            feats_list, y_list, sub_list = [], [], []
            with torch.no_grad():
                for x, m, yb, subs in loader:
                    x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
                    feats = model.forward_features(x, m).cpu().numpy()
                    feats_list.append(feats)
                    y_list.append(yb.numpy())
                    sub_list.extend(list(subs))
            Xp = np.concatenate(feats_list)
            yp = np.concatenate(y_list)
            subp = np.array(sub_list)
            return Xp, yp, subp

        Xtr_emb, ytr_emb, sub_tr_emb = collect_embeddings(dl_tr)
        Xva_emb, yva_emb, sub_va_emb = collect_embeddings(dl_va)
        Xte_emb, yte_emb, sub_te_emb = collect_embeddings(dl_te)

        # probe = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)

        probe = LogisticRegression(
                                            max_iter=2000,
                                            class_weight="balanced",
                                            solver="lbfgs"
                                        )

        probe.fit(Xtr_emb, ytr_emb)
        pva = probe.predict_proba(Xva_emb)[:, 1]
        pte = probe.predict_proba(Xte_emb)[:, 1]

        probe_val = dict(
            AUROC=roc_auc_score(yva_emb, pva),
            AUPRC=average_precision_score(yva_emb, pva)
        )
        probe_test = dict(
            AUROC=roc_auc_score(yte_emb, pte),
            AUPRC=average_precision_score(yte_emb, pte)
        )

        print(f"[Fold {fold_idx}] [Probe] Val  AUROC={probe_val['AUROC']:.3f}  "
              f"AUPRC={probe_val['AUPRC']:.3f}")
        print(f"[Fold {fold_idx}] [Probe] Test AUROC={probe_test['AUROC']:.3f}  "
              f"AUPRC={probe_test['AUPRC']:.3f}")

        with open(fold_dir / "probe_metrics_test.json", "w") as f:
            json.dump(probe_test, f, indent=2)

        cv_probe_metrics.append(probe_test)

    # --------- Summarize CV across folds ----------
    def summarize_cv(metric_list, keys):
        summary = {}
        for k in keys:
            vals = [m[k] for m in metric_list if k in m and not np.isnan(m[k])]
            if not vals:
                continue
            summary[k + "_mean"] = float(np.mean(vals))
            summary[k + "_std"] = float(np.std(vals))
        return summary

    main_keys = ["AUROC", "AUPRC", "F1", "F1_max",
                 "Accuracy", "BalancedAccuracy",
                 "Sensitivity", "Specificity"]
    probe_keys = ["AUROC", "AUPRC"]

    summary_main = summarize_cv(cv_metrics, main_keys)
    summary_probe = summarize_cv(cv_probe_metrics, probe_keys)
    summary_subject = summarize_cv(cv_subject_metrics, main_keys)
   


    cv_summary = {
        "main_test": summary_main,
        "probe_test": summary_probe,
        "n_folds": 5,
    }

    print("\n" + "=" * 60)
    print("[CV SUMMARY] (TEST over 5 folds)")
    for k, v in summary_main.items():
        print(f"  {k}: {v:.4f}")
    print("[CV SUMMARY] [Probe]")
    for k, v in summary_probe.items():
        print(f"  {k}: {v:.4f}")

    cv_summary["subject_test"] = summary_subject  
    with open(args.outdir / "metrics_cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    
if __name__ == "__main__":
    main()
