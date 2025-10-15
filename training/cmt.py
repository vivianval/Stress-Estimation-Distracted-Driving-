#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression

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
        p = k//2 if p is None else p
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
        mid = max(64, embed_dim//2)
        self.proj = nn.Conv1d(in_dim, mid, kernel_size=1)
        self.b1 = DepthwiseConv1d(mid, mid, k=3)
        self.b2 = DepthwiseConv1d(mid, mid, k=5)
        self.b3 = DepthwiseConv1d(mid, mid, k=7)
        self.out = nn.Conv1d(mid*3, embed_dim, kernel_size=1)
    def forward(self, x):          # x: [B,T,F]
        x = x.permute(0,2,1)       # [B,F,T]
        x = self.proj(x)
        y = [self.b1(x), self.b2(x), self.b3(x)]
        x = torch.cat(y, dim=1)
        x = self.out(x).permute(0,2,1)  # [B,T,E]
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, y, key_padding_mask_x=None, key_padding_mask_y=None):
        h, _ = self.attn(query=x, key=y, value=y, key_padding_mask=key_padding_mask_y)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ffn(x))
        return x

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layers=2, nhead=4, dropout=0.1, max_len=2000):
        super().__init__()
        self.stem = ConvStem(in_dim, embed_dim)
        self.pos  = PositionalEncoding(embed_dim, max_len)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                         dim_feedforward=4*embed_dim, dropout=dropout,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)
    def forward(self, x, pad_mask=None):
        z = self.stem(x)
        z = self.pos(z)
        z = self.enc(z, src_key_padding_mask=pad_mask)
        return z

class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d_model,d_model), nn.Tanh(), nn.Linear(d_model,1))
    def forward(self, x, pad_mask=None):
        w = self.a(x).squeeze(-1)
        if pad_mask is not None:
            w = w.masked_fill(pad_mask, -1e9)
        a = torch.softmax(w, dim=1)
        return torch.einsum('bte,bt->be', x, a), a

class CrossModalStressNet(nn.Module):
    def __init__(self, flame_dim, bio_dim, embed_dim=128, enc_layers=2, nhead=4, dropout=0.1, max_len=2000):
        super().__init__()
        self.enc_flame = ModalityEncoder(flame_dim, embed_dim, enc_layers, nhead, dropout, max_len)
        self.enc_bio   = ModalityEncoder(bio_dim,   embed_dim, enc_layers, nhead, dropout, max_len)
        self.xattn_ab  = CrossAttentionBlock(embed_dim, nhead, dropout)  # FLAME <- BIO
        self.xattn_ba  = CrossAttentionBlock(embed_dim, nhead, dropout)  # BIO   <- FLAME
        self.pool      = AttnPool(embed_dim)
        self.head      = nn.Sequential(nn.Linear(2*embed_dim, embed_dim), nn.LeakyReLU(0.1),
                                       nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    # features only (for linear probe / diagnostics)
    def forward_features(self, xf, xb, mask_f=None, mask_b=None):
        zf = self.enc_flame(xf, mask_f)
        zb = self.enc_bio(xb, mask_b)
        zf = self.xattn_ab(zf, zb, mask_f, mask_b)
        zb = self.xattn_ba(zb, zf, mask_b, mask_f)
        pf,_ = self.pool(zf, mask_f)
        pb,_ = self.pool(zb, mask_b)
        return torch.cat([pf,pb], dim=-1)  # [B, 2E]

    def forward(self, xf, xb, mask_f=None, mask_b=None):
        feats = self.forward_features(xf, xb, mask_f, mask_b)
        return self.head(feats).squeeze(-1)

# =========================
# Data utils (time-based)
# =========================
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def find_col(df, names):
    for c in df.columns:
        if c.lower() in [n.lower() for n in names]:
            return c
    return None

def detect_columns(df):
    subj_col = find_col(df, ["subject","subject_id","Subject"])
    frame_col = find_col(df, ["frame_idx","frame_id","frame","t_frame"])
    t_col = find_col(df, ["t_sec","time_sec","timestamp_s","time"])
    # label detection
    label_col = None
    candidates = [c for c in df.columns if c.lower() in {"label","stress","stress_label","y"}]
    for c in candidates:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if set(np.unique(vals)).issubset({0,1}): label_col=c; break
    if label_col is None:
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(vals)>0 and set(np.unique(vals)).issubset({0,1}):
                label_col=c; break
    if subj_col is None or t_col is None or label_col is None:
        raise ValueError(f"Could not detect columns (subject/time/label). Found: subj={subj_col}, t_sec={t_col}, label={label_col}")
    return subj_col, frame_col, t_col, label_col

def pick_features(df):
    # FLAME stream
    flame_cols = [c for c in df.columns if c.startswith("exp_")]
    flame_cols += [c for c in df.columns if c.startswith("pose_")]
    flame_cols += [c for c in df.columns if c.startswith("delta_pose")]
    # Bio/Gaze stream
    bio_candidates = [
        "Heart.Rate","Perinasal.Perspiration","Breathing.Rate",
        "Gaze.X.Pos","Gaze.Y.Pos",
        "GazeVel","GazeVel_X","GazeVel_Y",
        "GazeAcc","GazeAcc_X","GazeAcc_Y",
        "GazeDispersion_2s","GazeVel_mean_1s","GazeVel_std_1s","GazeVel_mean_3s","GazeVel_std_3s"
    ]
    bio_cols = [c for c in bio_candidates if c in df.columns]
    return flame_cols, bio_cols

def split_subjects(df, subj_col, seed=42, train_ratio=0.80, val_ratio=0.10):
    subs = sorted(df[subj_col].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_train = int(round(n*train_ratio))
    n_val   = int(round(n*val_ratio))
    n_train = min(n_train, n)
    n_val   = min(n_val, n - n_train)
    return subs[:n_train], subs[n_train:n_train+n_val], subs[n_train+n_val:]

def time_window_starts(t, win_sec, stride_sec):
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0: return []
    t0, t1 = float(t[0]), float(t[-1])
    if t1 - t0 < 1e-9: return [t0]
    last_start = t1 - win_sec
    if last_start < t0: last_start = t0
    starts = []
    s = t0
    while s <= last_start + 1e-9:
        starts.append(s)
        s += stride_sec
    if not starts: starts = [t0]
    return starts

def clean_subject_time(t_raw, target_hz=30.0):
    """
    t_raw: 1D array-like of t_sec for one subject (possibly strings/NaNs).
    Returns a strictly increasing float64 array with NaNs interpolated/extrapolated.
    """
    t = pd.to_numeric(pd.Series(t_raw), errors="coerce")
    # Interpolate internal NaNs on index
    t_interp = t.interpolate(method="linear", limit_direction="both")
    vals = t_interp.to_numpy(dtype=float)

    # If all NaN, create synthetic time based on target_hz
    if not np.isfinite(vals).any():
        step = 1.0 / float(target_hz)
        return np.arange(len(vals), dtype=float) * step

    # Estimate dt from finite diffs
    finite = np.isfinite(vals)
    diffs = np.diff(vals[finite])
    if diffs.size == 0 or not np.isfinite(diffs).any():
        dt = 1.0 / float(target_hz)
    else:
        # robust dt estimate
        dt = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 1.0/float(target_hz)

    # Fill any remaining NaNs at edges by extrapolating with dt
    # (should be rare after interpolate(..., both))
    for i in range(len(vals)):
        if not np.isfinite(vals[i]):
            vals[i] = (vals[i-1] + dt) if i > 0 and np.isfinite(vals[i-1]) else (0.0 if i==0 else vals[i-1] + dt)

    # Enforce strict monotonicity (fix tiny non-positive steps)
    for i in range(1, len(vals)):
        if not (vals[i] > vals[i-1]):
            vals[i] = vals[i-1] + dt

    return vals

def linear_resample(times_s, values, grid_s):
    t = np.asarray(pd.to_numeric(times_s, errors="coerce"), dtype=float)
    v = np.asarray(pd.to_numeric(values,  errors="coerce"), dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    if mask.sum() < 2:
        return np.zeros_like(grid_s, dtype=np.float32)
    t = t[mask]; v = v[mask]
    if np.any(np.diff(t) < 0):
        idx = np.argsort(t); t = t[idx]; v = v[idx]
    y = np.interp(grid_s, t, v, left=v[0], right=v[-1])
    return y.astype(np.float32)

class ZNorm:
    def __init__(self, eps=1e-8): self.mu=None; self.sigma=None; self.eps=eps
    def fit(self, X): self.mu=np.nanmean(X,axis=(0,1)); self.sigma=np.nanstd(X,axis=(0,1)); self.sigma[self.sigma<1e-6]=1.0
    def __call__(self, X): return (X - self.mu)/ (self.sigma + self.eps)

class WindowManifest:
    def __init__(self): self.rows=[]
    def add(self, **kw): self.rows.append(kw)
    def to_df(self): return pd.DataFrame(self.rows)

class StressWindowsDataset(Dataset):
    """
    Time-based windowing using t_sec, with linear resampling to a uniform grid.
    Adds a Δt channel (=1/target_hz) to each modality.
    """
    def __init__(self, df, subjects, flame_cols, bio_cols, subj_col, t_col, label_col,
                 win_sec, stride_sec, target_hz,
                 norm_f=None, norm_b=None, label_thresh=0.30):
        self.df = df[df[subj_col].isin(subjects)].copy()
        self.df.sort_values([subj_col, t_col], inplace=True)
        self.subj_col, self.t_col, self.label_col = subj_col, t_col, label_col
        self.flame_cols, self.bio_cols = flame_cols, bio_cols
        self.win_sec, self.stride_sec, self.hz = float(win_sec), float(stride_sec), float(target_hz)
        self.T = int(round(self.win_sec * self.hz))
        self.index = []  # (subject, start_time, end_time, label, stress_ratio)
        self.manifest = WindowManifest()

        ######

        verbose = False
        for sid, g in self.df.groupby(subj_col):
            tt = clean_subject_time(g[self.t_col].values, target_hz=self.hz)
            starts = time_window_starts(tt, self.win_sec, self.stride_sec)
            if verbose:
                print(f"[Windowing] {sid}: frames={len(g)} t0={tt[0]:.3f} t1={tt[-1]:.3f} "
                      f"span={tt[-1]-tt[0]:.3f}s starts={len(starts)}")
        #     ...


        # # inside StressWindowsDataset.__init__ loop
        # for sid, g in self.df.groupby(subj_col):
        #     # Clean per-subject time
        #     tt = clean_subject_time(g[self.t_col].values, target_hz=self.hz)
        #     # Build starts using cleaned time
        #     starts = time_window_starts(tt, self.win_sec, self.stride_sec)
            for s in starts:
                e = s + self.win_sec
                mask = (tt >= s) & (tt <= e)
                labs = pd.to_numeric(g.loc[mask, self.label_col], errors="coerce").fillna(0).to_numpy()
                stress_ratio = float((labs==1).mean()) if labs.size>0 else 0.0
                y = 1 if stress_ratio >= label_thresh else 0
                self.index.append((sid, float(s), float(e), y, stress_ratio))
                self.manifest.add(Subject=sid, start_sec=float(s), end_sec=float(e),
                                stress_ratio=stress_ratio, label=y)


        # Fit normalization on resampled train windows ONLY
                
        Xf, Xb = [], []
        for sid, s, e, y, r in self.index:
            g = self.df[self.df[self.subj_col] == sid]
            tt = clean_subject_time(g[self.t_col].values, target_hz=self.hz)
            grid = np.linspace(s, e, self.T, endpoint=False, dtype=np.float64)

            f_mat = [linear_resample(tt, g[c].values, grid) for c in self.flame_cols]
            b_mat = [linear_resample(tt, g[c].values, grid) for c in self.bio_cols]
            if not f_mat: f_mat = [np.zeros_like(grid, dtype=np.float32)]
            if not b_mat: b_mat = [np.zeros_like(grid, dtype=np.float32)]

            f = np.stack(f_mat, axis=-1)  # [T, Ff]
            b = np.stack(b_mat, axis=-1)  # [T, Fb]
            dt = np.full((self.T, 1), 1.0 / self.hz, dtype=np.float32)

            Xf.append(np.concatenate([f, dt], axis=-1))
            Xb.append(np.concatenate([b, dt], axis=-1))

        Xf = np.stack(Xf) if Xf else np.zeros((1, self.T, 1), np.float32)
        Xb = np.stack(Xb) if Xb else np.zeros((1, self.T, 1), np.float32)

        self.norm_f = norm_f or ZNorm()
        self.norm_b = norm_b or ZNorm()
        self.norm_f.fit(Xf)
        self.norm_b.fit(Xb)


    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        sid, s, e, y, r = self.index[i]
        g = self.df[self.df[self.subj_col] == sid]
        tt = clean_subject_time(g[self.t_col].values, target_hz=self.hz)
        grid = np.linspace(s, e, self.T, endpoint=False, dtype=np.float64)

        f_mat = [linear_resample(tt, g[c].values, grid) for c in self.flame_cols]
        b_mat = [linear_resample(tt, g[c].values, grid) for c in self.bio_cols]
        if not f_mat: f_mat = [np.zeros_like(grid, dtype=np.float32)]
        if not b_mat: b_mat = [np.zeros_like(grid, dtype=np.float32)]

        f = np.stack(f_mat, axis=-1)
        b = np.stack(b_mat, axis=-1)
        dt = np.full((self.T, 1), 1.0 / self.hz, dtype=np.float32)

        f = np.concatenate([f, dt], axis=-1)
        b = np.concatenate([b, dt], axis=-1)

        mask_f = np.zeros((self.T,), dtype=bool)
        mask_b = np.zeros((self.T,), dtype=bool)

        f = self.norm_f(f)
        b = self.norm_b(b)

        return (torch.from_numpy(f), torch.from_numpy(b),
                torch.from_numpy(mask_f), torch.from_numpy(mask_b),
                torch.tensor(y, dtype=torch.float32),
                sid, r)


# =========================
# Samplers / eval
# =========================
class FiftyFiftyBatchSampler(Sampler):
    """Each batch gets 50% positives, 50% negatives (oversamples minority)."""
    def __init__(self, labels, batch_size, shuffle=True):
        assert batch_size % 2 == 0, "Use even batch_size for 50/50."
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch  = batch_size
        self.half   = batch_size // 2
        self.shuffle= shuffle
        self.pos_idx = np.where(self.labels==1)[0].tolist()
        self.neg_idx = np.where(self.labels==0)[0].tolist()
        if len(self.pos_idx)==0 or len(self.neg_idx)==0:
            raise ValueError("Need both positive and negative windows.")
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.pos_idx); random.shuffle(self.neg_idx)
        n_batches = math.ceil(len(self.labels)/self.batch)
        p_ptr = n_ptr = 0
        for _ in range(n_batches):
            p, n = [], []
            for _ in range(self.half):
                p.append(self.pos_idx[p_ptr % len(self.pos_idx)]); p_ptr += 1
                n.append(self.neg_idx[n_ptr % len(self.neg_idx)]); n_ptr += 1
            batch = p + n
            random.shuffle(batch)
            yield from batch
    def __len__(self):
        return math.ceil(len(self.labels)/self.batch) * self.batch

def make_sampler(labels, mode, batch_size):
    if mode=="none": return None
    y = np.array(labels, dtype=np.int64)
    if mode=="pos_weight": return None
    if mode=="weighted_sampler":
        p = (y==1).mean()
        w_pos = 0.5/max(p, 1e-6); w_neg = 0.5/max(1-p, 1e-6)
        weights = np.where(y==1, w_pos, w_neg).astype(np.float32)
        return WeightedRandomSampler(weights=torch.tensor(weights), num_samples=len(y), replacement=True)
    if mode=="fifty_fifty":
        return FiftyFiftyBatchSampler(y, batch_size=batch_size, shuffle=True)
    return None

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for f,b,mf,mb,y,_,_ in loader:
            f,b,mf,mb = f.to(device), b.to(device), mf.to(device), mb.to(device)
            p = torch.sigmoid(model(f,b,mf,mb)).cpu().numpy()
            ys.append(y.numpy()); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)
    try:
        auroc = roc_auc_score(y, p); auprc = average_precision_score(y, p)
    except Exception:
        auroc, auprc = float("nan"), float("nan")
    prec, rec, _ = precision_recall_curve(y, p)
    f1s = 2*prec*rec/(prec+rec+1e-9)
    f1 = np.nanmax(f1s) if len(f1s) else float("nan")
    return dict(AUROC=auroc, AUPRC=auprc, F1=f1)

# =========================
# Reporting helpers
# =========================
def report_subject_stats(df, subj_col, t_col, label_col, title):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    rows = []
    for sid, g in df.groupby(subj_col):
        n = len(g)
        tmin, tmax = g[t_col].min(), g[t_col].max()
        span = float(tmax - tmin) if pd.notna(tmax) and pd.notna(tmin) else np.nan
        sr = pd.to_numeric(g[label_col], errors="coerce").fillna(0).mean()
        rows.append((str(sid), n, span, sr))
    tbl = pd.DataFrame(rows, columns=["Subject","num_frames","time_span_sec","stress_ratio"])
    print(tbl.sort_values("Subject").to_string(index=False))
    return tbl

def report_window_stats(manifest_df, title):
    print("\n" + "-"*80)
    print(title)
    print("-"*80)
    g = manifest_df.groupby("Subject").agg(
        num_windows=("label","size"),
        pos_frac=("label","mean"),
        sr_mean=("stress_ratio","mean"),
        start=("start_sec","min"),
        end=("end_sec","max")
    ).reset_index()
    print(g.sort_values("Subject").to_string(index=False))
    return g

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="all_subjects_merged.csv")
    ap.add_argument("--outdir", type=Path, default=Path("./runs/cmt_time_report"))
    # time-based windowing
    ap.add_argument("--win_sec", type=float, default=12.0)
    ap.add_argument("--stride_sec", type=float, default=3.0)
    ap.add_argument("--target_hz", type=float, default=30.0, help="resample rate (Hz)")
    # splits
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    # model
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    # train
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--balance", choices=["none","pos_weight","weighted_sampler","fifty_fifty"], default="pos_weight")
    ap.add_argument("--label_thresh", type=float, default=0.30, help="window stress_ratio -> 1")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # ---------- Load & detect columns
    df = pd.read_csv(args.csv, low_memory=False)
   
    subj_col, frame_col, t_col, label_col = detect_columns(df)
    flame_cols, bio_cols = pick_features(df)

    print(f"[Cols] subj={subj_col}, t_sec={t_col}, label={label_col}")
    print(f"[Dims] FLAME={len(flame_cols)}  BIO={len(bio_cols)}")
    print(f"[Split]  train={args.train_ratio*100:.0f}%  val={args.val_ratio*100:.0f}%  test={(1-args.train_ratio-args.val_ratio)*100:.0f}%")

    # ---------- NaN audit (already in your script)
    nan_rows = (pd.to_numeric(df[t_col], errors="coerce").isna())
    nan_report = df[nan_rows].groupby(subj_col).size().sort_values(ascending=False)
    if len(nan_report):
        print("\n[WARN] NaN t_sec rows per subject:")
        print(nan_report.to_string())

    # ---------- Time audit after cleaning  <<< ADD THIS BLOCK HERE
    print("\n[Time audit after cleaning]")
    for sid, g in df.groupby(subj_col):
        tt = clean_subject_time(g[t_col].values, target_hz=args.target_hz)
        tspan = float(tt[-1] - tt[0]) if len(tt) > 1 else 0.0
        dt_med = float(np.median(np.diff(tt))) if len(tt) > 1 else float("nan")
        print(f"  {sid}: frames={len(g):6d}  cleaned_span={tspan:8.3f}s  dt~{dt_med:.4f}s")



    # print(f"[Cols] subj={subj_col}, t_sec={t_col}, label={label_col}")
    # print(f"[Dims] FLAME={len(flame_cols)}  BIO={len(bio_cols)}")
    # print(f"[Split]  train={args.train_ratio*100:.0f}%  val={args.val_ratio*100:.0f}%  test={(1-args.train_ratio-args.val_ratio)*100:.0f}%")

    # ---------- Global subject-level stats (frames/time span/stress ratio)
    _ = report_subject_stats(df, subj_col, t_col, label_col, "GLOBAL PER-SUBJECT STATS (frames, time span, stress ratio)")

    # ---------- Subject split
    train_subs, val_subs, test_subs = split_subjects(
        df, subj_col, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    print("\nTRAIN SUBJECTS:", ", ".join(sorted(map(str, train_subs))))
    print("VAL   SUBJECTS:", ", ".join(sorted(map(str, val_subs))))
    print("TEST  SUBJECTS:", ", ".join(sorted(map(str, test_subs))))

    # ---------- Build datasets (time-based)
    ds_train = StressWindowsDataset(df, train_subs, flame_cols, bio_cols, subj_col, t_col, label_col,
                                    win_sec=args.win_sec, stride_sec=args.stride_sec, target_hz=args.target_hz,
                                    norm_f=None, norm_b=None, label_thresh=args.label_thresh)
    ds_val   = StressWindowsDataset(df, val_subs, flame_cols, bio_cols, subj_col, t_col, label_col,
                                    win_sec=args.win_sec, stride_sec=args.stride_sec, target_hz=args.target_hz,
                                    norm_f=ds_train.norm_f, norm_b=ds_train.norm_b, label_thresh=args.label_thresh)
    ds_test  = StressWindowsDataset(df, test_subs, flame_cols, bio_cols, subj_col, t_col, label_col,
                                    win_sec=args.win_sec, stride_sec=args.stride_sec, target_hz=args.target_hz,
                                    norm_f=ds_train.norm_f, norm_b=ds_train.norm_b, label_thresh=args.label_thresh)

    
    # ---------- Save manifests & report per-split window stats
    man_train = ds_train.manifest.to_df(); man_train.to_csv(args.outdir/"manifest_train.csv", index=False)
    man_val   = ds_val.manifest.to_df();   man_val.to_csv(args.outdir/"manifest_val.csv",   index=False)
    man_test  = ds_test.manifest.to_df();  man_test.to_csv(args.outdir/"manifest_test.csv",  index=False)

    report_window_stats(man_train, "TRAIN: per-subject windows / label balance / time coverage")
    report_window_stats(man_val,   "VAL:   per-subject windows / label balance / time coverage")
    report_window_stats(man_test,  "TEST:  per-subject windows / label balance / time coverage")
    

    # ---------- Dataloaders (balancing without breaking temporal succession inside windows)
    y_train = [int(ds_train.index[i][3]) for i in range(len(ds_train))]
    sampler = make_sampler(y_train, mode=args.balance, batch_size=args.batch)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=(sampler is None),
                          sampler=sampler, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=2)
    dl_test  = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalStressNet(
        flame_dim=len(flame_cols)+1,  # +1 for Δt
        bio_dim=len(bio_cols)+1,      # +1 for Δt
        embed_dim=args.embed_dim, enc_layers=args.layers,
        nhead=args.heads, dropout=args.dropout,
        max_len=int(round(args.win_sec*args.target_hz))+8
    ).to(device)

    # ---------- Loss & optim
    pos_frac = np.mean(y_train) if len(y_train)>0 else 0.5
    pos_weight = torch.tensor([max((1-pos_frac)/(pos_frac+1e-6), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if args.balance=="pos_weight" else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---------- Train
    best_val = -1.0; best_path = args.outdir/"best.pt"
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for f,b,mf,mb,y,_,_ in dl_train:
            f,b,mf,mb,y = f.to(device), b.to(device), mf.to(device), mb.to(device), y.to(device)
            logit = model(f,b,mf,mb)
            loss = criterion(logit, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        sched.step()
        val_metrics = evaluate(model, dl_val, device)
        print(f"[Epoch {epoch:03d}] loss={np.mean(losses):.4f} | val AUROC={val_metrics['AUROC']:.3f} AUPRC={val_metrics['AUPRC']:.3f} F1={val_metrics['F1']:.3f}")
        score = val_metrics["AUPRC"]
        if score > best_val:
            best_val = score
            torch.save({
                "model": model.state_dict(),
                "config": vars(args),
                "flame_cols": flame_cols,
                "bio_cols": bio_cols,
                "subj_col": subj_col, "t_col": t_col, "label_col": label_col
            }, best_path)
            print(f"  ↳ saved best → {best_path}")

    # ---------- Test
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, dl_test, device)
    print(f"[TEST] AUROC={test_metrics['AUROC']:.3f}  AUPRC={test_metrics['AUPRC']:.3f}  F1={test_metrics['F1']:.3f}")
    with open(args.outdir/"metrics_test.json","w") as f: json.dump(test_metrics, f, indent=2)

    # ---------- Linear probe on pooled embeddings (diagnostic: do embeddings carry signal?)
    print("\n[Probe] Extracting pooled embeddings for train/val/test …")
    def collect_embeddings(loader):
        model.eval()
        feats_list, y_list, sub_list = [], [], []
        with torch.no_grad():
            for f,b,mf,mb,y,subs,_ in loader:
                f,b,mf,mb = f.to(device), b.to(device), mf.to(device), mb.to(device)
                feats = model.forward_features(f,b,mf,mb).cpu().numpy()
                feats_list.append(feats)
                y_list.append(y.numpy())
                sub_list.extend(list(subs))
        X = np.concatenate(feats_list); y = np.concatenate(y_list)
        return X, y, np.array(sub_list)

    Xtr, ytr, str_sub = collect_embeddings(dl_train)
    Xva, yva, sva_sub = collect_embeddings(dl_val)
    Xte, yte, ste_sub = collect_embeddings(dl_test)

    probe = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)
    probe.fit(Xtr, ytr)
    pva = probe.predict_proba(Xva)[:,1]; pte = probe.predict_proba(Xte)[:,1]
    probe_val = dict(AUROC=roc_auc_score(yva,pva), AUPRC=average_precision_score(yva,pva))
    probe_test= dict(AUROC=roc_auc_score(yte,pte), AUPRC=average_precision_score(yte,pte))
    print(f"[Probe] Val  AUROC={probe_val['AUROC']:.3f}  AUPRC={probe_val['AUPRC']:.3f}")
    print(f"[Probe] Test AUROC={probe_test['AUROC']:.3f}  AUPRC={probe_test['AUPRC']:.3f}")

    # Per-subject probe metrics (macro)
    print("\n[Probe] Per-subject TEST AUROC:")
    aurocs = []
    for sid in sorted(set(ste_sub)):
        m = ste_sub==sid
        if m.sum()>=3:
            aurocs.append(roc_auc_score(yte[m], pte[m]))
            print(f"  {sid}: {aurocs[-1]:.3f}")
    if aurocs:
        print(f"  Macro-avg: {np.mean(aurocs):.3f}")

if __name__ == "__main__":
    main()
