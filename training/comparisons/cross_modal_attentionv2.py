#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-modal Transformer (feature-tokens) for window-level NPZ like FINAL.npz

NPZ expected:
  X             [N,F] float32
  y             [N] int
  feature_names [F] object/str
  subject_id    [N] object/str

We treat each feature as a token:
  - For modality A: tokens = selected visual features (Fa tokens)
  - For modality B: tokens = selected bio features (Fb tokens)
Each token is a scalar value -> projected to embedding -> + learned token embedding.
Then:
  - modality-specific Transformer encoders
  - bidirectional cross-attention blocks
  - attention pooling per modality
  - MLP head

Normalization options (train-only stats):
  --norm none | zscore | minmax

Bio selection:
  --bio_mode all | perinasal | heart | breathing | perinasal_only_veldiff (etc.)
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------
# Repro / CUDA
# ----------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# ----------------------------
# Utils
# ----------------------------
def to_str_list(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def dataset_pos_frac(ds) -> Tuple[float, int, int]:
    """
    Returns (pos_frac, pos_count, n) for a Dataset whose __getitem__ returns (xa, xb, y).
    Assumes y is 0/1 (or close).
    """
    ys = np.array([float(ds[i][2].item()) for i in range(len(ds))], dtype=np.float32)
    ys = (ys >= 0.5).astype(np.int32)
    n = int(len(ys))
    pos = int(ys.sum())
    frac = float(pos / max(n, 1))
    return frac, pos, n


def best_threshold_for_metric(y_true: np.ndarray, probs: np.ndarray, metric: str = "Acc"):
    """
    Returns (best_thr, best_score) for metric in {"Acc","BalAcc","F1"}.
    Sweeps thresholds from PR curve (plus 0.5 fallback).
    """
    y_true = y_true.astype(np.int32)

    # candidate thresholds: from PR curve gives meaningful breakpoints
    p, r, thr = precision_recall_curve(y_true, probs)
    cand = np.unique(np.concatenate([thr, np.array([0.5], dtype=np.float64)]))

    best_thr, best_score = 0.5, -1.0
    for t in cand:
        y_hat = (probs >= t).astype(np.int32)
        if metric == "Acc":
            score = accuracy_score(y_true, y_hat)
        elif metric == "BalAcc":
            score = balanced_accuracy_score(y_true, y_hat)
        elif metric == "F1":
            score = f1_score(y_true, y_hat, zero_division=0)
        else:
            raise ValueError(metric)

        if score > best_score:
            best_score = float(score)
            best_thr = float(t)

    return best_thr, best_score


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(np.int32)
    probs = sigmoid_np(logits.astype(np.float64))

    out = {}
    out["AUROC"] = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
    out["AUPRC"] = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")

    p, r, _ = precision_recall_curve(y_true, probs)
    f1s = 2 * p * r / (p + r + 1e-12)
    out["F1_max"] = float(np.nanmax(f1s))

    y_hat = (probs >= thr).astype(np.int32)
    out["thr"] = float(thr)
    out["Acc"] = float(accuracy_score(y_true, y_hat))
    out["BalAcc"] = float(balanced_accuracy_score(y_true, y_hat))
    out["F1"] = float(f1_score(y_true, y_hat, zero_division=0))
    return out


# def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
#     y_true = y_true.astype(np.int32)
#     probs = sigmoid_np(logits.astype(np.float64))

#     out = {}
#     try:
#         out["AUROC"] = float(roc_auc_score(y_true, probs))
#     except Exception:
#         out["AUROC"] = float("nan")

#     try:
#         out["AUPRC"] = float(average_precision_score(y_true, probs))
#     except Exception:
#         out["AUPRC"] = float("nan")

#     try:
#         p, r, _ = precision_recall_curve(y_true, probs)
#         f1s = 2 * p * r / (p + r + 1e-12)
#         out["F1_max"] = float(np.nanmax(f1s))
#     except Exception:
#         out["F1_max"] = float("nan")



#     y_hat = (probs >= 0.5).astype(np.int32)
#     out["Acc"] = float(accuracy_score(y_true, y_hat))
#     out["BalAcc"] = float(balanced_accuracy_score(y_true, y_hat))
#     out["F1"] = float(f1_score(y_true, y_hat, zero_division=0))
#     return out

# def select_indices_frame_vel(
#     frame_names: List[str],
#     vel_names: List[str],
#     bio_mode: str = "all",
#     emoca_mode: str = "all",
# ):
#     """
#     Returns indices & names for:
#       A_frame, B_frame selected from frame_feature_names
#       A_vel,   B_vel   selected from vel_feature_names

#     Keeps your same modes:
#       emoca_mode controls A (visual) selection
#       bio_mode controls B (biosignal) selection
#     """

#     fn = list(frame_names)
#     vn = list(vel_names)

#     def idx_of(names, want):
#         missing = [w for w in want if w not in names]
#         if missing:
#             raise RuntimeError(f"Missing features: {missing[:10]} ...")
#         return sorted([names.index(w) for w in want])

#     # -----------------
#     # BIO selection
#     # -----------------
#     bio_base = ["Perinasal.Perspiration", "Breathing.Rate","Heart.Rate"]     #"Breathing.Rate",
#     bio_vel  = [
#         "veldiff__Perinasal.Perspiration",
#         "veldiff__Breathing.Rate",
#                                                                  #"veldiff__Breathing.Rate",
#         "veldiff__Heart.Rate",
#     ]

#     if bio_mode == "all":
#         want_b_frame = bio_base
#         want_b_vel   = bio_vel
#     elif bio_mode == "perinasal":
#         want_b_frame = ["Perinasal.Perspiration"]
#         want_b_vel   = ["veldiff__Perinasal.Perspiration"]
#     elif bio_mode == "heart":
#         want_b_frame = ["Heart.Rate"]
#         want_b_vel   = ["veldiff__Heart.Rate"]
#     elif bio_mode == "breathing":
#         want_b_frame = ["Breathing.Rate"]
#         want_b_vel   = ["veldiff__Breathing.Rate"]
#     elif bio_mode == "base_only":
#         want_b_frame = bio_base
#         want_b_vel   = []
#     elif bio_mode == "veldiff_only":
#         want_b_frame = []
#         want_b_vel   = bio_vel
#     else:
#         raise ValueError(f"Unknown bio_mode={bio_mode}")

#     # -----------------
#     # EMOCA selection
#     # -----------------

#     ################ FEATURE SELECTION###################
#     # EXP_EXACT = {"exp_20","exp_40","exp_18","exp_03","exp_16","exp_05","exp_39","exp_07","exp_21"}
#     # exp_base = [c for c in fn if c in EXP_EXACT]
#     # POSE_EXACT = {"pose_00", "pose_03"}
#     # pose_base = [c for c in fn if c in POSE_EXACT]
#     # delta1_exact_Exp = {"delta1_exp_20", "delta1_exp_40", "delta1_exp_18", "delta1_exp_03", "delta1_exp_16", "delta1_exp_05", "delta1_exp_39", "delta1_exp_07", "delta1_exp_21"}    
#     # delta1_exp = [c for c in fn if c in delta1_exact_Exp]
#     # delta1_exact_pose = {"delta1_pose_00", "delta1_pose_03"}
#     # delta1_pose = [c for c in fn if c in delta1_exact_pose]
#     vel_exp_exact = { "veldiff__exp_20","veldiff__exp_40", "veldiff__exp_18", "veldiff__exp_03", "veldiff__exp_16", "veldiff__exp_05", "veldiff__exp_39", "veldiff__exp_07", "veldiff__exp_21"}
#     vel_exp = [c for c in fn if c in vel_exp_exact]
#     # vel_pose_exact = {"veldiff__pose_00", "veldiff__pose_03"}
#     # vel_pose = [c for c in fn if c in vel_pose_exact]
#     exp_base   = [c for c in fn if c.startswith("exp_")]
#     pose_base  = [c for c in fn if c.startswith("pose_")]
#     delta1_exp  = [c for c in fn if c.startswith("delta1_exp_")]
#     delta1_pose = [c for c in fn if c.startswith("delta1_pose_")]

#     # vel_exp  = [c for c in vn if c.startswith("veldiff__exp_")]
#     vel_pose = [c for c in vn if c.startswith("veldiff__pose_")]

#     if emoca_mode == "all":
#         want_a_frame = exp_base + pose_base + delta1_exp + delta1_pose
#         want_a_vel   = vel_exp + vel_pose
#     elif emoca_mode == "base":
#         want_a_frame = exp_base + pose_base
#         want_a_vel   = []
#     elif emoca_mode == "delta1":
#         want_a_frame = delta1_exp + delta1_pose
#         want_a_vel   = vel_exp + vel_pose                           # []
#     elif emoca_mode == "vel":
#         want_a_frame = []
#         want_a_vel   = vel_exp + vel_pose
#     elif emoca_mode == "exp_only":
#         want_a_frame = exp_base + delta1_exp
#         want_a_vel   = vel_exp
#     elif emoca_mode == "pose_only":
#         want_a_frame = pose_base + delta1_pose
#         want_a_vel   = vel_pose
#     elif emoca_mode == "exp_base":
#         want_a_frame = exp_base
#         want_a_vel   = []
#     elif emoca_mode == "pose_base":
#         want_a_frame = pose_base
#         want_a_vel   = []
#     elif emoca_mode == "exp_delta1":
#         want_a_frame = delta1_exp
#         want_a_vel   = []
#     elif emoca_mode == "pose_delta1":
#         want_a_frame = delta1_pose
#         want_a_vel   = []
#     elif emoca_mode == "exp_vel":
#         want_a_frame = []
#         want_a_vel   = vel_exp
#     elif emoca_mode == "pose_vel":
#         want_a_frame = []
#         want_a_vel   = vel_pose
#     else:
#         raise ValueError(f"Unknown emoca_mode={emoca_mode}")

#     # indices
#     a_frame_idx = idx_of(fn, want_a_frame) if want_a_frame else []
#     b_frame_idx = idx_of(fn, want_b_frame) if want_b_frame else []
#     a_vel_idx   = idx_of(vn, want_a_vel)   if want_a_vel   else []
#     b_vel_idx   = idx_of(vn, want_b_vel)   if want_b_vel   else []

#     names_a_frame = [fn[i] for i in a_frame_idx]
#     names_b_frame = [fn[i] for i in b_frame_idx]
#     names_a_vel   = [vn[i] for i in a_vel_idx]
#     names_b_vel   = [vn[i] for i in b_vel_idx]

#     # sanity: forbid gaze
#     if any("Gaze" in n for n in (names_a_frame + names_b_frame + names_a_vel + names_b_vel)):
#         raise RuntimeError("Gaze leakage detected.")

#     print("\n[INPUT SELECTION]")
#     print(f"  A/EMOCA frame ({emoca_mode}): {len(names_a_frame)}")
#     print(f"  A/EMOCA vel   ({emoca_mode}): {len(names_a_vel)}")
#     print(f"  B/BIO   frame ({bio_mode})  : {len(names_b_frame)} -> {names_b_frame}")
#     print(f"  B/BIO   vel   ({bio_mode})  : {len(names_b_vel)} -> {names_b_vel}")

#     return a_frame_idx, b_frame_idx, a_vel_idx, b_vel_idx, names_a_frame, names_b_frame, names_a_vel, names_b_vel


def select_indices_frame_vel(
    frame_names: List[str],
    vel_names: List[str],
    bio_mode: str = "all",
    emoca_mode: str = "all",
    gaze_mode: str = "all",
    modality_a: str = "emoca",
    modality_b: str = "bio",
):
    """
    Returns indices & names for:
      A_frame, B_frame selected from frame_feature_names
      A_vel,   B_vel   selected from vel_feature_names

    modality_a/modality_b in {"emoca","bio","gaze"}.
    Gaze is frame-only in this NPZ (no veldiff gaze features exist).
    """

    fn = list(frame_names)
    vn = list(vel_names)

    def idx_of(names, want):
        missing = [w for w in want if w not in names]
        if missing:
            raise RuntimeError(f"Missing features: {missing[:10]} ...")
        return sorted([names.index(w) for w in want])

    # -----------------
    # BIO selection (same as yours)
    # -----------------
    bio_base = ["Perinasal.Perspiration", "Breathing.Rate", "Heart.Rate"]
    bio_vel  = [
        "veldiff__Perinasal.Perspiration",
        "veldiff__Breathing.Rate",
        "veldiff__Heart.Rate",
    ]

    if bio_mode == "all":
        want_bio_frame = bio_base
        want_bio_vel   = bio_vel
    elif bio_mode == "perinasal":
        want_bio_frame = ["Perinasal.Perspiration"]
        want_bio_vel   = ["veldiff__Perinasal.Perspiration"]
    elif bio_mode == "heart":
        want_bio_frame = ["Heart.Rate"]
        want_bio_vel   = ["veldiff__Heart.Rate"]
    elif bio_mode == "breathing":
        want_bio_frame = ["Breathing.Rate"]
        want_bio_vel   = ["veldiff__Breathing.Rate"]
    elif bio_mode == "base_only":
        want_bio_frame = bio_base
        want_bio_vel   = []
    elif bio_mode == "veldiff_only":
        want_bio_frame = []
        want_bio_vel   = bio_vel
    else:
        raise ValueError(f"Unknown bio_mode={bio_mode}")

    # -----------------
    # EMOCA selection (same spirit as yours)
    # -----------------
    exp_base    = [c for c in fn if c.startswith("exp_")]
    pose_base   = [c for c in fn if c.startswith("pose_")]
    delta1_exp  = [c for c in fn if c.startswith("delta1_exp_")]
    delta1_pose = [c for c in fn if c.startswith("delta1_pose_")]

    vel_exp  = [c for c in vn if c.startswith("veldiff__exp_")]
    vel_pose = [c for c in vn if c.startswith("veldiff__pose_")]

    if emoca_mode == "all":
        want_emoca_frame = exp_base + pose_base + delta1_exp + delta1_pose
        want_emoca_vel   = vel_exp + vel_pose
    elif emoca_mode == "base":
        want_emoca_frame = exp_base + pose_base
        want_emoca_vel   = []
    elif emoca_mode == "delta1":
        want_emoca_frame = delta1_exp + delta1_pose
        want_emoca_vel   = vel_exp + vel_pose
    elif emoca_mode == "vel":
        want_emoca_frame = []
        want_emoca_vel   = vel_exp + vel_pose
    elif emoca_mode == "exp_only":
        want_emoca_frame = exp_base + delta1_exp
        want_emoca_vel   = vel_exp
    elif emoca_mode == "pose_only":
        want_emoca_frame = pose_base + delta1_pose
        want_emoca_vel   = vel_pose
    elif emoca_mode == "exp_base":
        want_emoca_frame = exp_base
        want_emoca_vel   = []
    elif emoca_mode == "pose_base":
        want_emoca_frame = pose_base
        want_emoca_vel   = []
    elif emoca_mode == "exp_delta1":
        want_emoca_frame = delta1_exp
        want_emoca_vel   = []
    elif emoca_mode == "pose_delta1":
        want_emoca_frame = delta1_pose
        want_emoca_vel   = []
    elif emoca_mode == "exp_vel":
        want_emoca_frame = []
        want_emoca_vel   = vel_exp
    elif emoca_mode == "pose_vel":
        want_emoca_frame = []
        want_emoca_vel   = vel_pose
    else:
        raise ValueError(f"Unknown emoca_mode={emoca_mode}")

    # -----------------
    # GAZE selection (NEW) — frame-only
    # -----------------
    # Note: NPZ already excluded Gaze.X.Pos / Gaze.Y.Pos at creation.
    # We'll just select the gaze columns that exist in feature_names_frame.
    gaze_all = [c for c in fn if c.startswith("Gaze")]

    if gaze_mode == "all":
        want_gaze_frame = gaze_all
    elif gaze_mode == "no_axes":
        # keep aggregate/stats terms; drop explicit _X/_Y channels
        want_gaze_frame = [c for c in gaze_all if not (c.endswith("_X") or c.endswith("_Y") or "_X_" in c or "_Y_" in c)]
    else:
        raise ValueError(f"Unknown gaze_mode={gaze_mode}")

    want_gaze_vel = []  # IMPORTANT: gaze has no veldiff features in this NPZ

    # -----------------
    # Route modality -> (want_frame, want_vel)
    # -----------------
    def want_for(mod: str):
        if mod == "emoca":
            return want_emoca_frame, want_emoca_vel
        if mod == "bio":
            return want_bio_frame, want_bio_vel
        if mod == "gaze":
            return want_gaze_frame, want_gaze_vel
        raise ValueError(mod)

    want_a_frame, want_a_vel = want_for(modality_a)
    want_b_frame, want_b_vel = want_for(modality_b)

    # indices
    a_frame_idx = idx_of(fn, want_a_frame) if want_a_frame else []
    b_frame_idx = idx_of(fn, want_b_frame) if want_b_frame else []
    a_vel_idx   = idx_of(vn, want_a_vel)   if want_a_vel   else []
    b_vel_idx   = idx_of(vn, want_b_vel)   if want_b_vel   else []

    names_a_frame = [fn[i] for i in a_frame_idx]
    names_b_frame = [fn[i] for i in b_frame_idx]
    names_a_vel   = [vn[i] for i in a_vel_idx]
    names_b_vel   = [vn[i] for i in b_vel_idx]

    # sanity: gaze should appear only if a/b is gaze
    selected_all = names_a_frame + names_b_frame + names_a_vel + names_b_vel
    has_gaze = any(n.startswith("Gaze") for n in selected_all)
    if has_gaze and ("gaze" not in [modality_a, modality_b]):
        raise RuntimeError("Gaze features selected but neither modality_a nor modality_b is gaze.")

    print("\n[INPUT SELECTION]")
    print(f"  modality_a={modality_a} (emoca_mode={emoca_mode} bio_mode={bio_mode} gaze_mode={gaze_mode})")
    print(f"  modality_b={modality_b} (emoca_mode={emoca_mode} bio_mode={bio_mode} gaze_mode={gaze_mode})")
    print(f"  A frame: {len(names_a_frame)} | A vel: {len(names_a_vel)}")
    print(f"  B frame: {len(names_b_frame)} | B vel: {len(names_b_vel)}")

    return a_frame_idx, b_frame_idx, a_vel_idx, b_vel_idx, names_a_frame, names_b_frame, names_a_vel, names_b_vel





# ----------------------------
# Normalization (train-only)
# ----------------------------
class FeatureNormalizer:
    def __init__(self, mode: str = "none"):
        self.mode = mode
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, X: np.ndarray):
        # X: [N,F]
        if self.mode == "none":
            return
        Xf = X.astype(np.float64)
        if self.mode == "zscore":
            self.mean = np.nanmean(Xf, axis=0)
            self.std = np.nanstd(Xf, axis=0)
            self.std[self.std < 1e-12] = 1.0
        elif self.mode == "minmax":
            self.min = np.nanmin(Xf, axis=0)
            self.max = np.nanmax(Xf, axis=0)
            # avoid divide-by-zero
            span = self.max - self.min
            span[span < 1e-12] = 1.0
        else:
            raise ValueError(f"Unknown norm mode: {self.mode}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return X
        Xf = X.astype(np.float32, copy=True)
        if self.mode == "zscore":
            return (Xf - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        if self.mode == "minmax":
            span = (self.max - self.min).astype(np.float32)
            span[span < 1e-12] = 1.0
            return (Xf - self.min.astype(np.float32)) / span
        return Xf


# ----------------------------
# Dataset
# ----------------------------
class WindowTokenDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, idx_a: List[int], idx_b: List[int]):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.idx_a = np.array(idx_a, dtype=np.int64)
        self.idx_b = np.array(idx_b, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        xa = x[self.idx_a]  # [Fa]
        xb = x[self.idx_b]  # [Fb]
        return torch.from_numpy(xa), torch.from_numpy(xb), torch.tensor(self.y[i], dtype=torch.float32)

class WindowIntraSeqDataset(Dataset):
    """
    One sample = one window.
      xa: [Tw, Fa]  (Tw=270)
      xb: [Tw, Fb]
      y : scalar window label
    We repeat X_vel across Tw and concat to frame features.
    """
    def __init__(self, X_frame, X_vel, y, a_frame_idx, b_frame_idx, a_vel_idx, b_vel_idx):
        self.Xf = X_frame.astype(np.float32)   # [Nw, Tw, Ff]
        self.Xv = X_vel.astype(np.float32)     # [Nw, Fv]
        self.y  = y.astype(np.float32)

        self.aF = np.array(a_frame_idx, dtype=np.int64)
        self.bF = np.array(b_frame_idx, dtype=np.int64)
        self.aV = np.array(a_vel_idx, dtype=np.int64)
        self.bV = np.array(b_vel_idx, dtype=np.int64)

    def __len__(self):
        return self.Xf.shape[0]

    def __getitem__(self, i):
        xf = self.Xf[i]  # [Tw, Ff]
        xaf = xf[:, self.aF] if len(self.aF) else xf[:, :0]
        xbf = xf[:, self.bF] if len(self.bF) else xf[:, :0]

        Tw = self.Xf.shape[1]

        xav = self.Xv[i, self.aV] if len(self.aV) else self.Xv[i, :0]        # [FaV]
        xbv = self.Xv[i, self.bV] if len(self.bV) else self.Xv[i, :0]        # [FbV]

        # repeat vel across time
        if xav.size > 0:
            xav = np.repeat(xav[None, :], Tw, axis=0)  # [Tw, FaV]
        else:
            xav = np.zeros((Tw, 0), dtype=np.float32)

        if xbv.size > 0:
            xbv = np.repeat(xbv[None, :], Tw, axis=0)  # [Tw, FbV]
        else:
            xbv = np.zeros((Tw, 0), dtype=np.float32)

        xa = np.concatenate([xaf, xav], axis=1)        # [Tw, Fa]
        xb = np.concatenate([xbf, xbv], axis=1)        # [Tw, Fb]
        return torch.from_numpy(xa), torch.from_numpy(xb), torch.tensor(self.y[i], dtype=torch.float32)

class SubjectFrameVelNormalizer:
    """
    Per-subject z-score normalization.

    Frame stats computed over all (windows * time) for that subject.
    Vel stats computed over all windows for that subject.

    modes:
      - per_split : fit separately on (train), (val), (test) subjects using each split's data only
      - train_only: fit only on training subjects; unseen subjects use global train stats fallback
    """
    def __init__(self, mode: str = "per_split"):
        assert mode in ["per_split", "train_only"]
        self.mode = mode

        # dict: subject -> (f_mean, f_std, v_mean, v_std)
        self.subj_stats = {}

        # global fallback (train)
        self.f_mean_g = None
        self.f_std_g  = None
        self.v_mean_g = None
        self.v_std_g  = None

    @staticmethod
    def _safe_std(x: np.ndarray) -> np.ndarray:
        s = np.nanstd(x, axis=0)
        s[s < 1e-12] = 1.0
        return s

    def fit(self, Xf: np.ndarray, Xv: np.ndarray, subjects: np.ndarray):
        """
        Xf: [Nw, Tw, Ff]
        Xv: [Nw, Fv]
        subjects: [Nw] array-like of subject IDs aligned with Xf/Xv
        """
        subjects = np.asarray(subjects)

        # global fallback from THIS fit set
        Xf2 = Xf.reshape(-1, Xf.shape[-1]).astype(np.float64)
        Xv2 = Xv.astype(np.float64)
        self.f_mean_g = np.nanmean(Xf2, axis=0)
        self.f_std_g  = self._safe_std(Xf2)
        self.v_mean_g = np.nanmean(Xv2, axis=0)
        self.v_std_g  = self._safe_std(Xv2)

        self.subj_stats = {}

        for s in np.unique(subjects):
            m = (subjects == s)
            Xfs = Xf[m]  # [ns, Tw, Ff]
            Xvs = Xv[m]  # [ns, Fv]

            if Xfs.shape[0] == 0:
                continue

            Xfs2 = Xfs.reshape(-1, Xfs.shape[-1]).astype(np.float64)
            f_mean = np.nanmean(Xfs2, axis=0)
            f_std  = self._safe_std(Xfs2)

            v_mean = np.nanmean(Xvs.astype(np.float64), axis=0)
            v_std  = self._safe_std(Xvs.astype(np.float64))

            self.subj_stats[str(s)] = (
                f_mean.astype(np.float32),
                f_std.astype(np.float32),
                v_mean.astype(np.float32),
                v_std.astype(np.float32),
            )

    def transform(self, Xf: np.ndarray, Xv: np.ndarray, subjects: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies per-subject z-score using stored stats; falls back to global train stats if missing.
        """
        subjects = np.asarray(subjects)

        Xf_out = Xf.astype(np.float32, copy=True)
        Xv_out = Xv.astype(np.float32, copy=True)

        # fast path: loop over subjects in this split
        for s in np.unique(subjects):
            key = str(s)
            m = (subjects == s)
            if key in self.subj_stats:
                f_mean, f_std, v_mean, v_std = self.subj_stats[key]
            else:
                # unseen subject fallback
                f_mean, f_std, v_mean, v_std = (
                    self.f_mean_g.astype(np.float32),
                    self.f_std_g.astype(np.float32),
                    self.v_mean_g.astype(np.float32),
                    self.v_std_g.astype(np.float32),
                )

            # frame: [ns, Tw, Ff]
            Xf_out[m] = (Xf_out[m] - f_mean[None, None, :]) / f_std[None, None, :]
            # vel: [ns, Fv]
            Xv_out[m] = (Xv_out[m] - v_mean[None, :]) / v_std[None, :]

        return Xf_out, Xv_out

class FrameVelNormalizer:
    """
    Fit on TRAIN only.
    - frame stats computed over (N_train * Tw) samples per feature
    - vel stats computed over N_train per feature
    """
    def __init__(self, mode="none"):
        self.mode = mode
        self.f_mean = self.f_std = None
        self.v_mean = self.v_std = None
        self.f_min = self.f_max = None
        self.v_min = self.v_max = None

    def fit(self, Xf_train: np.ndarray, Xv_train: np.ndarray):
        if self.mode == "none":
            return

        # frame: [N, Tw, Ff] -> [N*Tw, Ff]
        Xf2 = Xf_train.reshape(-1, Xf_train.shape[-1]).astype(np.float64)
        Xv2 = Xv_train.astype(np.float64)

        if self.mode == "zscore":
            self.f_mean = np.nanmean(Xf2, axis=0); self.f_std = np.nanstd(Xf2, axis=0)
            self.v_mean = np.nanmean(Xv2, axis=0); self.v_std = np.nanstd(Xv2, axis=0)
            self.f_std[self.f_std < 1e-12] = 1.0
            self.v_std[self.v_std < 1e-12] = 1.0

        elif self.mode == "minmax":
            self.f_min = np.nanmin(Xf2, axis=0); self.f_max = np.nanmax(Xf2, axis=0)
            self.v_min = np.nanmin(Xv2, axis=0); self.v_max = np.nanmax(Xv2, axis=0)
            f_span = self.f_max - self.f_min; f_span[f_span < 1e-12] = 1.0
            v_span = self.v_max - self.v_min; v_span[v_span < 1e-12] = 1.0
        else:
            raise ValueError(f"Unknown norm mode: {self.mode}")

    def transform(self, Xf: np.ndarray, Xv: np.ndarray):
        if self.mode == "none":
            return Xf, Xv

        Xf_out = Xf.astype(np.float32, copy=True)
        Xv_out = Xv.astype(np.float32, copy=True)

        if self.mode == "zscore":
            Xf_out = (Xf_out - self.f_mean.astype(np.float32)) / self.f_std.astype(np.float32)
            Xv_out = (Xv_out - self.v_mean.astype(np.float32)) / self.v_std.astype(np.float32)
        elif self.mode == "minmax":
            f_span = (self.f_max - self.f_min).astype(np.float32); f_span[f_span < 1e-12] = 1.0
            v_span = (self.v_max - self.v_min).astype(np.float32); v_span[v_span < 1e-12] = 1.0
            Xf_out = (Xf_out - self.f_min.astype(np.float32)) / f_span
            Xv_out = (Xv_out - self.v_min.astype(np.float32)) / v_span

        return Xf_out, Xv_out


# # ----------------------------
# # Model blocks
# # ----------------------------
# class TokenModalityEncoder(nn.Module):
#     """
#     Input: scalar tokens [B, Ftokens]
#     -> value projection (1->E) + learned token embedding (Ftoks,E)
#     -> Transformer encoder
#     Output: [B, Ftokens, E]
#     """
#     def __init__(self, num_tokens: int, embed_dim: int, num_layers: int, nhead: int, dropout: float):
#         super().__init__()
#         self.num_tokens = num_tokens
#         self.val_proj = nn.Linear(1, embed_dim)
#         self.tok_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
#         nn.init.trunc_normal_(self.tok_emb, std=0.02)

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=nhead,
#             dim_feedforward=4 * embed_dim,
#             dropout=dropout,
#             batch_first=True,
#             activation="gelu",
#             norm_first=True,
#         )
#         self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

#     def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
#         # x_tokens: [B,F]
#         x = x_tokens.unsqueeze(-1)          # [B,F,1]
#         z = self.val_proj(x)                # [B,F,E]
#         z = z + self.tok_emb[:, :z.size(1)] # [B,F,E]
#         z = self.enc(z)                     # [B,F,E]
#         return z


# class CrossAttentionBlock(nn.Module):
#     def __init__(self, embed_dim: int, nhead: int, dropout: float):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
#         self.ln1 = nn.LayerNorm(embed_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, 4 * embed_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(4 * embed_dim, embed_dim),
#             nn.Dropout(dropout),
#         )
#         self.ln2 = nn.LayerNorm(embed_dim)

#     def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
#         h, _ = self.attn(query=q, key=kv, value=kv)
#         q = self.ln1(q + h)
#         q = self.ln2(q + self.ffn(q))
#         return q


# class AttnPool(nn.Module):
#     def __init__(self, embed_dim: int):
#         super().__init__()
#         self.score = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Tanh(),
#             nn.Linear(embed_dim, 1)
#         )

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         # z: [B,T,E] where T = num tokens
#         w = self.score(z).squeeze(-1)   # [B,T]
#         a = torch.softmax(w, dim=1)     # [B,T]
#         pooled = torch.einsum("bte,bt->be", z, a)
#         return pooled


# class CrossModalTokenNet(nn.Module):
#     def __init__(self, num_a: int, num_b: int,
#                  embed_dim: int = 128, enc_layers: int = 2, nhead: int = 4,
#                  xattn_layers: int = 1, dropout: float = 0.2):
#         super().__init__()

#         self.enc_a = TokenModalityEncoder(num_a, embed_dim, enc_layers, nhead, dropout)
#         self.enc_b = TokenModalityEncoder(num_b, embed_dim, enc_layers, nhead, dropout)

#         self.a_from_b = nn.ModuleList([CrossAttentionBlock(embed_dim, nhead, dropout) for _ in range(xattn_layers)])
#         self.b_from_a = nn.ModuleList([CrossAttentionBlock(embed_dim, nhead, dropout) for _ in range(xattn_layers)])

#         self.pool = AttnPool(embed_dim)

#         self.head = nn.Sequential(
#             nn.Linear(2 * embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim, 1)
#         )

#     def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
#         # xa: [B,Fa], xb: [B,Fb]
#         za = self.enc_a(xa)  # [B,Fa,E]
#         zb = self.enc_b(xb)  # [B,Fb,E]

#         # bidirectional cross attention (can stack)
#         for blk in self.a_from_b:
#             za = blk(za, zb)  # A <- B
#         for blk in self.b_from_a:
#             zb = blk(zb, za)  # B <- A

#         pa = self.pool(za)  # [B,E]
#         pb = self.pool(zb)  # [B,E]
#         feats = torch.cat([pa, pb], dim=-1)
#         logits = self.head(feats).squeeze(-1)
#         return logits

# ----------------------------
# Model
# ----------------------------

class TemporalDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.rand(x.shape[:2], device=x.device) > self.p
        return x * mask.unsqueeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):  # [B,T,E]
        pe = self.pe[:, :x.size(1), :].to(device=x.device, dtype=x.dtype)
        return x + pe

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
        self.b2 = DepthwiseConv1d(mid, mid, k=5)
        self.b3 = DepthwiseConv1d(mid, mid, k=7)
        self.out = nn.Conv1d(mid * 3, embed_dim, kernel_size=1)

    def forward(self, x):  # x: [B,T,F]
        x = x.permute(0, 2, 1)      # [B,F,T]
        x = self.proj(x)
        y = [self.b1(x), self.b2(x), self.b3(x)]
        x = torch.cat(y, dim=1)
        x = self.out(x).permute(0, 2, 1)  # [B,T,E]
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, y, key_padding_mask_y=None):
        h, _ = self.attn(query=x, key=y, value=y, key_padding_mask=key_padding_mask_y)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ffn(x))
        return x



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

# class ModalityEncoder(nn.Module):
#     def __init__(self, in_dim, embed_dim, num_layers=2, nhead=4, dropout=0.1, max_len=2000):
#         super().__init__()
#         self.stem = ConvStem(in_dim, embed_dim)
#         self.pos = PositionalEncoding(embed_dim, max_len)
#         enc = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=nhead,
#             dim_feedforward=4 * embed_dim,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

#     def forward(self, x, pad_mask=None):
#         z = self.stem(x)
#         z = self.pos(z)

#         if pad_mask is not None:
#             pad_mask = pad_mask.bool()

#         # robust: keep z in autocast dtype (prevents fp32 leakage)
#         if z.is_cuda and torch.is_autocast_enabled():
#             z = z.to(torch.float16)

#         z = self.enc(z, src_key_padding_mask=pad_mask)
#         return z

# class AttnPool(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.a = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

#     def forward(self, x, pad_mask=None):
#         # Compute logits in fp32 for numerical stability under AMP
#         w = self.a(x).squeeze(-1)  # [B,T]
#         w_fp32 = w.float()

#         if pad_mask is not None:
#             # fp16-safe mask value (use fp32 min since we softmax in fp32)
#             w_fp32 = w_fp32.masked_fill(pad_mask, torch.finfo(w_fp32.dtype).min)

#         a = torch.softmax(w_fp32, dim=1).to(dtype=x.dtype)  # cast back to match x dtype
#         pooled = torch.einsum("bte,bt->be", x, a)
#         return pooled, a
class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.a = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, pad_mask=None):
        # x: [B,T,E]
        w = self.a(x).squeeze(-1)          # [B,T]
        w_fp32 = w.float()

        if pad_mask is not None:
            w_fp32 = w_fp32.masked_fill(pad_mask, torch.finfo(w_fp32.dtype).min)

        a = torch.softmax(w_fp32, dim=1).to(dtype=x.dtype)
        pooled = torch.einsum("bte,bt->be", x, a)
        return pooled, a




class CrossModalStressNet(nn.Module):
    def __init__(self, dim_a, dim_b, embed_dim=128, enc_layers=2, nhead=4,xattn_layers=1, dropout=0.2, max_len=2000):
        super().__init__()
        self.enc_a = ModalityEncoder(dim_a, embed_dim, enc_layers, nhead, dropout, max_len)
        self.enc_b = ModalityEncoder(dim_b, embed_dim, enc_layers, nhead, dropout, max_len)

        # self.xattn_a_from_b = CrossAttentionBlock(embed_dim, nhead, dropout)  # A <- B
        # self.xattn_b_from_a = CrossAttentionBlock(embed_dim, nhead, dropout)  # B <- A

        self.xattn_a_from_b = nn.ModuleList([CrossAttentionBlock(embed_dim, nhead, dropout) for _ in range(xattn_layers)])
        self.xattn_b_from_a = nn.ModuleList([CrossAttentionBlock(embed_dim, nhead, dropout) for _ in range(xattn_layers)])


        self.pool = AttnPool(embed_dim)
        self.temp_drop = TemporalDropout(p=0.35)
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, 2* embed_dim),
            # nn.GELU(),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(2*embed_dim, 1),
        )

    def forward(self, xa, xb, mask_a=None, mask_b=None):
        za = self.enc_a(xa, mask_a)
        zb = self.enc_b(xb, mask_b)

        # stability: use originals for both directions
        za0, zb0 = za, zb
        # --- bidirectional cross-attention (stable) ---
        for blkA, blkB in zip(self.xattn_a_from_b, self.xattn_b_from_a):
            za_new = blkA(za, zb, key_padding_mask_y=mask_b)  # A <- B
            zb_new = blkB(zb, za, key_padding_mask_y=mask_a)  # B <- A
            za, zb = za_new, zb_new


        # za = self.xattn_a_from_b(za0, zb0, key_padding_mask_y=mask_b)
        # zb = self.xattn_b_from_a(zb0, za0, key_padding_mask_y=mask_a)

        pa, _ = self.pool(za, mask_a)
        pb, _ = self.pool(zb, mask_b)
        feats = torch.cat([pa, pb], dim=-1)
        return self.head(feats).squeeze(-1)
# ----------------------------
# Train / Eval
# ----------------------------
def run_one_epoch(model, loader, optimizer, device, loss_fn, scaler=None):
    model.train()
    total, n = 0.0, 0
    use_amp = (device.type == "cuda")

    for xa, xb, y in loader:
        xa = xa.to(device, non_blocking=True)  # [B,T,Fa]
        xb = xb.to(device, non_blocking=True)  # [B,T,Fb]
        y  = y.to(device, non_blocking=True)   # [B]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(True):
                logits = model(xa, xb)         # [B]
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xa, xb)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += float(loss.item()) * xa.size(0)
        n += xa.size(0)

    return total / max(n, 1)



# @torch.no_grad()
# def eval_model(model, loader, device) -> Dict[str, float]:
#     model.eval()
#     all_y = []
#     all_logits = []

#     for xa, xb, y in loader:
#         xa = xa.to(device, non_blocking=True)
#         xb = xb.to(device, non_blocking=True)
#         # fp32 eval for stability
#         logits = model(xa, xb)
#         all_y.append(y.detach().cpu().numpy())

#         all_logits.append(logits.detach().cpu().numpy())

#     y_true = np.concatenate(all_y) if all_y else np.array([])
#     logits = np.concatenate(all_logits) if all_logits else np.array([])
#     if len(y_true) == 0:
#         return {}
#     return compute_metrics(y_true, logits)
@torch.no_grad()
def eval_model(model, loader, device, thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    all_y, all_logits = [], []
    for xa, xb, y in loader:
        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        logits = model(xa, xb)
        all_y.append(y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.array([])
    logits = np.concatenate(all_logits) if all_logits else np.array([])
    if len(y_true) == 0:
        return {}
    return compute_metrics(y_true, logits, thr=thr)

def build_adamw(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "ln" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )





def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, default="/home/vivib/emoca/emoca/training/comparisons/FINAL_WINDOWSEQ_BASELINEv1.npz")
    ap.add_argument("--out", type=str, default="/home/vivib/emoca/emoca/training/comparisons/crossmodal_tokennet")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epochs", type=int, default=13)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-2)

    ap.add_argument("--embed_dim", type=int, default=96)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--xattn_layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--norm", type=str, default="none", choices=["none", "zscore", "minmax"])
    ap.add_argument("--bio_mode", type=str, default="all",
                    choices=["all", "perinasal", "heart", "breathing", "base_only", "veldiff_only"])

    ap.add_argument("--emoca_mode", type=str, default="all", choices=[ "all", "base", "delta1", "vel", "exp_only", "pose_only", "exp_base", "pose_base", "exp_delta1",
                 "pose_delta1","exp_vel", "pose_vel",])


    # ap.add_argument("--select_metric", type=str, default="AUPRC", choices=["AUROC", "AUPRC"])
    ap.add_argument("--select_metric", type=str, default="AUPRC",
                choices=["AUROC","AUPRC","ValAcc_best","ValBalAcc_best"])

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--max_gap", type=int, default=0)
    ap.add_argument("--subject_norm", type=str, default="none",
                choices=["none", "per_split", "train_only"],
                help="Per-subject z-score normalization. per_split is strongest; train_only is stricter.")


    ap.add_argument("--modality_a", type=str, default="emoca",
                choices=["emoca", "bio", "gaze"],
                help="First input modality (A).")
    ap.add_argument("--modality_b", type=str, default="bio",
                    choices=["emoca", "bio", "gaze"],
                    help="Second input modality (B). Must be different from A.")

    ap.add_argument("--gaze_mode", type=str, default="all",
                    choices=["all", "no_axes"],
                    help="Gaze feature selection (frame-only; no veldiff in this NPZ).")




    args = ap.parse_args()

    set_seed(args.seed)
    setup_cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    X_frame = data["X_frame"].astype(np.float32)           # [Nw, Tw, Ff]  (Tw=270)
    X_vel   = data["X_vel"].astype(np.float32)             # [Nw, Fv]
    y       = data["y_window"].astype(np.float32).reshape(-1)  # [Nw]

    frame_feature_names = to_str_list(data["feature_names_frame"])
    vel_feature_names   = to_str_list(data["feature_names_vel"])


    subjects = np.array(to_str_list(data["subject_id"]), dtype=object)
    phases = np.array(to_str_list(data["phase"]), dtype=object)
    window_index = data["window_index"].astype(np.int64)

    Nw, Tw, Ff = X_frame.shape
    Nv, Fv = X_vel.shape
    assert Nv == Nw, "X_vel and X_frame must have same number of windows"

    print(f"[NPZ] X_frame={X_frame.shape} X_vel={X_vel.shape} y={y.shape} #subjects={len(np.unique(subjects))}")
    print(f"[NPZ] Tw={Tw} F_frame={Ff} F_vel={Fv}")

    # ---- select features ONCE (global, same for all folds) ----
    # aF_idx, bF_idx, aV_idx, bV_idx, names_aF, names_bF, names_aV, names_bV = select_indices_frame_vel(
    #     frame_feature_names, vel_feature_names,
    #     bio_mode=args.bio_mode,
    #     emoca_mode=args.emoca_mode
    # )

    aF_idx, bF_idx, aV_idx, bV_idx, names_aF, names_bF, names_aV, names_bV = select_indices_frame_vel(
    frame_feature_names, vel_feature_names,
    bio_mode=args.bio_mode,
    emoca_mode=args.emoca_mode,
    gaze_mode=args.gaze_mode,
    modality_a=args.modality_a,
    modality_b=args.modality_b,
    )


    Fa = len(aF_idx) + len(aV_idx)
    Fb = len(bF_idx) + len(bV_idx)
    print(f"[DIMS] Fa={Fa} (frame {len(aF_idx)} + vel {len(aV_idx)}) | Fb={Fb} (frame {len(bF_idx)} + vel {len(bV_idx)})")


    # splitter = GroupKFold(n_splits=5)
    



    fold_metrics = []




    def make_loader(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )


    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros((Nw, 1)), (y >= 0.5).astype(int), groups=subjects),
        start=1
    ):

    # for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros((Nw, 1)), y, groups=subjects), start=1):
        fold_dir = out_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # simple subject-wise val split from train subjects
        train_subjects = np.unique(subjects[train_idx])
        rng = np.random.RandomState(args.seed + fold)
        rng.shuffle(train_subjects)
        n_val = max(1, int(0.3 * len(train_subjects)))
        val_subjects = train_subjects[:n_val]
        tr_subjects = train_subjects[n_val:]

        tr_idx = train_idx[np.isin(subjects[train_idx], tr_subjects)]
        va_idx = train_idx[np.isin(subjects[train_idx], val_subjects)]


                # --- normalization ---
        if args.subject_norm == "none":
            # your existing global train-only normalization
            norm = FrameVelNormalizer(args.norm)
            norm.fit(X_frame[tr_idx], X_vel[tr_idx])

            Xf_tr, Xv_tr = norm.transform(X_frame[tr_idx], X_vel[tr_idx])
            Xf_va, Xv_va = norm.transform(X_frame[va_idx], X_vel[va_idx])
            Xf_te, Xv_te = norm.transform(X_frame[test_idx], X_vel[test_idx])

        elif args.subject_norm == "train_only":
            # per-subject stats learned ONLY on training subjects
            sn = SubjectFrameVelNormalizer(mode="train_only")
            sn.fit(X_frame[tr_idx], X_vel[tr_idx], subjects[tr_idx])

            Xf_tr, Xv_tr = sn.transform(X_frame[tr_idx], X_vel[tr_idx], subjects[tr_idx])
            Xf_va, Xv_va = sn.transform(X_frame[va_idx], X_vel[va_idx], subjects[va_idx])
            Xf_te, Xv_te = sn.transform(X_frame[test_idx], X_vel[test_idx], subjects[test_idx])

        elif args.subject_norm == "per_split":
            # compute per-subject stats separately for each split (no labels used)
            sn_tr = SubjectFrameVelNormalizer(mode="per_split")
            sn_tr.fit(X_frame[tr_idx], X_vel[tr_idx], subjects[tr_idx])
            Xf_tr, Xv_tr = sn_tr.transform(X_frame[tr_idx], X_vel[tr_idx], subjects[tr_idx])

            sn_va = SubjectFrameVelNormalizer(mode="per_split")
            sn_va.fit(X_frame[va_idx], X_vel[va_idx], subjects[va_idx])
            Xf_va, Xv_va = sn_va.transform(X_frame[va_idx], X_vel[va_idx], subjects[va_idx])

            sn_te = SubjectFrameVelNormalizer(mode="per_split")
            sn_te.fit(X_frame[test_idx], X_vel[test_idx], subjects[test_idx])
            Xf_te, Xv_te = sn_te.transform(X_frame[test_idx], X_vel[test_idx], subjects[test_idx])

        else:
            raise ValueError(args.subject_norm)



        # # --- normalization (fit on TRAIN only) ---
        # norm = FrameVelNormalizer(args.norm)
        # norm.fit(X_frame[tr_idx], X_vel[tr_idx])

        # Xf_tr, Xv_tr = norm.transform(X_frame[tr_idx], X_vel[tr_idx])
        # Xf_va, Xv_va = norm.transform(X_frame[va_idx], X_vel[va_idx])
        # Xf_te, Xv_te = norm.transform(X_frame[test_idx], X_vel[test_idx])


        train_ds = WindowIntraSeqDataset(Xf_tr, Xv_tr, y[tr_idx], aF_idx, bF_idx, aV_idx, bV_idx)
        val_ds   = WindowIntraSeqDataset(Xf_va, Xv_va, y[va_idx], aF_idx, bF_idx, aV_idx, bV_idx)
        test_ds  = WindowIntraSeqDataset(Xf_te, Xv_te, y[test_idx], aF_idx, bF_idx, aV_idx, bV_idx)

        def label_stats(name, idx):
            yy = y[idx].astype(np.float32)
            print(f"{name}: n={len(yy)} mean={yy.mean():.3f} "
                f"bin_pos={(yy>=0.5).mean():.3f} "
                f"min={yy.min():.3f} max={yy.max():.3f}")

        label_stats("TRAIN y_window", tr_idx)
        label_stats("VAL   y_window", va_idx)
        label_stats("TEST  y_window", test_idx)

        print("TEST subjects:", sorted(np.unique(subjects[test_idx]).tolist()))





                # ---- SEQUENCE-level class balance (what the model actually sees) ----
        tr_frac, tr_pos, tr_n = dataset_pos_frac(train_ds)
        va_frac, va_pos, va_n = dataset_pos_frac(val_ds)
        te_frac, te_pos, te_n = dataset_pos_frac(test_ds)

        print(f"[Fold {fold}] SEQ LABELS: "
            f"train pos={tr_pos}/{tr_n} ({tr_frac:.3f}) | "
            f"val pos={va_pos}/{va_n} ({va_frac:.3f}) | "
            f"test pos={te_pos}/{te_n} ({te_frac:.3f})")



        # def pr(name, idx):
        #     yy = y[idx].astype(int)
        #     print(f"{name}: n={len(idx)} pos={yy.sum()} rate={yy.mean():.6f}")

        # pr("TRAIN", tr_idx)
        # pr("VAL  ", va_idx)
        # pr("TEST ", test_idx)
        # pr("ALL  ", np.arange(len(y)))




        train_loader = make_loader(train_ds, shuffle=True)
        val_loader   = make_loader(val_ds, shuffle=False)
        test_loader  = make_loader(test_ds, shuffle=False)

        ytr = np.array([train_ds[i][2].item() for i in range(len(train_ds))], dtype=np.float32)
        pos = float(tr_pos)
        neg = float(tr_n - tr_pos)
        pos_weight = torch.tensor([neg / (pos + 1e-9)], device=device)
        loss_fn = nn.BCEWithLogitsLoss()   ################### PLAIN BCE

        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



        # # pos_weight from training labels
        # ytr_bin = (y[tr_idx] >= 0.5).astype(np.float32)
        # pos = float((ytr_bin == 1).sum())
        # neg = float((ytr_bin == 0).sum())
        # pos_weight = torch.tensor([neg / (pos + 1e-9)], device=device)
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        print(f"\n[Fold {fold}] train_subs={len(tr_subjects)} val_subs={len(val_subjects)} test_subs={len(np.unique(subjects[test_idx]))}")
        print(f"[Fold {fold}] windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | pos_rate_train={pos/(pos+neg+1e-9):.3f}")
        print(f"[Fold {fold}] dims: Fa={Fa} Fb={Fb} | norm={args.norm} | subject_norm={args.subject_norm} "f"| A={args.modality_a} B={args.modality_b} | emoca_mode={args.emoca_mode} bio_mode={args.bio_mode} gaze_mode={args.gaze_mode}")



        model = CrossModalStressNet(
            dim_a=Fa,
            dim_b=Fb,
            embed_dim=args.embed_dim,
            enc_layers=args.enc_layers,
            nhead=args.nhead,
            xattn_layers=args.xattn_layers,
            dropout=args.dropout,
            max_len=Tw
        ).to(device)



        optimizer = build_adamw(model, lr=args.lr, weight_decay=args.weight_decay)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_score = -1.0
        best_path = fold_dir / "best.pt"

        for epoch in range(1, args.epochs + 1):
            tr_loss = run_one_epoch(model, train_loader, optimizer, device, loss_fn, scaler=scaler)

            # raw logits on val for threshold tuning
            model.eval()
            vy, vlogits = [], []
            for xa, xb, yb in val_loader:
                xa = xa.to(device, non_blocking=True)
                xb = xb.to(device, non_blocking=True)
                lg = model(xa, xb)
                vy.append(yb.numpy())
                vlogits.append(lg.detach().cpu().numpy())
            vy = np.concatenate(vy).astype(np.int32)
            # vy = (np.concatenate(vy) >= 0.5).astype(np.int32)

            vlogits = np.concatenate(vlogits)
            vprobs = sigmoid_np(vlogits.astype(np.float64))

            best_thr_acc, best_acc = best_threshold_for_metric(vy, vprobs, metric="Acc")
            best_thr_bacc, best_bacc = best_threshold_for_metric(vy, vprobs, metric="BalAcc")
            best_thr_f1, best_f1 = best_threshold_for_metric(vy, vprobs, metric="F1")

            # best_thr, best_acc = best_threshold_for_metric(vy, vprobs, metric="Acc")
            # best_thr_b, best_bacc = best_threshold_for_metric(vy, vprobs, metric="BalAcc")

            val_m = compute_metrics(vy, vlogits, thr=best_thr_acc)
            val_m["best_thr_acc"] = best_thr_acc
            val_m["best_thr_bacc"] = best_thr_bacc
            val_m["best_thr_f1"] = best_thr_f1
            val_m["ValAcc_best"] = best_acc
            val_m["ValBalAcc_best"] = best_bacc
            val_m["ValF1_best"] = best_f1





            # val_m = eval_model(model, val_loader, device)
            sel = val_m.get(args.select_metric, float("nan"))

            print(f"[Fold {fold} | Epoch {epoch:03d}] loss={tr_loss:.4f} | "
                  f"VAL AUROC={val_m.get('AUROC', float('nan')):.3f} "
                  f"AUPRC={val_m.get('AUPRC', float('nan')):.3f} "
                  f"F1_max={val_m.get('F1_max', float('nan')):.3f} "
                  f"Acc={val_m.get('Acc', float('nan')):.3f} "
                  f"BalAcc={val_m.get('BalAcc', float('nan')):.3f}")
        


            if np.isfinite(sel) and sel > best_score:
                best_score = float(sel)
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "names_a_frame": names_aF,
                    "names_a_vel": names_aV,
                    "names_b_frame": names_bF,
                    "names_b_vel": names_bV,
                    "a_frame_idx": aF_idx,
                    "a_vel_idx": aV_idx,
                    "b_frame_idx": bF_idx,
                    "b_vel_idx": bV_idx,
                    "norm": args.norm,
                    "bio_mode": args.bio_mode,
                    "emoca_mode": args.emoca_mode,
                    "gaze_mode": args.gaze_mode,
                    "modality_a": args.modality_a,
                    "modality_b": args.modality_b,
                    "best_thr_acc": float(best_thr_acc),
                    "best_thr_bacc": float(best_thr_bacc),
                    "best_thr_f1": float(best_thr_f1),



                }, best_path)

                print(f"  ↳ saved best (by {args.select_metric}) → {best_path}")

        # test with best
        # ckpt = torch.load(best_path, map_location=device)
        # model.load_state_dict(ckpt["model"], strict=True)
        # test_m = eval_model(model, test_loader, device)
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

        thr_acc  = float(ckpt.get("best_thr_acc", 0.5))
        thr_bacc = float(ckpt.get("best_thr_bacc", 0.5))
        thr_f1   = float(ckpt.get("best_thr_f1", 0.5))

        test_acc  = eval_model(model, test_loader, device, thr=thr_acc)
        test_bacc = eval_model(model, test_loader, device, thr=thr_bacc)
        test_f1   = eval_model(model, test_loader, device, thr=thr_f1)

        # print(f"[Fold {fold}] TEST(thr_acc={thr_acc:.3f})  Acc={test_acc['Acc']:.3f} BalAcc={test_acc['BalAcc']:.3f} F1={test_acc['F1']:.3f}")
        # print(f"[Fold {fold}] TEST(thr_bacc={thr_bacc:.3f}) Acc={test_bacc['Acc']:.3f} BalAcc={test_bacc['BalAcc']:.3f} F1={test_bacc['F1']:.3f}")
        # print(f"[Fold {fold}] TEST(thr_f1={thr_f1:.3f})   Acc={test_f1['Acc']:.3f} BalAcc={test_f1['BalAcc']:.3f} F1={test_f1['F1']:.3f}")

        # and pick ONE for reporting (usually thr_acc or thr_bacc):
        test_m = eval_model(model, test_loader, device, thr=thr_bacc)

            
        fold_metrics.append(test_m)

        with open(fold_dir / "test_metrics.json", "w") as f:
            json.dump(test_m, f, indent=2)

        print(f"[Fold {fold}] TEST AUROC={test_m.get('AUROC', float('nan')):.3f} "
              f"AUPRC={test_m.get('AUPRC', float('nan')):.3f} "
              f"F1_max={test_m.get('F1_max', float('nan')):.3f} "
              f"Acc={test_m.get('Acc', float('nan')):.3f} "
              f"BalAcc={test_m.get('BalAcc', float('nan')):.3f}")

    # CV summary
    def mean_std(key: str) -> Tuple[float, float]:
        vals = np.array([m.get(key, float("nan")) for m in fold_metrics], dtype=np.float32)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    summary = {}
    for k in ["AUROC", "AUPRC", "F1_max", "F1", "Acc", "BalAcc"]:
        mu, sd = mean_std(k)
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd

    print("\n[CV SUMMARY] (TEST)")
    for k in ["AUROC", "AUPRC", "F1_max", "F1", "Acc", "BalAcc"]:
        print(f"  {k}: {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_std']:.4f}")

    with open(out_dir / "cv_summary_test.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
