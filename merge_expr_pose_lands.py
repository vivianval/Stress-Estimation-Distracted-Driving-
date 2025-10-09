# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import argparse, sys, re, csv, os, time
# from pathlib import Path
# from tqdm import tqdm
# import numpy as np

# def find_subject_dirs(root: Path):
#     root = root.expanduser().resolve()
#     if not root.exists():
#         raise FileNotFoundError(f"Root not found: {root}")
#     cand = [d for d in root.iterdir() if d.is_dir() and d.name.upper().startswith("T")]
#     if cand:
#         return sorted(cand)
#     subs = []
#     for d in root.iterdir():
#         if d.is_dir():
#             subs.extend([x for x in d.iterdir() if x.is_dir() and x.name.upper().startswith("T")])
#     return sorted(subs)

# def iter_frame_dirs(subject_dir: Path):
#     return sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("FRAME_")])

# _frame_re = re.compile(r"^FRAME_(\d+)_\d+$")
# def parse_frame_id(frame_dir: Path):
#     m = _frame_re.match(frame_dir.name)
#     if m: return int(m.group(1))
#     s = frame_dir.name.replace("FRAME_", "").split("_", 1)[0]
#     try: return int(s.lstrip("0") or "0")
#     except: return frame_dir.name

# def load_np(path: Path, mmap=True):
#     if not path.exists():
#         return None
#     try:
#         return np.load(str(path), allow_pickle=False, mmap_mode=("r" if mmap else None))
#     except Exception as e:
#         print(f"[WARN] Could not load {path}: {e}", file=sys.stderr)
#         return None

# def infer_landmark_len(frame_dirs, max_bytes: int):
#     for fd in frame_dirs:
#         p = fd / "landmarks.npy"
#         if not p.exists(): 
#             continue
#         try:
#             if p.stat().st_size > max_bytes:
#                 print(f"[WARN] {p} too large ({p.stat().st_size} bytes); skipping for schema.", file=sys.stderr)
#                 continue
#             arr = load_np(p, mmap=True)
#             if arr is not None and arr.size > 0:
#                 return int(np.asarray(arr).size)
#         except Exception as e:
#             print(f"[WARN] Inspect {p}: {e}", file=sys.stderr)
#     return 0

# def export_subject_dual_csv(subject_dir: Path, out_dir: Path, chunk_rows: int,
#                             fsync_every:int=0, lmk_max_bytes:int=4*1024*1024,
#                             write_landmarks: bool=True):
#     subj = subject_dir.name
#     fdirs = iter_frame_dirs(subject_dir)
#     if not fdirs:
#         print(f"[SKIP] No FRAME_* in {subject_dir}")
#         return None

#     # Columns
#     exp_cols  = [f"exp_{i:02d}"  for i in range(50)]
#     pose_cols = [f"pose_{i:02d}" for i in range(6)]
#     lmk_len = infer_landmark_len(fdirs, lmk_max_bytes) if write_landmarks else 0
#     lmk_cols = [f"lmk_{i:03d}" for i in range(lmk_len)] if lmk_len>0 else []

#     out_dir.mkdir(parents=True, exist_ok=True)
#     # Files
#     exp_pose_csv = out_dir / f"{subj}_exp_pose.csv"
#     lmk_csv      = out_dir / f"{subj}_landmarks.csv" if write_landmarks and lmk_len>0 else None
#     tmp_ep = exp_pose_csv.with_suffix(".csv.tmp")
#     tmp_lk = lmk_csv.with_suffix(".csv.tmp") if lmk_csv else None

#     # Writers
#     f_ep = tmp_ep.open("w", newline="")
#     w_ep = csv.writer(f_ep)
#     w_ep.writerow(["subject_id","frame_id"] + exp_cols + pose_cols)

#     if lmk_csv:
#         f_lk = tmp_lk.open("w", newline="")
#         w_lk = csv.writer(f_lk)
#         w_lk.writerow(["subject_id","frame_id"] + lmk_cols)
#     else:
#         f_lk = w_lk = None

#     buf_ep, buf_lk = [], []
#     flush_count_ep = flush_count_lk = 0

#     def flush_ep(force=False):
#         nonlocal buf_ep, flush_count_ep
#         if buf_ep:
#             w_ep.writerows(buf_ep); buf_ep = []
#             f_ep.flush(); flush_count_ep += 1
#             if fsync_every and (force or flush_count_ep % fsync_every == 0):
#                 os.fsync(f_ep.fileno())

#     def flush_lk(force=False):
#         nonlocal buf_lk, flush_count_lk
#         if f_lk is None or not buf_lk: return
#         w_lk.writerows(buf_lk); buf_lk = []
#         f_lk.flush(); flush_count_lk += 1
#         if fsync_every and (force or flush_count_lk % fsync_every == 0):
#             os.fsync(f_lk.fileno())

#     last_ping = time.time()
#     for i, fd in enumerate(tqdm(fdirs, desc=subj, unit="frame")):
#         exp  = load_np(fd / "exp.npy",  mmap=True)
#         pose = load_np(fd / "pose.npy", mmap=True)
#         if exp is None or pose is None:
#             continue
#         exp  = np.asarray(exp).reshape(-1)
#         pose = np.asarray(pose).reshape(-1)
#         if exp.size != 50:
#             print(f"[WARN] {fd}/exp.npy has {exp.size} elems (expected 50); writing anyway.", file=sys.stderr)
#         if pose.size != 6:
#             print(f"[WARN] {fd}/pose.npy has {pose.size} elems (expected 6); writing anyway.", file=sys.stderr)

#         frame_id = parse_frame_id(fd)
#         buf_ep.append([subj, frame_id] + exp.tolist() + pose.tolist())

#         if lmk_csv:
#             lp = fd / "landmarks.npy"
#             if lp.exists() and lp.stat().st_size <= lmk_max_bytes:
#                 lmk = load_np(lp, mmap=True)
#                 if lmk is not None and lmk.size > 0:
#                     lmk_flat = np.asarray(lmk).reshape(-1)
#                     # pad/trim to schema length
#                     if lmk_flat.size < lmk_len:
#                         lmk_flat = np.pad(lmk_flat, (0, lmk_len - lmk_flat.size))
#                     elif lmk_flat.size > lmk_len:
#                         lmk_flat = lmk_flat[:lmk_len]
#                     buf_lk.append([subj, frame_id] + lmk_flat.tolist())
#                 else:
#                     # write blanks for missing
#                     buf_lk.append([subj, frame_id] + ([""]*lmk_len))
#             else:
#                 buf_lk.append([subj, frame_id] + ([""]*lmk_len))

#         if len(buf_ep) >= chunk_rows: flush_ep()
#         if lmk_csv and len(buf_lk) >= chunk_rows: flush_lk()

#         if time.time() - last_ping > 10:
#             print(f"[{subj}] processed ~{i+1}/{len(fdirs)} frames")
#             last_ping = time.time()

#     # Final flush & close
#     flush_ep(force=True); f_ep.close(); tmp_ep.replace(exp_pose_csv)
#     if lmk_csv:
#         flush_lk(force=True); f_lk.close(); tmp_lk.replace(lmk_csv)

#     print(f"[OK] Wrote {exp_pose_csv}" + (f" and {lmk_csv}" if lmk_csv else ""))
#     return exp_pose_csv, lmk_csv

# def main():
#     ap = argparse.ArgumentParser(description="Export EMOCA → two CSVs per subject: exp+pose and landmarks.")
#     ap.add_argument("--root", required=True, help=".../emoca_results/EMOCA_v2_lr_mse_20")
#     ap.add_argument("--out_dir", default="emoca_csv_dual", help="Output directory")
#     ap.add_argument("--subjects", nargs="*", default=None, help="e.g., T001 T010 (default: all)")
#     ap.add_argument("--chunk_rows", type=int, default=5000, help="Rows per flush (default 5000)")
#     ap.add_argument("--fsync_every", type=int, default=0, help="0 disables fsync (fastest).")
#     ap.add_argument("--lmk_max_bytes", type=int, default=4*1024*1024, help="Skip huge landmarks files (> this).")
#     ap.add_argument("--no_landmarks", action="store_true", help="Do NOT write landmarks CSV.")
#     args = ap.parse_args()

#     root = Path(args.root).expanduser().resolve()
#     out_dir = Path(args.out_dir).expanduser().resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     subjects = find_subject_dirs(root)
#     if args.subjects:
#         want = set(s.upper() for s in args.subjects)
#         subjects = [d for d in subjects if d.name.upper() in want]
#         if not subjects:
#             print(f"[ERR] no requested subjects found under {root}", file=sys.stderr)
#             sys.exit(1)

#     for sdir in subjects:
#         try:
#             export_subject_dual_csv(
#                 sdir, out_dir, chunk_rows=args.chunk_rows,
#                 fsync_every=args.fsync_every,
#                 lmk_max_bytes=args.lmk_max_bytes,
#                 write_landmarks=(not args.no_landmarks),
#             )
#         except KeyboardInterrupt:
#             print("\n[INTERRUPTED] Stopping cleanly.", file=sys.stderr)
#             break
#         except Exception as e:
#             print(f"[ERR] {sdir.name}: {e}", file=sys.stderr)

#     print("[DONE]")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys, re, csv, os, time
from pathlib import Path
from tqdm import tqdm
import numpy as np

def find_subject_dirs(root: Path):
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    cand = [d for d in root.iterdir() if d.is_dir() and d.name.upper().startswith("T")]
    if cand: return sorted(cand)
    subs = []
    for d in root.iterdir():
        if d.is_dir():
            subs.extend([x for x in d.iterdir() if x.is_dir() and x.name.upper().startswith("T")])
    return sorted(subs)

def iter_frame_dirs(subject_dir: Path):
    return sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("FRAME_")])

_frame_re = re.compile(r"^FRAME_(\d+)_\d+$")
def parse_frame_id(frame_dir: Path):
    m = _frame_re.match(frame_dir.name)
    if m: return int(m.group(1))
    s = frame_dir.name.replace("FRAME_", "").split("_", 1)[0]
    try: return int(s.lstrip("0") or "0")
    except: return frame_dir.name

def load_np(path: Path, mmap=True):
    if not path.exists(): return None
    try:
        return np.load(str(path), allow_pickle=False, mmap_mode=("r" if mmap else None))
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}", file=sys.stderr)
        return None

def infer_points_2d(frame_dirs, filename: str, max_bytes: int):
    """Return number of points if file looks like (N,2) or can be inferred; else 0."""
    for fd in frame_dirs:
        p = fd / filename
        if not p.exists(): continue
        try:
            if p.stat().st_size > max_bytes:
                print(f"[WARN] {p} too large ({p.stat().st_size} bytes); skipping for schema.", file=sys.stderr)
                continue
            arr = load_np(p, mmap=True)
            if arr is None or arr.size == 0: continue
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape[1] == 2:
                return int(a.shape[0])
            # fallback: try to infer from size
            if a.size % 2 == 0:
                return int(a.size // 2)
        except Exception as e:
            print(f"[WARN] Inspect {p}: {e}", file=sys.stderr)
    return 0

def export_subject_landmarks_xy(
    subject_dir: Path, out_dir: Path, chunk_rows: int,
    lmk68_name: str, lmkmp_name: str, lmk_max_bytes:int=4*1024*1024,
    fsync_every:int=0
):
    subj = subject_dir.name
    fdirs = iter_frame_dirs(subject_dir)
    if not fdirs:
        print(f"[SKIP] No FRAME_* in {subject_dir}")
        return None

    n68 = infer_points_2d(fdirs, lmk68_name, lmk_max_bytes)
    nmp = infer_points_2d(fdirs, lmkmp_name, lmk_max_bytes)
    if n68 == 0 and nmp == 0:
        print(f"[SKIP] No {lmk68_name} or {lmkmp_name} found under {subject_dir}")
        return None

    # Column headers: paired x/y per point
    l68_cols = [c for i in range(1, n68+1) for c in (f"l68_x{i:03d}", f"l68_y{i:03d}")]
    mp_cols  = [c for i in range(1, nmp+1) for c in (f"mp_x{i:03d}",  f"mp_y{i:03d}")]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{subj}_landmarks_xy.csv"
    tmp_out = out_csv.with_suffix(".csv.tmp")

    f_out = tmp_out.open("w", newline="")
    w_out = csv.writer(f_out)
    w_out.writerow(["subject_id","frame_id"] + l68_cols + mp_cols)

    buf = []
    flush_count = 0
    def flush(force=False):
        nonlocal buf, flush_count
        if not buf: return
        w_out.writerows(buf); buf = []
        f_out.flush(); flush_count += 1
        if fsync_every and (force or flush_count % fsync_every == 0):
            os.fsync(f_out.fileno())

    last_ping = time.time()
    N = len(fdirs)
    for i, fd in enumerate(tqdm(fdirs, desc=subj, unit="frame")):
        frame_id = parse_frame_id(fd)

        # Prepare blanks
        l68_vals = [""] * (2*n68)
        mp_vals  = [""] * (2*nmp)

        # Load 68
        p68 = fd / lmk68_name
        if n68 > 0 and p68.exists() and p68.stat().st_size <= lmk_max_bytes:
            a = load_np(p68, mmap=True)
            if a is not None and a.size >= 2:
                a = np.asarray(a)
                if a.ndim == 1:
                    a = a.reshape(-1, 2)
                if a.shape[1] != 2:
                    # Try to coerce if someone saved flat
                    a = a.reshape(-1, 2)
                pts = a[:n68, :]
                if pts.shape[0] < n68:
                    pad = np.full((n68 - pts.shape[0], 2), np.nan)
                    pts = np.vstack([pts, pad])
                l68_vals = pts.reshape(-1).tolist()

        # Load MP
        pmp = fd / lmkmp_name
        if nmp > 0 and pmp.exists() and pmp.stat().st_size <= lmk_max_bytes:
            a = load_np(pmp, mmap=True)
            if a is not None and a.size >= 2:
                a = np.asarray(a)
                if a.ndim == 1:
                    a = a.reshape(-1, 2)
                if a.shape[1] != 2:
                    a = a.reshape(-1, 2)
                pts = a[:nmp, :]
                if pts.shape[0] < nmp:
                    pad = np.full((nmp - pts.shape[0], 2), np.nan)
                    pts = np.vstack([pts, pad])
                mp_vals = pts.reshape(-1).tolist()

        buf.append([subj, frame_id] + l68_vals + mp_vals)
        if len(buf) >= chunk_rows: flush()

        if time.time() - last_ping > 10:
            print(f"[{subj}] processed ~{i+1}/{N} frames")
            last_ping = time.time()

    flush(force=True)
    f_out.close()
    tmp_out.replace(out_csv)
    print(f"[OK] Wrote {out_csv}")
    return out_csv

def main():
    ap = argparse.ArgumentParser(description="Export 2D landmarks (68 + Mediapipe) with paired x/y columns per subject.")
    ap.add_argument("--root", required=True, help=".../emoca_results/EMOCA_v2_lr_mse_20")
    ap.add_argument("--out_dir", default="emoca_csv_landmarks_xy", help="Output directory")
    ap.add_argument("--subjects", nargs="*", default=None, help="e.g., T001 T010 (default: all)")
    ap.add_argument("--chunk_rows", type=int, default=5000, help="Rows per flush")
    ap.add_argument("--fsync_every", type=int, default=0, help="0 disables fsync (fastest).")
    ap.add_argument("--lmk_max_bytes", type=int, default=4*1024*1024, help="Skip files larger than this.")
    ap.add_argument("--lmk68_name", default="landmarks68.npy", help="Filename for 68-pt landmarks")
    ap.add_argument("--lmkmp_name", default="landmarks105_mp.npy", help="Filename for Mediapipe 105 landmarks")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_dirs(root)
    if args.subjects:
        want = set(s.upper() for s in args.subjects)
        subjects = [d for d in subjects if d.name.upper() in want]
        if not subjects:
            print(f"[ERR] no requested subjects found under {root}", file=sys.stderr)
            sys.exit(1)

    for sdir in subjects:
        try:
            export_subject_landmarks_xy(
                sdir, out_dir, chunk_rows=args.chunk_rows,
                lmk68_name=args.lmk68_name, lmkmp_name=args.lmkmp_name,
                lmk_max_bytes=args.lmk_max_bytes, fsync_every=args.fsync_every
            )
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopping cleanly.", file=sys.stderr)
            break
        except Exception as e:
            print(f"[ERR] {sdir.name}: {e}", file=sys.stderr)
    print("[DONE]")

if __name__ == "__main__":
    main()
