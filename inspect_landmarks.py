# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import argparse, sys, re, time
# from pathlib import Path
# import numpy as np

# RE = re.compile(r"^FRAME_(\d+)_\d+$")

# def parse_frame_id(name: str):
#     m = RE.match(name)
#     if not m: return None
#     return int(m.group(1))

# def load_lmk(p: Path):
#     try:
#         return np.load(str(p), allow_pickle=False, mmap_mode="r")
#     except Exception as e:
#         return f"LOAD_ERROR: {e}"

# def main():
#     ap = argparse.ArgumentParser(description="Inspect EMOCA landmarks.npy for a subject.")
#     ap.add_argument("--subject_dir", required=True,
#                     help="e.g. /home/.../emoca_results/EMOCA_v2_lr_mse_20/T001")
#     ap.add_argument("--max_bytes", type=int, default=4*1024*1024,
#                     help="Flag files larger than this (default 4MB).")
#     ap.add_argument("--samples", type=int, default=3,
#                     help="Show up to this many example frames with landmarks (default 3).")
#     args = ap.parse_args()

#     subj_dir = Path(args.subject_dir).expanduser().resolve()
#     if not subj_dir.exists():
#         print(f"[ERR] not found: {subj_dir}", file=sys.stderr); sys.exit(1)

#     frames = sorted([d for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith("FRAME_")],
#                     key=lambda d: parse_frame_id(d.name) or 10**12)
#     if not frames:
#         print("[ERR] no FRAME_* folders found"); sys.exit(1)

#     total = 0
#     have = 0
#     sizes = []
#     shapes = {}
#     dtypes = {}
#     bad = []   # (frame, reason)
#     nan_frames = []
#     huge = []
#     samples_left = args.samples

#     print(f"[INFO] scanning {len(frames)} frames in {subj_dir.name} ...")
#     t0 = time.time()

#     for fd in frames:
#         total += 1
#         p = fd / "landmarks.npy"
#         if not p.exists():
#             bad.append((fd.name, "MISSING"))
#             continue

#         sz = p.stat().st_size
#         if sz > args.max_bytes:
#             huge.append((fd.name, sz))

#         arr = load_lmk(p)
#         if isinstance(arr, str):
#             bad.append((fd.name, arr))
#             continue

#         a = np.asarray(arr)
#         shapes[a.shape] = shapes.get(a.shape, 0) + 1
#         dtypes[str(a.dtype)] = dtypes.get(str(a.dtype), 0) + 1
#         sizes.append(sz)
#         have += 1

#         if np.isnan(a).any() or np.isinf(a).any():
#             nan_frames.append(fd.name)

#         # show a few examples
#         if samples_left > 0:
#             flat = a.reshape(-1)
#             print(f"\n[EXAMPLE] {fd.name}")
#             print(f"  path     : {p}")
#             print(f"  file size: {sz/1024:.1f} KiB")
#             print(f"  dtype    : {a.dtype}, shape: {a.shape}, flat_len: {flat.size}")
#             print(f"  min/max  : {np.nanmin(flat):.4f} / {np.nanmax(flat):.4f}")
#             print(f"  first 12 : {flat[:12]}")
#             samples_left -= 1

#     dt = time.time() - t0
#     print("\n=== SUMMARY ===")
#     print(f"subject          : {subj_dir.name}")
#     print(f"frames scanned   : {total}")
#     print(f"with landmarks   : {have}")
#     print(f"missing/corrupt  : {len(bad)}")
#     print(f"NaN/inf frames   : {len(nan_frames)}")
#     print(f"shapes           : {shapes}")
#     print(f"dtypes           : {dtypes}")
#     if sizes:
#         print(f"size (KiB) min/avg/max : {min(sizes)/1024:.1f} / {sum(sizes)/len(sizes)/1024:.1f} / {max(sizes)/1024:.1f}")
#     if huge:
#         huge_sorted = sorted(huge, key=lambda x: -x[1])[:10]
#         print("\nTop large files (> threshold):")
#         for name, b in huge_sorted:
#             print(f"  {name}: {b/1024:.1f} KiB")
#     if bad:
#         print("\nBad/missing examples:")
#         for name, rsn in bad[:10]:
#             print(f"  {name}: {rsn}")
#     if nan_frames:
#         print("\nFrames with NaN/Inf (first 10):")
#         for name in nan_frames[:10]:
#             print(f"  {name}")

#     print(f"\n[INFO] done in {dt:.1f}s")

# if __name__ == "__main__":
#     main()
# import numpy as np, glob, os
# base="/home/vivib/emoca/emoca/emoca_results/EMOCA_v2_lr_mse_20/T018"
# f68=sorted(glob.glob(os.path.join(base,"FRAME_*","landmarks68.npy")))
# fvis=sorted(glob.glob(os.path.join(base,"FRAME_*","landmarks68_vis.npy")))
# f105=sorted(glob.glob(os.path.join(base,"FRAME_*","landmarks105_mp.npy")))
# print("68xy files:", len(f68), " | 68vis:", len(fvis), " | 105mp:", len(f105))
# if f68:  print("sample 68:",  np.load(f68[0]).shape)
# if fvis: print("sample vis:", np.load(fvis[0]).shape)
# if f105: print("sample 105:", np.load(f105[0]).shape)


import pandas as pd, sys
import numpy as np
p= '/home/vivib/emoca/emoca/emoca_results/CSV_STREAM/T017_exp_pose.csv'
p2 = '/home/vivib/emoca/emoca/emoca_results/CSV_STREAM/lands/T079_landmarks_xy.csv'
df=pd.read_csv(p, nrows=1)
df2 = pd.read_csv(p2, nrows=5)
print(df.columns.tolist())

# import pandas as pd, numpy as np
# p="/home/vivib/emoca/emoca/emoca_results/CSV_STREAM/lands/T079_landmarks_xy.csv"
# df=pd.read_csv(p, nrows=5)
xcols=[c for c in df2.columns if c.startswith(("l68_x","mp_x"))]
ycols=[c for c in df2.columns if c.startswith(("l68_y","mp_y"))]
xs=df2[xcols].to_numpy().astype(float); ys=df2[ycols].to_numpy().astype(float)
print("x min/max:", np.nanmin(xs), np.nanmax(xs))
print("y min/max:", np.nanmin(ys), np.nanmax(ys))
print(df2.columns.tolist())


# #!/usr/bin/env python3
# import os, sys, math, json, pickle, argparse
# from pathlib import Path
# import numpy as np
# import cv2

# # Rendering
# import trimesh
# import pyrender

# # Optional FLAME via chumpy loader
# try:
#     from smpl_webuser.serialization import load_model as load_flame_chumpy
#     HAS_CHUMPY_FLAME = True
# except Exception:
#     HAS_CHUMPY_FLAME = False

# # ------------- helpers -------------
# def log(s): print(f"[vis] {s}")

# def load_coeffs_any(path):
#     import pandas as pd, re
#     df = pd.read_csv(path)

#     # exp_00..exp_49  → (T, 50)
#     exp_cols = sorted([c for c in df.columns if re.fullmatch(r"exp_\d{2}", c)],
#                       key=lambda s: int(s.split("_")[1]))
#     exp = df[exp_cols].to_numpy(dtype=np.float32) if exp_cols else None

#     # pose_00..pose_05 → (T, 6) : global(3) + jaw(3)
#     pose_cols = sorted([c for c in df.columns if re.fullmatch(r"pose_\d{2}", c)],
#                        key=lambda s: int(s.split("_")[1]))
#     pose = df[pose_cols].to_numpy(dtype=np.float32) if pose_cols else None

#     T = len(df)
#     cam = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32)[None, :], (T, 1))
#     shape = None  # not in your CSV

#     return {"shape": shape, "exp": exp, "pose": pose, "cam": cam, "T": T}


#     # Normalize to arrays per key we care about, per-frame
#     # Expected keys (varies with EMOCA setup):
#     # - shape (betas) [T, n_shape] or [n_shape] for constant
#     # - exp / expression [T, n_exp]
#     # - pose [T, n_pose] (axis-angle chunks of 3)
#     # - cam / camera [T, 3] -> (s, tx, ty) weak perspective
#     # Sometimes they come stacked in one big vector per frame -> try to parse by name or prefix.

#     def get_series(key_options):
#         for k in key_options:
#             if k in data and isinstance(data[k], np.ndarray):
#                 return data[k]
#             if "__df__" in data:
#                 # columns like 'shape[0]' or 'shape_0'
#                 df = data["__df__"]
#                 cols = [c for c in df.columns if c.startswith(k+"[") or c.startswith(k+"_")]
#                 if cols:
#                     arr = df[cols].to_numpy()
#                     return arr
#         return None

#     shaped = {
#         "shape": get_series(["shape", "betas", "shape_params"]),
#         "exp":   get_series(["exp", "expression", "expcode", "expr"]),
#         "pose":  get_series(["pose", "posecode", "pose_params"]),
#         "cam":   get_series(["cam", "camera", "cam_params"]),
#     }

#     # Expand 1D constants to T-long
#     T = None
#     for v in shaped.values():
#         if v is not None:
#             T = v.shape[0] if v.ndim >= 2 else T
#     if T is None and "__df__" in data:
#         T = len(data["__df__"])
#     if T is None:
#         raise ValueError("Could not infer number of frames from coeffs.")

#     for k, v in list(shaped.items()):
#         if v is None: 
#             continue
#         v = np.asarray(v)
#         if v.ndim == 1:
#             v = np.tile(v[None, :], (T,1))
#         shaped[k] = v

#     # Some dumps store cam as [s, tx, ty] per frame; ensure shape (T,3)
#     if shaped["cam"] is not None:
#         cam = shaped["cam"]
#         if cam.shape[1] >= 3:
#             shaped["cam"] = cam[:, :3]
#         else:
#             # fill default s=1, tx=0, ty=0
#             tmp = np.zeros((T,3), dtype=np.float32)
#             tmp[:,0] = 1.0
#             tmp[:,:cam.shape[1]] = cam
#             shaped["cam"] = tmp
#     else:
#         tmp = np.zeros((T,3), dtype=np.float32)
#         tmp[:,0] = 1.0
#         shaped["cam"] = tmp

#     shaped["T"] = T
#     return shaped

# def load_landmarks_any(path, T_expected=None):
#     p = Path(path)
#     ext = p.suffix.lower()
#     lmks = None
#     if ext == ".npy":
#         lmks = np.load(p, allow_pickle=True)
#     elif ext == ".pkl":
#         with open(p, "rb") as f:
#             obj = pickle.load(f)
#         # Accept: list of arrays, dict with 'landmarks', or array
#         if isinstance(obj, dict) and "landmarks" in obj:
#             lmks = np.asarray(obj["landmarks"])
#         elif isinstance(obj, list):
#             lmks = np.asarray(obj)
#         else:
#             lmks = np.asarray(obj)
#     elif ext == ".csv":
#         import pandas as pd
#         df = pd.read_csv(p)
#         # expect columns like lmk_0_x, lmk_0_y, ..., lmk_67_x, lmk_67_y
#         xs = [c for c in df.columns if c.endswith("_x")]
#         ys = [c for c in df.columns if c.endswith("_y")]
#         xs = sorted(xs, key=lambda s: int(s.split("_")[1]))
#         ys = sorted(ys, key=lambda s: int(s.split("_")[1]))
#         X = df[xs].to_numpy()
#         Y = df[ys].to_numpy()
#         T = len(df)
#         lmks = np.stack([np.stack([X[t], Y[t]], axis=1) for t in range(T)], axis=0)
#     else:
#         raise ValueError(f"Unknown landmarks extension: {ext}")

#     lmks = np.asarray(lmks)
#     if lmks.ndim == 2 and lmks.shape[0] == 68 and lmks.shape[1] >= 2:
#         # single frame -> expand
#         lmks = lmks[None, :, :2]
#     elif lmks.ndim == 3:
#         lmks = lmks[:, :68, :2]  # (T,68,2)
#     else:
#         raise ValueError(f"Unexpected landmarks shape: {lmks.shape}")

#     if T_expected is not None and len(lmks) != T_expected:
#         log(f"WARNING: landmarks T={len(lmks)} but coeffs T={T_expected}. Will clamp to min.")
#         Tm = min(len(lmks), T_expected)
#         lmks = lmks[:Tm]
#     return lmks

# def weak_persp_project(X3, cam):
#     """
#     X3: (N,3) vertices in (x,y,z)
#     cam: (3,) [s, tx, ty]
#     returns (N,2)
#     """
#     s, tx, ty = cam
#     x2 = X3[:, :2] + np.array([tx, ty])[None, :]
#     return s * x2
# def load_coeffs_any(path):
#     import pandas as pd, re
#     p = Path(path)
#     ext = p.suffix.lower()
#     if ext != ".csv":
#         raise ValueError("For your files, use the CSV loader. Got: "+ext)

#     df = pd.read_csv(p)

#     # Collect exp_00..exp_49
#     exp_cols = sorted([c for c in df.columns if re.fullmatch(r"exp_\d{2}", c)],
#                       key=lambda s: int(s.split("_")[1]))
#     exp = df[exp_cols].to_numpy(dtype=np.float32) if exp_cols else None

#     # Collect pose_00..pose_05
#     pose_cols = sorted([c for c in df.columns if re.fullmatch(r"pose_\d{2}", c)],
#                        key=lambda s: int(s.split("_")[1]))
#     pose = df[pose_cols].to_numpy(dtype=np.float32) if pose_cols else None

#     # No camera columns in your CSV → default cam = [1,0,0]
#     T = len(df)
#     cam = np.tile(np.array([1.0,0.0,0.0], dtype=np.float32)[None,:], (T,1))

#     # (Optional) shape is not present → None (neutral identity)
#     shape = None

#     return {"shape": shape, "exp": exp, "pose": pose, "cam": cam, "T": T}
# def load_landmarks_any(path, T_expected=None, pane_w=None, pane_h=None):
#     import pandas as pd, re
#     p = Path(path)
#     if p.suffix.lower() != ".csv":
#         raise ValueError("Use the CSV landmarks loader for your file.")

#     df = pd.read_csv(p)

#     # Prefer l68_* columns and ignore mp_*.
#     # Build pairs (x_i, y_i) for i=1..68 (zero-padded in your file).
#     def col(i, xy):
#         return f"l68_{xy}{i:03d}"

#     missing = []
#     x_cols = []; y_cols = []
#     for i in range(1, 69):
#         cx, cy = col(i, "x"), col(i, "y")
#         if cx not in df.columns or cy not in df.columns:
#             missing.append((cx, cy))
#         x_cols.append(cx); y_cols.append(cy)

#     if missing:
#         raise ValueError(f"Landmark columns missing: {missing[:3]}{' ...' if len(missing)>3 else ''}")

#     X = df[x_cols].to_numpy(dtype=np.float32)
#     Y = df[y_cols].to_numpy(dtype=np.float32)
#     T = len(df)

#     # Stack to (T,68,2)
#     lmks = np.stack([X, Y], axis=-1)  # (T,68,2)

#     # Auto-detect normalized coords in [-1,1] and map to pixel space if we know target pane size
#     # (Your earlier message showed ~[-0.73, 0.78] etc., so this kicks in.)
#     if pane_w is not None and pane_h is not None:
#         x_min, x_max = float(np.nanmin(lmks[...,0])), float(np.nanmax(lmks[...,0]))
#         y_min, y_max = float(np.nanmin(lmks[...,1])), float(np.nanmax(lmks[...,1]))
#         if -1.5 <= x_min <= 1.5 and -1.5 <= x_max <= 1.5 and -1.5 <= y_min <= 1.5 and -1.5 <= y_max <= 1.5:
#             # map [-1,1] → [0,W] / [0,H]
#             lmks[...,0] = (lmks[...,0] * 0.5 + 0.5) * float(pane_w)
#             lmks[...,1] = (lmks[...,1] * 0.5 + 0.5) * float(pane_h)

#     if T_expected is not None and T != T_expected:
#         Tm = min(T, T_expected)
#         lmks = lmks[:Tm]

#     return lmks

# def to_trimesh(vertices, faces):
#     return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# # --- put near other imports
# import numpy as np, cv2
# try:
#     import pyrender
#     _HAVE_PYRENDER = True
# except Exception:
#     _HAVE_PYRENDER = False

# # try to import Open3D rendering
# try:
#     import open3d as o3d
#     from open3d.visualization import rendering as o3dr
#     _HAVE_O3D = True
# except Exception:
#     _HAVE_O3D = False

# class RenderBackend:
#     def __init__(self, w, h):
#         self.w, self.h = w, h
#         self.mode = None
#         self.scene = None
#         self.mesh_node = None
#         if _HAVE_PYRENDER:
#             # try EGL first
#             import os
#             os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
#             try:
#                 self.scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.3,0.3,0.3,1.0])
#                 self.camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(50.0))
#                 self.cam_node = pyrender.Node(camera=self.camera, matrix=np.eye(4))
#                 self.scene.add_node(self.cam_node)
#                 self.scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4))
#                 self.renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
#                 self.mode = "pyrender"
#             except Exception as e:
#                 self.scene = None

#         if self.scene is None and _HAVE_O3D:
#             # Open3D offscreen fallback
#             self.renderer = o3dr.OffscreenRenderer(w, h)
#             self.scene = self.renderer.scene
#             self.scene.set_background([0,0,0,0])
#             self.scene.scene.set_sun_light([1,1,1], 45000, 100000)  # some light
#             self.scene.scene.enable_sun_light(True)
#             self.mode = "open3d"
#         if self.scene is None:
#             raise RuntimeError("No headless renderer available (pyrender EGL and Open3D both unavailable).")

#     def set_camera_front(self, z=800.0):
#         if self.mode == "pyrender":
#             T = np.eye(4)
#             T[2,3] = z
#             self.scene.set_pose(self.cam_node, pose=T)
#         else:
#             cam = o3dr.Camera()
#             cam.set_projection(60.0/180.0*np.pi, self.w/self.h, 0.1, 5000.0, o3dr.FovType.Vertical)
#             center = [0,0,0]; eye = [0,0,z]; up=[0,1,0]
#             self.renderer.setup_camera(cam, center, eye, up)

#     def update_mesh(self, vertices, faces, color=(0.75,0.75,0.8)):
#         if self.mode == "pyrender":
#             import trimesh
#             m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
#             m.visual.vertex_colors = (np.array(color)*255).astype(np.uint8)
#             if self.mesh_node is not None:
#                 self.scene.remove_node(self.mesh_node)
#             self.mesh_node = self.scene.add(pyrender.Mesh.from_trimesh(m, smooth=True))
#         else:
#             # Open3D
#             mesh = o3d.geometry.TriangleMesh(
#                 o3d.utility.Vector3dVector(vertices),
#                 o3d.utility.Vector3iVector(faces)
#             )
#             mesh.compute_vertex_normals()
#             mat = o3dr.MaterialRecord()
#             mat.shader = "defaultLit"
#             mat.base_color = (*color, 1.0)
#             if self.mesh_node is not None:
#                 self.scene.remove(self.mesh_node)
#             self.mesh_node = "mesh"
#             self.scene.add(self.mesh_node, mesh, mat)

#     def render_bgr(self):
#         if self.mode == "pyrender":
#             rgba, _ = self.renderer.render(self.scene)
#             return cv2.cvtColor(rgba[:,:,:3], cv2.COLOR_RGB2BGR)
#         else:
#             img = self.renderer.render_to_image()
#             img = np.asarray(img)  # RGBA or RGB
#             if img.ndim==3 and img.shape[2]==4:
#                 img = img[:,:,:3]
#             return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# def colorize(gray=(0.75,0.75,0.8)):
#     return np.array(gray, dtype=np.float32)

# # ------------- FLAME wrapper -------------
# class FlameWrapper:
#     def __init__(self, flame_pkl=None):
#         self.ok = False
#         self.faces = None
#         self.n_shape = 300
#         self.n_exp = 100
#         self.pose_len = 3*15  # global+neck+jaw+eyes, adjust as needed

#         if flame_pkl and HAS_CHUMPY_FLAME and Path(flame_pkl).exists():
#             log(f"Loading FLAME from {flame_pkl}")
#             self.m = load_flame_chumpy(str(flame_pkl))
#             # try to get faces
#             self.faces = np.asarray(self.m.f)
#             self.ok = True
#         else:
#             self.m = None
#             log("FLAME not available; will render a fallback sphere mesh.")

#     def vertices(self, shape, exp, pose):
#         if not self.ok:
#             sphere = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
#             return np.asarray(sphere.vertices), np.asarray(sphere.faces)

#         m = self.m

#         # --- shape ---
#         try:
#             if shape is not None:
#                 shape_full = np.zeros_like(m.betas.r)  # same length as model
#                 n = min(len(shape_full), len(shape))
#                 shape_full[:n] = shape[:n]
#                 m.betas[:] = shape_full
#             else:
#                 m.betas[:] = 0
#         except Exception:
#             pass

#         # --- expression ---
#         try:
#             if exp is not None:
#                 expr_full = np.zeros_like(m.expr.r)    # pad to model size (e.g., 100)
#                 n = min(len(expr_full), len(exp))
#                 expr_full[:n] = exp[:n]
#                 m.expr[:] = expr_full
#             else:
#                 m.expr[:] = 0
#         except Exception:
#             pass

#         # --- pose (axis-angle, pad missing parts with zeros) ---
#         try:
#             pose_full = np.zeros_like(m.pose.r)
#             if pose is not None:
#                 n = min(len(pose_full), len(pose))
#                 pose_full[:n] = pose[:n]
#             m.pose[:] = pose_full
#         except Exception:
#             pass

#         v = np.asarray(m.r)
#         f = np.asarray(self.faces)
#         return v, f


# # ------------- main pipeline -------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--video", required=True, type=str)
#     ap.add_argument("--coeffs", required=True, type=str,
#                    help="EMOCA coeffs: .npz / .pkl / .csv")
#     ap.add_argument("--landmarks", required=True, type=str,
#                    help="68x2 per-frame: .npy / .pkl / .csv")
#     ap.add_argument("--flame_pkl", default="", type=str,
#                    help="Path to FLAME chumpy .pkl (generic_model.pkl)")
#     ap.add_argument("--out", required=True, type=str)
#     ap.add_argument("--fps", type=float, default=0)
#     ap.add_argument("--w", type=int, default=512, help="pane width (each side)")
#     ap.add_argument("--h", type=int, default=512, help="pane height (each side)")
#     ap.add_argument("--draw_lmks", action="store_true", help="also draw 68 on render (projected)")
#     args = ap.parse_args()

#     coeffs = load_coeffs_any(args.coeffs)
#     print("[vis] coeffs:", 
#       "exp" if coeffs.get("exp") is not None else "NO exp",
#       "pose" if coeffs.get("pose") is not None else "NO pose",
#       "T=", coeffs["T"])

#     lmks = load_landmarks_any(args.landmarks, T_expected=coeffs["T"], pane_w=args.w, pane_h=args.h)
#     T = min(coeffs["T"], len(lmks))
#     shape = coeffs.get("shape", None)
#     exp = coeffs.get("exp", None)
#     pose = coeffs.get("pose", None)
#     cam  = coeffs.get("cam",  None)

#     if shape is None or exp is None:
#         log("WARNING: Missing shape/exp in coeffs. Will drive FLAME with zeros for missing terms.")
#     if pose is None:
#         log("WARNING: Missing pose in coeffs. Using zeros.")
#     if cam is None:
#         cam = np.tile(np.array([1.0,0.0,0.0])[None,:], (T,1))

#     # Open video
#     cap = cv2.VideoCapture(args.video)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {args.video}")
#     in_fps = cap.get(cv2.CAP_PROP_FPS)
#     fps = args.fps if args.fps > 0 else (in_fps if in_fps > 0 else 25)

#     # Output writer
#     out_w, out_h = args.w*2, args.h
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     out_path = os.path.splitext(args.out)[0] + ".avi"
#     os.makedirs(Path(args.out).parent, exist_ok=True)
#     #writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
#     if not writer.isOpened():
#         raise RuntimeError(f"OpenCV could not open VideoWriter for {out_path}. Try MJPG/.avi or use imageio-ffmpeg.")

#     # FLAME
#     flame = FlameWrapper(args.flame_pkl)

#     # Renderer
#     renderer = RenderBackend(args.w, args.h)
    
#     mesh_node = None

#     # Main loop
#     t = 0
#     while t < T:
#         ok, frame = cap.read()
#         if not ok: break

#         # Resize original pane
#         frame_show = cv2.resize(frame, (args.w, args.h), interpolation=cv2.INTER_AREA)

#         # Draw 68 landmarks on original
#         lm = lmks[t]  # (68,2)
#         for (x,y) in lm.astype(int):
#             cv2.circle(frame_show, (x,y), 2, (0,255,0), -1)

#         # Get FLAME params for this frame
#         sh = shape[t] if shape is not None else np.zeros(300, dtype=np.float32)
#         ex = exp[t]   if exp   is not None else np.zeros(100, dtype=np.float32)
#         po = pose[t]  if pose  is not None else np.zeros(45,  dtype=np.float32)  # 15*3 default
#         ca = cam[t]   if cam   is not None else np.array([1.0,0.0,0.0], dtype=np.float32)

#         # Mesh vertices/faces

#         V, F = flame.vertices(sh, ex, po)
#         Vc = V - V.mean(axis=0, keepdims=True)
#         scale = 1.2 * max(args.w, args.h) / (np.ptp(Vc[:,0]) + 1e-6)
#         Vn = Vc * scale

#         renderer.update_mesh(Vn, F)
#         renderer.set_camera_front(z=800.0)
#         render_bgr = renderer.render_bgr()
#         # V, F = flame.vertices(sh, ex, po)

#         # Center mesh and scale roughly to view
#         # Vc = V - V.mean(axis=0, keepdims=True)
#         # scale = 1.2 * max(args.w, args.h) / (np.ptp(Vc[:,0]) + 1e-6)
#         # Vn = Vc * scale
#         # mesh = to_trimesh(Vn, F)
#         # mesh.visual.vertex_colors = (colorize()*255).astype(np.uint8)

#         # # Update scene
#         # if mesh_node is not None:
#         #     scene.remove_node(mesh_node)
#         # mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

#         # renderer.update_mesh(Vn, F)
#         # renderer.set_camera_front(z=800.0)
#         # render_bgr = renderer.render_bgr()


#         # # Camera pose: simple front look-at
#         # cam_pose = np.eye(4)
#         # cam_pose[2,3] = 800.0  # pull camera back
#         # scene.set_pose(cam_node, pose=cam_pose)

#         # # Render
#         # render_rgba, _ = renderer.render(scene)
#         # render_bgr = cv2.cvtColor(render_rgba[:,:,:3], cv2.COLOR_RGB2BGR)

#         # Optionally draw projected landmarks on the render pane using weak-persp
#         if args.draw_lmks:
#             # Take a subset of vertices as pseudo-landmarks? Here, use the provided 2D landmarks projected by cam.
#             # If you have FLAME 3D landmark vertices, replace this with those and project via weak_persp_project.
#             # For now, just draw the same 2D points (they will not align to the rendered mesh unless cam/mesh match).
#             for (x,y) in lm.astype(int):
#                 if 0 <= x < args.w and 0 <= y < args.h:
#                     cv2.circle(render_bgr, (x,y), 2, (0,255,255), -1)

#         # Compose side-by-side
#         canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
#         canvas[:, :args.w] = frame_show
#         canvas[:, args.w:] = cv2.resize(render_bgr, (args.w, args.h), interpolation=cv2.INTER_AREA)

#         # small HUD
#         cv2.putText(canvas, f"t={t}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

#         writer.write(canvas)
#         t += 1

#     writer.release()
#     cap.release()
#     log(f"Done. Wrote: {args.out}")

# if __name__ == "__main__":
#     main()
