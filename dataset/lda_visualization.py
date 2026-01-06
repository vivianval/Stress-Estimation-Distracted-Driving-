# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# LDA stress direction visualization (supervisor version):

# - NO neutral / mean face shown.
# - 2x1 grid:
#     top:  -3σ along LDA direction
#     bottom: +3σ along LDA direction
# - Includes BOTH expressions and pose by default.

# We interpret the LDA direction weights w_j in standardized coefficient space.
# We visualize a ±3σ move along the LDA axis in original EMOCA coefficient space:
#     Δx_j = (3 * σ_j) * w_j
# where σ_j is the empirical std of feature j in the dataset.

# Robust rendering fixes:
# - Fixed camera (set once from neutral mesh)
# - Clean EMOCA->FLAME mapping: global pose -> pose[0:3], jaw -> pose[6:9]
# - Pose clamping to avoid extreme global rotations

# Outputs:
#   - LDA_stress_minus_plus_grid.png  (2 rows: -3σ, +3σ)
#   - (optional) LDA_stress_plus_heatmap.png  (vertex displacement vs neutral)
# """

# from pathlib import Path
# import os
# import numpy as np
# import pandas as pd
# import cv2

# # optional but helpful in headless environments
# os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
# os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)

# import open3d as o3d
# from open3d.visualization import rendering as o3dr
# from smpl_webuser.serialization import load_model

# # =========================
# # CONFIG
# # =========================
# CSV_PATH = Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/phased_csvs/MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN.csv")

# LDA_DIR = Path("/home/vivib/emoca/emoca/LDA")
# LDA_DIRECTION_CSV = LDA_DIR / "lda_direction_stress.csv"

# FLAME_MODEL = Path("/home/vivib/emoca/emoca/assets/FLAME/geometry/generic_model.pkl")

# OUT_DIR = Path("/home/vivib/emoca/emoca/feature_visualization/lda_stress_direction_grid")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# IMAGE_W, IMAGE_H = 600, 600
# N_EXP = 50
# N_POSE = 6

# N_SIGMA = 3.0          # ±3σ
# AMPLIFY = 4.0          # keep 1.0 unless you need visibility
# INCLUDE_POSE = True    # supervisor wants pose included

# # Pose safety clamps (radians)
# POSE_GLOBAL_MAX_RAD = 10.0 #0.60   # clamp global pose dims (pose_00..02)
# POSE_JAW_MAX_RAD    = 10.0 #0.80   # clamp jaw pose dims (pose_03..05)
# JAW_GAIN = 1.0

# # Optional: displacement heatmap for +3σ (can be misleading if global pose dominates)
# SAVE_HEATMAP = True

# # =========================
# # FLAME + renderer helpers
# # =========================
# def build_o3d_mesh(vertices, faces, vertex_colors=None):
#     m = o3d.geometry.TriangleMesh()
#     m.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
#     m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
#     m.compute_vertex_normals()

#     if vertex_colors is not None:
#         m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=np.float64))

#     # dummy uvs/material ids
#     ntri = np.asarray(m.triangles).shape[0]
#     uvs = np.tile(np.array([[0, 0], [1, 0], [0, 1]], np.float32), (ntri, 1))
#     m.triangle_uvs = o3d.utility.Vector2dVector(uvs)
#     m.triangle_material_ids = o3d.utility.IntVector([0] * ntri)
#     return m


# class HeadlessRenderer:
#     """Fixed-camera offscreen renderer."""
#     def __init__(self, width, height, bg_rgba=(1, 1, 1, 1)):
#         self.r = o3dr.OffscreenRenderer(int(width), int(height))
#         self.scene = self.r.scene
#         self.scene.set_background(bg_rgba)

#         self.mat = o3dr.MaterialRecord()
#         self.mat.shader = "defaultLit"
#         self.mat.base_color = (0.9, 0.9, 0.9, 1.0)

#         try:
#             self.scene.scene.set_sun_light([0.0, 0.5, 0.8], [1, 1, 1], 25000, True)
#             self.scene.scene.enable_sun_light(True)
#         except Exception:
#             pass

#         self.name = "mesh"
#         self._cam_set = False
#         self._cam_center = None
#         self._cam_eye = None
#         self._cam_up = np.array([0, 1, 0], dtype=np.float64)

#     def set_camera_from_mesh(self, mesh, extra=1.7):
#         bbox = mesh.get_axis_aligned_bounding_box()
#         c = bbox.get_center()
#         extent = bbox.get_extent()
#         radius = float(np.linalg.norm(extent)) * 0.5
#         if radius <= 1e-6:
#             radius = 1.0
#         eye = c + np.array([0.0, 0.0, extra * radius])
#         self._cam_center = c
#         self._cam_eye = eye
#         self._cam_set = True
#         self.r.setup_camera(60.0, c, eye, self._cam_up)

#     def render(self, mesh):
#         try:
#             self.scene.remove_geometry(self.name)
#         except Exception:
#             pass
#         self.scene.add_geometry(self.name, mesh, self.mat)

#         if self._cam_set:
#             self.r.setup_camera(60.0, self._cam_center, self._cam_eye, self._cam_up)

#         img = self.r.render_to_image()
#         return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# def set_flame_from_emoca(model, expr50, pose6, jaw_gain=1.0):
#     """Clean EMOCA->FLAME mapping: global -> pose[0:3], jaw -> pose[6:9]."""
#     model.betas[:] = 0
#     model.betas[300:350] = np.asarray(expr50, np.float64)

#     p = np.asarray(pose6, np.float64).reshape(-1)
#     model.pose[:] = 0
#     model.pose[0:3] = p[0:3]
#     model.pose[6:9] = jaw_gain * p[3:6]


# def clamp_pose(pose6):
#     p = np.asarray(pose6, np.float64).reshape(-1).copy()
#     if p.size >= 3:
#         p[0:3] = np.clip(p[0:3], -POSE_GLOBAL_MAX_RAD, POSE_GLOBAL_MAX_RAD)
#     if p.size >= 6:
#         p[3:6] = np.clip(p[3:6], -POSE_JAW_MAX_RAD, POSE_JAW_MAX_RAD)
#     return p


# def displacement_to_colors(disp_norm):
#     """Simple blue->red map (0 blue, 1 red)."""
#     disp_norm = np.clip(disp_norm, 0.0, 1.0)
#     r = disp_norm
#     g = np.zeros_like(disp_norm)
#     b = 1.0 - disp_norm
#     return np.stack([r, g, b], axis=-1)


# # =========================
# # MAIN
# # =========================
# def main():
#     print(f"[Load] CSV: {CSV_PATH}")
#     df = pd.read_csv(CSV_PATH, low_memory=False)

#     exp_cols = [f"exp_{i:02d}" for i in range(N_EXP)]
#     pose_cols = [f"pose_{i:02d}" for i in range(N_POSE)]
#     for c in exp_cols + pose_cols:
#         if c not in df.columns:
#             raise RuntimeError(f"Missing column in CSV: {c}")

#     # empirical std per feature
#     exp_std = df[exp_cols].std().to_numpy()
#     pose_std = df[pose_cols].std().to_numpy()

#     print(f"[Load] LDA direction: {LDA_DIRECTION_CSV}")
#     df_dir = pd.read_csv(LDA_DIRECTION_CSV)

#     # LDA weights in standardized space
#     exp_dir = np.zeros(N_EXP, dtype=np.float64)
#     pose_dir = np.zeros(N_POSE, dtype=np.float64)

#     for _, row in df_dir.iterrows():
#         feat = str(row["feature"])
#         w = float(row["weight"])
#         if feat.startswith("exp_"):
#             idx = int(feat.split("_")[1])
#             if 0 <= idx < N_EXP:
#                 exp_dir[idx] = w
#         elif feat.startswith("pose_"):
#             idx = int(feat.split("_")[1])
#             if 0 <= idx < N_POSE:
#                 pose_dir[idx] = w

#     if not INCLUDE_POSE:
#         pose_dir[:] = 0.0

#     # normalize direction vector
#     full_dir = np.concatenate([exp_dir, pose_dir])
#     nrm = np.linalg.norm(full_dir)
#     if nrm > 1e-8:
#         full_dir = full_dir / nrm
#     exp_dir = full_dir[:N_EXP]
#     pose_dir = full_dir[N_EXP:]

#     sigma1 = float(np.nanmax(np.r_[exp_std, pose_std]))  # shared visualization scale

#     exp_delta  = (N_SIGMA * sigma1) * exp_dir #* AMPLIFY
#     pose_delta = (N_SIGMA * sigma1) * pose_dir #* AMPLIFY
#     pose_delta = clamp_pose(pose_delta)


#     # # Convert standardized direction into original coefficient deltas:
#     # # Δx_j = (3 * σ_j) * w_j, then optional amplify
#     # exp_delta = (N_SIGMA * exp_std) * exp_dir * AMPLIFY
#     # pose_delta = (N_SIGMA * pose_std) * pose_dir * AMPLIFY

#     # pose clamp for sanity (especially if INCLUDE_POSE=True)
#     pose_delta = clamp_pose(pose_delta)


#     print("pose_dir:", pose_dir)
#     print("pose_std:", pose_std)
#     print("pose_delta:", pose_delta)
#     print("pose_delta global norm:", np.linalg.norm(pose_delta[0:3]))
#     print("pose_delta jaw norm   :", np.linalg.norm(pose_delta[3:6]))
#     print("exp_delta norm        :", np.linalg.norm(exp_delta))


#     print("[Info] exp_delta abs max:", float(np.max(np.abs(exp_delta))))
#     print("[Info] pose_delta:", pose_delta)

#     print("[Load] FLAME model...")
#     flame = load_model(str(FLAME_MODEL))
#     renderer = HeadlessRenderer(IMAGE_W, IMAGE_H)

#     zero_exp = np.zeros(N_EXP, dtype=np.float64)
#     zero_pose = np.zeros(N_POSE, dtype=np.float64)

#     # set fixed camera from neutral
#     set_flame_from_emoca(flame, zero_exp, zero_pose, jaw_gain=JAW_GAIN)
#     verts0 = np.array(flame.r, dtype=np.float64)
#     faces = np.array(flame.f, dtype=np.int32)
#     mesh0 = build_o3d_mesh(verts0, faces)
#     renderer.set_camera_from_mesh(mesh0, extra=1.7)
#     print("[OK] Fixed camera set from neutral.")

#     # Render -3σ and +3σ only (2x1)
#     configs = [
#         ("LDA  -3*sigma", zero_exp - exp_delta, clamp_pose(zero_pose - pose_delta)),
#         ("LDA  +3*sigma", zero_exp + exp_delta, clamp_pose(zero_pose + pose_delta)),
#     ]

#     tiles = []
#     for tag, expr_vec, pose_vec in configs:
#         set_flame_from_emoca(flame, expr_vec, pose_vec, jaw_gain=JAW_GAIN)
#         verts = np.array(flame.r, dtype=np.float64)
#         mesh = build_o3d_mesh(verts, faces)
#         img = renderer.render(mesh)
#         cv2.putText(img, tag, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 0), 2, cv2.LINE_AA)
#         tiles.append(img)

#     grid = np.vstack(tiles)  # top: -3σ, bottom: +3σ
#     out_grid = OUT_DIR / "LDA_stress_minus_plus_grid.png"
#     cv2.imwrite(str(out_grid), grid)
#     print("[Save]", out_grid)

#         # Optional displacement heatmaps for ±3σ vs neutral
#     if SAVE_HEATMAP:
#         # Neutral vertices
#         set_flame_from_emoca(flame, zero_exp, zero_pose, jaw_gain=JAW_GAIN)
#         v_neutral = np.array(flame.r, dtype=np.float64)

#         # +3σ vertices
#         set_flame_from_emoca(flame, zero_exp + exp_delta, clamp_pose(zero_pose + pose_delta), jaw_gain=JAW_GAIN)
#         v_plus = np.array(flame.r, dtype=np.float64)

#         # -3σ vertices
#         set_flame_from_emoca(flame, zero_exp - exp_delta, clamp_pose(zero_pose - pose_delta), jaw_gain=JAW_GAIN)
#         v_minus = np.array(flame.r, dtype=np.float64)

#         # Displacements
#         disp_plus  = np.linalg.norm(v_plus  - v_neutral, axis=1)
#         disp_minus = np.linalg.norm(v_minus - v_neutral, axis=1)

#         # Use a COMMON normalization so colors are comparable across ±
#         dmax = float(max(disp_plus.max(), disp_minus.max(), 1e-12))
#         disp_plus_n  = disp_plus  / dmax
#         disp_minus_n = disp_minus / dmax

#         # ---- + heatmap ----
#         colors_plus = displacement_to_colors(disp_plus_n)
#         mesh_plus = build_o3d_mesh(v_plus, faces, vertex_colors=colors_plus)
#         img_plus = renderer.render(mesh_plus)
#         cv2.putText(img_plus, "LDA  +3*sigma (vertex displacement)", (15, 32),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
#         out_plus = OUT_DIR / "LDA_stress_plus_heatmap.png"
#         cv2.imwrite(str(out_plus), img_plus)
#         print("[Save]", out_plus)

#         # ---- - heatmap ----
#         colors_minus = displacement_to_colors(disp_minus_n)
#         mesh_minus = build_o3d_mesh(v_minus, faces, vertex_colors=colors_minus)
#         img_minus = renderer.render(mesh_minus)
#         cv2.putText(img_minus, "LDA  -3*sigma (vertex displacement)", (15, 32),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
#         out_minus = OUT_DIR / "LDA_stress_minus_heatmap.png"
#         cv2.imwrite(str(out_minus), img_minus)
#         print("[Save]", out_minus)


#     # # Optional displacement heatmap for +3σ vs neutral (note: global pose can dominate)
#     # if SAVE_HEATMAP:
#     #     set_flame_from_emoca(flame, zero_exp, zero_pose, jaw_gain=JAW_GAIN)
#     #     v_neutral = np.array(flame.r, dtype=np.float64)

#     #     set_flame_from_emoca(flame, zero_exp + exp_delta, clamp_pose(zero_pose + pose_delta), jaw_gain=JAW_GAIN)
#     #     v_plus = np.array(flame.r, dtype=np.float64)

#     #     disp = np.linalg.norm(v_plus - v_neutral, axis=1)
#     #     disp_norm = disp / disp.max() if disp.max() > 0 else disp

#     #     colors = displacement_to_colors(disp_norm)
#     #     mesh_hm = build_o3d_mesh(v_plus, faces, vertex_colors=colors)
#     #     img_hm = renderer.render(mesh_hm)
#     #     cv2.putText(img_hm, "LDA  +3*sigma (vertex displacement)", (15, 32),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

#     #     out_hm = OUT_DIR / "LDA_stress_plus_heatmap.png"
#     #     cv2.imwrite(str(out_hm), img_hm)
#     #     print("[Save]", out_hm)

#     # print("✅ Done.")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end LDA (supervised) from GTlabel + visualization on FLAME.

- Fits LDA using y = GTlabel (0/1) from your merged phased CSV.
- Extracts the discriminant direction w (feature weights).
- Visualizes deformation at -3*sigma and +3*sigma (no neutral shown).
- Produces displacement maps for BOTH -3σ and +3σ vs neutral, with common normalization.

Key point:
    LDA uses labels at:
        lda.fit(X_std, y)

Scaling for visualization:
    Uses a shared sigma1 = max empirical std across all (exp+pose) coefficients,
    so that deformations are visible and comparable (like your PCA sigma1 figure).
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import cv2

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ---- headless stability (optional but helpful) ----
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)

import open3d as o3d
from open3d.visualization import rendering as o3dr
from smpl_webuser.serialization import load_model

# =========================
# CONFIG
# =========================
CSV_PATH = Path("/home/vivib/emoca/emoca/dataset/paired_tests_EMOCA/phased_csvs/MD_ALL_SUBJECTS_FINAL_PERFRAME_PHASED_GTLABEL_CLEAN.csv")

OUT_DIR = Path("/home/vivib/emoca/emoca/feature_visualization/lda_from_gtlabel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLAME_MODEL = Path("/home/vivib/emoca/emoca/assets/FLAME/geometry/generic_model.pkl")

IMAGE_W, IMAGE_H = 600, 600
N_EXP = 50
N_POSE = 6

INCLUDE_POSE = True
N_SIGMA = 3.0
JAW_GAIN = 0.5

# pose clamps (radians) — prevent crazy global rotations
POSE_GLOBAL_MAX_RAD = 0.9     # ~52 degrees
POSE_JAW_MAX_RAD    = 0.15  # jaw can be larger

# Optional subsampling to keep LDA fast on 843k frames (safe + reproducible)
# Set to None to use all frames (may be slower / memory heavier).
MAX_SAMPLES_TOTAL = None#200_000     # None or int
RANDOM_SEED = 0

SAVE_HEATMAPS = True

# =========================
# Helpers
# =========================
def build_o3d_mesh(vertices, faces, vertex_colors=None):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.compute_vertex_normals()

    if vertex_colors is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=np.float64))

    # dummy UVs
    ntri = np.asarray(m.triangles).shape[0]
    uvs = np.tile(np.array([[0, 0], [1, 0], [0, 1]], np.float32), (ntri, 1))
    m.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    m.triangle_material_ids = o3d.utility.IntVector([0] * ntri)
    return m


class HeadlessRenderer:
    """Offscreen renderer with fixed camera."""
    def __init__(self, width, height, bg_rgba=(1, 1, 1, 1)):
        self.r = o3dr.OffscreenRenderer(int(width), int(height))
        self.scene = self.r.scene
        self.scene.set_background(bg_rgba)

        self.mat = o3dr.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.mat.base_color = (0.9, 0.9, 0.9, 1.0)

        try:
            self.scene.scene.set_sun_light([0.0, 0.5, 0.8], [1, 1, 1], 25000, True)
            self.scene.scene.enable_sun_light(True)
        except Exception:
            pass

        self.name = "mesh"
        self._cam_set = False
        self._cam_center = None
        self._cam_eye = None
        self._cam_up = np.array([0, 1, 0], dtype=np.float64)

    def set_camera_from_mesh(self, mesh, extra=1.7):
        bbox = mesh.get_axis_aligned_bounding_box()
        c = bbox.get_center()
        extent = bbox.get_extent()
        radius = float(np.linalg.norm(extent)) * 0.5
        if radius <= 1e-6:
            radius = 1.0
        eye = c + np.array([0.0, 0.0, extra * radius])
        self._cam_center = c
        self._cam_eye = eye
        self._cam_set = True
        self.r.setup_camera(60.0, c, eye, self._cam_up)

    def render(self, mesh):
        try:
            self.scene.remove_geometry(self.name)
        except Exception:
            pass
        self.scene.add_geometry(self.name, mesh, self.mat)
        if self._cam_set:
            self.r.setup_camera(60.0, self._cam_center, self._cam_eye, self._cam_up)
        img = self.r.render_to_image()
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def set_flame_from_emoca(model, expr50, pose6, jaw_gain=1.0):
    """EMOCA -> FLAME mapping: global -> pose[0:3], jaw -> pose[6:9]."""
    model.betas[:] = 0
    model.betas[300:350] = np.asarray(expr50, np.float64)

    p = np.asarray(pose6, np.float64).reshape(-1)
    model.pose[:] = 0
    model.pose[0:3] = p[0:3]
    model.pose[6:9] = jaw_gain * p[3:6]


def clamp_pose(pose6):
    p = np.asarray(pose6, np.float64).reshape(-1).copy()
    if p.size >= 3:
        p[0:3] = np.clip(p[0:3], -POSE_GLOBAL_MAX_RAD, POSE_GLOBAL_MAX_RAD)
    if p.size >= 6:
        p[3:6] = np.clip(p[3:6], -POSE_JAW_MAX_RAD, POSE_JAW_MAX_RAD)
    return p


def displacement_to_colors(disp_norm):
    """Blue->Red for displacement magnitude (no direction)."""
    disp_norm = np.clip(disp_norm, 0.0, 1.0)
    r = disp_norm
    g = np.zeros_like(disp_norm)
    b = 1.0 - disp_norm
    return np.stack([r, g, b], axis=-1)


def zscore_fit_transform(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd


# =========================
# MAIN
# =========================
def main():
    exp_cols = [f"exp_{i:02d}" for i in range(N_EXP)]
    pose_cols = [f"pose_{i:02d}" for i in range(N_POSE)]
    feat_cols = exp_cols + (pose_cols if INCLUDE_POSE else [])

    print(f"[Load] {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    if "GTlabel" not in df.columns:
        raise RuntimeError("CSV missing required column: GTlabel")

    for c in feat_cols:
        if c not in df.columns:
            raise RuntimeError(f"CSV missing required feature: {c}")

    # keep only labeled rows
    df = df.copy()
    df["GTlabel"] = pd.to_numeric(df["GTlabel"], errors="coerce")
    df = df[df["GTlabel"].isin([0, 1])].copy()
    df.dropna(subset=feat_cols, inplace=True)

    print(f"[Info] Rows after GTlabel filter + drop NaN features: {len(df):,}")

    # optional subsample (keeps class balance roughly)
    if MAX_SAMPLES_TOTAL is not None and len(df) > MAX_SAMPLES_TOTAL:
        rng = np.random.default_rng(RANDOM_SEED)
        idx0 = df.index[df["GTlabel"] == 0].to_numpy()
        idx1 = df.index[df["GTlabel"] == 1].to_numpy()
        n_half = MAX_SAMPLES_TOTAL // 2
        take0 = rng.choice(idx0, size=min(n_half, len(idx0)), replace=False)
        take1 = rng.choice(idx1, size=min(n_half, len(idx1)), replace=False)
        keep = np.concatenate([take0, take1])
        df = df.loc[keep].copy()
        print(f"[Info] Subsampled to: {len(df):,}  (0:{(df.GTlabel==0).sum():,} | 1:{(df.GTlabel==1).sum():,})")

    # Build X, y
    X = df[feat_cols].to_numpy(np.float64)
    y = df["GTlabel"].to_numpy(np.int64)

    # Standardize features (global)
    Xz, mu, sd = zscore_fit_transform(X)

    # -----------------------------
    # Fit LDA  (THIS is where labels are used)
    # -----------------------------
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(Xz, y)   # <---------------- LABELS ENTER HERE
    w = lda.coef_.reshape(-1)  # (d,)

    # normalize direction vector
    wn = w / (np.linalg.norm(w) + 1e-12)




    # save direction weights to CSV
    out_dir_csv = OUT_DIR / "lda_direction_stress_from_gtlabel.csv"
    df_w = pd.DataFrame({"feature": feat_cols, "weight": wn})
    df_w.to_csv(out_dir_csv, index=False)
    print("[Save]", out_dir_csv)


        # --- OPTIONAL but recommended: fix sign so + direction points toward stress (GTlabel=1) ---
    m0 = Xz[y == 0].mean(axis=0)
    m1 = Xz[y == 1].mean(axis=0)
    if np.dot((m1 - m0), wn) < 0:
        wn = -wn

    # --- base = global mean in ORIGINAL coefficient space ---
    base = mu.copy()                 # mu from zscore_fit_transform(X)
    sd_safe = sd.copy()
    sd_safe[sd_safe < 1e-12] = 1.0   # avoid divide-by-zero issues

    # --- ±3σ step in standardized space ---
    k = 1.0
    x_std_minus = -k * wn            # base_std is 0 when base==mu
    x_std_plus  =  k * wn

    # --- map back to original coefficient space ---
    x_minus = base + sd_safe * x_std_minus
    x_plus  = base + sd_safe * x_std_plus

    exp_minus = x_minus[:N_EXP]
    exp_plus  = x_plus[:N_EXP]

    if INCLUDE_POSE:
        pose_minus = clamp_pose(x_minus[N_EXP:N_EXP + N_POSE])
        pose_plus  = clamp_pose(x_plus [N_EXP:N_EXP + N_POSE])
    else:
        pose_minus = np.zeros(N_POSE)
        pose_plus  = np.zeros(N_POSE)


    # split to exp / pose direction
    exp_dir = wn[:N_EXP]
    pose_dir = wn[N_EXP:] if INCLUDE_POSE else np.zeros(N_POSE, dtype=np.float64)

    # empirical std in ORIGINAL coefficient space (for sigma1)
    exp_std = df[exp_cols].std().to_numpy()
    pose_std = df[pose_cols].std().to_numpy() if INCLUDE_POSE else np.zeros(N_POSE)

    sigma1 = float(np.nanmax(np.r_[exp_std, pose_std]))  # shared visualization scale

    exp_delta  = (N_SIGMA * sigma1) * exp_dir * 0.3
    pose_delta = (N_SIGMA * sigma1) * pose_dir 
    if INCLUDE_POSE:
        pose_delta = clamp_pose(pose_delta)

    print("[Debug] sigma1:", sigma1)
    print("[Debug] exp_delta norm:", float(np.linalg.norm(exp_delta)))
    print("[Debug] pose_delta norm:", float(np.linalg.norm(pose_delta)))

    # -----------------------------
    # Render
    # -----------------------------
    print("[Load] FLAME...")
    flame = load_model(str(FLAME_MODEL))
    renderer = HeadlessRenderer(IMAGE_W, IMAGE_H)

    zero_exp = np.zeros(N_EXP, dtype=np.float64)
    zero_pose = np.zeros(N_POSE, dtype=np.float64)

    # fixed camera from neutral
    # set_flame_from_emoca(flame, zero_exp, zero_pose, jaw_gain=JAW_GAIN)


    
    # --- BASE mesh at global mean (mu) ---
    set_flame_from_emoca(flame, base[:N_EXP], clamp_pose(base[N_EXP:N_EXP+N_POSE]), jaw_gain=JAW_GAIN)
    v_base = np.array(flame.r, dtype=np.float64)

    verts0 = np.array(flame.r, dtype=np.float64)
    faces = np.array(flame.f, dtype=np.int32)
    mesh0 = build_o3d_mesh(verts0, faces)
    renderer.set_camera_from_mesh(mesh0, extra=1.7)
    print("[OK] Fixed camera set.")

    # (a) -3σ deformation + displacement
    set_flame_from_emoca(flame, zero_exp - exp_delta, clamp_pose(zero_pose - pose_delta), jaw_gain=JAW_GAIN)
    v_minus = np.array(flame.r, dtype=np.float64)
    mesh_minus = build_o3d_mesh(v_minus, faces)
    img_minus = renderer.render(mesh_minus)
    cv2.putText(img_minus, "LDA  -3*sigma", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0,0,0), 2, cv2.LINE_AA)

    # (b) +3σ deformation + displacement
    set_flame_from_emoca(flame, zero_exp + exp_delta, clamp_pose(zero_pose + pose_delta), jaw_gain=JAW_GAIN)
    v_plus = np.array(flame.r, dtype=np.float64)
    mesh_plus = build_o3d_mesh(v_plus, faces)
    img_plus = renderer.render(mesh_plus)
    cv2.putText(img_plus, "LDA  +3*sigma", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0,0,0), 2, cv2.LINE_AA)

    # save stacked deformation-only grid (top -3σ, bottom +3σ)
    grid = np.vstack([img_minus, img_plus])
    out_grid = OUT_DIR / "LDA_stress_minus_plus_grid.png"
    cv2.imwrite(str(out_grid), grid)
    print("[Save]", out_grid)

    # -----------------------------
    # Heatmaps for BOTH ±3σ vs neutral (common normalization)
    # -----------------------------
    if SAVE_HEATMAPS:
        # neutral
        # --- BASE mesh at global mean (mu) ---
        set_flame_from_emoca(flame, base[:N_EXP], clamp_pose(base[N_EXP:N_EXP+N_POSE]), jaw_gain=JAW_GAIN)
        v_base = np.array(flame.r, dtype=np.float64)

        disp_minus = np.linalg.norm(v_minus - v_base, axis=1)
        disp_plus  = np.linalg.norm(v_plus  - v_base, axis=1)

        dmax = float(max(disp_minus.max(), disp_plus.max(), 1e-12))
        dm = disp_minus / dmax
        dp = disp_plus / dmax


        # disp_minus = np.linalg.norm(v_minus - v_neutral, axis=1)
        # disp_plus  = np.linalg.norm(v_plus  - v_neutral, axis=1)

        # dmax = float(max(disp_minus.max(), disp_plus.max(), 1e-12))
        # dm = disp_minus / dmax
        # dp = disp_plus / dmax

        # minus heatmap
        colors_m = displacement_to_colors(dm)
        mesh_m = build_o3d_mesh(v_minus, faces, vertex_colors=colors_m)
        img_m = renderer.render(mesh_m)
        cv2.putText(img_m, "vertex displacement (vs neutral)", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,0,0), 2, cv2.LINE_AA)
        out_m = OUT_DIR / "LDA_stress_minus_heatmap.png"
        cv2.imwrite(str(out_m), img_m)
        print("[Save]", out_m)

        # plus heatmap
        colors_p = displacement_to_colors(dp)
        mesh_p = build_o3d_mesh(v_plus, faces, vertex_colors=colors_p)
        img_p = renderer.render(mesh_p)
        cv2.putText(img_p, "vertex displacement (vs neutral)", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,0,0), 2, cv2.LINE_AA)
        out_p = OUT_DIR / "LDA_stress_plus_heatmap.png"
        cv2.imwrite(str(out_p), img_p)
        print("[Save]", out_p)

        # optional combined figure like your final layout:
        # (a) -3σ + map, (b) +3σ + map
        combo_a = np.hstack([img_minus, img_m])
        combo_b = np.hstack([img_plus,  img_p])
        combo = np.vstack([combo_a, combo_b])
        out_combo = OUT_DIR / "LDA_stress_minus_plus_with_maps.png"
        cv2.imwrite(str(out_combo), combo)
        print("[Save]", out_combo)

    print("✅ Done.")


if __name__ == "__main__":
    main()
