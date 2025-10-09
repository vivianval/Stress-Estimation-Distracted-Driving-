

# -*- coding: utf-8 -*-
"""
Left = the original EMOCA-landmark overlay (unchanged mapping)
  + a small HUD with FLAME pose & top-|exp| values.
Right = FLAME mesh rendered from EMOCA exp/pose for the same frame.
"""

import os, sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import mediapipe as mp
from skimage.transform import estimate_transform, warp

# ---------- FLAME right-pane deps ----------
try:
    import open3d as o3d
    from open3d.visualization import rendering as o3dr
    from smpl_webuser.serialization import load_model
except Exception as e:
    raise RuntimeError("Needs open3d>=0.15 and smpl_webuser for FLAME rendering") from e

# ==========================
# USER CONFIG
# ==========================
SUBJECT_ID     = "T019"
VIDEO_FILE     = f"/media/storage/vivib/Structured Study Data/dataset/converted_mp4/{SUBJECT_ID}-007.mp4"
LANDMARKS_CSV  = f"/home/vivib/emoca/emoca/emoca_results/CSV_STREAM/lands/{SUBJECT_ID}_landmarks_xy.csv"
EXP_POSE_CSV   = f"/home/vivib/emoca/emoca/emoca_results/CSV_STREAM/{SUBJECT_ID}_exp_pose.csv"
FLAME_MODEL    = f"/home/vivib/emoca/emoca/assets/FLAME/geometry/generic_model.pkl"

OUTPUT_VIDEO   = f"/home/vivib/emoca/emoca/feature_visualization/{SUBJECT_ID}_overlay_side_by_side.avi"
FPS_FALLBACK   = 30

# EMOCA crop size used during processing (224 by default; sometimes 256)
IMAGE_SIZE = 224

# EMOCA-like crop params (match your run if you know them)
SCALE = 1.25               # EMOCA default
BB_CENTER_SHIFT_X = 0.00   # fraction of bbox width
BB_CENTER_SHIFT_Y = 0.08   # fraction of bbox height; ↑increase to move landmarks DOWN

# If your raw video is mirrored, flip horizontally at the end
FLIP_X = False

# Draw options
DRAW_L68          = True
DRAW_MP           = True
DRAW_L68_CONTOURS = True
L68_COLOR = (0, 255, 255)  # BGR yellow
MP_COLOR  = (255, 255, 0)  # BGR cyan
L68_RADIUS = 3
MP_RADIUS  = 2

# HUD options
TOPK_EXP = 6          # how many largest |exp| to show
HUD_POS  = (10, 10)   # x,y in left frame

# ==========================
# EMOCA helpers (from repo)
# ==========================
def bbox2point(left, right, top, bottom, type='mediapipe'):
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y =  bottom - (bottom - top) / 2.0
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
    elif type == "mediapipe":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0
    else:
        raise NotImplementedError(f"bbox2point not implemented for {type}")
    return old_size, np.array([center_x, center_y], dtype=float)

def point2bbox(center, size):
    size2 = size / 2.0
    return np.array([[center[0] - size2, center[1] - size2],
                     [center[0] - size2, center[1] + size2],
                     [center[0] + size2, center[1] - size2]], dtype=float)

def point2transform(center, size, target_size_h, target_size_w=None):
    target_size_w = target_size_w or target_size_h
    src_pts = point2bbox(center, size)
    dst_pts = np.array([[0, 0],
                        [0, target_size_w - 1],
                        [target_size_h - 1, 0]], dtype=float)
    return estimate_transform('similarity', src_pts, dst_pts)

def bbpoint_warp(image, center, size, target_size_h, target_size_w=None,
                 inv=True, landmarks=None, order=3):
    target_size_w = target_size_w or target_size_h
    tform = point2transform(center, size, target_size_h, target_size_w)
    tf_img = tform.inverse if inv else tform
    out_img = warp(image, tf_img, output_shape=(target_size_h, target_size_w),
                   order=order, preserve_range=True)
    if landmarks is None:
        return out_img
    tf_lmk = tform if inv else tform.inverse
    if isinstance(landmarks, np.ndarray):
        dst_landmarks = tf_lmk(landmarks[:, :2])
    else:
        raise ValueError("landmarks must be a numpy array")
    return out_img, dst_landmarks

# 68-pt contour topology (0-based)
L68_CONTOURS = [
    list(range(0, 17)),        # jaw
    list(range(17, 22)),       # left brow
    list(range(22, 27)),       # right brow
    list(range(27, 31)),       # nose bridge
    list(range(31, 36)),       # lower nose
    list(range(36, 42)) + [36],# left eye loop
    list(range(42, 48)) + [42],# right eye loop
    list(range(48, 60)) + [48],# outer lip
    list(range(60, 68)) + [60] # inner lip
]

def draw_emoca_square(img, center, size, color=(180,180,180), thickness=1):
    x0 = int(round(center[0] - size/2))
    y0 = int(round(center[1] - size/2))
    x1 = int(round(center[0] + size/2))
    y1 = int(round(center[1] + size/2))
    cv2.rectangle(img, (x0,y0), (x1,y1), color, thickness, cv2.LINE_AA)

def draw_l68_contours(img, pts, color, thickness=1):
    pts = np.asarray(pts, dtype=int)
    for idxs in L68_CONTOURS:
        poly = pts[idxs].reshape(-1,1,2)
        cv2.polylines(img, [poly], False, color, thickness, cv2.LINE_AA)

# ==========================
# MediaPipe face mesh
# ==========================
class MPFaceBox:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5)

    def bbox_from_frame(self, bgr):
        """Returns (left, right, top, bottom) in RAW pixels using MP 468-landmark extremes."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lmk = res.multi_face_landmarks[0].landmark
        xs = np.array([p.x for p in lmk], dtype=float) * w
        ys = np.array([p.y for p in lmk], dtype=float) * h
        left, right = float(xs.min()), float(xs.max())
        top, bottom = float(ys.min()), float(ys.max())
        return left, right, top, bottom

# ==========================
# FLAME rendering utilities (RIGHT pane)
# ==========================
def build_o3d_mesh(vertices, faces):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.compute_vertex_normals()
    # simple UVs to avoid warnings
    ntri = np.asarray(m.triangles).shape[0]
    uvs = np.tile(np.array([[0,0],[1,0],[0,1]], np.float32), (ntri,1))
    m.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    m.triangle_material_ids = o3d.utility.IntVector([0]*ntri)
    return m

class HeadlessRenderer:
    def __init__(self, width, height, bg_rgba=(0,0,0,1)):
        self.r = o3dr.OffscreenRenderer(int(width), int(height))
        self.scene = self.r.scene
        self.scene.set_background(bg_rgba)
        self.mat = o3dr.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.mat.base_color = (0.9, 0.9, 0.9, 1.0)
        try:
            self.scene.scene.set_sun_light([0.0,0.5,0.8], [1,1,1], 25000, True)
            self.scene.scene.enable_sun_light(True)
        except Exception:
            pass
        self.name = "mesh"

    def _look_at(self, mesh, extra=1.7):
        bbox = mesh.get_axis_aligned_bounding_box()
        c = bbox.get_center()
        extent = bbox.get_extent()
        radius = float(np.linalg.norm(extent)) * 0.5
        if radius <= 1e-6: radius = 1.0
        eye = c + np.array([0.0, 0.0, extra * radius])
        self.r.setup_camera(60.0, c, eye, [0,1,0])

    def render(self, mesh):
        try: self.scene.remove_geometry(self.name)
        except Exception: pass
        self.scene.add_geometry(self.name, mesh, self.mat)
        self._look_at(mesh)
        img = self.r.render_to_image()
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# def set_flame_from_emoca(model, expr50, pose6):
#     model.betas[:] = 0
#     model.betas[300:350] = np.asarray(expr50, dtype=np.float64)
#     model.pose[:] = 0
#     r = np.asarray(pose6, dtype=np.float64).reshape(-1)
#     if r.size >= 3:
#         model.pose[0:3] = r[:3]    # global
#     if r.size >= 6:
#         model.pose[3:6] = r[3:6]   # jaw

def set_flame_from_emoca(model, expr50, pose6):
    """pose6 = [global(3), jaw(3)] in axis-angle (radians)."""
    JAW_GAIN = 1.0   # visualization only
 
    model.betas[:] = 0
    model.betas[300:350] = np.asarray(expr50, np.float64)

    model.pose[:] = 0
    p = np.asarray(pose6, np.float64).reshape(-1)
    jaw = JAW_GAIN * p[3:6]        # amplify
    model.pose[:] = 0
    model.pose[0:3] = p[:3]        # global
    model.pose[6:9] = jaw     
  #  p = np.asarray(pose6, np.float64).reshape(-1)
    # global head
    model.pose[0:3] = p[:3]
    # jaw — skip neck triplet [3:6] and write to [6:9]
   # model.pose[6:9] = p[3:6]
    print("pose len:", len(model.pose.r))
    print("pose vec:", model.pose.r)  # should show nonzeros at [0:3] and [6:9]



# HUD on LEFT
def draw_flame_hud(img, expr_vec, pose_vec, x=10, y=10, line=18):
    h, w = img.shape[:2]
    p = np.asarray(pose_vec, float).reshape(-1)
    e = np.asarray(expr_vec, float).reshape(-1)
    lines = []
    if p.size >= 6:
        lines += [f"pose g:[{p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}]",
                  f"pose j:[{p[3]:+.2f},{p[4]:+.2f},{p[5]:+.2f}]"]
    idxs = np.argsort(-np.abs(e))[:TOPK_EXP]
    lines += ["exp (top |val|):"] + [f"  exp_{k:02d}={e[k]:+.3f}" for k in idxs]

    # translucent box
    pad = 8
    block_w = 260
    block_h = pad*2 + line*len(lines)
    x0, y0 = x, y
    x1, y1 = min(w-1, x0+block_w), min(h-1, y0+block_h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    yy = y0 + pad + 12
    for s in lines:
        cv2.putText(img, s, (x0+pad, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        yy += line

# ==========================
# Main
# ==========================
def main():
    # --- Landmarks CSV (unit-center) ---
    df = pd.read_csv(LANDMARKS_CSV)
    if "subject_id" in df.columns:
        df = df[df["subject_id"] == SUBJECT_ID].copy()
    df["frame_id"] = df["frame_id"].astype(int)
    df.set_index("frame_id", inplace=True)

    # columns
    L68X_COLS = [f"l68_x{i:03d}" for i in range(1,69)]
    L68Y_COLS = [f"l68_y{i:03d}" for i in range(1,69)]
    MPX_COLS  = [f"mp_x{i:03d}"  for i in range(1,106)]
    MPY_COLS  = [f"mp_y{i:03d}"  for i in range(1,106)]


    # --- Exp/Pose CSV ---
    df_ep = pd.read_csv(EXP_POSE_CSV)
    if "subject_id" in df_ep.columns:
        df_ep = df_ep[df_ep["subject_id"] == SUBJECT_ID].copy()
    df_ep["frame_id"] = df_ep["frame_id"].astype(int)
    df_ep.set_index("frame_id", inplace=True)
    EXPR_COLS = [f"exp_{i:02d}"  for i in range(50)]
    POSE_COLS = [f"pose_{i:02d}" for i in range(6)]
    print("exp abs max:", np.nanmax(np.abs(df_ep[EXPR_COLS].to_numpy())))
    print("pose abs max:", np.nanmax(np.abs(df_ep[POSE_COLS].to_numpy())))


    # --- Video I/O ---
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_FILE}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK

    # side-by-side writer (2x width)
    out_size = (width*2, height)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, float(fps), out_size)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, float(fps), out_size)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    mpbox = MPFaceBox()
    print(f"[INFO] Writing -> {OUTPUT_VIDEO}  ({out_size[0]}x{out_size[1]} @ {fps}fps)")

    # FLAME right-pane setup
    print("Loading FLAME…")
    flame = load_model(FLAME_MODEL)
    renderer = HeadlessRenderer(width, height)
    print("FLAME ready.")

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # ---------- LEFT: your original pipeline ----------
        bbox = mpbox.bbox_from_frame(frame_bgr)
        left_frame = frame_bgr.copy()

        l68_raw = mp_raw = None
        if bbox is not None:
            left, right, top, bottom = bbox
            old_size, center = bbox2point(left, right, top, bottom, type='mediapipe')
            center = center.copy()
            center[0] += abs(right - left)  * BB_CENTER_SHIFT_X
            center[1] += abs(bottom - top) * BB_CENTER_SHIFT_Y
            size = int(old_size * SCALE)

            # landmarks for this frame
            if frame_idx in df.index:
                row = df.loc[frame_idx]

                if DRAW_L68 and all(c in df.columns for c in L68X_COLS+L68Y_COLS):
                    l68_xu = np.array([row[c] for c in L68X_COLS], dtype=float)
                    l68_yu = np.array([row[c] for c in L68Y_COLS], dtype=float)
                    l68_crop = np.c_[ (l68_xu*0.5 + 0.5) * IMAGE_SIZE,
                                      (l68_yu*0.5 + 0.5) * IMAGE_SIZE ]
                    _, l68_raw = bbpoint_warp(np.zeros((IMAGE_SIZE, IMAGE_SIZE)),
                                              center, size, IMAGE_SIZE, IMAGE_SIZE,
                                              inv=False, landmarks=l68_crop)

                if DRAW_MP and all(c in df.columns for c in MPX_COLS+MPY_COLS):
                    mp_xu = np.array([row[c] for c in MPX_COLS], dtype=float)
                    mp_yu = np.array([row[c] for c in MPY_COLS], dtype=float)
                    mp_crop = np.c_[ (mp_xu*0.5 + 0.5) * IMAGE_SIZE,
                                     (mp_yu*0.5 + 0.5) * IMAGE_SIZE ]
                    _, mp_raw = bbpoint_warp(np.zeros((IMAGE_SIZE, IMAGE_SIZE)),
                                             center, size, IMAGE_SIZE, IMAGE_SIZE,
                                             inv=False, landmarks=mp_crop)

            if FLIP_X:
                if l68_raw is not None: l68_raw[:,0] = (width - 1) - l68_raw[:,0]
                if mp_raw is not None:  mp_raw[:,0]  = (width - 1) - mp_raw[:,0]

            draw_emoca_square(left_frame, center, size, color=(180,180,180), thickness=1)

            if l68_raw is not None:
                pts68 = []
                for (x, y) in l68_raw:
                    if np.isfinite(x) and np.isfinite(y):
                        ix, iy = int(round(x)), int(round(y))
                        if 0 <= ix < width and 0 <= iy < height:
                            pts68.append((ix, iy))
                            cv2.circle(left_frame, (ix, iy), L68_RADIUS, L68_COLOR, -1, cv2.LINE_AA)
                if DRAW_L68_CONTOURS and len(pts68) == 68:
                    draw_l68_contours(left_frame, np.array(pts68, dtype=int), L68_COLOR, 1)

            if mp_raw is not None:
                for (x, y) in mp_raw:
                    if np.isfinite(x) and np.isfinite(y):
                        ix, iy = int(round(x)), int(round(y))
                        if 0 <= ix < width and 0 <= iy < height:
                            cv2.circle(left_frame, (ix, iy), MP_RADIUS, MP_COLOR, -1, cv2.LINE_AA)

        cv2.putText(left_frame, f"frame {frame_idx}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # ---------- LEFT HUD: FLAME values ----------
        if frame_idx in df_ep.index:
            r = df_ep.loc[frame_idx]
            expr = r[[*EXPR_COLS]].to_numpy(np.float64)
            pose = r[[*POSE_COLS]].to_numpy(np.float64)
            gx, gy, gz = float(pose[0]), float(pose[1]), float(pose[2])
            jx, jy, jz = float(pose[3]), float(pose[4]), float(pose[5])
            #print("global:", gx, gy, gz, "jaw:", jx, jy, jz)
            draw_flame_hud(left_frame, expr, pose, x=HUD_POS[0], y=HUD_POS[1])

        # ---------- RIGHT: FLAME mesh render ----------
        if frame_idx in df_ep.index:
            r = df_ep.loc[frame_idx]
            JAW_SCALE = 6.0  # try 4–8
            # pose = df_ep.loc[frame_idx, POSE_COLS].to_numpy(float, copy=True)
            # pose[3:6] *= JAW_SCALE
            # set_flame_from_emoca(flame, expr, pose)
            expr = r[[*EXPR_COLS]].to_numpy(np.float64)
            pose = r[[*POSE_COLS]].to_numpy(np.float64)
            set_flame_from_emoca(flame, expr, pose)
            verts = np.array(flame.r, dtype=np.float64)
            faces = np.array(flame.f, dtype=np.int32)
            mesh = build_o3d_mesh(verts, faces)
            right_frame = renderer.render(mesh)
        else:
            right_frame = np.zeros_like(left_frame)

        # compose side-by-side
        side = np.hstack([left_frame, right_frame])
        writer.write(side)
        frame_idx += 1

    cap.release()
    writer.release()
    print("✅ Done:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
