#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Radek Danecek (original), modified with:
- safe per-video cleanup
- landmarks saving: 68x2, 68x visibility, and 105x2 (mediapipe)
- debug prints of visdict/vals/batch keys and shapes (once)
- strict gating so image tensors are never saved as landmarks
"""

from pathlib import Path
import argparse
import shutil
import sys
from typing import Optional, Any, Dict

import numpy as np
import torch
from tqdm import auto

import gdl
from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test

# ---------- Landmarks gating/extraction ----------
from typing import Optional, Tuple
import numpy as np
import torch

# ------------------------ helpers ------------------------

def subject_from_video_name(p: Path) -> str:
    stem = p.stem
    return stem.split('-', 1)[0] if '-' in stem else stem

def str2bool(v):
    if isinstance(v, bool): return v
    s = str(v).lower()
    if s in ('yes','true','t','y','1'): return True
    if s in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def _resolve_processed_top(dm_output_dir: Path, video_stem: str) -> Path:
    p = Path(dm_output_dir)
    if p.name.startswith("processed_"): return p
    if p.parent.name.startswith("processed_"): return p.parent
    if p.name == video_stem and p.parent.name.startswith("processed_"): return p.parent
    return p  # cleanup is guarded by name check



def _looks_like_landmarks_2d(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor): return False
    shp = tuple(x.shape)
    if len(shp) == 3 and shp[0] in (1,3) and shp[1] >= 64 and shp[2] >= 64:
        return False
    if shp in ((68,2),(1,68,2),(68,3)): return True
    if len(shp) == 2 and shp[1] == 2 and 20 <= shp[0] <= 500:
        return True
    return False

def _to_numpy_68xy_from_pred(x: torch.Tensor) -> Optional[np.ndarray]:
    if not isinstance(x, torch.Tensor): return None
    if x.shape == (1,68,2): x = x[0]
    if x.shape != (68,2): return None
    return x.detach().cpu().to(torch.float32).numpy()

def _to_numpy_68_from_lmk2d(x: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """landmarks2d: (68,3) -> ((68,2) xy, (68,) vis)"""
    if not isinstance(x, torch.Tensor) or x.shape != (68,3):
        return (None, None)
    xy  = x[:, :2].detach().cpu().to(torch.float32).numpy()
    vis = x[:,  2].detach().cpu().to(torch.float32).numpy()
    return (xy, vis)

def _to_numpy_105_from_mp(x: torch.Tensor) -> Optional[np.ndarray]:
    if not isinstance(x, torch.Tensor) or x.shape != (105,2): return None
    return x.detach().cpu().to(torch.float32).numpy()



def _debug_print_once(tag: str, container: Any):
    print(f"\n[DEBUG] {tag} keys/shapes:")
    if container is None:
        print(f"[DEBUG] {tag}: None"); return
    if isinstance(container, dict):
        keys = list(container.keys())
        print(f"[DEBUG] {tag} keys:", keys)
        for k in keys:
            v = container[k]
            if isinstance(v, torch.Tensor):
                print(f"[DEBUG] {tag}.{k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                try:
                    shape = tuple(v.shape); print(f"[DEBUG] {tag}.{k}: type={type(v)}, shape={shape}")
                except Exception:
                    print(f"[DEBUG] {tag}.{k}: type={type(v)}")
    else:
        try:
            keys = list(container.keys()); print(f"[DEBUG] {tag} keys:", keys)
            for k in keys:
                v = container[k]
                if isinstance(v, torch.Tensor):
                    print(f"[DEBUG] {tag}.{k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
                else:
                    try:
                        shape = tuple(v.shape); print(f"[DEBUG] {tag}.{k}: type={type(v)}, shape={shape}")
                    except Exception:
                        print(f"[DEBUG] {tag}.{k}: type={type(v)}")
        except Exception:
            print(f"[DEBUG] {tag}: type={type(container)} (no keys())")


# ------------------------ core ------------------------

def reconstruct_video(args, input_video_path: Path):
    path_to_models = args.path_to_models
    model_name = args.model_name
    output_folder = args.output_folder + "/" + model_name
    image_type = args.image_type
    black_background = args.black_background
    include_original = args.include_original
    include_rec = args.include_rec
    cat_dim = args.cat_dim
    use_mask = args.use_mask
    include_transparent = bool(args.include_transparent)
    processed_subfolder = args.processed_subfolder
    mode = args.mode

    # 1) Data module per video
    dm = TestFaceVideoDM(
        str(input_video_path),
        output_folder,
        processed_subfolder=processed_subfolder,
        batch_size=4,
        num_workers=4
    )
    dm.prepare_data(); dm.setup()

    dm_output_dir = Path(getattr(dm, "output_dir", output_folder))
    processed_top = _resolve_processed_top(dm_output_dir, input_video_path.stem)
    print(f"[INFO] processed_top resolved to: {processed_top}")

    # 2) Load model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca = emoca.cuda().eval()

    # SUBJECT folder
    subject_id = subject_from_video_name(input_video_path)
    subject_out_root = Path(output_folder) / subject_id
    subject_out_root.mkdir(parents=True, exist_ok=True)

    # 3) Data loader
    dl = dm.test_dataloader()

    try:
        with torch.no_grad():
            peeked = False
            for _, batch in enumerate(auto.tqdm(dl)):
                vals, visdict = test(emoca, batch)

                if not peeked:
                    _debug_print_once("visdict", visdict)
                    _debug_print_once("vals", vals)
                    try: _debug_print_once("batch", batch)
                    except Exception: print("[DEBUG] batch: <unprintable>")
                    sys.stdout.flush()
                    peeked = True

                B = batch["image"].shape[0]
                for i in range(B):
                    raw_name  = str(batch["image_name"][i])   # e.g. "000001_000"
                    frame_dir = (subject_out_root / f"FRAME_{raw_name}"); frame_dir.mkdir(parents=True, exist_ok=True)

                    if args.save_mesh:
                        save_obj(emoca, str(frame_dir / "mesh_coarse.obj"), vals, i)
                    if args.save_images:
                        save_images(str(subject_out_root), f"FRAME_{raw_name}", visdict, i)
                    if args.save_codes:
                        save_codes(subject_out_root, f"FRAME_{raw_name}", vals, i)

                    # -------- Save landmarks: 68x2 (+vis) and 105x2 (mediapipe) --------
                    if args.save_landmarks:
                        # 68x2: prefer predicted_landmarks, else landmarks2d[:,:2]
                        saved68 = False
                        if "predicted_landmarks" in vals and vals["predicted_landmarks"] is not None:
                            item = vals["predicted_landmarks"][i]
                            arr  = _to_numpy_68xy_from_pred(item)
                            if arr is not None:
                                np.save(str(frame_dir / "landmarks68.npy"), arr.astype("float32"))
                                saved68 = True
                        if (not saved68) and "landmarks2d" in vals and vals["landmarks2d"] is not None:
                            item = vals["landmarks2d"][i]
                            xy, vis = _to_numpy_68_from_lmk2d(item)
                            if xy is not None:
                                np.save(str(frame_dir / "landmarks68.npy"), xy.astype("float32"))
                                saved68 = True
                            if vis is not None:
                                np.save(str(frame_dir / "landmarks68_vis.npy"), vis.astype("float32"))

                        # 105x2: mediapipe
                        if "predicted_landmarks_mediapipe" in vals and vals["predicted_landmarks_mediapipe"] is not None:
                            item = vals["predicted_landmarks_mediapipe"][i]
                            arr  = _to_numpy_105_from_mp(item)
                            if arr is not None:
                                np.save(str(frame_dir / "landmarks105_mp.npy"), arr.astype("float32"))
                    # --------------------------------------------------------------------

    finally:
        if getattr(args, "cleanup_processed", True):
            try:
                if processed_top.exists() and processed_top.name.startswith("processed_"):
                    shutil.rmtree(processed_top, ignore_errors=True)
                    print(f"[CLEANUP] Deleted {processed_top}")
                else:
                    print(f"[CLEANUP] Skipped (not a processed_* folder): {processed_top}")
            except Exception as e:
                print(f"[WARN] Cleanup skipped at {processed_top}: {e}")


# ------------------------ CLI ------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
        help="Video file OR directory of videos for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output",
        help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20',
        help='Model name.')
    parser.add_argument('--path_to_models', type=str,
        default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail",
        choices=["detail", "coarse"])

    parser.add_argument('--save_images', type=str2bool, default=False,
        help="Save rendered images (disk heavy).")
    parser.add_argument('--save_codes', type=str2bool, default=True,
        help="Save FLAME codes (exp/pose/etc).")
    parser.add_argument('--save_mesh', type=str2bool, default=False,
        help="Save meshes.")

    parser.add_argument('--save_landmarks', type=str2bool, default=True,
        help="Save 68x2 (and visibility) + 105x2 mediapipe landmarks.")

    # video options (disabled by default)
    parser.add_argument('--image_type', type=str, default='geometry_detail',
        choices=["geometry_detail", "geometry_coarse", "out_im_detail", "out_im_coarse"])
    parser.add_argument('--processed_subfolder', type=str, default=None)
    parser.add_argument('--cat_dim', type=int, default=0)
    parser.add_argument('--include_rec', type=str2bool, default=True)
    parser.add_argument('--include_transparent', type=str2bool, default=True)
    parser.add_argument('--include_original', type=str2bool, default=True)
    parser.add_argument('--black_background', type=str2bool, default=False)
    parser.add_argument('--use_mask', type=str2bool, default=True)

    parser.add_argument('--cleanup_processed', type=str2bool, default=True,
        help="Delete processed_<timestamp> after each video.")

    parser.add_argument('--logger', type=str, default="", choices=["", "wandb"])
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if input_path.is_file():
        reconstruct_video(args, input_path)
    elif input_path.is_dir():
        exts = {".mp4",".mov",".avi",".mkv",".mpg",".mpeg",".m4v"}
        videos = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])
        if not videos:
            raise RuntimeError(f"No videos found under: {input_path}")
        for v in videos:
            try:
                reconstruct_video(args, v)
            except Exception as e:
                print(f"[SKIP] {v.name}: {e}")
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path}")

    print("Done")


if __name__ == '__main__':
    main()
