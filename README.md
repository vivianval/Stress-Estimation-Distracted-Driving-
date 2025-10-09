# Stress-Estimation-Distracted-Driving

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](#)
[![CUDA 11.6](https://img.shields.io/badge/CUDA-11.6-brightgreen.svg)](#)
[![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-red.svg)](#)
[![PyTorch3D 0.6.2](https://img.shields.io/badge/PyTorch3D-0.6.2-orange.svg)](#)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

Code for **EMOCA-based facial feature extraction** (FLAME expression/pose + robust 2D landmarks) and **side-by-side Open3D visualization** in a reproducible setup.  
This repo stays lightweight (models and large CSVs are excluded) and focuses on the exact scripts used in our distracted-driving stress pipeline.



## Environment (Docker summary)
A reproducible image pins **CUDA 11.6 + cuDNN 8, Python 3.8, PyTorch 1.12.1 (cu116), PyTorch3D 0.6.2, mmcv-full 1.5.0**, ensuring PyTorch ↔ PyTorch3D binary compatibility and stable builds.  
Docker Compose bind-mounts your local **EMOCA_v2** checkout and dataset so extraction and visualization run identically across NVIDIA GPU hosts.

## Feature Extraction (EMOCA + landmarks)
Run **`gdl_apps/EMOCA/demos/test_emoca_on_vids_lnm.py`**.  
**What it does (additions):**
- **Memory-efficient per-subject CSVs** for FLAME **expression (50)** and **pose (6)**.
- **Landmarks saving:** 68×2 (x,y) in **unit-center crop space** + **105×2** MediaPipe subset; optional 68-pt visibility.
- **Strict gating** so only numeric arrays are saved (prevents corrupt rows) and **safe per-video cleanup** after CSV flush.
> Ensure **FLAME models** exist under `assets/` before extraction.

**Expected outputs (CSV_stream):**
- `T019_exp_pose.csv` → columns: `subject_id, frame_id, exp_00..exp_49, pose_00..pose_05`  
- `lands/T019_landmarks_xy.csv` (if enabled) → `l68_x*, l68_y*, mp_x*, mp_y*` (and optional `l68_vis*`)

## Visualization (side-by-side)
Script: **`visualize_emoca_flame_side_by_side.py`**.  
- **Left:** original frame with EMOCA-aligned **68-pt** + **MediaPipe** landmarks, plus a small HUD with **FLAME pose** and top-|exp|.  
- **Right:** **Open3D** render of the **FLAME mesh** driven by the same frame’s **expr/pose**; writes a side-by-side `.avi` to `feature_visualization/`.  
Notes: landmarks are restored to raw-pixel space via EMOCA-style crop transforms (`bbox2point`, `point2transform`, `bbpoint_warp`). If your raw video is mirrored, set `FLIP_X = True`. Adjust `IMAGE_SIZE`, `SCALE`, `BB_CENTER_SHIFT_X/Y` to match your EMOCA run.
cd "$REPO"
if ! grep -q "docs/demo.mp4" README.md; then
  awk '
    BEGIN { inserted=0 }
    { print }
    /^## Visualization \(side-by-side\)/ && !inserted {
      print "";
      print "<video src=\"docs/demo.mp4\" controls muted loop width=\"720\"></video>";
      print "";
      inserted=1
    }
  ' README.md > README.tmp && mv README.tmp README.md
fi

## Models / assets
Download FLAME and related weights into `assets/` (gitignored). Example path expected by scripts:
assets/FLAME/geometry/generic_model.pkl
You may symlink or set env vars if models live elsewhere.

## Tips 
- If a video won’t play, re-encode to H.264/yuv420p (see `convert_to_mp4.sh`).  
- EMOCA pose ordering is `[global(3), jaw(3)]`; the renderer maps jaw to `pose[6:9]` and exposes a gain for visibility.  


## Acknowledgements
Please cite and credit:
- **EMOCA v2** (feature extraction pipeline)  
- **FLAME** 3D morphable model

## License
MIT © 2025 Vivi Valergaki

## Changelog
- 2025-10-09: Initial curated release (Docker-pinned env; EMOCA demo with landmarks; side-by-side visualization; clean history).
