# Kappazunder Wien 3D Gaussian Splatting – Project Guide

This repository contains data-preparation tools for a 3D reconstruction workflow based on:

- image selection + COLMAP export
- LiDAR conversion to PLY
- mask generation/combination
- YOLO segmentation dataset preparation/fine-tuning support
- optional MoGe3 depth/normal extraction experiments

The scripts are designed as standalone utilities with editable config blocks at the top of each file.

## Repository structure

```text
Kappazunder-Wien-3D-Gaussian-Splatting/
├── colmap_pipeline/
│   ├── build_colmap_selection.py
│   ├── rotation_conversion.py
│   ├── laz_to_ply.py
│   ├── combine_masks.ipynb
│   ├── reduce_kg19_data.ipynb
│   └── reduce_stadtpark_data.ipynb
├── yolo_segmentation/
│   ├── prepare_yolo_database.py
│   ├── mask_car_parts.ipynb
│   └── yolo_finetune_images/
│       └── data.yaml
└── MoGe3_pipeline/
    └── get_normal_depth.ipynb
```

## What each part does

### `colmap_pipeline/`

- **`build_colmap_selection.py`**  
  Main camera/image export step. It:
  1. loads image metadata and interior orientation,
  2. builds frustum polygons per image,
  3. selects images intersecting a region of interest,
  4. optionally copies and renames selected images/masks to `frame_######.jpg`,
  5. exports COLMAP text files (`cameras.txt`, `images.txt`, empty `points3D.txt`),
  6. writes `scene_origin.txt` offset for consistent point-cloud alignment.

- **`rotation_conversion.py`**  
  Pose conversion logic used by `build_colmap_selection.py` to create COLMAP-compatible world-to-camera transforms.

- **`laz_to_ply.py`**  
  Converts one or more LAZ files to a single colored PLY, applies the same `scene_origin` offset, and writes `points3D_init.ply` for COLMAP initialization.

- **`combine_masks.ipynb` / `reduce_kg19_data.ipynb` / `reduce_stadtpark_data.ipynb`**  
  Notebook-based utilities for data reduction/mask preparation.

### `yolo_segmentation/`

- **`prepare_yolo_database.py`**  
  Builds a YOLO segmentation dataset from images + binary masks:
  - converts mask contours into YOLO polygon labels,
  - splits train/val,
  - writes expected YOLO folder layout and `data.yaml`,
  - saves a visual sanity-check image.

- **`mask_car_parts.ipynb`**  
  Notebook for generating/editing segmentation masks.

- **`yolo_finetune_images/data.yaml`**  
  YOLO training data config (generated/maintained by the dataset prep flow).

### `MoGe3_pipeline/`

- **`get_normal_depth.ipynb`**  
  Experimental notebook pipeline for depth/normal extraction.

## Recommended tool usage order

Use this order for the core geometry pipeline:

1. **(Optional) ROI/data reduction notebooks**  
   Run `reduce_*.ipynb` if you need to spatially reduce raw inputs first.

2. **`colmap_pipeline/build_colmap_selection.py`**  
   This is the central first script for image/camera export.  
   Output: `colmap_export/images`, `colmap_export/sparse/0/{cameras.txt,images.txt,points3D.txt}`, `colmap_export/scene_origin.txt`.

3. **`colmap_pipeline/laz_to_ply.py`**  
   Run after Step 2 so `scene_origin.txt` exists.  
   Output: `colmap_export/sparse/0/points3D_init.ply` aligned to exported cameras.

4. **(Optional) mask combination**  
   Use `combine_masks.ipynb` if you need merged/refined masks for later training or filtering.

Use this order for the segmentation pipeline:

1. **`yolo_segmentation/mask_car_parts.ipynb`** (if masks are not ready)
2. **`yolo_segmentation/prepare_yolo_database.py`** to build train/val + polygon labels
3. **YOLO training** using generated `yolo_finetune_images/data.yaml` (outside this repo)

Use MoGe3 notebook independently as an optional experimental branch:

1. **`MoGe3_pipeline/get_normal_depth.ipynb`**

## Typical outputs you should expect

- `colmap_pipeline/colmap_export/selection_preview.png`
- `colmap_pipeline/colmap_export/scene_origin.txt`
- `colmap_pipeline/colmap_export/sparse/0/cameras.txt`
- `colmap_pipeline/colmap_export/sparse/0/images.txt`
- `colmap_pipeline/colmap_export/sparse/0/points3D.txt` (empty placeholder)
- `colmap_pipeline/colmap_export/sparse/0/points3D_init.ply`
- `yolo_segmentation/yolo_finetune_images/images/{train,val}/*`
- `yolo_segmentation/yolo_finetune_images/labels/{train,val}/*`
- `yolo_segmentation/yolo_finetune_images/data.yaml`
- `yolo_segmentation/yolo_finetune_images/sanity_check.jpg`

## Setup notes

- Edit each script’s **CONFIG** section before running.
- Paths in scripts currently point to local dataset folders not included in this repository.
- Required Python packages vary by script and include:
  - `numpy`, `pandas`
  - `geopandas`, `shapely`, `matplotlib`, `Pillow`, `tqdm`
  - `opencv-python`
  - `laspy`, `open3d`

## Quick start (core geometry flow)

From `colmap_pipeline/`:

1. configure `build_colmap_selection.py`
2. run `python3 build_colmap_selection.py`
3. configure `laz_to_ply.py` LAZ inputs
4. run `python3 laz_to_ply.py`

---

If you want, I can refine this README further with an exact end-to-end command sequence for your specific dataset layout (which folders/files you actually use first).
