# Kappazunder Wien 3D Gaussian Splatting – Project Guide

This repository contains data-preparation tools for a 3D Gaussian splatting reconstruction workflow of [Kappazunder](https://digitales.wien.gv.at/projekt/kappazunder/) (City of Vienna) data based on:

- coordinate-based image selection + COLMAP export
- LiDAR conversion to PLY
- mask generation/combination
- YOLO segmentation dataset preparation/fine-tuning support
- optional MoGe3 depth/normal extraction

The scripts are designed as standalone utilities with editable config blocks at the top of each file.

## General Kappazunder data structure

Upon request, the City of Vienna provides a zipped dataset containing:

```
Los_*
├── Bild-Meta/
|  ├── image_meta.txt
|  ├── interior_orientation.txt
|  └── multisys.txt
├── Bild-Rohdaten/
|  └── Trajektorie_*/
|     └── Sensor_*/
|        └── *.jpg
├── Scan-Meta/
|  └── scan_meta.txt
├── Scan-Punktwolken/
|  └── Trajektorie_*/
|     └── Sensor_*/
|        └── scandata_*.laz
└── Verortung/
   └── trajectory_*.txt
```

The file `Bild-Meta/image_meta.txt` contains meta data about the images in `Bild-Rohdaten/`, i.e. (selected for our purposes)
- the trajectory ID (continuous capture of images and LiDAR), 
- the sensor ID (one of six cameras), 
- the image name (which is present up to six times, once per camera),
- the camera position in metres in EPSG:31256 (MGI / Austria GK East) coordinates, and
- the camera orientation vector as (r_x, r_y, r_z) in radians, where r_z marks the orientation of the heading vector in the horizontal plane and the sensor role determines whether the image top should align with the driving direction (vertical cameras, i.e. facing up/down) or with world +z (horizontal cameras, i.e. facing left/right/forward/backward).

The file `Bild-Meta/interior_orientation.txt` lists the camera IDs, their focal lengths, pixel sizes, image sizes, mounting heights, and the pitch of the camera relative to the vehicle.

These two files contain most of the information needed for a successful 3DGS reconstruction.

## The different pipelines

The main idea of this project is to automatically handle Kappazunder data to produce usable 3DGS datasets. The user provides the raw data and a region of interest (ROI) polygon, and the scripts will produce a COLMAP export, a PLY point cloud, and optionally several masks. `colmap_pipeline/` is the main pipeline for geometry and image selection, while `yolo_segmentation/` is a secondary pipeline for mask generation and YOLO dataset preparation. The optional `MoGe3_pipeline/` can be used to generate depth and normal masks for Spirula Studio or Lichtfeld Studio.

### `colmap_pipeline/`

- **`build_colmap_selection.py`**  
  Main camera/image export step. It:
  1. loads image metadata and interior orientation,
  2. builds frustum polygons per image,
  3. selects images with corresponding camera frustums intersecting a region of interest,
  4. optionally copies and renames selected images/masks to `frame_######.jpg`,
  5. exports COLMAP text files (`cameras.txt`, `images.txt`, empty `points3D.txt`),
  6. writes `scene_origin.txt` offset for consistent point-cloud alignment.

- **`rotation_conversion.py`**  
  Pose conversion logic used by `build_colmap_selection.py` to create COLMAP-compatible world-to-camera transforms.

- **`laz_to_ply.py`**  
  Converts one or more LAZ files to a single colored PLY, applies the same `scene_origin` offset, and writes `points3D_init.ply` for COLMAP initialisation.

- **`combine_masks.ipynb` / `reduce_*_data.ipynb`**  
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
  Notebook pipeline for depth/normal extraction using MoGe3.

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

Use MoGe3 notebook to generate depth/normal masks for training in Spirula Studio and Lichtfeld Studio:

1. **`MoGe3_pipeline/get_normal_depth.ipynb`**

## Setup notes

- Edit each script’s **CONFIG** section before running.
- Paths in scripts point to local dataset folders not included in this repository (downloaded from Kappazunder).
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
