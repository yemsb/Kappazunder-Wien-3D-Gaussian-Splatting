---
title: Introduction
weight: 10
bookToC: true
---

# Introduction

This project provides a complete pipeline for creating 3D Gaussian Splatting (3DGS) reconstructions from the City of Vienna's Kappazunder street-view dataset. The Kappazunder project captured Vienna's streets with synchronized multi-camera arrays and LiDAR sensors, providing high-quality georeferenced images and point clouds.

## What This Pipeline Does

The pipeline transforms raw Kappazunder data into camera poses and point clouds ready for 3DGS training:

{{% columns %}}
- **Spatial selection** — filter images by region of interest (ROI) using camera frustum intersection
- **Pose export** — convert Kappazunder metadata to COLMAP format with correct coordinate transforms
- **Point cloud alignment** — convert LAZ LiDAR files to PLY with matching coordinate system
- **Mask generation** (optional) — prepare segmentation masks for dynamic object removal
- **3DGS training** — load the prepared data into Spirula Studio or Lichtfeld Studio
{{% /columns %}}

{{% hint warning %}}
Please note that the Kappazunder dataset is fundamentally challenging for 3DGS due to its relatively sparse and single-height coverage, dynamic objects, dynamic lighting (including change of seasons), and limited camera overlap.
{{% /hint %}}

## Why This Pipeline Exists

Kappazunder data uses a custom structure with trajectory-based organization, Austria-specific coordinate systems (**EPSG:31256**), and vehicle-mounted camera orientations. This pipeline handles:

- **Coordinate transforms** — Kappazunder uses MGI / Austria GK East (EPSG:31256) with vehicle-centric rotations; COLMAP expects world-to-camera quaternions
- **Camera frustum filtering** — efficiently select only images that see your ROI from hundreds of captures
- **Sensor geometry** — six cameras mounted on the vehicle (front/back/left/right/up/down)
- **Point cloud alignment** — apply consistent origin offsets to both cameras and LiDAR for numerical stability

## Project Structure

```
colmap_pipeline/
├── configs/                    # YAML configs per scene
│   └── <scene_name>/
│       ├── config.yaml
│       └── AoI.csv
├── build_colmap_selection.py  # Main camera/pose export script
├── laz_to_ply.py              # LiDAR conversion
├── rotation_conversion.py     # Pose math utilities
└── reduce_*.ipynb             # Optional spatial reduction notebooks

yolo_segmentation/
├── prepare_yolo_database.py   # YOLO dataset builder
└── mask_car_parts.ipynb       # Mask generation notebook
```

## Typical Workflow

For a standard 3DGS reconstruction:

1. **Request data** from Vienna's [Geoportal](https://geoportal.wien.gv.at/map/main/geodaten/) for your region
2. **Configure** your scene (ROI polygon + paths) in `configs/<scene>/`
3. **Run** `build_colmap_selection.py` → COLMAP export with selected images
4. **Run** `laz_to_ply.py` → aligned point cloud
5. **(Optional)** Generate masks for dynamic objects
6. **Train** in Spirula Studio or Lichtfeld Studio

## Next Steps

- [Setup](../setup/requirements) — install dependencies and prepare your environment
- [Data Access](../setup/data-access) — how to request Kappazunder data from Vienna
- [Pipeline Overview](../../pipeline/overview) — detailed workflow explanation
