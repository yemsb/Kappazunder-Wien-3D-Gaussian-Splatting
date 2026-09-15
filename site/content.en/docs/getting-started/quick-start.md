---
title: Quick Start
weight: 40
bookToC: true
---

# Quick Start

This guide walks through creating your first 3DGS reconstruction from Kappazunder data.

## Prerequisites

- Python 3.8+ with dependencies installed ([see Requirements](../setup/requirements))
- Kappazunder data downloaded for your region ([see Data Access](../setup/data-access))
- Basic understanding of coordinate systems and 3D reconstruction

## Overview

The complete workflow takes 4 steps:

1. **Prepare config** — define your scene with ROI polygon and data paths
2. **Export cameras** — run `build_colmap_selection.py` to filter images and generate COLMAP poses
3. **Convert LiDAR** — run `laz_to_ply.py` to create aligned point cloud
4. **Train 3DGS** — load the export into Spirula Studio or Lichtfeld Studio

Estimated time: 15-30 minutes for steps 1-3, depending on ROI size. 3DGS training (step 4) takes 30 minutes to several hours depending on scene complexity and GPU.

## Step 1: Prepare Your Config

Create a directory for your scene under `colmap_pipeline/configs/`:

```bash
cd colmap_pipeline/configs
mkdir my_scene
cd my_scene
```

### Define Your ROI

Create `AoI.csv` with your region of interest polygon as EPSG:31256 coordinates (one point per line: `x,y`):

```csv
341200.5,4320.3
341250.8,4340.1
341240.2,4380.6
341190.4,4360.2
341200.5,4320.3
```

{{% hint info %}}
**Finding Coordinates**: Use [QGIS](https://qgis.org/) or [geojson.io](http://geojson.io/) to draw your polygon in EPSG:31256. The polygon should close (first point = last point).
{{% /hint %}}

### Create Config File

Create `config.yaml`:

```yaml
EPSG: 31256
data_path: F:\kappazunder_data\Los_XXX  # Path to your downloaded Kappazunder data
max_frustum_distance: 10.0              # Maximum distance (m) from camera to ROI
include_bottom_facing_cameras: false    # Usually false (exclude downward cameras)
use_AoI: true                           # Set to false to export ALL images
invert_y_axis: true                     # Coordinate system adjustment
invert_z_axis: false                    # Usually false
voxel_size: 0.1                         # Point cloud voxel grid size (m)
```

**Key parameters:**
- `data_path` — absolute path to your unzipped Kappazunder `Los_*` directory
- `max_frustum_distance` — cameras further than this from your ROI are excluded (smaller = fewer images, faster processing)
- `include_bottom_facing_cameras` — set to true if you want downward-facing cameras (mostly unusable)

## Step 2: Export Cameras and Images

From the repository root:

```bash
cd colmap_pipeline
python build_colmap_selection.py --config configs/my_scene/config.yaml [--skip-images] [--skip-ply]
```

This script:
1. Loads image metadata and camera interior orientation from Kappazunder files
2. Computes camera frustums for each image
3. Filters to images where the frustum intersects your ROI polygon
4. Copies selected images to `colmap_export/images/` as `frame_######.jpg`
5. Writes COLMAP text format: `cameras.txt`, `images.txt`, `points3D.txt` (empty)
6. Saves `scene_origin.txt` for coordinate alignment

## Step 3: Convert LiDAR Point Cloud

The LiDAR conversion will be done by `build_colmap_selection.py`, unless you specify the `--skip-ply` flag. If you skipped or want to run manually, use:

```bash
python laz_to_ply.py --config configs/my_scene/config.yaml
```

This reads `scene_origin.txt` from step 2 and applies the same coordinate offset to the LiDAR data, producing `colmap_export/sparse/0/points3D_init.ply`.

## Step 4: Train 3DGS

Your `colmap_export/` directory now contains everything needed for 3DGS training:

```
colmap_export/
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── sparse/0/
    ├── cameras.txt
    ├── images.txt
    ├── points3D.txt
    └── points3D.ply
```

### Import into Spirula Studio or Lichtfeld Studio

{{< hint info >}}
Both tools support real-time preview during training. You can pause and resume training at any time.
{{< /hint >}}

## Verification

After training completes:
- **Visual quality**: Navigate the trained splat in the tool's viewer — check for artifacts, floaters, or missing geometry
- **Coverage**: Verify your ROI is fully reconstructed
- **Performance**: Real-time rendering (30+ FPS) is expected for scenes under 10M gaussians

## Next Steps

- [Understanding Kappazunder Data](../pipeline/kappazunder-data) — deep dive into the data format
- [Pipeline Details](../pipeline/camera-selection) — how image selection and pose conversion work

See the [Showcases](/showcases/) page for rendered results.
