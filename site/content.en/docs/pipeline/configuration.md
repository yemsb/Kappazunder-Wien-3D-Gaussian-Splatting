---
title: Configuration Reference
weight: 15
bookToC: true
---

# Configuration Reference

The pipeline uses YAML configuration files to specify processing parameters, replacing the original approach of editing script headers directly.

## Overview

Each scene configuration lives in `colmap_pipeline/configs/<scene_name>/` and contains:
- `config.yaml` — all processing parameters  
- `AoI.csv` — region of interest polygon (tab-separated X, Y coordinates in EPSG:31256)  

This approach enables:
- Reproducible experiments  
- Version control of processing parameters  
- Quick switching between configurations  
- Sharing configurations without modifying scripts  

---

## Configuration File Structure

A typical `config.yaml`:

```yaml
# Coordinate system
EPSG: "31256"

# Data locations
data_path: F:\kappazunder_data\Los_XXXXXX

# Processing parameters
max_frustum_distance: 10.0          # meters
include_bottom_facing_cameras: false
use_AoI: true
invert_y_axis: true
invert_z_axis: false
voxel_size: 0.1                     # meters
```

---

## Parameter Reference

### Coordinate System

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EPSG` | string | `"31256"` | Coordinate system code (documented only; does not affect calculations) |

All Kappazunder data uses **EPSG:31256** (MGI / Austria GK East), a projected system with units in meters.

### Data Locations

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data_path` | string | **Yes** | Root directory containing `Los_*` trajectory directories |

The path should point to the directory **containing** the `Los_*` subdirectories, not to a specific `Los_*` directory itself.

### Camera Selection Parameters

These parameters affect `build_colmap_selection.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_frustum_distance` | float | `10.0` | Maximum distance (meters) from camera to ROI for frustum calculation. Cameras farther than this are excluded. <br><br>**Typical values:**<br>- `5.0`: Very selective (nearby cameras only)<br>- `10.0`: Default (good balance)<br>- `20.0`: Less selective (more coverage)<br>- `50.0`+: Nearly all cameras considered |
| `include_bottom_facing_cameras` | bool | `false` | Whether to include downward-facing cameras (sensor ID % 10 == 5). <br><br>**Enable when:**<br>- Reconstructing building interiors or underground areas<br>- Roof details matter<br><br>**Disable when** (default):<br>- Street-level reconstruction<br>- Downward cameras capture only vehicle hood and immediate ground |
| `use_AoI` | bool | `true` | Master switch for ROI filtering. <br><br>`true`: Only process images whose frustum intersects `AoI.csv` polygon by >25%<br>`false`: Process ALL images from all trajectories in `data_path` (ignores `AoI.csv`) |
| `invert_y_axis` | bool | `true` | Whether to negate Y during coordinate conversion. <br><br>**Almost always `true`**: Compensates for the axis difference between Kappazunder (EPSG:31256) and COLMAP conventions. |
| `invert_z_axis` | bool | `false` | Whether to negate Z during coordinate conversion. <br><br>**Almost always `false`**: Both systems use +Z up. Only enable if debugging coordinate alignment issues. |

### LiDAR Processing Parameters

These parameters affect `laz_to_ply.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voxel_size` | float | `0.1` | Voxel grid size for LiDAR downsampling (meters). Larger values = more aggressive downsampling = faster processing but less detail. <br><br>**Guidelines:**<br>- `0.05` (5cm): High detail<br>- `0.10` (10cm): Default balance<br>- `0.20` (20cm): Lower detail<br>- `0.50` (50cm): Very coarse (terrain only)<br><br>**Rule of thumb:** Adapt voxel size so 10–50% of the final Gaussian count is initialized from the LiDAR point cloud. |

---

## Next Steps

- [Pipeline Overview](./overview) — how configuration fits into the end-to-end workflow
- [Camera Selection Details](./camera-selection) — deep dive into frustum calculation and filtering  
- [LiDAR Processing Details](./lidar-conversion) — voxelization and coordinate alignment  
- [Masking Guide](./masking) — removing dynamic objects for cleaner reconstructions
