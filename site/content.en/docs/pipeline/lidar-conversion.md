---
title: LiDAR Conversion & Alignment
weight: 30
bookToC: true
---

# LiDAR Conversion & Alignment

The script `laz_to_ply.py` merges LAZ LiDAR scans, applies the same scene origin offset as used for camera poses, optionally adds a sky dome, flips Y and Z axes, voxel‑downsamples, and exports a colored PLY point cloud for 3DGS initialization.

## Overview

While `build_colmap_selection.py` handles image selection and camera export, `laz_to_ply.py` processes the complementary LiDAR data to create an initial point cloud. This point cloud serves two purposes:

1. **Initialization** — provides a rough geometric scaffold for 3DGS training  
2. **Scale & Alignment** — establishes real‑world scale and ensures LiDAR and camera coordinate systems match precisely  

The script reads LAZ‑compressed LiDAR files, applies the scene origin offset from `scene_origin.txt`, performs a fixed Y/Z axis flip, adds a hemispheric sky dome, voxel‑downsamples, and exports the result as `points3D.ply` (binary PLY) and `points3D.txt` (ASCII).

---

## 1. Input Files

The script expects the same directory structure as the camera selection script, focusing on the LiDAR components:

```
data_path/
├── Scan-Meta/
│  └─ scan_meta.txt       # LiDAR metadata (similar format to image_meta.txt)
├── Scan-Punktwolken/
│  └─ Trajektorie_*/
│     └─ Sensor_*/
│        └─ scandata_*.laz  # Compressed LiDAR point clouds
└─ Verortung/
   └─ trajectory_*.txt    # Optional ground truth trajectories
```

### scandata_*.laz Files

Each LAZ file contains:
- **XYZ coordinates** (in EPSG:31256 meters)
- **RGB colors** (normalized to [0,1] internally; stored as 8‑bit or 16‑bit in the file)

The Kappazunder dataset typically includes both XYZ and RGB.

### scan_meta.txt Format

Similar to `image_meta.txt` but for LiDAR scans:

| Column | Name | Description |
|--------|------|-------------|
| 0 | trajectory_id | Vehicle trajectory identifier |
| 1 | sensor_trajectory_id | Sensor trajectory identifier |
| 2 | datafile_id | Unique data file identifier |
| 3 | epoch_start_s | Start timestamp in seconds since epoch |
| 4 | epoch_end_s | End timestamp in seconds since epoch |
| 5 | scandata_name | Filename (matches `Scan-Punktwolken/`) |

The script does not use scan_meta.txt directly — it discovers LAZ files by scanning the directory structure.

---

## 2. Processing Steps

### 2.1 Dependency on Camera Selection

`laz_to_ply.py` requires that `build_colmap_selection.py` has been run first for the same configuration because it needs:
- `scene_origin.txt` — the metric offset calculated from camera poses  
- Implicit trust that the ROI and trajectory selection match  

The script verifies `scene_origin.txt` exists in the dataset output directory.

### 2.2 LAZ File Discovery Logic

The script checks for LAZ files in two locations, with priority given to user-provided files:

1. **Root directory check** — First scans `data_path` for any `*.laz` files  
   - If found, uses only these files (assumes user preprocessed/merged them)  
   - Filters out any `*.copc.laz` files if both versions exist  
2. **Fallback discovery** — If no LAZ files in root, recursively scans `Los_*/**/scandata_*.laz`  
   - One file per LiDAR scan trajectory/sensor combination  

This allows users to provide their own preprocessed LAZ file in the root directory to override the default discovery behavior.

### 2.3 Reading the Scene Origin

The file `scene_origin.txt` contains three lines with the offset:

```
offset_x_m -341200.5
offset_y_m -4320.3
offset_z_m 0.0
```

These values are read as `(ox, oy, oz)` and represent the **negative** centroid of selected camera positions.

### 2.3 Discovering LAZ Files

The script scans for LAZ files in two places:
1. Directly under `data_path` (if any `.laz` files exist there)  
2. Recursively under `Los_*/**/scandata_*.laz` (one file per LiDAR scan)

If both locations contain files, only the non‑`.copc.laz` files are used from the first location.

### 2.4 Loading and Merging Point Clouds

Each LAZ file is read with `laspy` to obtain:
- `xyz`: N×3 array of float64 coordinates (already in EPSG:31256)  
- `rgb`: N×3 array of float32 colors in the range [0,1]  

All trajectory/sensor files are concatenated into a single large point cloud.

### 2.5 Applying Scene Origin Offset

The scene origin offset is subtracted from all points to center the scene near the origin:

```python
xyz_centered = xyz - np.array([ox, oy, oz])
```

This mirrors the offset applied to camera poses in `build_colmap_selection.py`.

### 2.6 Adding a Sky Dome

To help with sky removal during 3DGS training, a hemisphere of 2000 grey points is generated:
- **Radius**: 2 × the maximum distance of any LiDAR point from the scene origin  
- **Distribution**: Uniform (deterministic Fibonacci spiral) on the upper hemisphere (z ≥ 0)  
- **Color**: Grey (RGB = 0.5, 0.5, 0.5)  

These points are appended to the point cloud before downsampling.

### 2.7 Axis Flip (Y and Z Negation)

The following fixed transformation is applied:
```python
xyz_centered[:, 1] *= -1  # Negate Y
xyz_centered[:, 2] *= -1  # Negate Z
```

This corresponds to a world‑axis remap of `(x, y, z) → (x, −y, −z)` and aligns the LiDAR point cloud with the camera coordinate system after the offsets from `build_colmap_selection.py`.

### 2.8 Voxel Downsampling

To reduce point cloud density for efficiency, voxel grid downsampling is applied:
```python
pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
```
The voxel size (in meters) is taken from `config.yaml`. Typical values range from 0.05 m (fine detail) to 0.2 m (coarse, less guidance).

### 2.9 Saving as PLY

The final point cloud is saved in two formats:
- **Binary PLY** (`points3D.ply`) — compact, preferred for 3DGS training  
- **ASCII PLY** (`points3D.txt`) — human‑readable, includes point ID and RGB  

{{< hint info >}}
Spirula Studio looks for either `points3D.txt` or `points3D.bin` when loading the initial point cloud (not the `.ply` file), while Lichtfeld Studio can read the binary `.ply` directly. Both formats contain the same data.
{{< /hint >}}

Each vertex contains:
- `x, y, z`: Position in EPSG:31256 meters (after all transformations)  
- `r, g, b`: Color as integers 0–255  

---

## 3. Configuration Parameters

All parameters come from the same `config.yaml` used by `build_colmap_selection.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EPSG` | string | `"31256"` | Coordinate system code (documented only) |
| `data_path` | string | *(required)* | Root path to Kappazunder `Los_*` directories |
| `voxel_size` | float | `0.1` | Voxel grid size for LiDAR downsampling (meters) |
| `use_AoI` | bool | `true` | Same parameters as the setting used in camera selection |

Note: LiDAR processing does not use `max_frustum_distance` or camera‑specific flags, since LiDAR provides 360° coverage per scan.

---

## 4. Validation and Quality Checks

The script performs the following checks:
- **File existence** — verifies LAZ files exist for the selected trajectories  
- **Data integrity** — checks for NaN/inf values in point coordinates  
- **Bounds reporting** — prints min/max of XYZ before and after offset  
- **Density reporting** — shows raw point count, post‑sky‑dome count, and final downsampled count  

Typical console output:
```
Total merged points: 12,458,932 points
Applying voxel downsampling (voxel_size=0.1 m): 12,458,932 -> 624,318 points
Wrote datasets/stadtpark/sparse/0/points3D.ply (624,318 points)
```

---

## 5. Coordinate System Alignment Verification

To verify that cameras and LiDAR are properly aligned:

1. **Visual check** — In Spirula/Lichtfeld Studio, the initialized point cloud should:  
   - Follow visible geometry (building facades, ground planes, vegetation)  
   - Not float freely in space or be offset from image features  
   - Have reasonable density matching image resolution  

2. **Quantitative check** — Project a subset of LiDAR points into each camera image; the mean reprojection error should be < 1–2 pixels for good alignment.  

3. **Scale check** — Known real‑world dimensions should match:  
   - Lane widths: ~3.5 m  
   - Building heights: typical Vienna buildings 10–30 m  
   - Tree trunk diameters: 0.3–1.0 m  

---

## 6. Running the Script

`laz_to_ply.py` is normally invoked automatically by `build_colmap_selection.py` unless `--skip-ply` is specified. To run it manually:

```bash
cd colmap_pipeline
python laz_to_ply.py --config configs/<scene_name>/config.yaml
```

There are no additional command‑line arguments; all settings come from `config.yaml`.

---

## 7. Output Files

Output is saved to `datasets/<scene_name>/sparse/0/`:

```
datasets/<scene_name>/sparse/0/
├── points3D.ply        # Binary PLY: position + color for each point
└── points3D.txt        # ASCII PLY: ID X Y Z R G B (one line per point)
```

The binary PLY is the file used by Spirula Studio and Lichtfeld Studio as the initial point cloud for 3DGS training.

---

## 8. Example: Stadtpark Processing

Continuing from the camera selection example:

```bash
# After running build_colmap_selection.py (which created scene_origin.txt)
cd colmap_pipeline

# Process LiDAR with the same Stadtpark config
python laz_to_ply.py --config configs/stadtpark/config.yaml

# Sample output:
# Total merged points: 12,458,932 points
# Applying voxel downsampling (voxel_size=0.1 m): 12,458,932 -> 624,318 points
# Wrote datasets/stadtpark/sparse/0/points3D.ply (624,318 points)
```

The resulting `points3D.ply` contains approximately 600K points covering the Stadtpark area, aligned to the camera poses exported in the previous step. This provides an excellent initialization for 3DGS training in Spirula or Lichtfeld Studio.

---

## 9. Next Steps

- [Masking Guide](./masking) — learn how to generate and use segmentation masks for dynamic object removal  
- [Configuration Reference](./configuration) — detailed explanation of all config parameters and their interactions  
- [Troubleshooting](../../troubleshooting/common-issues) — fixing common problems with LiDAR processing  
- [Training Guide](../../training/overview) — loading the export into Spirula Studio or Lichtfeld Studio