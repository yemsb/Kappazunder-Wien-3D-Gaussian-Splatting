---
title: Requirements
weight: 10
bookToC: true
---

{{% hint warning %}}
This page is WIP!
{{% /hint %}}

# Requirements

## Hardware

### Minimum
- **CPU**: 4 cores, x86_64
- **RAM**: 16 GB
- **Storage**: 50 GB free (for one medium-sized dataset + outputs)
- **GPU**: Not required for pipeline scripts

### Recommended
- **CPU**: 8+ cores for faster processing
- **RAM**: 32 GB for large datasets
- **Storage**: 200+ GB if working with multiple scenes
- **GPU**: NVIDIA GPU with 8+ GB VRAM for 3DGS training (RTX 3060 Ti or better)

{{< hint info >}}
The pipeline scripts (`build_colmap_selection.py`, `laz_to_ply.py`) run on CPU only. GPU is only needed for downstream 3DGS training in Spirula/Lichtfeld Studio.
{{< /hint >}}

## Software

### Operating System
- **Linux** (Ubuntu 20.04+ recommended)
- **Windows** 10/11 (tested on Windows 11)
- **macOS** (Intel or Apple Silicon)

### Python
- **Python 3.8 or newer**
- `pip` package manager

Verify your Python version:
```bash
python --version  # or python3 --version
```

### Required Python Packages

Install via pip:

```bash
pip install numpy pandas geopandas shapely matplotlib Pillow tqdm opencv-python laspy open3d pyyaml
```

Or use the provided requirements file (if available):

```bash
pip install -r requirements.txt
```

**Package purposes:**
- `numpy`, `pandas` — data manipulation
- `geopandas`, `shapely` — geospatial operations (ROI polygon intersection)
- `matplotlib`, `Pillow` — visualization and image handling
- `tqdm` — progress bars
- `opencv-python` — image operations (optional, for masking pipeline)
- `laspy` — LAZ point cloud reading
- `open3d` — PLY point cloud writing and visualization
- `pyyaml` — config file parsing

### Optional: YOLO Segmentation

If you plan to generate masks for dynamic object removal:

```bash
pip install ultralytics  # YOLO v8/v11
pip install torch torchvision  # PyTorch (for YOLO training)
```

### Optional: MoGe3 Depth/Normal

For depth and normal map generation (experimental):

```bash
pip install moge3  # Or follow MoGe3 installation instructions
```

## 3DGS Training Tools

One of the following for the final training step:

### Spirula Studio (Recommended)
- **Website**: [spirulastudio.com](https://spirulastudio.com)
- **Requirements**: Windows 10/11, NVIDIA GPU (RTX 2060 or better)
- **License**: Commercial tool (free trial available)
- **Strengths**: User-friendly GUI, real-time preview, automatic mask generation

### Lichtfeld Studio
- **Website**: [lichtfeldstudio.com](https://lichtfeldstudio.com)
- **Requirements**: Windows 10/11, NVIDIA GPU
- **License**: Commercial tool
- **Strengths**: Advanced training parameters, high-quality results

{{< hint warning >}}
Both Spirula and Lichtfeld Studio are Windows-only. If you're working on Linux/macOS, you can run the pipeline scripts there and transfer the `colmap_export/` directory to a Windows machine for training.
{{< /hint >}}

## Verification

Test your environment:

```bash
# Check Python packages
python -c "import numpy, pandas, geopandas, shapely, laspy, open3d, yaml; print('All packages OK')"

# Check GPU (if available)
nvidia-smi  # Should show your NVIDIA GPU
```

## Disk Space Planning

Approximate storage requirements per scene:

| Component | Size (Medium Scene) | Notes |
|-----------|---------------------|-------|
| Raw Kappazunder data | 10-50 GB | Depends on ROI size and trajectory count |
| Selected images | 1-5 GB | After ROI filtering |
| LiDAR PLY | 100-500 MB | After voxel downsampling |
| COLMAP export (total) | 1-5 GB | Images + metadata |
| Trained 3DGS model | 500 MB - 2 GB | Depends on scene complexity |

Budget ~20-50 GB per scene including working files and outputs.

## Next Steps

- [Data Access](../data-access) — request Kappazunder data from Vienna
- [Quick Start](../../getting-started/quick-start) — run your first reconstruction
