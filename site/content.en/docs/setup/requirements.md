---
title: Requirements
weight: 10
bookToC: true
---

# Requirements

## Software

### Operating System
- **Linux**
- **Windows** 10/11 (recommended, tested on Windows 11)
- **macOS** (Lichtfeld Studio [not supported](https://github.com/MrNeRF/LichtFeld-Studio/wiki#requirements))

---

## Environment Setup

The pipeline supports three environment configurations depending on which components you need.

### Base Pipeline (Camera Selection + LiDAR Conversion)

For running `build_colmap_selection.py` and `laz_to_ply.py`:

**Conda environment** (use the provided `environment_kappazunder3DGS.yml`):
```yaml
name: kappazunder3DGS
channels:
  - conda-forge
  - ngsolve
  - defaults
dependencies:
  - pip
  - earthengine-api
  - notebook
  - conda-forge::geemap
  - matplotlib
  - geopandas
  - laspy
  - open3d
  - contextily
  - pip:
      - scipy
```

**Installation:**
```bash
conda env create -f environment_kappazunder3DGS.yml
conda activate kappazunder3DGS
```

### YOLO Segmentation (Optional)

For generating masks with `prepare_yolo_database.py`:

**Conda environment** (use the provided `environment_yolo.yml`):
```yaml
name: yolo
channels:
  - pytorch
  - nvidia
  - conda-forge
  - ngsolve
  - defaults
dependencies:
  - pip
  - pytorch
  - pytorch-cuda=12.1
  - torchvision
  - ultralytics
  - ipykernel
```

**Installation:**
```bash
conda env create -f environment_yolo.yml
conda activate yolo
```

### MoGe Depth/Normal

For depth and normal map generation with `get_depth_and_normal_data.py`:

**Installation:**

```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe
uv sync
```

---

## 3DGS Training Tools

One of the following for the final training step:

### Spirula Studio (Recommended)
- **Website**: [Spirula Studio GitHub](https://github.com/harry7557558/spirula-studio)
- **OS**: Windows 10/11, Linux, macOS
- **Hardware**: NVIDIA, AMD, Inter, and Apple GPUs supported
- **License**: Open-source (GPL-3.0)
- **Strengths**: User-friendly GUI, handy dataset generation tools, floater suppression and robustness modes, free pre-built binaries

### Lichtfeld Studio
- **Website**: [lichtfeld.io](https://lichtfeld.io)
- **OS**: Windows 10/11 (recommended), Linux (possible)
- **Hardware**: NVIDIA GPU with CUDA support
- **License**: Open-source (GPL-3.0)
- **Strengths**: Advanced training parameters, region of interest, plugin marketplace, large community and Discord server

---

## Next Steps

- [Data Access](./data-access) — request Kappazunder data from Vienna
- [Quick Start](../getting-started/quick-start) — run your first reconstruction
