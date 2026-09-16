# Kappazunder Wien 3D Gaussian Splatting

Small collection of scripts and notebooks for turning parts of the City of Vienna's [Kappazunder](https://digitales.wien.gv.at/projekt/kappazunder/) street-view data into assets that can be used for 3D Gaussian Splatting (3DGS). It is mainly a practical data-preparation project: image selection, COLMAP pose export, LiDAR conversion, and a few optional masking and depth/normal tools.

The pipeline is useful for experimenting with selected areas, but Kappazunder data has limited overlap, changing lighting, and many dynamic objects. Results therefore vary by capture and area.

### Showcase

You can find splats created using this pipeline on the [Showcases](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/showcases/) page of this project's documentation or directly on [SuperSplat](https://superspl.at/user/yems). Details of the reconstructions can be found under [Walkthroughs](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/docs/captures/).

## Getting started

1. Request a small area containing both imagery and LiDAR through Vienna's [Geoportal](https://geoportal.wien.gv.at/map/main/geodaten/) (`Mobile Mapping Download` -> `Kappazunder LIDAR` and `Kappazunder PANO`).
2. Extract the download and create a scene directory under `colmap_pipeline/configs/` with a `config.yaml` and an `AoI.csv` region-of-interest polygon.
3. Install the environment for the part of the pipeline you want to use.
4. Run the camera selection and LiDAR conversion scripts below.

The [project documentation](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/) contains the longer setup, data, configuration, and pipeline notes.

## Kappazunder data structure

The downloaded data is organized into trajectory directories. The main files used here are image metadata and camera intrinsics in `Bild-Meta/`, images in `Bild-Rohdaten/`, and LAZ point clouds in `Scan-Punktwolken/`:

```text
data_path/
|-- Los_*/
|   |-- Bild-Meta/
|   |   |-- image_meta.txt
|   |   |-- interior_orientation.txt
|   |   `-- multisys.txt
|   |-- Bild-Rohdaten/Trajektorie_*/Sensor_*/*.jpg
|   |-- Scan-Meta/scan_meta.txt
|   `-- Scan-Punktwolken/Trajektorie_*/Sensor_*/scandata_*.laz
`-- Verortung/trajectory_*.txt
```

Coordinates are in EPSG:31256 (MGI / Austria GK East). The configuration's `data_path` should point to the directory containing the `Los_*` directories, not to one individual trajectory.

## Installation

For the core camera and LiDAR pipeline, use the supplied Conda environment:

```bash
conda env create -f environment_kappazunder3DGS.yml
conda activate kappazunder3DGS
```

The optional YOLO mask workflow has its own environment:

```bash
conda env create -f environment_yolo.yml
conda activate yolo
```

The MoGe dependencies can also be installed with `uv sync` in a cloned [MoGe](https://github.com/microsoft/MoGe) checkout. The notebooks in `MoGe3_pipeline/` use that setup.

## Usage

Put the scene parameters and paths in `colmap_pipeline/configs/<scene_name>/config.yaml`, then run:

```bash
cd colmap_pipeline
python build_colmap_selection.py --config configs/<scene_name>/config.yaml
# Optional (is executed automatically by build_colmap_selection):
python laz_to_ply.py --config configs/<scene_name>/config.yaml
```

`build_colmap_selection.py` selects images whose camera frustums intersect the AoI, exports COLMAP camera files, and writes the scene origin used for alignment. `laz_to_ply.py` merges and voxel-downsamples the corresponding LAZ data into a colored PLY point cloud. The second command is normally run after the first; the first script can also run it automatically unless `--skip-ply` is supplied.

Optional tools include `yolo_segmentation/prepare_yolo_database.py`, the mask notebooks, and `MoGe3_pipeline/get_normal_depth.ipynb`. The resulting COLMAP export and point cloud can then be imported into a 3DGS tool such as [Spirula Studio](https://github.com/harry7557558/spirula-studio) or [Lichtfeld Studio](https://lichtfeld.io).

## Documentation

For details, see the [Kappazunder 3DGS documentation](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/), especially:

- [Quick start](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/docs/getting-started/quick-start/)
- [Requirements and installation](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/docs/setup/requirements/)
- [Data access](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/docs/setup/data-access/)
- [Pipeline overview](https://yemsb.github.io/Kappazunder-Wien-3D-Gaussian-Splatting/docs/pipeline/overview/)
