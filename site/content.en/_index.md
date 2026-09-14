---
layout: landing
---

<br />

# Street View to 3DGS {anchor=false}

This page belongs to the GitHub repository [Kappazunder Wien 3D Gaussian Splatting](https://github.com/yemsb/Kappazunder-Wien-3D-Gaussian-Splatting).
With this open-source project, I will explain how to create a 3D Gaussian Splatting model from publicly available street view images and LiDAR data of Vienna, Austria.
In the city's [Kappazunder project](https://digitales.wien.gv.at/projekt/kappazunder/), the city of Vienna has made a large amount of street view images and LiDAR data from 2020 available to the public.

{{<button href="/docs/getting-started/introduction">}}Documentation{{</button>}}

<br />
<br />

{{< video src="https://github.com/yemsb/Kappazunder-Wien-3D-Gaussian-Splatting/releases/download/assets/2026-09-08_20-08-56.mp4" >}}

<br />

## What This Project Does

This repository contains tools to transform Vienna's Kappazunder street-view dataset into 3D Gaussian Splatting reconstructions:

- **Geometry Pipeline**: Convert synchronized multi-camera images and LiDAR scans into camera poses and point clouds
- **Region of Interest Selection**: Automatically select only images that see your area of interest using camera frustum intersection
- **Coordinate Transformation**: Handle Kappazunder's Austria-specific coordinate system (EPSG:31256) and vehicle-mounted camera orientations
- **Point Cloud Alignment**: Convert LAZ LiDAR files to PLY with proper alignment to camera poses
- **Masking Support**: Generate segmentation masks to remove dynamic objects like vehicles and pedestrians
- **3DGS Ready Output**: Produce COLMAP-format exports compatible with Spirula Studio and Lichtfeld Studio

## Key Features

🎯 **Precise ROI Selection** - Process only the imagery you need from hundreds of thousands of captures  
📐 **Accurate Georeferencing** - Maintain real-world scale and positioning through careful coordinate handling  
🚗 **Dynamic Object Removal** - Optional masking eliminates ghosting and floaters from moving vehicles/pedestrians  
🔧 **Standalone Scripts** - No compilation required, just Python with standard scientific libraries  
📊 **Configurable Workflow** - YAML-based configuration makes experiments reproducible and shareable  

## Typical Workflow

1. **Request Data** - Use Vienna's geoportal to get Kappazunder imagery and LiDAR for your area
2. **Configure** - Set up your scene with ROI polygon and data paths in YAML config
3. **Export Cameras** - Run `build_colmap_selection.py` to filter images and generate COLMAP poses
4. **Convert LiDAR** - Run `laz_to_ply.py` to create aligned point cloud for initialization
5. **(Optional) Mask** - Generate segmentation masks for dynamic object removal
6. **Train 3DGS** - Load the export into Spirula Studio or Lichtfeld Studio for final reconstruction

## Get Started

Begin with the [Introduction](../docs/getting-started/introduction) to understand the pipeline, then follow the [Quick Start](../docs/getting-started/quick-start) guide for your first reconstruction.