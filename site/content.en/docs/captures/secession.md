---
title: Secession Capture Walkthrough
weight: 40
bookToC: true
---

# Secession Capture Walkthrough

The Secession is one of the world's oldest independent exhibition venues expressly dedicated to contemporary art.
Designed by Joseph Maria Olbrich, it was built in 1897 as an architectural manifesto for the Vienna Secession, a group of artists who had seceded from the Association of Austrian Artists.
The building's conception has a [long history](https://artsandculture.google.com/story/uAUBHhOdD2y9LA) and contributed greatly to the development of Jugendstil (Art Nouveau), with its golden dome and the motto "To every age its art, to art its freedom" (Ludwig Hevesi) inscribed above the entrance.

## Overview

The Secession features:
- The Secession Building
- A small park area behind the building
- Wide streets surrounding the building (which is lucky for us, as it allows for a good number of images to be captured from different angles)
- The [Academy of Fine Arts Vienna](https://www.akbild.ac.at/) (Akademie der bildenden Künste Wien) right next to the Secession
- Mixed seasons (June and October)
- Mixed lighting conditions
- Lots of transient objects (cars, people, bicycles, etc.)

{{< image src="images/secession_front_facing-ezgif.com-optimize.gif">}} 

{{% hint warning %}}
**Medium difficulty** due to the size of the area (3.4 ha compared to 1.2 ha for [Kolonitzplatz](./kolonitzpark/) and 0.3 ha for [Stadtpark](./stadtpark/)), the change of seasons and lighting, and the presence of many transient objects.
SfM calibration using COLMAP with prior positions used here again (see [previous reconstruction](kolonitzpark/#revisited-using-colmap-with-prior-positions-for-sfm)).
{{% /hint %}}

### Area of Interest (AoI)
```
# EPSG: 31256
# x_m		y_m
2312.7901	340325.4572
2340.5447	340360.1103
2371.1069	340335.8852
2397.2572	340323.8529
2486.4570	340294.4939
2503.9440	340349.8427
2533.3029	340339.8960
2538.2763	340337.6500
2521.1102	340283.7450
2550.6295	340275.5630
2580.7906	340307.8097
2613.9999	340285.5098
2655.7120	340228.2358
2617.6898	340201.4438
2605.6575	340189.8927
2565.0683	340160.8547
2518.3828	340119.3029
2459.5045	340184.5985
2462.2319	340186.0424
2432.0708	340222.9416
2399.6637	340256.1509
2345.9192	340300.1090
2312.7901	340325.4572
```

{{< image src="images/secession_selection_preview.png" >}}

After [coordinate-based selection](../pipeline/camera-selection/) and an acceptance border of 10 meters around the AoI, 2500 images (~8 GB) were selected.

## Pre- and Post-processing

### Prior-informed SfM

Expanding upon the prior-informed SfM used in the [Kolonitzpark](kolonitzpark/#revisited-using-colmap-with-prior-positions-for-sfm) walkthrough, I want to detail the specific commands and additional scripts used to generate the poses and the match graph for this reconstruction.

First, the "vanilla" sparse model with images is created using the prior positions from Kappazunder directly (see [Camera Selection & Pose Export](../pipeline/camera-selection/#5-running-the-script)):

```bash
python build_colmap_selection.py --config configs/secession/config.yaml
```

{{% details "Click to expand the script output" %}}

This generates the following output (truncated for brevity):
```text
< Print of the config file >
Processing F:\Documents\kappazunder_secession\Los_6A
Loaded 3024 image records.
FOV by sensor (deg): {110010: np.float64(90.0), 110011: np.float64(90.0), 110012: np.float64(90.0), 110013: np.float64(90.0), 110014: np.float64(90.0), 110015: np.float64(90.0), 110020: np.float64(90.0), 110021: np.float64(90.0), 110022: np.float64(90.0), 110023: np.float64(90.0), 110024: np.float64(90.0), 110025: np.float64(90.0), 110030: np.float64(90.0), 110031: np.float64(90.0), 110032: np.float64(90.0), 110033: np.float64(90.0), 110034: np.float64(90.0), 110035: np.float64(90.0)}
Using coordinate-based selection (skipping view direction pruning)...
Selected 2500 / 3024 images (coordinate-based selection, excluding down-facing cameras).
Excluded 504 down-facing camera images.
Saved selection preview: datasets\secession\selection_preview.png
Scene origin: (np.float64(2481.586586), np.float64(340244.25486399996), np.float64(17.692314))
Created sensor subdirectories: ['logs', 'Sensor_110010', 'Sensor_110011', 'Sensor_110012', 'Sensor_110013', 'Sensor_110014', 'Sensor_110020', 'Sensor_110021', 'Sensor_110022', 'Sensor_110023', 'Sensor_110024', 'Sensor_110030', 'Sensor_110031', 'Sensor_110032', 'Sensor_110033', 'Sensor_110034']
Wrote datasets\secession\spirula_positions.txt
Wrote datasets\secession\spirula_positions_zup.txt
Wrote datasets\secession\sparse\0\cameras.txt
Wrote datasets\secession\sparse\0\images.txt
Wrote datasets\secession\scene_origin.txt  (offset: 2481.587, 340244.255, 17.692)
Processing LAZ to PLY...
F:\Documents\kappazunder_secession\secession_AoI.laz: 1,592,385 points
Warning: DOM CRS is PROJCS["MagWien_MGI_M34",GEOGCS["unnamed",DATUM["unnamed",SPHEROID["Bessel 1841"...
  Please verify this matches your expected EPSG:31256 projection.
Filtered DOM points to AoI: 75,753 points remain (from 25,010,001)
Warning: DOM CRS is PROJCS["MagWien_MGI_M34",GEOGCS["unnamed",DATUM["unnamed",SPHEROID["Bessel 1841"...
  Please verify this matches your expected EPSG:31256 projection.
Filtered DOM points to AoI: 61,512 points remain (from 25,010,001)
Added 137,265 DOM points to point cloud
Total merged points: 1,729,650
Voxel downsampled (0.25 m): 1,731,650 -> 1,522,337 points
Wrote datasets\secession\sparse\0\points3D.ply (1,522,337 points)

Z convention check -- the rig height is constant, so the tighter spread wins:
    invert_z_axis=False  n= 1910  median   3.974 m   spread p10-p90   3.373 m
    invert_z_axis=True   n= 2470  median   2.062 m   spread p10-p90   0.050 m
    expected rig height (dh_m): 2.090 m
    -> DISAGREES with the provisional export (False): re-exporting the poses with invert_z_axis=True

Wrote datasets\secession\spirula_positions.txt
Wrote datasets\secession\spirula_positions_zup.txt
Wrote datasets\secession\sparse\0\cameras.txt
Wrote datasets\secession\sparse\0\images.txt
Wrote datasets\secession\scene_origin.txt  (offset: 2481.587, 340244.255, 17.692)
```

{{% /details %}}

Next, SIFT features are extracted assuming a pinhole camera model with a single focal length of 3565 pixels (as provided by Kappazunder) and a maximum of 8192 features per image:

```bash
$S = "Kappazunder-Wien-3D-Gaussian-Splatting\colmap_pipeline\datasets\secession"
New-Item -ItemType Directory -Force -Path "$S\sift_prior\logs" | Out-Null
colmap feature_extractor --image_path "$S\images_by_sensor" --database_path "$S\sift_prior\database.db" --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --ImageReader.camera_params "3565,3565,3565,3565" --SiftExtraction.max_num_features 8192 --FeatureExtraction.use_gpu 1 --log_path "$S\sift_prior\logs"
```

Next, the prior camera positions from Kappazunder are used to seed the database with the initial camera poses and to create a match graph for COLMAP. The `prepare_prior_db.py` script is used to prepare the database for COLMAP's spatial and transitive matching:

```bash
python prepare_prior_db.py --workspace "$S\sift_prior" --stages priors,focal
```

Based on the prior positions, spatial matching is performed with a maximum distance of 30 metres and a maximum of 50 neighbours:

```bash
colmap spatial_matcher --database_path "$S\sift_prior\database.db" --SpatialMatching.max_distance 30 --SpatialMatching.max_num_neighbors 50 --SpatialMatching.ignore_z 1 --FeatureMatching.use_gpu 1 --log_path "$S\sift_prior\logs"
```

To strengthen the match graph and connect regions that may be visible from afar, transitive matching is performed with 3 iterations:

```bash
colmap transitive_matcher --database_path "$S\sift_prior\database.db" --TransitiveMatching.num_iterations 3 --FeatureMatching.use_gpu 1 --log_path "$S\sift_prior\logs"
```

Finally, the sparse reconstruction is performed using the prior positions and the match graph generated above. 
The bundle adjustment is configured to not refine the focal length, principal point, or extra parameters, and GPU acceleration is enabled:

```bash
New-Item -ItemType Directory -Force -Path "$S\sift_prior\sparse" | Out-Null
colmap pose_prior_mapper --database_path "$S\sift_prior\database.db" --image_path "$S\images_by_sensor" --output_path "$S\sift_prior\sparse" --Mapper.ba_refine_focal_length=0 --Mapper.ba_refine_principal_point=0 --Mapper.ba_refine_extra_params=0 --Mapper.ba_use_gpu=1 --log_path "$S\sift_prior\logs"
```

In principle, the model is ready to be trained.
However, to properly georeference the model, the sparse reconstruction is aligned to the survey points using the `align_sfm_to_survey.py` script.
A distance- and angle-based filtering between prior and posterior positions is applied to remove outliers:

```bash
python align_sfm_to_survey.py --config configs/secession/config.yaml --model datasets/secession/sift_prior/sparse/0 --output datasets/secession/sift_prior/georef/0 --filter-tau 1.0 --filter-deg 1.0
```

Tau is the maximum distance in metres between the prior and posterior positions, and deg is the maximum angle in degrees between the prior and posterior camera orientations.

The georeferencing is then applied to the point cloud:

```bash
python export_sfm_points.py --config configs/secession/config.yaml --sfm-model datasets/secession/sift_prior/sparse/0 --target datasets/secession/sift_prior/georef/0_clean/
```

### Clean-up via Clean-GS

[Clean-GS](https://github.com/smlab-niser/clean-gs), while originally conceived to isolate a central object from a cluttered background, can also be used to remove e.g. the sky in a large model.

Previously, step 1 of the Clean-GS pipeline **whitelisted** Gaussians once their projection onto a specific view was in the camera's view frustum and overlapping with the acceptance mask.
However, in a large scene like ours, this results in practically all Gaussians being whitelisted from some view (since there is no occlusion check).

Instead, the rule was changed to **blacklist** Gaussians that are in the view frustum of a specific camera _and_ outside the acceptance mask.
This makes the filtering meaningful for our sky-masking: without fine-grained occlusion checks, the only distinction we can make is whether a region of the image is occluded by an object in the foreground (e.g. a building) or not (e.g. the sky).

There is one caveat, however: since 45% of blacklistings rely only on a single view's vote, the blacklist is not very robust.
To mitigate this, a removal threshold of 3 votes is used (via `--min_blacklist`).
Higher values result in more Gaussians being kept, but also more sky remaining in the model.

Other changes to the Clean-GS code involve speed-ups by turning loops into vectorized operations.

## Reconstruction

First, the Secession itself:

{{< image src="images/secession_view14.jpg" >}}

{{< image src="images/secession_view1.jpg" >}}

{{< image src="images/secession_view4.jpg" >}}

{{< image src="images/secession_view13.jpg" >}}

{{< image src="images/secession_view12.jpg" >}}

The close-by Naschmarkt area (left) and a [prominent building](https://www.bezirksmuseum.at/wp-content/uploads/2024/05/getreidemarkt_1.pdf) (centre; architect: Carl Stephann, 1907):

{{< image src="images/secession_view2.jpg" >}}

{{< image src="images/secession_view8.jpg" >}}

The Academy of Fine Arts during renovations:

{{< image src="images/secession_view5.jpg" >}}

The Audimax building (centre front) and the Campus Getreidemarkt of the Technical University of Vienna:

{{< image src="images/secession_view6.jpg" >}}

Other views:

{{< image src="images/secession_view7.jpg" >}}

{{< image src="images/secession_view10.jpg" >}}

{{< image src="images/secession_view11.jpg" >}}


*Note that cars and people were masked in this reconstruction.*

### Model Availability

[SuperSplat](https://superspl.at/scene/4d402ce1)

---

## Next Steps

- [Data Access](./../setup/data-access) — request Kappazunder data from Vienna
- [Quick Start](../getting-started/quick-start) — run your first reconstruction
- [Pipeline Overview](./../pipeline/overview) — understand the end-to-end workflow