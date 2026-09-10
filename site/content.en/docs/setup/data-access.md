---
title: Data Access
weight: 20
bookToC: true
---

# Data Access

How to obtain Kappazunder street-view and LiDAR data from the City of Vienna.

## The Kappazunder Project

The [Kappazunder project](https://digitales.wien.gv.at/projekt/kappazunder/) captured Vienna's streets in 2020 using vehicle-mounted multi-camera arrays and LiDAR sensors. The dataset includes:

- **Images**: Six synchronized cameras (front, back, left, right, up, down)
- **LiDAR**: Full 360° field of view
- **Metadata**: GPS/IMU trajectories, camera intrinsics/extrinsics, timing information
- **Coverage**: Major streets, parks, and public spaces across Vienna

## Accessing the Data

Vienna provides access to this dataset through their geoportal. The process is:


1. Go to: [https://geoportal.wien.gv.at/map/main/geodaten/](https://geoportal.wien.gv.at/map/main/geodaten/)
2. Click on **"Mobile Mapping Download"**
3. Expand **"Dataset"** → select **"Kappazunder LIDAR"** & **Kappazunder PANO**
4. Use the map navigation tools to zoom to your area of interest
5. Draw a surface polygon
6. Request the download and enter your email address
7. Verify your email address (you will receive a confirmation email)
8.  Processing typically takes **a few hours to days** depending on area size and system load
9.  An email will contain a **download link** valid for **7 days**
10. Download and extract the ZIP file, which has the following structure:

```
your_data_directory/
├── Los_*/                     # One directory per trajectory
│   ├─ Bild-Meta/
│   │  ├─ image_meta.txt       # Camera poses and metadata
│   │  ├─ interior_orientation.txt  # Camera intrinsics
│   │  └─ multisys.txt         # Sensor-to-vehicle transforms
│   ├─ Bild-Rohdaten/
│   │  └─ Trajektorie_*/       # Images organized by trajectory/sensor
│   │     └─ Sensor_*/
│   │        └─ *.jpg
│   ├─ Scan-Meta/
│   │  └─ scan_meta.txt        # LiDAR metadata
│   └─ Scan-Punktwolken/
│     └─ Trajektorie_*/
│       └─ Sensor_*/
│         └─ scandata_*.laz    # Compressed LiDAR point clouds
└─ Verortung/
   └─ trajectory_*.txt         # Ground truth trajectories
```

## What You'll Receive

### Image Metadata (`Bild-Meta/image_meta.txt`)

Tab-separated file with one row per image:

| Column | Description | Units |
|--------|-------------|-------|
| `trajectory_id` | Vehicle trajectory identifier | integer |
| `sensor_id` | Camera identifier (0-5) | integer |
| `image_id` | Unique image identifier | integer |
| `epoch_s` | Timestamp in seconds since epoch | seconds |
| `image_name` | Matches the filenames in `Bild-Rohdaten` (one image for the six sensors) | string |
| `x_m` | Camera position X | meters (EPSG:31256) |
| `y_m` | Camera position Y | meters (EPSG:31256) |
| `z_m` | Camera position Z | meters (EPSG:31256) |
| `rx_rad` | Rotation X (-pi, -pi/2, or 0) | radians |
| `ry_rad` | Rotation Y (always 0) | radians |
| `rz_rad` | Rotation Z (orientation compared to north) | radians |

### Camera Interior Orientation (`Bild-Meta/interior_orientation.txt`)

One row per sensor (camera):

| Column | Description | Units |
|--------|-------------|-------|
| `sensor_id` | Camera identifier (0-5) | integer |
| `type_projection` | Camera projection type (always `p` for perspective) | string |
| `c_mm` | Focal length (always 16.399 mm) | millimeters |
| `psu_mm` | Pixel size (always 0.0046 mm) | millimeters |
| `psv_mm` | Pixel size (always 0.0046 mm) | millimeters |
| `pix_u` | Image width (always 7130) | pixels |
| `pix_v` | Image height (always 7130) | pixels |
| `dh_m` | Sensor mounting height (2.04 or 2.09 m) | meters |
| `pitch_rad` | Sensor mounting pitch (-pi/2, 0, or pi/2) | radians |

{{% hint warning %}}
There are 9 column headers but always 10 values per row (the last column contains a trailing 0).
{{% /hint %}}

### LiDAR Metadata (`Scan-Meta/scan_meta.txt`)

The following columns describe each LiDAR scan:

| Column | Description | Units |
|--------|-------------|-------|
| `trajectory_id` | Vehicle trajectory identifier | integer |
| `sensor_trajectory_id` | Sensor trajectory identifier | integer |
| `datafile_id` | Unique data file identifier | integer |
| `epoch_start_s` | Start timestamp in seconds since epoch | seconds |
| `epoch_end_s` | End timestamp in seconds since epoch | seconds |
| `scandata_name` | Matches the filenames in `Scan-Punktwolken` (one scan for each sensor) | string |

### LiDAR Data (`Scan-Punktwolken/*/scandata_*.laz`)

LAZ-compressed LiDAR point clouds containing:
- XYZ coordinates (EPSG:31256)
- RGB colors

{{% hint info %}}
Inspect the LAZ files using [QGIS](https://www.qgis.org/) or [CloudCompare](https://www.danielgm.net/cc/).
{{% /hint %}}

## Tips for Successful Requests

### Area Size
Keep your polygon small (e.g., a few city blocks) to avoid long processing times.

### Data Selection
- For 3DGS, you typically need **both** images and LiDAR
- Images alone work for pose estimation but lack scene scale
- LiDAR alone gives geometry but no texture/color

Typically, areas with tight camera coverage with multiple passes work best, like in this example of the MAK (museum of applied arts):

{{< image src="images/mak_coverage.png" >}}

### Coordinate System
All coordinates are in **EPSG:31256** (MGI / Austria GK East), a projected coordinate system using meters.

### File Naming
Trajectory IDs like `Los_6A` correspond to specific vehicle runs. You may receive multiple trajectories covering overlapping areas.

## Next Steps

Once you have your data:
1. Follow the [Quick Start](../getting-started/quick-start) guide
2. Or dive into [Pipeline Details](../../pipeline/overview) for deeper understanding