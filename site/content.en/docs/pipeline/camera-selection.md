---
title: Camera Selection & Pose Export
weight: 20
bookToC: true
---

# Camera Selection & Pose Export

The script `build_colmap_selection.py` filters raw Kappazunder image captures according to an Area of Interest (AoI) and exports camera poses in COLMAP format.

## Overview

Kappazunder data contains continuous image streams from six vehicle-mounted cameras across entire trajectories. Processing all images across a city would be computationally prohibitive and unnecessary for a localized 3D reconstruction.

`build_colmap_selection.py` handles:
1. **Camera Intrinsics & FOV** — Computes horizontal Field of View (FOV) per sensor from `interior_orientation.txt`.
2. **Frustum Construction** — Computes 2D circular sector frustums on the horizontal plane for each capture.
3. **Spatial Filtering** — Selects images whose frustum intersects the AoI polygon by more than 25% of the frustum area.
4. **Coordinate Transformation** — Converts Kappazunder poses (EPSG:31256 + vehicle heading) into world-to-camera quaternions and translation vectors for COLMAP.
5. **Image Staging** — Copies and renames selected images chronologically to sequential filenames (`frame_000001.jpg`, `frame_000002.jpg`, ...).
6. **COLMAP Export** — Writes `cameras.txt`, `images.txt`, and computes `scene_origin.txt`.

---

## 1. Camera Intrinsics & FOV Calculation

Each sensor's horizontal Field of View (FOV) in radians is calculated from the focal length {{< katex >}}c{{< /katex >}}, pixel size {{< katex >}}psu{{< /katex >}}, and image width in pixels {{< katex >}}pix\_u{{< /katex >}}:

{{< katex >}}
\text{FOV}_{\text{rad}} = 2 \cdot \arctan\left(\frac{psu\_mm \cdot pix\_u}{2 \cdot c\_mm}\right)
{{</ katex >}}

For standard Kappazunder perspective cameras ({{< katex >}}c = 16.399\text{ mm}{{< /katex >}}, {{< katex >}}psu = 0.0046\text{ mm}{{< /katex >}}, {{< katex >}}pix\_u = 7130\text{ px}{{< /katex >}}), this yields a horizontal FOV of approximately **90°**.

---

## 2. Sensor Roles & Frustum Geometry

The vehicle array consists of six sensors identified by `sensor_id` (determined by `sensor_id % 10`):

| Sensor ID Modulo | Role | Orientation | Frustum Type |
|---|---|---|---|
| `0` | **Up** | Pointing upwards (+Z) | Circle |
| `1` | **Front** | Facing driving direction | Sector oriented along heading direction (`rz_rad` in `Bild-Meta/image_meta.txt`) |
| `2` | **Right** | Facing vehicle right | Sector oriented along `rz_rad` - pi/2 |
| `3` | **Back** | Facing rear | Sector oriented along `rz_rad` + pi |
| `4` | **Left** | Facing vehicle left | Sector oriented along `rz_rad` + pi/2 |
| `5` | **Down** | Pointing towards ground (-Z) | Circle |

Exemplary frustums for two front-facing cameras (red and blue arcs), and a front/down-facing camera (blue circle):

{{< image src="images/frustum_example.png" alt="Sensor Roles" >}}

### Circular Sector Frustums

For horizontal cameras, a 2D circular sector polygon is constructed on the horizontal plane from the camera position (x_m, y_m), heading angle `rz_rad`, sensor FOV, and configured `max_frustum_distance`:

```python
angles = heading_rad + np.linspace(-fov_rad / 2, fov_rad / 2, n_arc_pts)
arc_x = x + max_dist * np.sin(angles)
arc_y = y + max_dist * np.cos(angles)
```

Vertical sensors (`up`, `down`) cover the full circular area (2 pi) around the vehicle position.

{{% hint info %}}
Down-facing cameras (Sensor 5) are excluded by default (`include_bottom_facing_cameras: false`) as they primarily capture the vehicle hood and immediate asphalt.
{{% /hint %}}

---

## 3. Spatial Selection Criteria

An image is selected if the intersection between its frustum polygon and the AoI polygon covers **more than 25%** of the frustum area:

```python
intersection_area = frustum.intersection(roi_polygon).area
selected = intersection_area > 0.25 * frustum.area
```

A preview plot (`selection_preview.png`) showing all camera frustums, colors per sensor role, the AoI polygon, and OpenStreetMap context (requires `contextily`) is automatically generated and saved to the dataset output directory.

---

## 4. Coordinate Transformation & COLMAP Format

COLMAP uses a **world-to-camera** convention:

{{< katex >}}
X_{\text{cam}} = R \cdot X_{\text{world}} + T
{{</ katex >}}

Where:
- Camera coordinates: {{< katex >}}+X{{</ katex >}} points right, {{< katex >}}+Y{{</ katex >}} points down, {{< katex >}}+Z{{</ katex >}} points forward along optical axis.
- {{< katex >}}R{{</ katex >}} is represented as a unit quaternion {{< katex >}}(q_w, q_x, q_y, q_z){{</ katex >}}
- {{< katex >}}T{{</ katex >}} is the camera translation in camera frame: {{< katex >}}T = -R \cdot X_{\text{world}}{{</ katex >}}

### Heading to Rotation Matrix

The transformation in `rotation_conversion.py` constructs the orthonormal camera basis in world coordinates:

1. **Heading vector**: {{< katex >}}r_z = 0\text{ rad}{{</ katex >}} points North ({{< katex >}}+Y{{</ katex >}} in EPSG:31256):
   {{< katex >}}
   \vec{h} = (\sin(r_z), \cos(r_z), 0)
   {{</ katex >}}
2. **Camera Forward ({{< katex >}}\vec{f}{{</ katex >}})**: Defined by vehicle heading for horizontal sensors, {{< katex >}}+Z{{</ katex >}} for `up`, and {{< katex >}}-Z{{</ katex >}} for `down`.
3. **Camera Up ({{< katex >}}\vec{u}{{</ katex >}})**: World {{< katex >}}+Z{{</ katex >}} for horizontal sensors; vehicle heading direction for vertical sensors.
4. **Camera Right ({{< katex >}}\vec{r}{{</ katex >}})** & **Down ({{< katex >}}\vec{d}{{</ katex >}})**:
   {{< katex >}}
   \vec{r} = \frac{\vec{f} \times \vec{u}}{\|\vec{f} \times \vec{u}\|}, \quad \vec{d} = \frac{\vec{f} \times \vec{r}}{\|\vec{f} \times \vec{r}\|}
   {{</ katex >}}
5. **World-to-Camera Rotation**: {{< katex >}}R_{\text{world}\to\text{cam}} = R_{\text{cam}\to\text{world}}^T = [\vec{r} \mid \vec{d} \mid \vec{f}]^T{{</ katex >}}

### Scene Origin Centering

As a cautionary practice, we center the scene origin around the cameras' mean position (e.g. EPSG:31256 coordinates around {{< katex >}}x \approx 341000\text{ m}, y \approx 4300\text{ m}{{</ katex >}}):

```python
scene_origin = (
    selected_gdf["x_m"].mean(),
    selected_gdf["y_m"].mean(),
    selected_gdf["z_m"].mean()
)
```

This origin (o_x, o_y, o_z) is subtracted from all camera coordinates and recorded in `scene_origin.txt` so that `laz_to_ply.py` can apply the exact same offset to LiDAR point clouds.

---

## 5. Running the Script

Execute `build_colmap_selection.py` with your scene configuration:

```bash
cd colmap_pipeline
python build_colmap_selection.py --config configs/<scene_name>/config.yaml
```

### CLI Arguments

| Argument | Type | Description |
|---|---|---|
| `--config`      | string | Path to `config.yaml` |
| `--skip-images` | flag   | Skip image copying/renaming and COLMAP text export |
| `--skip-ply`    | flag   | Skip automatic execution of `laz_to_ply` point cloud conversion |

---

## 6. Output Files

Output is saved to `datasets/<scene_name>/`:

```
datasets/<scene_name>/
├── selection_preview.png   # 2D visual verification plot
├── scene_origin.txt        # Metric offset applied to center the scene
├── images/                 # Renamed image files (frame_000001.jpg, ...)
├── masks/                  # Associated mask files (if existing)
└── sparse/0/
    ├── cameras.txt         # COLMAP camera intrinsics
    ├── images.txt          # COLMAP camera poses (quaternions + translations)
    ├── points3D.txt        # Point cloud in COLMAP text format
    └── points3D.ply        # Point cloud in binary PLY format (from laz_to_ply)
```