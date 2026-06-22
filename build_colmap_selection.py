"""
Full pipeline:
  1. Read image_meta.txt, build a frustum polygon per image.
  2. Keep images whose frustum intersects the region-of-interest polygon.
  3. Visualize the selection against the ROI.
  4. Copy selected images into one flat folder with linear naming (frame_*).
  5. Export poses for the selection in COLMAP format (cameras.txt, images.txt,
     empty points3D.txt -- you're supplying the point cloud separately).

Adjust the CONFIG section, then run:
    python3 build_colmap_selection.py
"""

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from rotation_conversion import colmap_pose_from_survey, sensor_role_from_pitch

# ----------------------------- CONFIG ---------------------------------

EPSG = "EPSG:31256"

IMAGE_META_PATH = "../LiDAR_kappazunder_stadtpark/Los_6A/Bild-Meta/image_meta.txt"
INTERIOR_ORIENTATION_PATH = "../LiDAR_kappazunder_stadtpark/Los_6B/Bild-Meta/interior_orientation.txt"
# Raw images live at: <RAW_IMAGES_ROOT>/Trajektorie_<trajectory_id>/Sensor_<sensor_id>/<image_name>
RAW_IMAGES_ROOT = "../LiDAR_kappazunder_stadtpark/Los_6A/Bild-Rohdaten"

OUTPUT_DIR = Path("./colmap_export")
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_SPARSE_DIR = OUTPUT_DIR / "sparse" / "0"

MAX_FRUSTUM_DIST = 10.0  # meters
COPY_IMAGES = True  # set True once paths point at your real data

reduction_polygon_coords = np.array([
    [3599.8874170715394, 340907.17366641754],
    [3648.3580812379278, 340882.9493204322],
    [3616.96347532236,   340824.96835747146],
    [3571.3034212154816, 340861.6948829265],
    [3599.8874170715394, 340907.17366641754],
])
REDUCTION_POLYGON = Polygon(reduction_polygon_coords)

INCLUDE_BOTTOM_FACING_CAMERAS = False
# INVERT_Y_AXIS = True # on export

# ------------------------------------------------------------------------


def horizontal_fov_rad(c_mm: float, psu_mm: float, pix_u: int) -> float:
    sensor_width_mm = psu_mm * pix_u
    return 2 * np.arctan((sensor_width_mm / 2) / c_mm)


def make_frustum(x, y, heading_rad, fov_rad, max_dist, n_arc_pts=12):
    """Circular-sector frustum polygon, same convention as your plotting code."""
    half_fov = fov_rad / 2
    angles = heading_rad + np.linspace(-half_fov, half_fov, n_arc_pts)
    arc_x = x + max_dist * np.sin(angles)
    arc_y = y + max_dist * np.cos(angles)
    poly_x = np.concatenate(([x], arc_x))
    poly_y = np.concatenate(([y], arc_y))
    return Polygon(zip(poly_x, poly_y))


def load_image_meta(path):
    df = pd.read_csv(path, sep="\t")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_m"], df["y_m"]),
        crs=EPSG,
    )
    return gdf


def load_interior_orientation(path):
    df = pd.read_csv(path, sep="\t", index_col=False)
    
    # If the last data column is an all-zero column (the stray trailing 0), drop it
    _last_col = df.columns[-1]
    _last_col_num = pd.to_numeric(df[_last_col], errors="coerce")
    if _last_col_num.notna().all() and (_last_col_num == 0).all():
        df = df.iloc[:, :-1]
    
    # Ensure sensor_id is a regular column (not the dataframe index)
    if "sensor_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "sensor_id"})
    else:
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=True)
    return df


def load_fov_per_sensor(path):
    """Returns {sensor_id: fov_rad} from interior_orientation.txt."""
    df = load_interior_orientation(path)
    fovs = {}
    for _, row in df.iterrows():
        fovs[row.sensor_id] = horizontal_fov_rad(row.c_mm, row.psu_mm, row.pix_u)
    return fovs


def select_images(image_meta_gdf, roi_polygon, fov_by_sensor, max_dist):
    """Adds a 'selected' boolean column based on frustum intersection."""
    selected = []
    for _, row in image_meta_gdf.iterrows():
        fov = fov_by_sensor.get(row.sensor_id, np.radians(90))  # fallback
        sensor_role = sensor_role_from_pitch(row.sensor_id)
        # If up- or down-facing, use fov = 2*pi to make the frustum a full circle
        if sensor_role in {"up", "down"}:
            fov = 2 * np.pi
        frustum = make_frustum(row.x_m, row.y_m, row.rz_rad, fov, max_dist)
        if sensor_role == "down" and not INCLUDE_BOTTOM_FACING_CAMERAS:
            selected.append(False)
            continue
        selected.append(frustum.intersects(roi_polygon))
    image_meta_gdf = image_meta_gdf.copy()
    image_meta_gdf["selected"] = selected
    return image_meta_gdf


def plot_selection(image_meta_gdf, roi_polygon, fov_by_sensor, max_dist, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    px, py = roi_polygon.exterior.xy
    ax.plot(px, py, "r-", linewidth=1.5, label="region of interest")

    for is_selected, group in image_meta_gdf.groupby("selected"):
        # label = "selected" if is_selected else "not selected"
        color_dict = {"front": "k", "right": "r", "back": "b", "left": "g", "up": "m", "down": "c"}
        for _, row in group.iterrows():
            fov = fov_by_sensor.get(row.sensor_id, np.radians(90))
            sensor_role = sensor_role_from_pitch(row.sensor_id)
            # If up- or down-facing, use fov = 2*pi to make the frustum a full circle
            if sensor_role in {"up", "down"}:
                fov = 2 * np.pi
            frustum = make_frustum(row.x_m, row.y_m, row.rz_rad, fov, max_dist * (1 + list(color_dict).index(sensor_role) * 0.05))
            fx, fy = frustum.exterior.xy
            if is_selected:
                color = color_dict.get(sensor_role, "0.75")  # default gray if role unknown
            else:
                color = "0.75"
            ax.plot(fx, fy, c=color, linewidth=0.5, alpha=0.7)
        # ax.scatter(group.x_m, group.y_m, s=5, c=color, label=label, zorder=5)

    # de-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys())

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def export_colmap(selected_gdf, interior_df, output_sparse_dir, frame_name_map,
                   scene_origin):
    """
    Writes cameras.txt, images.txt, points3D.txt (empty) in COLMAP text format.
    frame_name_map: dict mapping (sensor_id, image_id) -> "frame_000001.jpg"
    scene_origin: (ox, oy, oz) subtracted from every position before export,
                  to keep coordinates near zero for numerical stability.
    """
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    ox, oy, oz = scene_origin

    # --- cameras.txt: one entry per sensor_id (they share intrinsics) ---
    cameras_path = output_sparse_dir / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for _, row in interior_df.iterrows():
            sensor_id = int(row.sensor_id)
            width = int(row.pix_u)
            height = int(row.pix_v)
            focal_px = row.c_mm / row.psu_mm  # convert focal length to pixels
            cx, cy = width / 2, height / 2
            # PINHOLE model: fx, fy, cx, cy
            f.write(f"{sensor_id} PINHOLE {width} {height} "
                    f"{focal_px:.6f} {focal_px:.6f} {cx:.2f} {cy:.2f}\n")

    # --- images.txt: one entry per selected image ---
    images_path = output_sparse_dir / "images.txt"
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] (empty -- no 2D-3D correspondences supplied)\n")
        for idx, row in enumerate(selected_gdf.itertuples(), start=1):
            qw, qx, qy, qz, tx, ty, tz = colmap_pose_from_survey(
                row.x_m - ox, row.y_m - oy, row.z_m - oz,
                row.sensor_id, row.rz_rad
            )
            out_name = frame_name_map[(row.sensor_id, row.image_id)]
            f.write(f"{idx} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.6f} {ty:.6f} {tz:.6f} {row.sensor_id} {out_name}\n")
            f.write("\n")  # empty POINTS2D line

    # --- points3D.txt: empty, point cloud supplied separately ---
    points_path = output_sparse_dir / "points3D.txt"
    with open(points_path, "w") as f:
        f.write("# 3D point list -- intentionally empty.\n")
        f.write("# Point cloud is supplied separately (converted from LAZ).\n")

    # --- scene_origin.txt: record the offset for later reuse ---
    origin_path = output_sparse_dir.parent.parent / "scene_origin.txt"
    with open(origin_path, "w") as f:
        f.write(f"# Subtracted from EPSG:31256 x_m,y_m,z_m to center the scene.\n")
        f.write(f"# Apply the SAME offset to your cropped LiDAR PLY so it lines\n")
        f.write(f"# up with these camera poses (see laz_to_ply step).\n")
        f.write(f"offset_x_m {ox:.6f}\n")
        f.write(f"offset_y_m {oy:.6f}\n")
        f.write(f"offset_z_m {oz:.6f}\n")
        f.write(f"# world_axis_remap: whether (x,y,z) -> (x,z,-y) was applied\n")
        f.write(f"# to these camera poses. laz_to_ply.py's swap-Y/Z-and-negate\n")
        f.write(f"# step MUST match this, or cameras and points will be rotated\n")
        f.write(f"# relative to each other.\n")

    print(f"Wrote {cameras_path}")
    print(f"Wrote {images_path}")
    print(f"Wrote {points_path}")
    print(f"Wrote {origin_path}  (offset: {ox:.3f}, {oy:.3f}, {oz:.3f})")


def copy_and_rename_images(selected_gdf, raw_images_root, output_images_dir,
                            copy=True):
    """
    Builds linear frame_* names independent of sensor, and (optionally)
    copies the actual files. Returns the name map used by export_colmap.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    # Remove any existing files in the output_images_dir to avoid confusion
    for existing_file in output_images_dir.glob("frame_*.jpg"):
        existing_file.unlink()
    name_map = {}
    # Sort by epoch_s so frame numbers roughly follow capture order
    ordered = selected_gdf.sort_values("epoch_s")
    for i, row in enumerate(ordered.itertuples(), start=1):
        out_name = f"frame_{i:06d}.jpg"
        name_map[(row.sensor_id, row.image_id)] = out_name
        if copy:
            src = (Path(raw_images_root)
                   / f"Trajektorie_{row.trajectory_id}"
                   / f"Sensor_{row.sensor_id}"
                   / row.image_name)
            dst = output_images_dir / out_name
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  [warning] source image not found, skipped: {src}")
    return name_map


if __name__ == "__main__":
    image_meta_gdf = load_image_meta(IMAGE_META_PATH)
    interior_df = load_interior_orientation(INTERIOR_ORIENTATION_PATH)
    fov_by_sensor = load_fov_per_sensor(INTERIOR_ORIENTATION_PATH)
    pitch_by_sensor = interior_df.set_index("sensor_id")["pitch_rad"].to_dict()

    image_meta_gdf = image_meta_gdf.copy()
    image_meta_gdf["pitch_rad"] = image_meta_gdf["sensor_id"].map(pitch_by_sensor)

    print(f"Loaded {len(image_meta_gdf)} image records.")
    print(f"FOV by sensor (deg): "
          f"{ {k: round(np.degrees(v),1) for k,v in fov_by_sensor.items()} }")

    image_meta_gdf = select_images(
        image_meta_gdf, REDUCTION_POLYGON, fov_by_sensor, MAX_FRUSTUM_DIST
    )
    n_selected = image_meta_gdf["selected"].sum()
    print(f"Selected {n_selected} / {len(image_meta_gdf)} images "
          f"(frustum intersects ROI).")

    plot_path = OUTPUT_DIR / "selection_preview.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_selection(image_meta_gdf, REDUCTION_POLYGON, fov_by_sensor,
                   MAX_FRUSTUM_DIST, plot_path)
    print(f"Saved selection preview: {plot_path}")

    selected_gdf = image_meta_gdf[image_meta_gdf["selected"]].copy()

    # debug_csv_path = OUTPUT_DIR / "debug_camera_positions_qgis.csv"
    # export_debug_csv_for_qgis(selected_gdf, debug_csv_path)

    if len(selected_gdf) == 0:
        print("No images selected -- check ROI polygon / MAX_FRUSTUM_DIST.")
    else:
        # Center the scene on the selected images' centroid (in x,y; z kept
        # at the mean camera height so vertical values also stay small).
        scene_origin = (
            selected_gdf["x_m"].mean(),
            selected_gdf["y_m"].mean(),
            selected_gdf["z_m"].mean(),
        )
        print(f"Scene origin (EPSG:31256): "
              f"x={scene_origin[0]:.3f} y={scene_origin[1]:.3f} z={scene_origin[2]:.3f}")

        name_map = copy_and_rename_images(
            selected_gdf, RAW_IMAGES_ROOT, OUTPUT_IMAGES_DIR, copy=COPY_IMAGES
        )
        export_colmap(selected_gdf, interior_df, OUTPUT_SPARSE_DIR, name_map,
                      scene_origin)
        print(f"\nDone. COLMAP-format export at: {OUTPUT_DIR}")
        print(f"  {OUTPUT_DIR}/images/        <- (renamed) images"
              f"{'(not copied -- COPY_IMAGES=False)' if not COPY_IMAGES else ''}")
        print(f"  {OUTPUT_DIR}/sparse/0/      <- cameras.txt, images.txt, points3D.txt")
        print(f"  {OUTPUT_DIR}/scene_origin.txt  <- offset, reuse for LiDAR PLY export")