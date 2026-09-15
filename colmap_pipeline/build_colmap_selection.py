import shutil
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from PIL import Image
import argparse
from rotation_conversion import colmap_pose_from_survey, sensor_role_from_pitch
import laz_to_ply
from utils import load_reduction_polygon, load_and_verify_config


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


def load_image_meta(path, epsg="EPSG:31256"):
    df = pd.read_csv(path, sep="\t")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_m"], df["y_m"]),
        crs=epsg,
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


def select_images(image_meta_gdf, roi_polygon, fov_by_sensor, max_dist, raw_images_root, include_bottom_facing_cameras=False):
    """Adds a 'selected' boolean column when the ROI covers over 25% of a frustum."""
    selected = []
    for _, row in image_meta_gdf.iterrows():
        fov = fov_by_sensor.get(row.sensor_id, np.radians(90))  # fallback
        sensor_role = sensor_role_from_pitch(row.sensor_id)
        # If up- or down-facing, use fov = 2*pi to make the frustum a full circle
        if sensor_role in {"up", "down"}:
            fov = 2 * np.pi
        frustum = make_frustum(row.x_m, row.y_m, row.rz_rad, fov, max_dist)
        if sensor_role == "down" and not include_bottom_facing_cameras:
            selected.append(False)
            continue
        intersection_area = frustum.intersection(roi_polygon).area
        selected.append(intersection_area > 0.25 * frustum.area)
    image_meta_gdf = image_meta_gdf.copy()
    image_meta_gdf["selected"] = selected

    # Check for each image whether there is a corresponding mask image file in <image_file_path>/<image_name>.jpg/masks/<image_name>.jpg
    # Make column for mask paths
    image_meta_gdf["mask_path"] = None
    for idx, row in image_meta_gdf.iterrows():
        image_file_path = Path(raw_images_root) / f"Trajektorie_{row.trajectory_id}" / f"Sensor_{row.sensor_id}" / row.image_name
        mask_file_path = image_file_path.parent / "masks" / row.image_name
        # Turn into absolute path
        mask_file_path = mask_file_path.resolve()
        if mask_file_path.exists():
            image_meta_gdf.at[idx, "mask_path"] = str(mask_file_path)

    print(f"Found {image_meta_gdf['mask_path'].notna().sum()} mask images for selected images.")
    return image_meta_gdf


def plot_selection(image_meta_gdf, roi_polygon, fov_by_sensor, max_dist, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    px, py = roi_polygon.exterior.xy
    ax.plot(px, py, "r-", linewidth=1.5, label="region of interest", zorder=10000)

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

    # In the background, plot openstreetmap tiles for context
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs="EPSG:31256", source=ctx.providers.OpenStreetMap.Mapnik, headers={'User-Agent': 'kappazunderUser/1.0 (https://github.com/yemsb/Kappazunder-Wien-3D-Gaussian-Splatting)'})
    except ImportError:
        print("contextily not installed, skipping background map tiles.")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    plt.tight_layout()
    plt.savefig(out_path, dpi=70)
    plt.close(fig)


def export_colmap(selected_gdf, interior_df, output_sparse_dir, frame_name_map,
                   scene_origin, invert_y_axis=True, invert_z_axis=False):
    """
    Writes cameras.txt, images.txt, points3D.txt (empty) in COLMAP text format.
    frame_name_map: dict mapping (sensor_id, image_id) -> "frame_000001.jpg"
    scene_origin: (ox, oy, oz) subtracted from every position before export,
                  to keep coordinates near zero for numerical stability.
    invert_y_axis: if True, mirror world Y during pose export to match
                   alternate downstream axis conventions.
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
                row.sensor_id, row.rz_rad, invert_y_axis=invert_y_axis, invert_z_axis=invert_z_axis
            )
            out_name = frame_name_map[(row.sensor_id, row.image_id)]
            f.write(f"{idx} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.6f} {ty:.6f} {tz:.6f} {row.sensor_id} {out_name}\n")
            f.write("\n")  # empty POINTS2D line

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
        f.write(f"invert_y_axis {int(invert_y_axis)}\n")

    print(f"Wrote {cameras_path}")
    print(f"Wrote {images_path}")
    print(f"Wrote {origin_path}  (offset: {ox:.3f}, {oy:.3f}, {oz:.3f})")


def copy_and_rename_images(selected_gdf, 
                           raw_images_root, 
                           output_images_dir, 
                           output_masks_dir,
                           copy=True,
                           invert_masks=False):
    """
    Builds linear frame_* names independent of sensor, and (optionally)
    copies the actual files. Returns the name map used by export_colmap.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    # Remove any existing files in the output_images_dir to avoid confusion
    for existing_file in output_images_dir.glob("frame_*.jpg"): 
        existing_file.unlink()
    for existing_mask in output_masks_dir.glob("frame_*.png"): 
        existing_mask.unlink()
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

            # Copy mask if it exists
            if row.mask_path:
                mask_src = Path(row.mask_path)
                mask_dst = output_masks_dir / out_name
                if mask_src.exists():
                    shutil.copy2(mask_src, mask_dst)
                else:
                    print(f"  [warning] source mask not found, skipped: {mask_src}")

                if invert_masks:
                    # Invert the mask image (assuming it's a black and white jpg)
                    try:
                        mask_image = Image.open(mask_dst)
                        inverted_mask = Image.eval(mask_image, lambda x: 255 - x)
                        inverted_mask.save(mask_dst)
                    except Exception as e:
                        print(f"  [warning] failed to invert mask {mask_dst}: {e}")
    return name_map


def process_los_directory(los_dir, config, reduction_polygon_):
    """
    Process a single Los_* directory: load metadata, select images, copy and rename,
    and export COLMAP files.
    """
    print(f"Processing {los_dir}")
    image_meta_path = los_dir / "Bild-Meta" / "image_meta.txt"
    interior_orientation_path = los_dir / "Bild-Meta" / "interior_orientation.txt"
    raw_images_root = los_dir / "Bild-Rohdaten"
    output_dir = config["dataset_output_dir"]
    # Load metadata
    image_meta_gdf = load_image_meta(image_meta_path, config["EPSG"])
    interior_df = load_interior_orientation(interior_orientation_path)
    fov_by_sensor = load_fov_per_sensor(interior_orientation_path)

    print(f"Loaded {len(image_meta_gdf)} image records.")
    print(f"FOV by sensor (deg): "
          f"{ {k: round(np.degrees(v),1) for k,v in fov_by_sensor.items()} }")

    if reduction_polygon_ is None:
        selected_gdf = image_meta_gdf.copy()
    else:
        # Select images based on frustum intersection with AoI
        image_meta_gdf = select_images(
            image_meta_gdf, reduction_polygon_, fov_by_sensor, config.get("max_frustum_distance"), raw_images_root
        )
        n_selected = image_meta_gdf["selected"].sum()
        print(f"Selected {n_selected} / {len(image_meta_gdf)} images "
            f"(frustum intersects ROI).")

        selection_preview_path = output_dir / "selection_preview.png"
        plot_selection(image_meta_gdf, reduction_polygon_, fov_by_sensor,
                    config.get("max_frustum_distance"), selection_preview_path)
        print(f"Saved selection preview: {selection_preview_path}")

        selected_gdf = image_meta_gdf[image_meta_gdf["selected"]].copy()

    if len(selected_gdf) == 0:
        print("No images selected -- check ROI polygon / MAX_FRUSTUM_DIST.")
        return selected_gdf

    # Center the scene on the selected images' centroid (in x,y; z kept
    # at the mean camera height so vertical values also stay small).
    scene_origin = (
        selected_gdf["x_m"].mean(),
        selected_gdf["y_m"].mean(),
        selected_gdf["z_m"].mean()
    )
    print(f"Scene origin: {scene_origin}")

    name_map = copy_and_rename_images(
        selected_gdf, raw_images_root, output_dir / "images", output_dir / "masks",
        copy=True, invert_masks=False
    )

    export_colmap(selected_gdf, interior_df, output_dir / "sparse" / "0", name_map,
                scene_origin, invert_y_axis=config.get("invert_y_axis", True),
                invert_z_axis=config.get("invert_z_axis", False))


def parse():
    # Parse input arguments and load config
    parser = argparse.ArgumentParser(description="COLMAP selection and export pipeline.")
    parser.add_argument("--config", type=str, help="Path to the config.yaml file.")
    parser.add_argument("--skip-images", action="store_true", default=False, help="Whether to skip copying and rename images (default: False).")
    parser.add_argument("--skip-ply", action="store_true", default=False, help="Whether to skip processing the PLY point cloud (default: False).")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    config_dir, config = load_and_verify_config(args)

    reduction_polygon = load_reduction_polygon(config_dir, config)

    print(f"Using AoI polygon: {reduction_polygon is not None}")

    if not args.skip_images:
        # Loop over the Los_* directories and process each one
        for los_dir in Path(config["data_path"]).glob("Los_*"):
            print(f"Processing {los_dir}")
            # Process each Los_* directory
            process_los_directory(los_dir, config, reduction_polygon)

    # PLY processing is costly
    if not args.skip_ply:
        print("Processing LAZ to PLY...")
        laz_to_ply.process_lidar_data(config, reduction_polygon)