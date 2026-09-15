from pathlib import Path
import laspy
import numpy as np
import open3d as o3d
import pandas as pd
import rasterio
from utils import load_config, load_reduction_polygon, load_and_verify_config
import argparse


def get_scene_origin(config):
    """Read the scene origin from the colmap_export/scene_origin.txt file."""
    f = open(Path(config["dataset_output_dir"]) / "scene_origin.txt", "r")
    lines = f.readlines()[3:6]
    return pd.DataFrame([line.strip().split() for line in lines], columns=["axis", "value"]).values[:, 1].astype(float)


def read_laz_as_arrays(path):
    """Returns (xyz: Nx3 float64, rgb: Nx3 float32 in [0,1])."""
    with laspy.open(path) as f:
        las = f.read()

    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    # LAS color channels are typically stored as 16-bit (0-65535), but some
    # producers write 8-bit values (0-255) into the same field. Detect and
    # normalize accordingly rather than assuming.
    red, green, blue = las.red, las.green, las.blue
    max_val = max(red.max(), green.max(), blue.max())
    denom = 65535.0 if max_val > 255 else 255.0
    rgb = np.column_stack([red, green, blue]).astype(np.float64) / denom

    return xyz, rgb


def augment_dom(config, xyz, rgb, aoi_polygon=None):
    """Augment point cloud with DOM (Digital Orthophoto Model) data.

    Args:
        config: Configuration dictionary
        xyz: Existing point cloud coordinates (Nx3)
        rgb: Existing point cloud colors (Nx3)
        aoi_polygon: Optional Shapely Polygon to filter DOM points to the area of interest

    Returns:
        tuple: (xyz, rgb) arrays with DOM points added, or original arrays if skipped
    """
    if config.get("dom_path") is None:
        return xyz, rgb

    dom_path = Path(config["dom_path"])
    if not dom_path.exists():
        print(f"Warning: DOM path {dom_path} does not exist. Skipping DOM augmentation.")
        return xyz, rgb

    # Check if file is zipped and extract if necessary
    if dom_path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(dom_path, 'r') as zip_ref:
            zip_ref.extractall(dom_path.parent)
        # The archive is commonly named <id>_tif.zip while the extracted raster
        # is named <id>.tif, so replacing only the suffix is not sufficient.
        extracted_stem = dom_path.stem.removesuffix("_tif")
        candidates = [
            dom_path.parent / f"{extracted_stem}.tif",
            dom_path.parent / f"{extracted_stem}.tiff",
        ]
        dom_path = next((path for path in candidates if path.exists()), candidates[0])

    with rasterio.open(dom_path) as src:
        dom_data = src.read(1)
        dom_transform = src.transform
        dom_crs = src.crs

    # Warning about CRS - user should verify it matches
    print(f"Warning: DOM CRS is {dom_crs.to_string()[:80]}...")
    print(f"  Please verify this matches your expected EPSG:{config['EPSG']} projection.")

    # Turn the raster data into a point cloud and merge it with the existing point cloud.
    # Use array operations instead of iterating over every raster cell.
    rows, cols = np.nonzero(~np.isnan(dom_data))
    z = dom_data[rows, cols]
    # Use rasterio's xy method to correctly transform pixel coordinates to spatial coordinates
    x, y = rasterio.transform.xy(dom_transform, rows, cols)
    dom_xyz = np.column_stack((x, y, z))

    # Filter DOM points to AoI polygon if provided
    if aoi_polygon is not None:
        from matplotlib.path import Path as MplPath
        # Vectorized point-in-polygon test (fast even for millions of points)
        # The AoI polygon and DOM coords are in the same CRS (verified by the warning above)
        mpl_path = MplPath(aoi_polygon.exterior.coords)
        mask = mpl_path.contains_points(np.column_stack([x, y]).astype(np.float64))
        total = dom_xyz.shape[0]
        dom_xyz = dom_xyz[mask]
        print(f"Filtered DOM points to AoI: {len(dom_xyz):,} points remain (from {total:,})")

    # Merge the DOM points with the existing point cloud
    if len(dom_xyz) > 0:
        xyz = np.vstack([xyz, dom_xyz])
        rgb = np.vstack([rgb, np.ones((dom_xyz.shape[0], 3)) * 0.5])  # Grey color for DOM points
        print(f"Added {len(dom_xyz):,} DOM points to point cloud")

    return xyz, rgb


def process_lidar_data(config: dict, aoi_polygon=None):
    all_xyz = []
    all_rgb = []

    # If there is already a laz file in the dataset_output_dir, take this instead of searching for LAZ files in the data_path => files were already processed in a previous run

    if Path(config["data_path"]).glob("*.laz"):
        laz_paths = list(Path(config["data_path"]).glob("*.laz"))
        # If there are two files <file>.copc.laz and <file>.laz, only take the <file>.laz file
        if len(laz_paths) > 1:
            laz_paths = [path for path in laz_paths if not path.name.endswith(".copc.laz")]
    else:
        laz_paths = list(Path(config["data_path"]).glob("Los_*/**/scandata_*.laz"))

    for path in laz_paths:
        xyz, rgb = read_laz_as_arrays(path)
        print(f"{path}: {len(xyz):,} points")
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    xyz, rgb = augment_dom(config, xyz, rgb, aoi_polygon)


    print(f"Total merged points: {len(xyz):,}")

    # Apply the same offset used for the camera poses
    ox, oy, oz = get_scene_origin(config)
    xyz_centered = xyz - np.array([ox, oy, oz])

    # To deal with the sky, put a hemisphere of 1k-2k grey points around the scene origin
    sky_dome_radius = 2 * max(np.linalg.norm(xyz_centered, axis=1))  # 2x the furthest point
    num_sky_points = 2000
    # Uniform (deterministic, not random!) distribution of points on a hemisphere
    # Spiral (Fibonacci) Sphere Method
    indices = np.arange(0, num_sky_points, dtype=float) + 0.5
    # Restrict the dome to the upper hemisphere (z >= 0).
    phi = np.arccos(1 - indices / num_sky_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.ones(num_sky_points) * sky_dome_radius

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Combine the sky points with the existing point cloud
    sky_xyz = np.column_stack([x, y, z])
    xyz_centered = np.vstack([xyz_centered, sky_xyz])
    rgb = np.vstack([rgb, np.ones((num_sky_points, 3)) * 0.5])  # Grey color for sky points

    # if APPLY_WORLD_AXIS_REMAP:
    #     # (x,y,z) -> (x,z,-y): swap Y/Z and negate new Z. Proper rotation
    #     # (det=+1) but mirrors the top-down view -- only enable this if you
    #     # specifically need Y-up world space AND have also set
    #     # APPLY_WORLD_AXIS_REMAP=True in build_colmap_selection.py, AND you
    #     # account for the top-down mirroring elsewhere.
    #     xyz_centered = xyz_centered[:, [0, 2, 1]]
    #     xyz_centered[:, 2] *= -1
    xyz_centered[:, 1] *= -1
    xyz_centered[:, 2] *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_centered)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    if config.get("voxel_size") is not None:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
        print(f"Voxel downsampled ({config['voxel_size']} m): {before:,} -> {len(pcd.points):,} points")

    output_ply_path = Path(config["dataset_output_dir"]) / "sparse" / "0" / "points3D.ply"
    output_txt_path = Path(config["dataset_output_dir"]) / "sparse" / "0" / "points3D.txt"

    o3d.io.write_point_cloud(str(output_ply_path), pcd, write_ascii=False)
    print(f"Wrote {output_ply_path} ({len(pcd.points):,} points)")

    with open(output_txt_path, "w") as f:
        f.write("# 3D points\n")
        f.write(f"# Number of points: {len(pcd.points):,}\n")
        f.write("# Columns: POINT3D_ID X Y Z R G B\n")
        for i, (point, color) in enumerate(zip(pcd.points, pcd.colors)):
            f.write(f"{i} {point[0]} {point[1]} {point[2]} {int(color[0]*255)} {int(color[1]*255)} {int(color[2]*255)}\n")


def parse():
    parser = argparse.ArgumentParser(description="Convert LAZ to PLY with scene origin offset.")
    parser.add_argument("--config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    config_dir, config = load_and_verify_config(args)

    reduction_polygon = load_reduction_polygon(config_dir, config)

    process_lidar_data(config, reduction_polygon)