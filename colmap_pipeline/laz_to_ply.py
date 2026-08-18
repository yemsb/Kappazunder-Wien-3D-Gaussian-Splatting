"""
Convert LAZ point cloud(s) to a colored PLY, applying the SAME scene origin
offset used in build_colmap_selection.py so the point cloud lines up with
the exported camera poses.

Usage:
    python3 laz_to_ply.py

Adjust the CONFIG section below. Supports merging multiple LAZ files
(e.g. several scandata_*.laz tiles covering your region) into one PLY.
"""

from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
import pandas as pd

# ----------------------------- CONFIG ---------------------------------

# One or more LAZ files to merge. For your real data this would be the
# scandata_*.laz files whose epoch range overlaps your selected images
# (see scan_meta.txt), or simply every LAZ tile inside your QGIS-cropped
# region if you already did spatial cropping there.
LAZ_PATHS = [
    Path(__file__).resolve().parent / "data" / "Trajektorie_15767_AoI_thin.laz",
]

OUTPUT_PLY = Path("./colmap_export/sparse/0/points3D_init.ply")

# Voxel size in meters for downsampling. None = no downsampling.
VOXEL_SIZE = 0.03

# ------------------------------------------------------------------------


def get_scene_origin():
	"""Read the scene origin from the colmap_export/scene_origin.txt file."""
	f = open("colmap_export/scene_origin.txt", "r")
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


def main():
    all_xyz = []
    all_rgb = []

    for path in LAZ_PATHS:
        xyz, rgb = read_laz_as_arrays(path)
        print(f"{path}: {len(xyz):,} points")
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    print(f"Total merged points: {len(xyz):,}")

    # Apply the same offset used for the camera poses
    ox, oy, oz = get_scene_origin()
    xyz_centered = xyz - np.array([ox, oy, oz])

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

    if VOXEL_SIZE is not None:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print(f"Voxel downsampled ({VOXEL_SIZE} m): {before:,} -> {len(pcd.points):,} points")

    OUTPUT_PLY.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(OUTPUT_PLY), pcd, write_ascii=False)
    print(f"Wrote {OUTPUT_PLY} ({len(pcd.points):,} points)")


if __name__ == "__main__":
    main()
