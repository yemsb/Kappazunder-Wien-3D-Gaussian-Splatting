#!/usr/bin/env python3
"""
Copy the SfM reconstruction's own point cloud into the georeferenced model, in the
SAME frame as the aligned cameras.

Why this is not a plain file copy: the SfM model's `points3D.bin` lives in Spirula's
own gauge, while `georef/<model>/images.txt` holds cameras that the alignment step
already moved by `(s, R, t)`. Those two are not identical - the kolonitzplatz fit is
scale 1.005936 with a 0.43 degree rotation, which displaces points by ~1-2 m across a
150 m scene. Copying the file verbatim would seed the Gaussians 1-2 m away from the
very cameras that must render them.

The transform is read from `scene_transform.json`, so only a transform actually
computed by the aligner is ever applied - never a hand-written one.

Writes into the target model directory:
    points3D.bin       COLMAP binary, transformed      (what a sparse reader picks up)
    points3D_sfm.ply   the same points, for viewing

NOTE on coexistence: a LiDAR cloud may also be present as `points3D.ply` in the same
directory (written by laz_to_ply.py). That is a DIFFERENT cloud in the same frame.
Which one a trainer initialises from depends on its own reader order, so check which
it picks before drawing conclusions from a training run.

Usage:
    python export_sfm_points.py --config configs/kolonitzplatz/config.yaml
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_points3D_bin(path):
    """
    Read a COLMAP points3D.bin.

    Layout per point: id (u64), xyz (3 x f64), rgb (3 x u8), error (f64),
    track_length (u64), then track_length entries of (image_id u32, point2D_idx u32).
    """
    pts, cols = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                                    # point3D_id
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)                                    # reprojection error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, 1)                     # (image_id, point2D_idx) pairs
            pts.append(xyz)
            cols.append(rgb)
    return np.array(pts, dtype=float), np.array(cols, dtype=np.uint8)


def write_points3D_bin(path, xyz, rgb):
    """Write a COLMAP points3D.bin with empty tracks (2D observations are not needed
    for initialising Gaussians, and we do not have them post-transform)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i, (p, c) in enumerate(zip(xyz, rgb)):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *p))
            f.write(struct.pack("<3B", *c))
            f.write(struct.pack("<d", 0.0))              # error
            f.write(struct.pack("<Q", 0))                # empty track


def write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(xyz)}\n".encode())
        for k in ("x", "y", "z"):
            f.write(f"property double {k}\n".encode())
        for k in ("red", "green", "blue"):
            f.write(f"property uchar {k}\n".encode())
        f.write(b"end_header\n")
        rec = np.empty(len(xyz), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sfm-model", default=None,
                    help="SfM model dir (default <dataset>/sfm/sparse/0).")
    ap.add_argument("--target", default=None,
                    help="Where to write (default <dataset>/georef/0_clean).")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    dataset = Path(__file__).resolve().parent / "datasets" / config_path.parent.name

    sfm_dir = Path(args.sfm_model) if args.sfm_model else dataset / "sfm" / "sparse" / "0"
    pts_bin = sfm_dir / "points3D.bin"
    if not pts_bin.exists():
        sys.exit(f"No SfM point cloud at {pts_bin}")

    target = Path(args.target) if args.target else dataset / "georef" / "0_clean"
    if not (target / "images.txt").exists():
        sys.exit(f"Target {target} has no images.txt - run the aligner first, so a "
                 f"matching scene_transform.json exists.")

    xform = json.loads((target / "scene_transform.json").read_text())["train_from_world"]
    s = xform["scale"]
    R = np.array(xform["rotation"]["matrix_3x3"])
    t = np.array(xform["translation"])
    print(f"transform: scale {s:.6f}, rotation {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))):.2f} deg, "
          f"|t| {np.linalg.norm(t):.3f} m")

    xyz, rgb = read_points3D_bin(pts_bin)
    print(f"read {len(xyz):,} points from {pts_bin}")

    xyz_aligned = s * (xyz @ R.T) + t
    print(f"point displacement from the transform: "
          f"median {np.median(np.linalg.norm(xyz_aligned - xyz, axis=1)):.3f} m, "
          f"max {np.max(np.linalg.norm(xyz_aligned - xyz, axis=1)):.3f} m")

    write_points3D_bin(target / "points3D.bin", xyz_aligned, rgb)
    write_ply(target / "points3D_sfm.ply", xyz_aligned, rgb)
    print(f"wrote {target / 'points3D.bin'} and {target / 'points3D_sfm.ply'}")

    if (target / "points3D.ply").exists():
        print(f"\nNote: {target / 'points3D.ply'} also exists (the LiDAR cloud, same "
              f"frame). A reader may prefer either file - confirm which one your "
              f"trainer initialises from.")


if __name__ == "__main__":
    main()
