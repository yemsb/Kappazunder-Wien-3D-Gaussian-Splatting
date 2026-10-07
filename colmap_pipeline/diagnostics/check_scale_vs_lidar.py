#!/usr/bin/env python3
"""
Measure a reconstructed model's VERTICAL scale against the LiDAR cloud.

Why this exists: SfM alone cannot tell you whether a reconstruction is
anisotropically distorted. For a rig whose cameras are near-yaw-only over a
near-planar trajectory, there is a near-degeneracy between each camera's `fy` and
a vertical scaling of the scene - shrinking the world along the up axis is almost
exactly compensated by enlarging `fy`, so reprojection error stays low either way.
Neither orientation residuals nor camera-position residuals catch it, because a
vertical scale preserves every camera's forward/up/right directions and barely
moves a trajectory whose own vertical spread is under a metre.

The LiDAR is the independent ruler. The camera-to-ground distance is a known,
physically fixed quantity (~2 m for this rig), so measuring it exposes a vertical
scale error directly:

    correct vertical scale  ->  gap matches the survey's own gap
    world squashed by 1/k   ->  gap comes out gap/k

Measured on the expanded kolonitzplatz model this reads 2.012 m against the
survey's 2.019 m (ratio 1.003) - i.e. that model's vertical scale is sound, which
is what a similarity fit alone could not have told us.

FRAME CONVENTION: the pipeline's export frame is Z-DOWN (+Z points physically
down), so "below the camera" means LARGER z and the ground is the deepest surface.
Do not flip this to Z-up without flipping the comparison too.

Usage:
    python check_scale_vs_lidar.py --poses <georef/0/images.txt> \
        --lidar <points3D.ply> [--reference-images <survey images.txt>]

    # compare a model against the survey reference in one call
    python check_scale_vs_lidar.py \
        --poses datasets/kolonitzplatz_SS_expanded/georef/0/images.txt \
        --lidar datasets/kolonitzplatz_SS_expanded/georef/0/points3D.ply \
        --reference-images datasets/kolonitzplatz/sparse/0/images.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Lives in diagnostics/ but imports its siblings from the parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colmap_io import quat_to_rotmat  # noqa: E402


def read_ply_xyz(path):
    """
    Read xyz from a binary_little_endian PLY with double x/y/z and uchar rgb.

    Written by laz_to_ply.py and export_sfm_points.py, both of which emit that exact
    layout (27 bytes per vertex). Parsed by hand so this does not need open3d, whose
    DLLs only load inside an activated conda env.
    """
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    if marker not in raw:
        sys.exit(f"{path}: not a binary PLY with an end_header marker")
    header_end = raw.index(marker) + len(marker)
    header = raw[:header_end].split(b"\n")

    n = None
    for line in header:
        if line.startswith(b"element vertex"):
            n = int(line.split()[-1])
    if n is None:
        sys.exit(f"{path}: no 'element vertex' in the header")

    names = [l.split()[-1].decode() for l in header if l.startswith(b"property")]
    if names[:3] != ["x", "y", "z"]:
        sys.exit(f"{path}: expected x,y,z as the first properties, got {names[:3]}")

    # Layout: 3 doubles then one uchar per remaining property (rgb), tightly packed.
    offsets, off = [0, 8, 16], 24
    fields = ["<f8", "<f8", "<f8"]
    for _ in names[3:]:
        offsets.append(off)
        fields.append("u1")
        off += 1
    dt = np.dtype({"names": names, "formats": fields,
                   "offsets": offsets, "itemsize": off})
    V = np.frombuffer(raw, dtype=dt, count=n, offset=header_end)
    return np.column_stack([V["x"], V["y"], V["z"]])


def read_text_poses(path):
    """
    Read camera centres from a COLMAP-text images.txt, keyed by image name.

    Assumes the POINTS2D line is empty (as build_colmap_selection exports it and as
    align_sfm_to_survey writes it), so every non-blank, non-comment line is a pose.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 10 or not p[0].isdigit():
            continue
        q = np.array([float(x) for x in p[1:5]])
        t = np.array([float(x) for x in p[5:8]])
        out[p[9]] = -quat_to_rotmat(q).T @ t
    if not out:
        sys.exit(f"{path}: no poses parsed (expected an empty POINTS2D line)")
    return out


def ground_gap(centers, cloud, radius, depth_pct, min_points):
    """Camera-to-ground distance per camera, in metres (Z-down: ground is larger z)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud[:, :2])
    gaps = []
    for C in centers:
        idx = tree.query_ball_point(C[:2], radius)
        if len(idx) < min_points:
            continue
        z = cloud[idx, 2]
        below = z[z > C[2]]                      # Z-down: below == larger z
        if len(below) < min_points:
            continue
        gaps.append(np.percentile(below, depth_pct) - C[2])
    if not gaps:
        sys.exit("No camera had enough ground points nearby - check the frames "
                 "line up, or raise --radius.")
    return np.array(gaps)


def report(gaps, label):
    print(f"{label:<40} n={len(gaps):>5}  median {np.median(gaps):>6.3f} m   "
          f"mean {gaps.mean():>6.3f} m   p10 {np.percentile(gaps, 10):>6.3f}   "
          f"p90 {np.percentile(gaps, 90):>6.3f}")
    return float(np.median(gaps))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", required=True,
                    help="COLMAP-text images.txt whose camera VERTICAL scale to measure.")
    ap.add_argument("--lidar", required=True,
                    help="LiDAR PLY (binary_little_endian, double xyz) in the same frame.")
    ap.add_argument("--reference-images", default=None,
                    help="Optional second images.txt to use as the baseline, e.g. the "
                         "survey poses. Its median gap becomes the denominator.")
    ap.add_argument("--radius", type=float, default=2.0,
                    help="Horizontal search radius for ground points, metres (default 2).")
    ap.add_argument("--depth-pct", type=float, default=90.0,
                    help="Percentile of the below-camera depths taken as the ground "
                         "(default 90: robust to stray low points).")
    ap.add_argument("--min-points", type=int, default=5,
                    help="Minimum nearby points required (default 5).")
    args = ap.parse_args()

    cloud = read_ply_xyz(args.lidar)
    print(f"LiDAR cloud: {len(cloud):,} points\n")

    poses = read_text_poses(args.poses)

    # If the two files name the same images, restrict BOTH to that common set. A
    # model and its filtered copy are different POPULATIONS, and a median over a
    # subset can differ from the whole by several percent for reasons that have
    # nothing to do with scale - which would otherwise be misread as a scale error.
    # Only the common subset is a fair like-for-like comparison.
    ref = None
    if args.reference_images:
        ref = read_text_poses(args.reference_images)
        common = sorted(set(poses) & set(ref))
        if common:
            if len(common) < len(poses) or len(common) < len(ref):
                print(f"common images: {len(common)} (of {len(poses)} model, "
                      f"{len(ref)} reference) - restricted to these for a fair "
                      f"comparison\n")
            poses = {k: poses[k] for k in common}
            ref = {k: ref[k] for k in common}
        else:
            print("no shared image names - comparing as independent sets "
                  "(a subset comparison here would be misleading)\n")

    ref_med = None
    if ref:
        ref_med = report(ground_gap(np.array(list(ref.values())), cloud,
                                    args.radius, args.depth_pct, args.min_points),
                         "reference (assumed correct)")
    med = report(ground_gap(np.array(list(poses.values())), cloud,
                            args.radius, args.depth_pct, args.min_points),
                 f"model ({Path(args.poses).parent.name})")

    if ref_med:
        r = med / ref_med
        print(f"\nratio model/reference = {r:.4f}")
        # Resolution note: the survey's own median gap moves by several percent
        # between camera subsets (2.08 m over 1043 images vs 2.23 m over 741 here),
        # because camera height above the scanned ground genuinely varies with road
        # slope, vehicle pitch and LiDAR coverage. So a few percent is measurement
        # noise, not a scale error - only a deviation well beyond that is real.
        if abs(r - 1.0) < 0.05:
            print("  Vertical scale is sound (within the ~5% resolution of this test).")
        else:
            print(f"  VERTICAL SCALE IS OFF by {abs(1 - r) * 100:.1f}% - beyond the "
                  f"~5% noise floor.")
            print(f"  A scene squashed vertically by 1/k shows a gap smaller by the")
            print(f"  same k, and is the signature of a camera `fy` that drifted: with")
            print(f"  near-yaw-only cameras the two are almost exactly interchangeable,")
            print(f"  so reprojection error stays low while the world is wrong.")
            print(f"  Try --camera-model pinhole with a free fy (not simple-pinhole).")


if __name__ == "__main__":
    main()
