#!/usr/bin/env python3
"""
Test the survey trajectory against the LiDAR, which is an independent reference.

Why this exists: the SfM-to-survey alignment leaves large residuals (up to ~40 m)
even though the fitted transform is provably rigid (pairwise camera distances are
preserved to 1e-6 relative). A rigid transform cannot change a trajectory's SHAPE,
so a large residual that survives a rigid fit means the two trajectories genuinely
differ in shape - one of them is distorted.

The LiDAR breaks the tie. It was scanned from the same vehicle in the survey frame,
so its coverage footprint IS the drive path. If the survey positions fall inside the
scanned corridor, the survey is fine and the SfM is distorted. If the survey
wanders off the scanned ground, the survey is the one that is wrong.

Usage:
    python check_trajectory_vs_lidar.py --config configs/kolonitzplatz/config.yaml
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
# Lives in diagnostics/ but imports its siblings from the parent directory.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import align_sfm_to_survey as A
from colmap_io import read_cameras_bin, quat_to_rotmat  # noqa: F401


def read_ply_xyz(path):
    """Read x,y,z from a binary little-endian PLY (Open3D writes float64 here)."""
    path = Path(path)
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        text = header.decode("ascii", errors="replace")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        props = re.findall(r"property (\S+) (\S+)", text)
        dtype_map = {"double": "<f8", "float": "<f4", "uchar": "u1"}
        dt = np.dtype([(name, dtype_map[kind]) for kind, name in props])
        arr = np.fromfile(f, dtype=dt, count=n)
    return np.column_stack([arr["x"], arr["y"], arr["z"]])


def instant_positions(records, use_mean=True):
    """Collapse per-sensor poses to one position per capture instant (basename)."""
    groups = {}
    for r in records:
        groups.setdefault(Path(r["name"]).name, []).append(r)
    out = {}
    for name, rs in groups.items():
        C = np.array([-(quat_to_rotmat(r["qvec"]).T @ r["tvec"]) for r in rs])
        out[name] = C.mean(0) if use_mean else C[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="SfM model dir (default <dataset>/sfm/sparse/0)")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    dataset = Path(__file__).resolve().parent / "datasets" / config_path.parent.name

    ply = dataset / "sparse" / "0" / "points3D.ply"
    if not ply.exists():
        sys.exit(f"No LiDAR cloud at {ply}")
    xyz = read_ply_xyz(ply)
    print(f"LiDAR cloud: {len(xyz):,} points from {ply}")

    survey = instant_positions(A.read_survey_images_txt_all(dataset / "sparse" / "0" / "images.txt"))
    print(f"survey instants: {len(survey)}")

    # Nearest-LiDAR-point distance for each survey instant. The cloud is dense
    # (0.1 m voxels) along the drive, so a position on scanned ground is metres from
    # nothing; a position off the drive is far from every point.
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz[:, :2])          # horizontal only: we are testing the path
    names = sorted(survey)
    sp = np.array([survey[n] for n in names])
    dist, _ = tree.query(sp[:, :2], k=1)
    print()
    print("distance from each SURVEY instant to the nearest LiDAR point (horizontal):")
    print(f"  median {np.median(dist):8.3f} m   mean {dist.mean():8.3f} m   "
          f"p90 {np.percentile(dist, 90):8.3f} m   max {dist.max():8.3f} m")
    for thr in (1.0, 2.0, 5.0, 10.0, 20.0):
        print(f"  within {thr:5.1f} m: {int((dist <= thr).sum()):4d} / {len(dist)}")

    # Same measure for the aligned SfM model, if it exists.
    geo = dataset / "georef" / "0" / "images.txt"
    if geo.exists():
        gm = instant_positions(A.read_survey_images_txt_all(geo))
        gp = np.array([gm[n] for n in names if n in gm])
        gd, _ = tree.query(gp[:, :2], k=1)
        print()
        print("distance from each ALIGNED MODEL instant to the nearest LiDAR point:")
        print(f"  median {np.median(gd):8.3f} m   mean {gd.mean():8.3f} m   "
              f"p90 {np.percentile(gd, 90):8.3f} m   max {gd.max():8.3f} m")
        for thr in (1.0, 2.0, 5.0, 10.0, 20.0):
            print(f"  within {thr:5.1f} m: {int((gd <= thr).sum()):4d} / {len(gd)}")

    # Plot: LiDAR density with both trajectories on top.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    lim = 140
    for a, (title, _) in zip(ax, [("", None)]):
        pass

    a = ax[0]
    keep = (np.abs(xyz[:, 0]) < lim) & (np.abs(xyz[:, 1]) < lim)
    a.hexbin(xyz[keep, 0], xyz[keep, 1], gridsize=320, bins="log", cmap="Greys", mincnt=1)
    a.plot(sp[:, 0], sp[:, 1], "-", color="tab:blue", lw=2.2, label="survey trajectory")
    a.plot(sp[dist > 2, 0], sp[dist > 2, 1], "o", color="tab:red", ms=7,
           label=f"survey instants >2 m from any LiDAR point ({(dist > 2).sum()})")
    a.set_aspect("equal")
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_title("SURVEY trajectory over LiDAR coverage")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.legend(fontsize=9)

    a = ax[1]
    a.hexbin(xyz[keep, 0], xyz[keep, 1], gridsize=320, bins="log", cmap="Greys", mincnt=1)
    if geo.exists():
        a.plot(gp[:, 0], gp[:, 1], "-", color="tab:green", lw=2.2, label="aligned SfM trajectory")
        a.plot(gp[gd > 2, 0], gp[gd > 2, 1], "o", color="tab:red", ms=7,
               label=f"model instants >2 m from any LiDAR point ({(gd > 2).sum()})")
    a.set_aspect("equal")
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_title("ALIGNED SfM trajectory over LiDAR coverage")
    a.set_xlabel("x (m)")
    a.legend(fontsize=9)

    fig.suptitle("Which trajectory lies on the scanned ground? "
                 "(LiDAR coverage is the drive path)", fontsize=13)
    fig.tight_layout()
    out = dataset / "georef" / "0" / "trajectory_vs_lidar.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
