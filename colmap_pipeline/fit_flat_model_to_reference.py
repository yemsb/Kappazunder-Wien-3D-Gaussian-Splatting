#!/usr/bin/env python3
"""
Fit the similarity taking a flat-named SfM model into a reference model's frame.

Used to place the `kolonitzplatz_SS` reconstruction (and therefore its trained splat) into the
georeferenced frame of `sift_prior/georef/0_clean`.

THE MAPPING IS RECOVERED, NOT GUESSED. `kolonitzplatz_SS` names its images with a running index
(`frame_000166.jpg`) because the rig's five sensors share basenames, so a flat layout cannot
carry the sensor. The instant number is NOT recoverable from that index -- `frame_000166` is
`Sensor_110011/frame_000043.jpg` -- and comparing equal numbers gives a plausible-looking but
wrong pairing (the flat set is frustum-pruned toward the front-facing sensor, so same-number
comparison scores misleadingly well and even "confirms" itself).

What does recover it: `build_colmap_selection.py` wrote both sets with `shutil.copy2`, which
preserves size and mtime from the raw capture file. So each flat image and its sensor-qualified
counterpart share an identical (size, mtime) pair, and that pair is unique across the 1355
sensor images. The mapping falls out of the filesystem -- no content matching, no replaying the
selection. NOTE it only works against `datasets/kolonitzplatz/images`, which kept the raw
mtimes; `kolonitzplatz_SS/images` is a later copy that did not.

The fit is a robust similarity (Umeyama with iterative outlier rejection). Sanity-check it: on
kolonitzplatz it should land ~0.05 m median on ~75% of cameras, with a rotation of about 177
degrees -- the expected near-flip between Spirula's +z-up normalised frame and the survey's
Z-DOWN one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_images_bin, quat_to_rotmat


def sensor_index(sensor_root: Path) -> dict:
    """(size, mtime) -> `Sensor_<id>/<name>`, the identity the raw copies preserved."""
    idx = {}
    for dirpath, _, files in os.walk(sensor_root):
        for fn in files:
            if not fn.lower().endswith(".jpg"):
                continue
            p = Path(dirpath) / fn
            st = p.stat()
            rel = p.relative_to(sensor_root).as_posix()
            idx[(st.st_size, round(st.st_mtime, 3))] = rel
    return idx


def read_reference(path: Path) -> dict:
    """COLMAP text images.txt -> {name: camera centre}."""
    out = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 10 and f[0].isdigit():
            K = quat_to_rotmat(np.array([float(x) for x in f[1:5]]))
            out[f[9]] = -K.T @ np.array([float(x) for x in f[5:8]])
    return out


def umeyama(A, B):
    """Similarity mapping A -> B (B ~= s R A + t)."""
    mA, mB = A.mean(0), B.mean(0)
    A0, B0 = A - mA, B - mB
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = (S * np.array([1.0, 1.0, d])).sum() / (A0 ** 2).sum()
    return s, R, mB - s * R @ mA


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path,
                    help="flat-named model dir containing images.bin")
    ap.add_argument("--reference", required=True, type=Path,
                    help="reference dir containing images.txt (already in the target frame)")
    ap.add_argument("--flat-dir", action="append", required=True, type=Path,
                    help="flat image dir(s) holding the raw mtimes (repeatable)")
    ap.add_argument("--sensor-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help=".npz to write")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="inlier threshold in metres for the robust refit (default 0.5)")
    args = ap.parse_args()

    idx = sensor_index(args.sensor_dir)
    ref = read_reference(args.reference)
    imgs = read_images_bin(args.model / "images.bin")

    A, B, names = [], [], []
    unresolved = 0
    for im in imgs.values():
        hit = None
        for root in args.flat_dir:
            p = root / im["name"]
            if p.exists():
                st = p.stat()
                hit = idx.get((st.st_size, round(st.st_mtime, 3)))
                if hit:
                    break
        if hit is None or hit not in ref:
            unresolved += 1
            continue
        A.append(-quat_to_rotmat(im["qvec"]).T @ im["tvec"])
        B.append(ref[hit])
        names.append((im["name"], hit))

    print(f"model images {len(imgs)}  ->  correspondences {len(A)}  (unresolved {unresolved})")
    if len(A) < 3:
        sys.exit("not enough correspondences; are the flat dirs the ones that kept the raw mtimes?"
                 " (kolonitzplatz/images does; kolonitzplatz_SS/images does not)")
    A, B = np.array(A), np.array(B)

    keep = np.ones(len(A), bool)
    for _ in range(10):
        s, R, t = umeyama(A[keep], B[keep])
        res = np.linalg.norm((s * (R @ A.T).T + t) - B, axis=1)
        new = res < max(args.tau, 3 * np.median(res[keep]))
        if new.sum() == keep.sum():
            break
        keep = new

    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"\nsimilarity: rotation {ang:.4f} deg, scale {s:.6f}, |t| {np.linalg.norm(t):.3f}")
    print(f"inliers {keep.sum()}/{len(A)}   residual median {np.median(res[keep]):.4f} m  "
          f"p90 {np.percentile(res[keep], 90):.4f}  max {res[keep].max():.4f}")
    if (~keep).any():
        print(f"outliers {int((~keep).sum())}   median residual {np.median(res[~keep]):.2f} m")

    np.savez(args.out, s=np.array(s), R=R, t=t, A=A, B=B, keep=keep,
             names=np.array([f"{a}>{b}" for a, b in names]))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
