#!/usr/bin/env python3
"""
Write a copy of the SfM model with the alignment-flagged bad cameras removed and
EVERYTHING ELSE - poses, intrinsics, point cloud - left exactly as it was.

Why this exists: `georef/0_clean` differs from `sfm/sparse/0` in two ways at once -
144 cameras are dropped, and the survivors carry an (s, R, t) alignment. When that
model dissolved in training while the unfiltered SfM did not, the two changes were
confounded and neither could be blamed.

Together with the models that already exist, this completes a 2x2 in which each cell
differs from its neighbours in exactly one factor:

                           no transform        transform applied
    no filtering      sfm/sparse/0          georef/0   (+ SfM points)
    filtering         THIS SCRIPT           georef/0_clean (+ SfM points)

THE POINT CLOUD IS DELIBERATELY NOT FILTERED. It would be tempting to drop points
whose only observations came from removed cameras, but that introduces a second
difference and re-confounds the very comparison this model exists to make. The SfM
points are copied verbatim, so this model differs from sfm/sparse/0 in the camera
set and in nothing else. In particular it does NOT match the point cloud in
georef/0_clean, which carries the alignment transform.

Which cameras count as bad comes from the aligner's own per-camera verdict
(`is_inlier` in pose_alignment_per_image.csv), so the two models disagree about
nothing except which cameras they contain.

Usage:
    python filter_sfm_model.py --config configs/kolonitzplatz/config.yaml
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np


def read_images_bin_full(path):
    """
    Read images.bin INCLUDING the 2D observations.

    `ingest_spirula_model.read_images_bin` seeks past them, which is fine for
    diagnostics but not here: dropping observations would not reproduce the model
    faithfully.
    """
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name += b
            num = struct.unpack("<Q", f.read(8))[0]
            pts = np.frombuffer(f.read(num * 24), dtype=np.dtype(
                [("x", "<f8"), ("y", "<f8"), ("id", "<u8")]))
            out[image_id] = dict(image_id=image_id, qvec=qvec, tvec=tvec,
                                 camera_id=camera_id, name=name.decode("utf-8"),
                                 points2D=pts)
    return out


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<I", im["image_id"]))
            f.write(struct.pack("<4d", *im["qvec"]))
            f.write(struct.pack("<3d", *im["tvec"]))
            f.write(struct.pack("<I", im["camera_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im["points2D"])))
            f.write(im["points2D"].tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sfm-model", default=None,
                    help="Source model (default <dataset>/sfm/sparse/0).")
    ap.add_argument("--kept", default=None,
                    help="Explicit list of cameras to keep, as written by the aligner "
                         "(default <dataset>/georef/0_clean/cameras_kept.txt).")
    ap.add_argument("--out", default=None,
                    help="Destination (default <dataset>/sfm/sparse/0_filtered).")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    dataset = Path(__file__).resolve().parent / "datasets" / config_path.parent.name

    src = Path(args.sfm_model) if args.sfm_model else dataset / "sfm" / "sparse" / "0"
    kept_path = (Path(args.kept) if args.kept
                 else dataset / "georef" / "0_clean" / "cameras_kept.txt")
    dst = Path(args.out) if args.out else dataset / "sfm" / "sparse" / "0_filtered"

    for p in (src / "images.bin", src / "cameras.bin", src / "points3D.bin", kept_path):
        if not p.exists():
            sys.exit(f"Missing input: {p}")

    # Take the camera set from the aligner's own record of what it kept, rather than
    # recomputing it. The per-image CSV's `is_inlier` column is the ALIGNMENT inlier
    # flag (position <= 0.5 m AND orientation <= 2 deg) - a much stricter set than
    # the 5 m camera filter that produced georef/0_clean. Using it here would yield a
    # model whose camera set differs from 0_clean by ~250 images and confound the
    # very comparison this script exists to enable.
    keep_names = {l.strip() for l in kept_path.read_text().splitlines() if l.strip()}
    print(f"aligner kept {len(keep_names)} cameras (from {kept_path.name})")

    images = read_images_bin_full(src / "images.bin")
    print(f"source model: {len(images)} images")

    kept = [im for im in images.values() if im["name"] in keep_names]
    if not kept:
        sys.exit("No images matched the kept-camera list - refusing an empty model.")
    missing = len(keep_names) - len(kept)
    print(f"keeping {len(kept)}, dropping {len(images) - len(kept)}"
          + (f"  ({missing} named in the list were not in this model)" if missing else ""))

    dst.mkdir(parents=True, exist_ok=True)
    # Verbatim copies: intrinsics, point cloud, gauge. Only images.bin is rewritten.
    shutil.copy2(src / "cameras.bin", dst / "cameras.bin")
    shutil.copy2(src / "points3D.bin", dst / "points3D.bin")
    for extra in ("gauge.txt",):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)
    write_images_bin(dst / "images.bin", kept)

    unchanged = all(np.array_equal(im["qvec"], images[im["image_id"]]["qvec"])
                    and np.array_equal(im["tvec"], images[im["image_id"]]["tvec"])
                    for im in kept)
    print(f"\nposes bit-identical to the source: {unchanged}")
    print(f"point cloud: copied verbatim ({len(images)} images' worth of tracks kept)")
    print(f"wrote {dst}")
    print("\nThis differs from sfm/sparse/0 in the CAMERA SET AND NOTHING ELSE - no "
          "alignment, no point filtering.")


if __name__ == "__main__":
    main()
