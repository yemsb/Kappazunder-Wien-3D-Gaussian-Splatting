#!/usr/bin/env python3
"""
Build Clean-GS's inputs from a georeferenced pipeline model.

Clean-GS (`F:\\Documents\\clean-gs`) does not read a COLMAP model. It wants a
`cameras.json` of per-image `{image_name, width, height, R, T, K}` with R/T in
CAMERA-TO-WORLD, plus a FLAT directory of masks, and it finds a camera's mask by
`Path(image_name).stem` joined to that flat directory.

Two mismatches make the pipeline's own layout unusable as-is:

  * Mask filenames collide. A rig capture shares `frame_<n>.png` across the five
    sensors, so a flat mask directory can hold only one of each instant's five
    masks -- and `Path("Sensor_110010/frame_000001.jpg").stem` is just
    `frame_000001`, which matches whichever sensor happened to win.
  * The names are folder-qualified (`Sensor_<id>/frame_<n>.jpg`), which the stem
    lookup flattens away.

So both sides are rewritten to a single unambiguous form, `Sensor_<id>_frame_<n>`,
and the masks are hardlinked into a flat directory under that name. Nothing is
copied twice: hardlinks keep the mask bytes in one place.

Note the masks must mark what should be KEPT (Clean-GS whitelists bright pixels),
not what should be removed -- its `object_mask` is `bright > 0.05`.

Usage, from `colmap_pipeline/`:
    python prepare_clean_gs_inputs.py --scene secession \\
        --model datasets/secession/sift_prior/georef/0_clean \\
        --masks datasets/secession/masks_sky \\
        --out-dir F:/Documents/clean-gs/data/datasets/secession

It also flattens the original photographs, hardlinked, into `photos_flat`. Clean-GS
needs those for its colour-validation stage: it compares each Gaussian's colour
against the masked image at the pixel it projects to, and on a binary mask every
object pixel is white, so that test degenerates into "is this Gaussian white?".
Passed alongside each other -- `--masked_images masks_flat --photos photos_flat` --
the whitelist comes from the mask and the colour from the photograph, which is what
feeding Clean-GS genuinely masked photographs would have done.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import quat_to_rotmat

PIPELINE = Path(__file__).resolve().parent
DATASETS = PIPELINE / "datasets"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def read_cameras_txt(path):
    cams = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        cam_id, model, w, h = int(f[0]), f[1], int(f[2]), int(f[3])
        params = [float(x) for x in f[4:]]
        cams[cam_id] = (model, w, h, params)
    return cams


def intrinsics(model, w, h, params):
    if model == "PINHOLE" and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL") and len(params) >= 3:
        fx = fy = params[0]; cx, cy = params[1], params[2]
    elif len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    else:
        fx = fy = max(w, h) / 2; cx, cy = w / 2, h / 2
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def find_image(directory, stem):
    if not directory.is_dir():
        return None
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", type=Path, default=None,
                    help="georef model dir (default datasets/<scene>/sift_prior/georef/0_clean)")
    ap.add_argument("--masks", type=Path, default=None,
                    help="mask root, sensor subdirectories included "
                         "(default datasets/<scene>/masks_sky)")
    ap.add_argument("--photos", type=Path, default=None,
                    help="original photographs, sensor subdirectories included "
                         "(default datasets/<scene>/images_by_sensor)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Clean-GS scene dir, e.g. <clean-gs>/data/datasets/<scene>")
    ap.add_argument("--copy-masks", action="store_true",
                    help="copy instead of hardlinking the masks")
    args = ap.parse_args()

    model = args.model or DATASETS / args.scene / "sift_prior" / "georef" / "0_clean"
    masks = args.masks or DATASETS / args.scene / "masks_sky"
    photos = args.photos or DATASETS / args.scene / "images_by_sensor"
    images_txt, cameras_txt = model / "images.txt", model / "cameras.txt"
    for p in (images_txt, cameras_txt, masks):
        if not p.exists():
            sys.exit(f"missing: {p}")
    if not photos.exists():
        print(f"  WARNING           no photographs at {photos}; Clean-GS will fall "
              f"back to colour-validating against the masks")
        photos = None

    cams = read_cameras_txt(cameras_txt)
    flat_masks = args.out_dir / "masks_flat"
    flat_masks.mkdir(parents=True, exist_ok=True)
    flat_photos = args.out_dir / "photos_flat"
    if photos is not None:
        flat_photos.mkdir(parents=True, exist_ok=True)

    entries, linked, missing, shots, shots_missing = [], 0, [], 0, []
    for line in images_txt.read_text().splitlines():
        f = line.split()
        if len(f) != 10 or not f[0].isdigit():
            continue
        R_w2c = quat_to_rotmat(np.array([float(x) for x in f[1:5]]))
        t_w2c = np.array([float(x) for x in f[5:8]])
        camera_id, name = int(f[8]), f[9]
        if camera_id not in cams:
            sys.exit(f"image names camera {camera_id}, absent from cameras.txt")
        model_name, w, h, params = cams[camera_id]

        # camera-to-world, which is what Clean-GS's projection inverts back
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        flat = name.replace("/", "_")          # Sensor_<id>_frame_<n>.jpg
        entries.append({"image_name": flat, "width": w, "height": h,
                        "R": R_c2w.tolist(), "T": t_c2w.tolist(),
                        "K": intrinsics(model_name, w, h, params).tolist()})

        src = masks / Path(name).parent / (Path(name).stem + ".png")
        if src.exists():
            dst = flat_masks / (Path(flat).stem + ".png")
            if not dst.exists():
                if args.copy_masks:
                    shutil.copy2(src, dst)
                else:
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
            linked += 1
        else:
            missing.append(str(src))

        if photos is not None:
            shot = find_image(photos / Path(name).parent, Path(name).stem)
            if shot is None:
                shots_missing.append(str(name))
            else:
                dst = flat_photos / (Path(flat).stem + shot.suffix)
                if not dst.exists():
                    try:
                        os.link(shot, dst)
                    except OSError:
                        shutil.copy2(shot, dst)
                shots += 1

    json_path = args.out_dir / "cameras.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(entries, indent=2))
    print(f"  cameras.json      {len(entries)} cameras -> {json_path}")
    print(f"  masks             {linked} linked into {flat_masks}")
    if photos is not None:
        print(f"  photos            {shots} linked into {flat_photos}")
    if missing:
        print(f"  WARNING           {len(missing)} images have no mask, e.g. {missing[:3]}")
        print(f"                    those cameras contribute nothing to the whitelist")
    if shots_missing:
        print(f"  WARNING           {len(shots_missing)} images have no photograph, "
              f"e.g. {shots_missing[:3]}")
    print(f"\n  note: Clean-GS whitelists BRIGHT mask pixels, so masks must mark what to KEEP")
    print(f"  run:  --masked_images {flat_masks} --photos {flat_photos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
