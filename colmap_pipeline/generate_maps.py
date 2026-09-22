#!/usr/bin/env python3
"""
Generate per-image normals, depth and masks for a Kappazunder scene.

Wraps `spirula geometry` (normals, optionally depth) and `spirula sam mask`
(static-shape masks). Both write trees that MIRROR the image directory structure:

    images_by_sensor/Sensor_110010/frame_000023.jpg
    normals/Sensor_110010/frame_000023.png
    depths/Sensor_110010/frame_000023.png
    masks/Sensor_110010/frame_000023.png

so a single run over images_by_sensor covers every sensor. There is no need to
point the tool at one sensor at a time and move files afterwards.

ONE REQUIREMENT WORTH STATING LOUDLY. These tools locate images by the names in the
COLMAP model, resolved relative to --image-dir. The model at <dataset>/sparse/0/
must therefore name its images with the sensor prefix:

    Sensor_110010/frame_000023.jpg      <- resolves
    frame_000023.jpg                    <- matches nothing

A bare-basename model does not error: the run simply processes zero images and
reports success. This script checks for that up front rather than letting it pass.

A note on cost/benefit: normals come from MoGe, computed MONOCULARLY (from a single
image each). Camera poses are therefore irrelevant to the result - the model only
supplies the list of images. Depth is off by default upstream because it roughly
doubles the reading a training run does; pass --depth when you want it.

Usage:
    python generate_maps.py --config configs/kolonitzplatz/config.yaml
    python generate_maps.py --config configs/kolonitzplatz/config.yaml --no-depth
    python generate_maps.py --config configs/kolonitzplatz/config.yaml --masks \
        --mask-shape "ellipse 0.5,0.5,0.49,0.49"
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_SPIRULA_EXE = r"E:\Downloads\spirula-2026.9.13-windows-vulkan-x86_64\spirula.exe"


def read_model_image_names(images_txt):
    """Return the NAME field of every pose line in a COLMAP text model."""
    names = []
    if not Path(images_txt).exists():
        return names
    for line in Path(images_txt).read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            names.append(parts[9])
    return names


def check_model_matches_images(dataset, image_dir):
    """
    Fail early if the model's image names will not resolve against the image tree.

    Returns the number of model images that exist on disk.
    """
    images_txt = dataset / "sparse" / "0" / "images.txt"
    names = read_model_image_names(images_txt)
    if not names:
        sys.exit(f"No COLMAP model at {images_txt} (expected a text model with NAME "
                 f"fields). Run build_colmap_selection.py first.")
    found = sum(1 for n in names if (image_dir / n).exists())
    if found == 0:
        sample = names[0]
        sys.exit(
            f"None of the {len(names)} images named in {images_txt} exist under "
            f"{image_dir}.\n"
            f"  first model name : {sample}\n"
            f"  would need file  : {image_dir / sample}\n\n"
            f"This is the bare-basename problem: with a rig capture, names must carry "
            f"the sensor prefix (Sensor_110010/frame_000023.jpg). Re-run "
            f"build_colmap_selection.py --skip-ply to write a sensor-qualified model, "
            f"otherwise this generation silently processes zero images."
        )
    if found < len(names):
        print(f"  note: {len(names) - found} / {len(names)} model images have no file "
              f"under {image_dir} (not registered, or filtered out) - they are skipped")
    return found


def run_tool(cmd, log_path, label):
    """Run a spirula subcommand, streaming to console and a log file."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("-" * 72)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n### {label}\n" + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    print(f"-> exit {code}")
    return code


def count_files(d):
    d = Path(d)
    return sum(1 for _ in d.rglob("*.png")) if d.exists() else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to the scene config.yaml.")
    ap.add_argument("--spirula-exe", default=DEFAULT_SPIRULA_EXE)
    ap.add_argument("--image-dir", default="images_by_sensor",
                    help="Image folder, relative to the dataset (default images_by_sensor).")
    ap.add_argument("--normals", dest="normals", action="store_true", default=True,
                    help="Write surface normals (default: on).")
    ap.add_argument("--no-normals", dest="normals", action="store_false",
                    help="Skip the geometry pass entirely (see note below).")
    ap.add_argument("--depth", dest="depth", action="store_true", default=True,
                    help="Also write depth maps (default: on).")
    ap.add_argument("--no-depth", dest="depth", action="store_false",
                    help="Normals only - faster to generate and to train on.")
    ap.add_argument("--masks", dest="masks", action="store_true", default=False,
                    help="Also generate static-shape masks via `spirula sam mask`.")
    ap.add_argument("--no-masks", dest="masks", action="store_false")
    ap.add_argument("--mask-shape", default=None,
                    help="Shape spec for sam mask, e.g. 'ellipse 0.5,0.5,0.49,0.49'. "
                         "Required with --masks unless the frames have a detectable "
                         "border; arbitrary shapes can be layered with ';'.")
    ap.add_argument("--mask-model", default=None,
                    help="SAM .onnx for object masking. Reserved: object masks need "
                         "`spirula sam track`, which this wrapper does not drive yet.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute maps already on disk. Without it a run resumes "
                         "where the last one stopped.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    with open(config_path) as f:
        yaml.safe_load(f)

    # Derive from the script location so the working directory does not matter.
    dataset = Path(__file__).resolve().parent / "datasets" / config_path.parent.name
    image_dir = dataset / args.image_dir
    if not image_dir.is_dir():
        sys.exit(f"Image directory not found: {image_dir}")

    exe = args.spirula_exe
    if not shutil.which(exe) and not Path(exe).exists():
        sys.exit(f"spirula executable not found: {exe}")

    print(f"dataset   : {dataset}")
    print(f"image dir : {image_dir}")
    n_images = check_model_matches_images(dataset, image_dir)
    print(f"images resolved from the model: {n_images}")

    # geometry always writes normals; --depth only adds depth maps on top. There is
    # no depth-without-normals mode, so say so rather than pretending otherwise.
    run_geometry = args.normals or args.depth
    if args.depth and not args.normals:
        print("  note: --no-normals with --depth still writes normals - `geometry` "
              "always emits them and --depth merely adds depth maps.")

    log_path = dataset / "generate_maps.log"
    if log_path.exists() and not args.overwrite:
        log_path.unlink()

    if run_geometry:
        cmd = [str(exe), "geometry", str(dataset), "--image-dir", args.image_dir,
               "--lang", "en"]
        if args.depth:
            cmd.append("--depth")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.dry_run:
            print("\n[dry run] " + " ".join(cmd))
        else:
            code = run_tool(cmd, log_path, "spirula geometry (normals"
                            + (" + depth" if args.depth else "") + ")")
            if code != 0:
                sys.exit(f"geometry failed (exit {code}); see {log_path}")
    else:
        print("\nSkipping geometry (both normals and depth disabled).")

    if args.masks:
        if not args.mask_shape:
            sys.exit(
                "--masks needs --mask-shape. `sam mask` searches for a static border "
                "(fisheye rim, watermark, rig in shot); these frames are full-frame "
                "panoramas with no such border, so a shape must be named explicitly. "
                "For object masks (sky, cars) use `spirula sam track` instead."
            )
        out_dir = dataset / "masks"
        existing = count_files(out_dir)
        if existing and not args.overwrite:
            sys.exit(
                f"Refusing to run: {out_dir} already holds {existing} mask files (your "
                f"GUI-generated ones, most likely). `sam mask` writes into that same "
                f"directory and would overwrite them. Re-run with --overwrite if that "
                f"is really what you want."
            )
        cmd = [str(exe), "sam", "mask", str(image_dir), "--shape", args.mask_shape,
               "--out", str(out_dir), "--lang", "en"]
        if args.dry_run:
            print("\n[dry run] " + " ".join(cmd))
        else:
            code = run_tool(cmd, log_path, "spirula sam mask (static shapes)")
            if code != 0:
                sys.exit(f"sam mask failed (exit {code}); see {log_path}")

    if args.dry_run:
        print("\nDry run -- nothing written.")
        return

    print(f"\n{'=' * 72}\nRESULT\n{'=' * 72}")
    for label, d in (("normals", dataset / "normals"), ("depths", dataset / "depths"),
                     ("masks", dataset / "masks")):
        print(f"  {label:8s}: {count_files(d):5d} files  ({d})")
    print(f"  log     : {log_path}")


if __name__ == "__main__":
    main()
