#!/usr/bin/env python3
"""
Run Spirula's SfM on a prepared Kappazunder scene.

This replaces the old repo-root `run_spirula_rig.py`, which could never work: it
died on an unused `from rotation_conversion import ...` import, pointed at a
dataset path that did not exist, and hardcoded a rig sensor list referencing a
sensor with no images.

Settings are taken from the `.resume/*.sig` files left by the previous successful
run. Those are the authoritative record: Spirula's GUI runs this same in-process
code path, so these flags reproduce GUI behaviour rather than approximating it.

Why the flags matter here:

  --rig       locks each trajectory's cameras into one rigid frame. This is only
              usable because build_colmap_selection.py now names images by capture
              instant, so a given instant shares a basename across every Sensor_*/
              folder. Epoch_s is shared by all sensors of a rig frame (verified:
              6 images per (trajectory_id, epoch_s)).
  --metric-positions  georeferences the model. NOTE the `_zup` variant: this
              pipeline's export frame is Z-DOWN, and Spirula assumes a positions
              file's +Z is up, so the plain file makes it fit an inverted gauge
              (previously observed as a 177.52 deg up-axis disagreement). The _zup
              file negates Z so +Z is physically up.

Exit codes matter: 4 means "model written, but not in the metric frame asked for",
which is the signal the previous run silently produced while reporting success.
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Spirula's own exit codes (from `spirula sfm auto --help`).
SPIRULA_EXIT_MEANINGS = {
    0: "a reconstruction that looks sound",
    1: "usage or runtime error",
    2: "no reconstruction at all",
    3: "partial: under half the images registered, or over 2 px mean reprojection",
    4: "the model was written, but NOT in the metric frame that was asked for",
}

DEFAULT_SPIRULA_EXE = r"E:\Downloads\spirula-2026.9.13-windows-vulkan-x86_64\spirula.exe"


def discover_rigs(images_dir):
    """
    Group Sensor_<id> subdirectories into rigs.

    Sensor ids are 6-digit: the final digit is the camera role (0 up, 1 front,
    2 right, 3 back, 4 left, 5 down) and the leading digits identify the rig, so
    `sensor_id // 10` groups one trajectory's cameras together.

    Returns a list of member lists, each already sorted and filtered to
    directories that actually contain images. Spirula requires >= 2 members.
    """
    rigs = defaultdict(list)
    for sensor_dir in sorted(images_dir.glob("Sensor_*")):
        if not sensor_dir.is_dir():
            continue
        if not any(sensor_dir.glob("*.jpg")):
            continue
        try:
            sensor_id = int(sensor_dir.name.split("_")[-1])
        except ValueError:
            print(f"  [warn] skipping unrecognised directory: {sensor_dir.name}")
            continue
        rigs[sensor_id // 10].append(sensor_dir.name)

    return [sorted(members) for _, members in sorted(rigs.items()) if len(members) >= 2]


def build_command(spirula_exe, images_dir, workspace, rigs, positions_path, masks_dir=None,
                  final_free_rig=False, rig_min_frames=None, extra=None):
    cmd = [
        str(spirula_exe), "sfm", "auto", str(images_dir),
        "-o", str(workspace),
        # --- settings recorded in the previous successful run's .resume/*.sig ---
        "--lang", "en",
        "--camera-mode", "folder",
        "--camera-model", "pinhole",
        "--focal", "3565",
        "--features", "aliked-n16rot",
        "--max-image-size", "1600",
        "--max-features", "8192",
        "--pairs", "prefilter",
        "--ratio", "1",
    ]
    for members in rigs:
        cmd += ["--rig", ",".join(members)]
    # Release the rig at the very end so every image settles on its own pose. The rig
    # constraint helps while solving, but a member the rig could not properly support
    # (a sky-facing lens has too few features to earn one) is then free to sit tens of
    # metres from where it belongs. Freeing the rig afterwards lets those poses relax
    # onto the imagery instead of staying pinned to a wrong rig estimate.
    if final_free_rig:
        cmd.append("--final-free-rig")
    if rig_min_frames is not None:
        cmd += ["--rig-min-frames", str(rig_min_frames)]
    if positions_path is not None:
        cmd += ["--metric-positions", str(positions_path)]

    # Mask handling is explicit on purpose. Spirula auto-discovers a `masks`
    # directory near the dataset, and the scene has an EMPTY one at
    # datasets/<scene>/masks. Left to `auto` that aborts the whole run with
    # "No mask ... matches any image", because Spirula (correctly) refuses to
    # silently continue unmasked. Mask support is deferred, so default to
    # --no-masks and make the real thing an explicit opt-in.
    if masks_dir:
        cmd += ["--masks", str(masks_dir)]
    else:
        cmd += ["--no-masks"]
    # Verbatim passthrough for flags this wrapper does not model. Placed LAST so it
    # can override anything above, which is what makes it useful for experiments.
    if extra:
        cmd += list(extra)
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run Spirula SfM for a prepared Kappazunder scene."
    )
    parser.add_argument("--config", required=True,
                        help="Path to the scene config.yaml.")
    parser.add_argument("--spirula-exe", default=DEFAULT_SPIRULA_EXE,
                        help="Path to spirula.exe (or `spirula` on PATH).")
    parser.add_argument("--images-dir", default=None,
                        help="Override the image directory (default: <dataset>/images_by_sensor).")
    parser.add_argument("--workspace", default=None,
                        help="Override the output workspace (default: <dataset>/sfm).")
    parser.add_argument("--positions", choices=["zup", "plain", "none"], default="zup",
                        help="Which positions file to georeference with. `zup` (default) "
                             "negates Z so +Z is physically up, which is what Spirula "
                             "assumes; `plain` is the pipeline's Z-down frame; `none` "
                             "skips georeferencing so the model stays in its own gauge.")
    parser.add_argument("--no-rig", action="store_true",
                        help="Do not pass --rig (falls back to the settings known to "
                             "reach 468/526 registration).")
    parser.add_argument("--masks", default=None,
                        help="Directory of masks. Omitted by default, which passes "
                             "--no-masks: the scene has an EMPTY masks/ dir that "
                             "Spirula would auto-discover and then abort on.")
    parser.add_argument("--final-free-rig", action="store_true",
                        help="Pass --final-free-rig: one last bundle adjustment with the "
                             "rig set aside, so every image settles on its own pose. "
                             "Lets a poorly-supported member (e.g. a sky-facing lens) "
                             "relax onto the imagery instead of staying pinned to a "
                             "wrong rig estimate.")
    parser.add_argument("--rig-min-frames", type=int, default=None,
                        help="Pass --rig-min-frames N (Spirula default 3): frames in "
                             "which a member and its rig reference must both register "
                             "before the member's extrinsic is trusted. Lower it to let "
                             "a weak member earn support.")
    parser.add_argument("--extra", default=None,
                        help='Extra flags passed verbatim to `spirula sfm auto`, placed '
                             'LAST so they override this wrapper\'s defaults. Quote the '
                             'whole group, e.g. --extra "--matcher lightglue '
                             '--aliked-max-features 8192".')
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command without running it.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Derive the dataset directory from the script's own location so the script
    # works regardless of the current working directory. `utils.load_and_verify_config`
    # resolves "./datasets" against the CWD instead, which is a easy way to write a
    # second dataset tree by accident.
    scene_name = config_path.parent.name
    dataset_dir = Path(__file__).resolve().parent / "datasets" / scene_name

    images_dir = Path(args.images_dir) if args.images_dir else dataset_dir / "images_by_sensor"
    workspace = Path(args.workspace) if args.workspace else dataset_dir / "sfm"

    if not images_dir.is_dir():
        sys.exit(
            f"Image directory not found: {images_dir}\n"
            f"Run build_colmap_selection.py first (it writes images_by_sensor/)."
        )

    positions_path = None
    if args.positions != "none":
        suffix = "_zup" if args.positions == "zup" else ""
        positions_path = dataset_dir / f"spirula_positions{suffix}.txt"
        if not positions_path.exists():
            sys.exit(
                f"Positions file not found: {positions_path}\n"
                f"Run build_colmap_selection.py first (it writes both variants)."
            )

    rigs = [] if args.no_rig else discover_rigs(images_dir)

    spirula_exe = args.spirula_exe
    if not shutil.which(spirula_exe) and not Path(spirula_exe).exists():
        sys.exit(f"spirula executable not found: {spirula_exe}")

    cmd = build_command(spirula_exe, images_dir, workspace, rigs, positions_path,
                        masks_dir=args.masks, final_free_rig=args.final_free_rig,
                        rig_min_frames=args.rig_min_frames,
                        extra=shlex.split(args.extra) if args.extra else None)

    print("=" * 72)
    print("SPIRULA SfM")
    print("=" * 72)
    print(f"images dir   : {images_dir}")
    print(f"workspace    : {workspace}")
    print(f"positions    : {positions_path if positions_path else '(none - ungeoreferenced)'}")
    if rigs:
        for members in rigs:
            print(f"rig          : {', '.join(members)}")
    else:
        print("rig          : (disabled)")
    print("-" * 72)
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 72)

    if args.dry_run:
        print("Dry run -- not executing.")
        return

    workspace.parent.mkdir(parents=True, exist_ok=True)
    log_path = dataset_dir / "spirula_sfm_run.log"

    # Stream to both the terminal and a log file, so the run leaves a record
    # (the previous runs' logs were the only reason their settings were recoverable).
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()

    print("=" * 72)
    print(f"Wrote log: {log_path}")
    print(f"Exit code: {returncode} -- {SPIRULA_EXIT_MEANINGS.get(returncode, 'unknown')}")

    if returncode == 4:
        print()
        print("!! The model was NOT written in the requested metric frame. Spirula")
        print("!! reports this but still exits having written the model, so it is easy")
        print("!! to miss. Check the 'up axis' line above for a ~180 deg disagreement,")
        print("!! which means the positions file's +Z is not physically up.")
    elif returncode == 0:
        print()
        print("Check the up-axis line above: it should be near 0 deg. Historically")
        print("this pipeline produced 177.52 deg with the plain (Z-down) positions file.")

    sys.exit(returncode)


if __name__ == "__main__":
    main()
