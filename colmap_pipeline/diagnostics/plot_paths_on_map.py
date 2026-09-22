#!/usr/bin/env python3
"""
Overlay the survey and aligned-SfM trajectories on an OpenStreetMap basemap.

Why: residual statistics alone cannot say which trajectory is wrong - only which two
disagree. A street map can, because the vehicle drove on roads, so a trajectory
running through a block interior cannot be right.

The OSM tile servers require a real, identifying User-Agent; without one they return
403 ("App is not following the usage policy of OpenStreetMap's volunteer-run
servers"). The header below matches the one used in build_colmap_selection.py.

Coordinate chain. Camera poses live in the pipeline's export frame: EPSG:31256 with
the scene origin subtracted and Y mirrored (see scene_origin.txt):

    x_export = x_m - offset_x
    y_export = -(y_m - offset_y)        (invert_y_axis 1)
    z_export = z_m - offset_z           (invert_z_axis 0)

Inverting that recovers EPSG:31256, which is plotted directly - contextily's
`crs=` argument reprojects the tiles to match.

Usage:
    python plot_paths_on_map.py --config configs/kolonitzplatz/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
# Lives in diagnostics/ but imports its siblings from the parent directory.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import align_sfm_to_survey as A
from colmap_io import quat_to_rotmat

OSM_HEADERS = {
    "User-Agent": ("kappazunderUser/1.0 "
                   "(https://github.com/yemsb/Kappazunder-Wien-3D-Gaussian-Splatting)")
}


def read_scene_origin(path):
    vals = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            vals[parts[0]] = float(parts[1])
    return (vals["offset_x_m"], vals["offset_y_m"], vals["offset_z_m"],
            int(vals.get("invert_y_axis", 0)), int(vals.get("invert_z_axis", 0)))


def instant_positions(path):
    groups = {}
    for r in A.read_survey_images_txt_all(path):
        groups.setdefault(Path(r["name"]).name, []).append(
            -(quat_to_rotmat(r["qvec"]).T @ r["tvec"]))
    return {k: np.mean(v, axis=0) for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    dataset = Path(__file__).resolve().parent / "datasets" / config_path.parent.name

    ox, oy, oz, inv_y, inv_z = read_scene_origin(dataset / "scene_origin.txt")
    print(f"scene origin: ({ox:.3f}, {oy:.3f}, {oz:.3f}) invert_y={inv_y} invert_z={inv_z}")

    def to_survey(p):
        """Invert the export transform: pipeline frame -> EPSG:31256."""
        x = p[:, 0] + ox
        y = (-p[:, 1] if inv_y else p[:, 1]) + oy
        z = (-p[:, 2] if inv_z else p[:, 2]) + oz
        return np.column_stack([x, y, z])

    survey = instant_positions(dataset / "sparse" / "0" / "images.txt")
    names = sorted(survey)
    S = to_survey(np.array([survey[n] for n in names]))

    geo = dataset / "georef" / "0" / "images.txt"
    M = None
    if geo.exists():
        model = instant_positions(geo)
        keep = [n for n in names if n in model]
        M = to_survey(np.array([model[n] for n in keep]))
        d = np.linalg.norm(M - S[:len(M)], axis=1)
        print(f"aligned model instants: {len(M)}; named offset "
              f"median {np.median(d):.2f} m, max {d.max():.2f} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import contextily as ctx

    fig, ax = plt.subplots(1, 2, figsize=(22, 11))
    for a, (xlim, ylim, title) in zip(ax, [
        (None, None, "full extent"),
        ((S[:, 0].min(), S[:, 0].max()), (S[:, 1].min(), S[:, 1].max()), "detail"),
    ]):
        a.plot(S[:, 0], S[:, 1], "-", color="blue", lw=2.6,
               label="survey trajectory", zorder=4)
        a.plot(S[:, 0], S[:, 1], "o", color="blue", ms=3.5, zorder=4)
        if M is not None:
            a.plot(M[:, 0], M[:, 1], "-", color="red", lw=2.0,
                   label="aligned SfM trajectory", zorder=5)
            a.plot(M[:, 0], M[:, 1], "s", color="black", ms=3.5, zorder=6)
        pad = 30 if xlim else 60
        a.set_xlim((xlim[0] - pad, xlim[1] + pad) if xlim
                   else (S[:, 0].min() - pad, S[:, 0].max() + pad))
        a.set_ylim((ylim[0] - pad, ylim[1] + pad) if ylim
                   else (S[:, 1].min() - pad, S[:, 1].max() + pad))
        try:
            ctx.add_basemap(a, crs="EPSG:31256",
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            headers=OSM_HEADERS)
        except Exception as e:
            print(f"  [warn] basemap fetch failed: {e}")
        a.set_title(f"{title}  (blue = survey, red = aligned SfM)")
        a.set_xlabel("EPSG:31256 eastings (m)")
        a.set_ylabel("EPSG:31256 northings (m)")
        a.legend(loc="upper left", fontsize=11)

    fig.suptitle("Which trajectory follows the streets?", fontsize=14)
    fig.tight_layout()
    out = Path(args.out) if args.out else dataset / "georef" / "0" / "paths_on_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
