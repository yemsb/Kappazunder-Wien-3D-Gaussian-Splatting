#!/usr/bin/env python3
"""
Plot a scene's match graph on its own terms: how strong, how far, and how connected.

Replaces the earlier two-scene comparison script. A comparison was useful once, when the
question was "is this graph better than the appearance-based one it replaced"; the
question now is whether a scene's graph is sound, which a single-scene figure answers
without needing a second scene to lean on.

Four panels:
    A  match strength    -- share of verified pairs by inlier count
    B  pair distance     -- share by separation of the two cameras, in metres
    C  instant gap       -- share by frames between the two captures
    D  connectivity      -- images OUTSIDE the largest connected component, by sensor

Panel D is the one that decides whether the mapper has work to do: images inside the
largest component can be registered by geometry, and the stragglers are the ones that
depend on prior-driven structure-less registration. Grouped by sensor because the
explanation is almost always a single sensor rather than a stretch of route.

And one plan view: every verified pair drawn between its two cameras' survey positions,
strength-encoded, with the stranded images marked -- the connectivity of panel D, located.

Restricted to pairs with >= 15 inliers: that is Mapper.min_num_matches, so anything below
it is invisible to the mapper and counting it would flatter the graph.

Palette and marks follow the dataviz reference (light surface); the blue/orange pair is the
validated categorical slots 1-2.

Usage:
    python plot_match_graph.py --dataset secession
"""

from __future__ import annotations

import argparse
import collections
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

P = 2147483647
PAT = re.compile(r"^Sensor_(\d+)[/_]frame_(\d+)\.jpg$")
MIN_INLIERS = 15

PIPELINE = Path(__file__).resolve().parent
DATASETS = PIPELINE / "datasets"

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BAR, MARK = "#2a78d6", "#eb6834"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def key(name):
    m = PAT.match(str(name).replace("\\", "/"))
    return (m.group(1), int(m.group(2))) if m else (str(name), -1)


def load(db):
    con = sqlite3.connect(str(db))
    c = con.cursor()
    name = {r[0]: r[1] for r in c.execute("SELECT image_id, name FROM images")}
    pos = {r[0]: np.array(struct.unpack("<3d", r[1]))
           for r in c.execute("SELECT corr_data_id, position FROM pose_priors")}
    rows = c.execute("SELECT pair_id, rows FROM two_view_geometries").fetchall()
    con.close()
    k = {i: key(n) for i, n in name.items()}
    all_pairs = [(divmod(p, P)[0], divmod(p, P)[1], r) for p, r in rows]
    return name, pos, k, [e for e in all_pairs if e[2] >= MIN_INLIERS], len(all_pairs)


def components(ids, edges):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = {i: j for j, i in enumerate(ids)}
    A = coo_matrix((np.ones(len(edges)), ([idx[a] for a, b in edges], [idx[b] for a, b in edges])),
                   shape=(len(ids), len(ids)))
    ncomp, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    return ncomp, lab, idx, sizes


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8.5, labelpad=4)


def bars(ax, labels, vals, log_y=False, direct_max=None):
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.62, color=BAR, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, color=MUTED)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(max(min(v for v in vals if v > 0) * 0.5, 0.5), max(vals) * 3)
    else:
        ax.set_ylim(0, max(vals) * 1.25)
    if direct_max is not None:
        for i in np.argsort(vals)[-direct_max:]:
            if vals[i] > 0:
                ax.annotate(f"{vals[i]:,.0f}", (x[i], vals[i]), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8, color=INK_2)


def binned(vals, edges, labels):
    v = np.asarray(vals)
    return np.array([int(((v >= edges[i]) & (v < edges[i + 1])).sum())
                     for i in range(len(labels))], dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="scene name under datasets/")
    ap.add_argument("--db", type=Path, default=None,
                    help="database to read (default datasets/<dataset>/sift_prior/database.db)")
    args = ap.parse_args()

    db = args.db or (DATASETS / args.dataset / "sift_prior" / "database.db")
    if not db.exists():
        sys.exit(f"no database at {db}")
    name, pos, k, edges, n_proposed = load(db)
    ids = sorted(name)

    dist = np.array([np.linalg.norm(pos[a] - pos[b]) for a, b, _ in edges])
    gaps = np.array([abs(k[a][1] - k[b][1]) for a, b, _ in edges])
    inl = np.array([r for _, _, r in edges])
    same_sensor = sum(1 for a, b, _ in edges if k[a][0] == k[b][0])
    same_inst = sum(1 for a, b, _ in edges if k[a][1] == k[b][1])
    ncomp, lab, idx, sizes = components(ids, [(a, b) for a, b, _ in edges])
    main_comp = int(np.argmax(sizes))
    stranded = [i for i in ids if lab[idx[i]] != main_comp]
    deg = collections.Counter()
    for a, b, _ in edges:
        deg[a] += 1; deg[b] += 1

    headline = (f"{args.dataset}: {len(name):,} images   ·   {len(edges):,} verified pairs "
                f"of {n_proposed:,} proposed ({100*len(edges)/max(n_proposed,1):.0f}%)   ·   "
                f"mean degree {np.mean([deg[i] for i in ids]):.1f}   ·   "
                f"{ncomp} components, largest {int(sizes[main_comp]):,} "
                f"({100*sizes[main_comp]/len(ids):.1f}%)")

    stranded_sensors = collections.Counter(k[i][0] for i in stranded)
    print(f"\ndatabase: {db}")
    for label, value in [
        ("images", f"{len(name):,}"),
        ("proposed pairs", f"{n_proposed:,}"),
        (f"effective pairs (>={MIN_INLIERS} inl)", f"{len(edges):,}  "
                                                  f"({100*len(edges)/max(n_proposed,1):.1f}% of proposed)"),
        ("cross-sensor %", f"{100*(len(edges)-same_sensor)/max(len(edges),1):.1f}"),
        ("same-instant %", f"{100*same_inst/max(len(edges),1):.1f}"),
        ("instant gap median / p90", f"{np.median(gaps):.1f} / {np.percentile(gaps,90):.1f}"),
        ("inliers per pair median / p90", f"{np.median(inl):.0f} / {np.percentile(inl,90):.0f}"),
        ("connected components", f"{ncomp}   largest {int(sizes[main_comp]):,} of {len(ids)} "
                                 f"({100*sizes[main_comp]/len(ids):.1f}%)"),
        ("images with no edge", f"{sum(1 for i in ids if deg[i] == 0):,}"),
        ("stranded (outside largest)", f"{len(stranded):,}"),
        ("mean degree", f"{np.mean([deg[i] for i in ids]):.1f}"),
    ]:
        print(f"  {label:<32} {value}")
    if stranded_sensors:
        print("  stranded by sensor:")
        for s in sorted(stranded_sensors, key=lambda s: -stranded_sensors[s]):
            tot = collections.Counter(k[i][0] for i in ids)[s]
            print(f"      {s:<10} {stranded_sensors[s]:5d} of {tot}")

    # ------------------------------------------------------------------ four panels
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    fig.patch.set_facecolor(SURFACE)

    specs = [
        (axes[0, 0], "A  Match strength",
         binned(inl, [15, 30, 60, 120, 250, 500, 1000, np.inf],
                ["15-30", "30-60", "60-120", "120-250", "250-500", "500-1k", "1k+"]),
         ["15-30", "30-60", "60-120", "120-250", "250-500", "500-1k", "1k+"],
         "inliers per verified pair"),
        (axes[0, 1], "B  Camera-pair distance",
         binned(dist, [0, 3, 6, 12, 30, 60, 120, np.inf],
                ["0-3", "3-6", "6-12", "12-30", "30-60", "60-120", "120+"]),
         ["0-3", "3-6", "6-12", "12-30", "30-60", "60-120", "120+"],
         "metres between the two cameras"),
        (axes[1, 0], "C  Capture-instant gap",
         binned(gaps, [0, 1, 3, 6, 14, 51, np.inf], ["0", "1-2", "3-5", "6-13", "14-50", "51+"]),
         ["0", "1-2", "3-5", "6-13", "14-50", "51+"],
         "instants between the two frames"),
    ]
    for ax, title, counts, labels, xlab in specs:
        bars(ax, labels, 100 * counts / counts.sum())
        style(ax, title, xlab, "% of verified pairs")

    ax = axes[1, 1]
    per_sensor = collections.Counter(k[i][0] for i in stranded)
    totals = collections.Counter(k[i][0] for i in ids)
    # Only the sensors that actually strand images: listing all 15 collides the tick
    # labels and the zeros carry no information. Each label carries its own total, so
    # "55 of 314" is readable without a second series.
    shown = sorted(per_sensor, key=lambda s: -per_sensor[s])
    bars(ax, [f"{s}\n{per_sensor[s]} of {totals[s]}" for s in shown],
         [per_sensor[s] for s in shown])
    style(ax, f"D  Stranded — images outside the largest component ({len(stranded)})",
          "sensor, and its share of that sensor's frames", "images stranded")
    fig.suptitle(headline, color=INK, fontsize=11.5, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.006,
             f"Verified pairs only (>= {MIN_INLIERS} inliers, the mapper's min_num_matches).   "
             f"cross-sensor {100*(len(edges)-same_sensor)/max(len(edges),1):.0f}%   ·   "
             f"same-instant {100*same_inst/max(len(edges),1):.1f}%   ·   "
             f"instant gap median {np.median(gaps):.0f} p90 {np.percentile(gaps,90):.0f}   ·   "
             f"inliers median {np.median(inl):.0f} p90 {np.percentile(inl,90):.0f}",
             color=MUTED, fontsize=7.6, ha="left")
    fig.tight_layout(rect=(0, 0.028, 1, 0.945))
    out1 = DATASETS / args.dataset / f"match_graph_{args.dataset}.png"
    fig.savefig(out1, dpi=150, facecolor=SURFACE)
    print("wrote", out1)

    # ------------------------------------------------------------------ plan view
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    norm = LogNorm(vmin=MIN_INLIERS, vmax=max(inl.max(), MIN_INLIERS + 1))
    fig2, ax2 = plt.subplots(figsize=(9.5, 8.2))
    fig2.patch.set_facecolor(SURFACE)
    ax2.set_facecolor(SURFACE)
    pts = np.array([pos[i] for i in ids])
    for a, b, r in edges:
        pa, pb = pos[a], pos[b]
        ax2.plot([pa[0], pb[0]], [pa[1], pb[1]], color=cmap(norm(r)), lw=0.5, alpha=0.5,
                 zorder=1, solid_capstyle="round")
    ax2.scatter(pts[:, 0], pts[:, 1], s=1.8, c=MUTED, alpha=0.6, linewidths=0, zorder=2)
    if stranded:
        sx = np.array([pos[i] for i in stranded])
        ax2.scatter(sx[:, 0], sx[:, 1], s=26, facecolors="none", edgecolors=MARK,
                    linewidths=1.3, zorder=4, label=f"outside the largest component ({len(stranded)})")
    ax2.set_aspect("equal")
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(colors=MUTED, labelsize=8, length=0)
    ax2.set_xlabel("east / m (survey frame)", color=MUTED, fontsize=9)
    ax2.set_ylabel("north / m (survey frame)", color=MUTED, fontsize=9)
    ax2.set_title(f"{args.dataset}: every verified pair, drawn between its two cameras\n"
                  f"colour = match strength", color=INK, fontsize=11, loc="left", pad=9,
                  linespacing=1.5)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig2.colorbar(sm, ax=ax2, fraction=0.035, pad=0.02)
    cb.set_label("inliers per pair", color=INK_2, fontsize=8.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    out2 = DATASETS / args.dataset / f"match_graph_spatial_{args.dataset}.png"
    fig2.savefig(out2, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out2)


if __name__ == "__main__":
    main()
