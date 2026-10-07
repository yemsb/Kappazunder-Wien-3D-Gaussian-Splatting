#!/usr/bin/env python3
"""
Visualise match-graph structure for the two kolonitzplatz databases.

Companion to the structural table: shows WHY the graphs differ, not just that they do.
Two figures:

  graph_comparison.png -- 4 panels, 2 series (MegaLoc vs SIFT+spatial):
      A share of verified pairs by match strength (inliers)
      B share by camera-pair distance    -- exposes the 30 m cap that spatial matching imposes
      C share by capture-instant gap     -- shows local vs long-range reach
      D cameras the LoMaB solve got WRONG, by how many links they have into the good region
        -- the bridging failure, which is the actual verdict

  graph_spatial.png -- plan view, one panel per graph, every verified pair drawn as an edge
      between the two cameras' survey positions, strength-encoded on one shared sequential
      ramp so the two panels are directly comparable.

Restricted to pairs with >= 15 inliers: that is Mapper.min_num_matches, so pairs below it are
invisible to the mapper and counting them would flatter whichever graph proposes more junk.

Palette and mark specs follow the dataviz reference (light surface #fcfcfb); the two-series
categorical pair #2a78d6 / #eb6834 passes all six validator checks.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

P = 2147483647
PAT = re.compile(r"^Sensor_(\d+)[/_]frame_(\d+)\.jpg$")

DATASETS = Path(__file__).resolve().parent / "datasets" / "kolonitzplatz"
DBS = {
    "MegaLoc": (DATASETS / "prior_run_lomab" / "database.db", "flat"),
    "SIFT + spatial + transitive": (DATASETS / "sift_prior" / "database.db", "folder"),
}
KEPT = DATASETS / "prior_run_lomab" / "georef" / "0_clean" / "cameras_kept.txt"

MIN_INLIERS = 15

# --- palette (dataviz reference instance, light mode) -----------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]           # slots 1, 2 -- validated adjacent pair
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def to_key(name, style):
    s = str(name).replace("\\", "/")
    if style == "flat":
        s = s.replace("/", "_")
    m = PAT.match(s)
    return (m.group(1), int(m.group(2)))


def load(db, style):
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    name = {r[0]: r[1] for r in cur.execute("SELECT image_id, name FROM images")}
    pos = {r[0]: np.array(struct.unpack("<3d", r[1]))
           for r in cur.execute("SELECT corr_data_id, position FROM pose_priors")}
    allp = cur.execute("SELECT pair_id, rows FROM two_view_geometries").fetchall()
    con.close()
    edges = []
    for pid, rows in allp:
        i1, i2 = divmod(pid, P)
        if rows >= MIN_INLIERS:
            edges.append((i1, i2, int(rows)))
    k = {i: to_key(n, style) for i, n in name.items()}
    return name, pos, k, edges, len(allp)


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8.5, labelpad=4)


def grouped_bars(ax, labels, vals_a, vals_b, pct=True, direct_label_max=None):
    x = np.arange(len(labels))
    w = 0.38
    for off, vals, col in ((-w / 2 - 0.005, vals_a, SERIES[0]),
                           (w / 2 + 0.005, vals_b, SERIES[1])):
        ax.bar(x + off, vals, w, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=MUTED)
    if pct:
        ax.set_ylim(0, max(max(vals_a), max(vals_b)) * 1.25)
    if direct_label_max is not None:
        for i in np.argsort(vals_b)[-direct_label_max:]:
            ax.annotate(f"{vals_b[i]:,.0f}", (x[i] + w / 2, vals_b[i]),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8, color=INK_2)


def bin_counts(vals, bins, labels):
    vals = np.asarray(vals)
    out = []
    for i, lab in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        out.append(int(((vals >= lo) & (vals < hi)).sum()))
    return np.array(out, dtype=float)


def print_table(data):
    """The structural comparison as numbers, using the same definitions as the figures."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    per = {}
    for label, d in data.items():
        ids = sorted(d["name"])
        idx = {i: j for j, i in enumerate(ids)}
        e = d["edges"]
        A = coo_matrix((np.ones(len(e)), ([idx[a] for a, b, r in e], [idx[b] for a, b, r in e])),
                       shape=(len(ids), len(ids)))
        ncomp, lab = connected_components(A, directed=False)
        same_sens = sum(1 for a, b, r in e if d["k"][a][0] == d["k"][b][0])
        same_inst = sum(1 for a, b, r in e if d["k"][a][1] == d["k"][b][1])
        links = list(d["links"].values())
        per[label] = [
            ("proposed pairs", d["n_proposed"], "d"),
            (f"effective pairs (>={MIN_INLIERS} inl)", len(e), "d"),
            ("  % of proposed", 100 * len(e) / max(d["n_proposed"], 1), ".1f"),
            ("cross-sensor %", 100 * (len(e) - same_sens) / max(len(e), 1), ".1f"),
            ("same-instant %", 100 * same_inst / max(len(e), 1), ".1f"),
            ("instant-gap median", float(np.median(d["gaps"])), ".1f"),
            ("instant-gap p90", float(np.percentile(d["gaps"], 90)), ".1f"),
            ("connected components", int(ncomp), "d"),
            ("largest component", int(np.bincount(lab).max()), "d"),
            ("images with no edge", sum(1 for v in d["deg"].values() if v == 0), "d"),
            ("mean degree", float(np.mean(list(d["deg"].values()))), ".1f"),
            ("LoMaB-dropped cameras", len(d["dropped"]), "d"),
            ("  with ZERO link to kept", sum(1 for v in links if v == 0), "d"),
            ("  mean links to kept", float(np.mean(links)), ".2f"),
        ]

    labels = list(data)
    w = max(len(m[0]) for m in per[labels[0]]) + 2
    cw = max(16, max(len(l) for l in labels) + 2)
    print("\n" + " " * w + "".join(f"{l:>{cw}s}" for l in labels))
    for i, (name, _, fmt) in enumerate(per[labels[0]]):
        print(f"{name:<{w}s}" + "".join(f"{per[l][i][1]:>{cw}{fmt}}" for l in labels))


def main():
    kept = {to_key(l.strip(), "folder") for l in KEPT.read_text().splitlines() if l.strip()}

    data = {}
    for label, (db, style) in DBS.items():
        name, pos, k, edges, n_prop = load(db, style)
        dropped = [i for i in name if k[i] not in kept]
        link_to_kept = {i: 0 for i in dropped}
        dist, gaps = [], []
        deg = {i: 0 for i in name}
        for a, b, r in edges:
            dist.append(np.linalg.norm(pos[a] - pos[b]))
            gaps.append(abs(k[a][1] - k[b][1]))
            deg[a] += 1
            deg[b] += 1
            if a in link_to_kept and k[b] in kept:
                link_to_kept[a] += 1
            if b in link_to_kept and k[a] in kept:
                link_to_kept[b] += 1
        data[label] = dict(name=name, pos=pos, k=k, edges=edges, dist=np.array(dist),
                           gaps=np.array(gaps), links=link_to_kept, dropped=dropped,
                           n_proposed=n_prop, deg=deg)

    print_table(data)

    # ---------------------------------------------------------------- figure 1: structure
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    fig.patch.set_facecolor(SURFACE)

    inl_bins = [15, 30, 60, 120, 250, 500, 1000, np.inf]
    inl_lab = ["15–30", "30–60", "60–120", "120–250", "250–500", "500–1k", "1k+"]
    dist_bins = [0, 3, 6, 12, 30, 60, 120, np.inf]
    dist_lab = ["0–3", "3–6", "6–12", "12–30", "30–60", "60–120", "120+"]
    gap_bins = [0, 1, 3, 6, 14, 51, np.inf]
    gap_lab = ["0", "1–2", "3–5", "6–13", "14–50", "51+"]

    panels = [
        (axes[0, 0], "A  Match strength",
         lambda d: bin_counts([r for _, _, r in d["edges"]], inl_bins, inl_lab), inl_lab,
         "inliers per verified pair", True),
        (axes[0, 1], "B  Camera-pair distance",
         lambda d: bin_counts(d["dist"], dist_bins, dist_lab), dist_lab,
         "metres between the two cameras", True),
        (axes[1, 0], "C  Capture-instant gap",
         lambda d: bin_counts(d["gaps"], gap_bins, gap_lab), gap_lab,
         "instants between the two frames", True),
    ]
    LA, LB = list(data)[0], list(data)[1]
    for ax, title, fn, labs, xlab, pct in panels:
        a = fn(data[LA]); b = fn(data[LB])
        if pct:
            a = 100 * a / a.sum(); b = 100 * b / b.sum()
        grouped_bars(ax, labs, a, b, pct=pct)
        style_axes(ax, title, xlab, "% of verified pairs" if pct else None)

    # D: the bridging verdict -- identical denominator (528) for both graphs
    ax = axes[1, 1]
    link_bins = [0, 1, 3, 6, np.inf]
    link_lab = ["0", "1–2", "3–5", "6+"]
    a = bin_counts([v for v in data[LA]["links"].values()], link_bins, link_lab)
    b = bin_counts([v for v in data[LB]["links"].values()], link_bins, link_lab)
    grouped_bars(ax, link_lab, a, b, pct=False, direct_label_max=1)
    style_axes(ax, "D  Bridging — cameras the LoMaB solve got wrong",
               "verified links into the 827 well-solved cameras", "cameras (of 528)")

    handles = [Line2D([], [], color=c, lw=6) for c in SERIES]
    fig.legend(handles, list(DBS), loc="upper right", frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.995, 0.985))
    fig.suptitle("Match-graph structure: appearance-based retrieval vs geometry-seeded matching "
                 "(after transitive extension)",
                 color=INK, fontsize=12, x=0.01, ha="left", y=0.985)
    fig.text(0.01, 0.005,
             f"Verified pairs only (>= {MIN_INLIERS} inliers, the mapper's min_num_matches).",
             color=MUTED, fontsize=7.6, ha="left")
    fig.tight_layout(rect=(0, 0.025, 1, 0.955))
    out1 = DATASETS / "graph_comparison.png"
    fig.savefig(out1, dpi=150, facecolor=SURFACE)
    print("wrote", out1)

    # ---------------------------------------------------------------- figure 2: spatial
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    all_inl = np.concatenate([[r for _, _, r in d["edges"]] for d in data.values()])
    norm = LogNorm(vmin=MIN_INLIERS, vmax=all_inl.max())

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.4, 6.4))
    fig2.patch.set_facecolor(SURFACE)
    for ax, (label, d) in zip(axes2, data.items()):
        ax.set_facecolor(SURFACE)
        pts = np.array([d["pos"][i] for i in d["name"]])
        ax.scatter(pts[:, 0], pts[:, 1], s=1.6, c=MUTED, alpha=0.55, linewidths=0, zorder=2)
        for a, b, r in d["edges"]:
            pa, pb = d["pos"][a], d["pos"][b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=cmap(norm(r)),
                    lw=0.5, alpha=0.5, zorder=1, solid_capstyle="round")
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        zero = sum(1 for v in d["links"].values() if v == 0)
        n_eff = len(d["edges"])
        ax.set_title(f"{label}\n{n_eff:,} verified pairs   ·   {zero} unbridged",
                     color=INK, fontsize=10.5, loc="left", pad=8, linespacing=1.5)
        ax.set_xlabel("east / m (survey frame)", color=MUTED, fontsize=8.5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig2.colorbar(sm, ax=axes2, fraction=0.021, pad=0.015)
    cb.set_label("inliers per pair (match strength)", color=INK_2, fontsize=8.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig2.suptitle("Where the connections are: every verified pair drawn between its two cameras",
                  color=INK, fontsize=12, x=0.012, ha="left", y=0.975)
    fig2.text(0.012, -0.015,
              "\"Unbridged\" = cameras the LoMaB solve misplaced (>5 m from the survey) with no verified "
              "pair into the 827 it solved well.  Reading it as a verdict is a mistake: that partition "
              "came from MegaLoc's own solve, so the metric flatters MegaLoc.  The unbridged cameras "
              "here average degree 27.6 -- embedded in their own dense clusters, not isolated.",
              color=MUTED, fontsize=7.6, ha="left")
    fig2.savefig(DATASETS / "graph_spatial.png", dpi=150, facecolor=SURFACE,
                 bbox_inches="tight")
    print("wrote", DATASETS / "graph_spatial.png")


if __name__ == "__main__":
    main()
