# SfM and georeferencing experiments — findings

Record of the work done on the `kolonitzplatz` scene to improve camera poses for 3DGS.
Kept because most of it is negative results and hard-won diagnostics that would be
expensive to rediscover.

Short version: **the rig-based Spirula pipeline did not beat the simpler unrigged
approach on splat quality.** Poses are what determine splat quality, and the rig-aware
pipeline produced *worse* poses — not better poses with worse coverage, which was the
expectation going in. Georeferencing placed the model correctly in the survey frame,
but that is a separate concern from pose accuracy and did not improve it.

---

## Setup

`colmap_pipeline/build_colmap_selection.py` selects images whose centres fall in a
buffered AoI (no view-direction pruning), writes them per sensor as
`images_by_sensor/Sensor_<id>/frame_<instant>.jpg`, and exports survey poses in COLMAP
text format. `frame_<instant>` is a dense rank over `epoch_s`, which all sensors of a
rig frame share — so one capture instant has the same basename in every sensor folder.
That matters: it is what makes `--rig` pairing possible at all.

Survey reference: 1355 poses in 271 instants, 5 non-down-facing sensors each.

---

## Findings

### 1. The exported frame is Z-DOWN, and that is a convention, not a bug

Camera up-vectors in `sparse/0/images.txt` are `-Z` for every horizontal sensor
(measured: roles 1–4 all give mean up `[0, 0, -1.000]`), and the sky dome that
`laz_to_ply.py` appends lands at `z <= 0`. Cameras and point cloud agree. This comes
from a genuine axis-convention difference between Kappazunder's LAZ and its camera
metadata; `laz_to_ply.py` mirrors to match. **Do not "fix" it.** Anything consuming
these poses should read the flags in `scene_origin.txt` rather than assume `+Z` is up.

### 2. Spirula's up-axis diagnostic is actively misleading

`spirula sfm auto --metric-positions` fits a Sim(3) over camera **centres only** and
then reports "fitted up axis vs the cameras' mean up". On this scene that reads
**174–177°**, which looks like a 180° roll. It is not one: the fit is position-only,
and the ~180° figure is a consequence of the Z-down frame meeting Spirula's
assumption that a positions file's `+Z` is up. Negating Z to make a `_zup` variant
moved it only from 177.52° to 174.22°, because the trajectory's z spread is ~0.7 m
over ~100 m — the flip barely changes the point set.

The number is **report-only**: nothing gates on it, so a wrong gauge still writes
`oriented 1` and exits 0. Do not trust it; compute the alignment yourself.

### 3. The up-facing camera breaks the rig — this was the dominant defect

`Sensor_110010` (role 0, sky-facing) finds too few features to earn a rig extrinsic.
`rigs.txt` shows the rig calibrated from only **4–20 of 271 frames**, and the member
is simply absent from the rig definition.

Consequence, measured: in **133 of 271 instants the rig members are spread > 3 m**, and
`Sensor_110010` is the outlier in **125 of them (94%)** — median 47.5 m from its
instant's centre, max **163.7 m**. The four horizontal sensors stay together. Filtering
at 5 m dropped **125 of its 202 frames** versus **2 of 249** for each horizontal sensor.

This single fact explains the large alignment residuals, the "random poses here and
there", and the smeared regions in early training runs.

### 4. Per-instant positions must be reduced robustly, not averaged

Because the survey gives one position per instant, per-sensor poses are reduced to one
point per instant. Using the **mean** let a single 160 m outlier drag the instant
position ~32 m off, corrupting the fit. Switching to the largest coincident cluster
raised alignment inliers from **39.4% to 69.4%** with no filtering applied.

### 5. Survey positions are per-instant, not per-sensor

All 5 sensors of an instant share the **identical** survey position (spread
0.0000 m) — it records the vehicle reference point, not each camera. Comparing a
per-sensor model position against it therefore measures the **lever arm** (~1.5 m), not
error. Position is only comparable at instant level; orientation genuinely is
per-sensor. This is why an earlier "82% outlier" result was an artefact of my own
comparison, not a bad reconstruction.

### 6. Depth and normal maps were ruled out

Two training runs, one stable and one dissolving, used maps that are effectively
identical where they correspond (`p50 2473` vs `2512`, `p99 18631` vs `18630` for
matched views). An initial apparent difference was an artefact of comparing
non-corresponding views. The maps are not the cause.

### 7. A 2×2 that inverted — filtering has no consistent causal role

Holding the pipeline fixed and varying only (a) whether low-quality poses were filtered
and (b) whether the alignment transform was applied:

|                | no transform          | transform applied    |
|----------------|-----------------------|----------------------|
| **no filtering** | stable (10k+)       | stable (15k)         |
| **filtering**    | dissolved (5k)      | dissolved            |

Then with **masked** SfM (sky excluded), the same grid **completely inverted**:
the filtered cells survived (11k, 17k) and the unfiltered ones dissolved (6k).

Same structure, opposite outcome. The honest reading is that depth/normal-weighted
training here is **marginally stable**, and which run dissolves is substantially luck.
No clean causal story about filtering survives.

*Caveat:* the filtered cells also carried `points3D.bin` copied verbatim, whose tracks
reference removed image IDs, while the unfiltered cells did not. That remains a
confound if anyone wants to push this further.

### 8. `--final-free-rig` gave the best single SfM result

| run | settings | registered | mean repr. |
|-----|----------|-----------|-----------|
| `spirula_sfm` | sift, 2400 px | 373/526 (71%) | 1.35 px |
| `spirula_sfm_lg` | aliked, 1600 px | 468/526 (89%) | 2.52 px |
| `sfm` | + `--rig` | 1307/1355 (96%) | 2.59 px |
| `sfm_masked` | + masks (sky) | 972/1355 (72%), fragmented | — |
| **`sfm_free`** | **+ `--final-free-rig`** | **1311/1355 (97%)** | **2.39 px** |
| `sfm_lg` | + lightglue, 8192 feats, 2h+ | ~97% | 2.65 px |

Two things worth noting. **Sky masking cost 25 points of coverage** (72% vs 97%) — the
sky-facing camera needs those features. And **more compute did not help**: lightglue
plus a doubled feature budget took over two hours and produced *worse* reprojection
(2.65 vs 2.39) and no visible splat improvement.

---

## Conclusion

Rig-based SfM is not worth pursuing further on this dataset in its current form. Its
best configuration (`sfm_free`: 97% coverage, 2.39 px, one right-facing sequence
detaching ~2 m) still lost to the manual, unrigged GUI reconstruction
(`kolonitzplatz_SS`) on splat quality — less sharp, fuzzier in many areas, with gains
only in details and in areas that benefited from wider coverage.

The dominant limitation is the sky-facing camera, and no amount of matching quality or
feature budget fixed it: it fails for lack of *features*, and masking the sky removes
more of them.

**Most promising unexplored direction:** seeding SfM with survey pose priors. Spirula
does not support this (a `sfm-cam-prior` branch exists but reportedly "didn't improve").
The project already contains a working implementation of the idea —
`refine_poses_with_colmap.py`, driving COLMAP's `pose_prior_mapper` — whose earlier run
reached **0.061 m mean deviation** over 406 images. That is the supported route and the
natural next step.

## What is still worth keeping

The georeferencing stack is correct, self-tested, and independent of the SfM quality
question: `align_sfm_to_survey.py` (orientation-aware Sim(3) with roll disambiguation,
verified against an exactly-collinear adversarial case), `generate_maps.py`,
`export_sfm_points.py`, `filter_sfm_model.py`. If a future SfM is better, these turn it
into a georeferenced, trainer-ready dataset without rework.
