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

### 9. `pose_prior_mapper` runs, but does not emit the survey frame

The route works end to end. On a 160-image probe it registered 158/160 in two models at
1.14 px; on the full 1355-image set, 1352/1355 in a **single** model at 1.19 px. So the
fragmentation that plagued rig-aware Spirula is gone.

But the premise that "the reconstruction comes out in the survey frame" is **wrong**, and
this matters before anyone trusts the output's frame. The prior residual measures
`prior_position - (-R^T t)` — the offset from the camera centre — and the Sim(3) that maps
the reconstruction onto the priors is a *nuisance parameter re-estimated inside bundle
adjustment*, not a transform applied to the output. So the model is related to the survey
only by a Sim(3).

Measured on the probe model: **rotation 1.43 deg, scale 1.021, |t| 1.98 m**. Removing that
transform took the median deviation from 0.61 m to **0.059 m** (117/128 inliers, p90
0.118 m) — the same order as the older 0.061 m result. The full model's fitted scale was
1.0023. In both cases `align_sfm_to_survey.py` is still required as the final step; on the
full model it lands 0.12 m median, 0.34 deg orientation on the inliers.

Two consequences of the same mechanism, both observed:

- A structurally wrong region need not pay a prior penalty, because the Sim(3) can absorb a
  change of gauge.
- The prior residual is 3-DoF on **position only**. Nothing constrains the points, so a
  region can hold plausible camera centres while its structure is wrong.

### 10. Position priors cannot repair a badly-bridged match graph

The full run registered 1352/1355 in one model, but filtering against the survey kept only
**827 (61.2%)**. The threshold barely matters — 60.3% at 0.5 m, 61.2% at 5 m, 61.9% at
10 m — so the discarded cameras are grossly wrong, not marginal.

The cause is the graph, and it is measurable. Dropped versus kept cameras:

| | pairs to *any* camera | pairs to *kept* cameras |
|---|---|---|
| kept (827) | 24.4 | 22.0 |
| dropped (528) | 23.6 | **3.8** |

The dropped cameras are **not starved** — they hold as many pairs as the kept ones. Their
connectivity is just concentrated among *other dropped* cameras; **31% have no pair at all**
to the kept set. Pulling such a cluster into place would require re-triangulating through
correspondences that do not exist, which is why no prior weight or solver setting fixes it.

This is a property of the graph DGSfM built. MegaLoc pairs by *appearance*, and
similar-looking views overwhelmingly come from the same camera: **57% of its 16,619 pairs
are same-sensor**. That leaves the rig's five cameras weakly tied to one another and the
route split into communities. A SIFT-based graph does not show the failure — consistent
with the older `refine_poses_with_colmap.py` result, whose SIFT + exhaustive graph reached
0.061 m over 406/526 with **no gross outliers**: sound structure, simply lower registration.

Coverage is also fragmented rather than a uniform 61%: 146 instants fully kept, 58 partial,
**67 entirely empty**, in 8 separate stretches with holes up to 32 instants. Instants 1–22
were 100% intact while 23–271 lost ~43%.

**Conclusion:** on this dataset the match graph, not the solver, is the binding constraint.
Build the graph from *geometry* (prior-driven spatial matching) rather than appearance.

### 11. COLMAP 4.1.0.dev2 CLI and database facts that each cost time

- `--log_path` is a **directory**, not a file. COLMAP creates it and writes a timestamped log
  inside; passing a file path fails with "Cannot create a file when that file already exists"
  and the tool never starts.
- The mappers' `--output_path` must **already exist**, else "`output_path` is not a directory".
- `--ImageReader.camera_params` are in **post-resize** pixel units and providing them
  **suppresses** `--FeatureExtraction.max_image_size` entirely.
- `--FeatureExtraction.max_image_size` is **inert** in this build: with no camera params and
  `max_image_size 1000` on a 7130 px image, the resulting camera was still 7130 px wide.
  COLMAP will not downscale for you — feed it the resolution you want.
- Supplying `camera_params` also sets `prior_focal_length = 1` automatically, which is what
  keeps bundle adjustment from refining the focal.
- The `pose_priors` table is the `PosePrior` schema: `(pose_prior_id, corr_data_id,
  corr_sensor_id, corr_sensor_type, position, position_covariance, gravity,
  coordinate_system)`. The `data_t` address is normalised into three columns —
  `corr_data_id` = image id, `corr_sensor_id` = camera id, `corr_sensor_type` = 0 (CAMERA).
  `position` and `gravity` are 3 float64, `position_covariance` is 9 float64, and unset
  gravity is **three NaNs, not NULL**.
- **`PosePrior::CoordinateSystem` is `UNDEFINED = -1, WGS84 = 0, CARTESIAN = 1`** — CARTESIAN
  is 1, not 0. Reversed, the rows parse cleanly and are then interpreted as lat/lon/alt.
- A 3-column keypoint blob is rejected outright ("Keypoint format not supported", a
  `LOG(FATAL_THROW)` that aborts both the verifier and the database cache). COLMAP accepts
  0/2/4/6 columns; its own writer emits 2 for a plain (x, y) array.

### 12. Two silent bugs in `align_sfm_to_survey.py` blocked this route

Both came from assuming a per-sensor *folder* layout, and both are now fixed and covered by
the self-test:

- `name_key` read the sensor from the parent directory, so the flat
  `Sensor_<id>_<basename>` naming produced an empty sensor token that never matched the
  survey side's camera id — **zero pairs, silently**, which looks like "no overlap" rather
  than a bug.
- The model writers rebuilt names as `{parent}/{basename}`, writing `/frame_000010.jpg` for a
  flat model: an invalid path *and* ambiguous, since the rig's five sensors share each
  basename. The same assumption was fixed in `cameras_kept.txt` and the CSV's sensor column.

### 13. Rebuilding the graph from geometry fixed it: 99.7% coverage

Finding 10 said the graph, not the solver, was the constraint — so the graph was rebuilt. COLMAP
SIFT at full 7130 px with **one shared PINHOLE at the vendor focal 3565** (`single_camera`;
`prior_focal_length` is set automatically when `camera_params` are passed), the same survey
priors, pairs from `spatial_matcher` (`max_distance 30`, 50 neighbours), then
`transitive_matcher` (3 iterations) to restore reach beyond the distance cap.

Spatial matching alone was **not** the fix — it made bridging worse (364 unbridged vs MegaLoc's
165), because 67% of its proximity-guessed pairs verify nothing and a 30 m cap makes the graph a
chain rather than a mesh. Transitive extension is what turned it around: effective pairs
12,272 → 37,326, mean degree 18 → 55, instant-gap median 4 → 18.

| | MegaLoc graph | SIFT + spatial + transitive |
|---|---|---|
| proposed → effective pairs | 16,619 → 16,333 | 152,683 → 37,326 |
| mean degree | 24.1 | 55.1 |
| instant-gap median / p90 | 13 / 185 | 18 / 195 |
| registered | 1352 / 1355 | 1305 / 1355 |
| mean reprojection error | 1.185 px | **1.021 px** |
| within-instant spread (median) | 0.143 m | **0.024 m** |
| instants with spread > 3 m | 125 / 271 | **4 / 271** |
| kept after the 5 m filter | 827 (61.2%) | **1301 (99.7%)** |
| alignment inliers | 779/1352 (57.6%), 161 instants | **1298/1305 (99.5%), 271/271 instants** |
| position median / mean | 0.120 / 0.129 m | 0.117 / 0.121 m |
| orientation median | 0.343 deg | 0.326 deg |
| fitted scale | 1.002258 | 1.000020 |

Two controls make this readable rather than a coincidence. The **priors were identical** in both
runs, so "the priors were simply echoed back" is excluded — the same priors kept 61% with one
graph and 99.7% with the other. And the two strongest signals are **prior-independent**:
reprojection error, and within-instant spread, which measures only whether the rig's five cameras
agree with *each other*. That spread falling 0.143 → 0.024 m is what shows the "shrunk and
embedded" clusters are gone.

Not perfect: **50 images did not register** (1305/1355 against MegaLoc's 1352). 34 of them are the
sky-facing `Sensor_110010`, 11 `Sensor_110014`, 5 `Sensor_110012` — consistent with the
sky-facing camera's lack of features (finding 3) rather than with the graph. 3DGS normally
discards the sky view anyway.

This is the best SfM result the project has produced, georeferenced and trainer-ready at
`datasets/kolonitzplatz/sift_prior/georef/0_clean` (1301 cameras + LiDAR points). Whether it beats
`kolonitzplatz_SS` on *splat* quality is now a training question, not an SfM one.

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

**Pose priors plus a geometry-built graph: this is the answer.** Seeding COLMAP's
`pose_prior_mapper` with the survey camera centres works, but the priors were never the binding
constraint — the match graph was (finding 10). Rebuild that graph from geometry rather than
appearance (SIFT at the vendor focal, `spatial_matcher` for pair selection, `transitive_matcher`
for reach beyond the distance cap) and the result is **1305/1355 registered in one model at
1.021 px, with 1301 cameras (99.7%) surviving the survey filter at 0.12 m median and every one
of the 271 instants an inlier** (finding 13). The same priors over DGSfM's appearance-based graph
kept 61%.

The lesson generalises beyond this dataset: for a rig capture, pair selection by *proximity* and
reach extended by *transitivity* beat retrieval by appearance, because appearance-based retrieval
preferentially pairs a camera with itself and leaves the rig's members weakly tied to one another.

What remains is no longer an SfM question. 50 images did not register, most of them the sky-facing
`Sensor_110010`, which is the known feature-poor sensor (finding 3) and a view 3DGS discards
anyway. Whether this reconstruction beats `kolonitzplatz_SS` on *splat* quality is settled by
training, not by reprojection error.

## What is still worth keeping

The georeferencing stack is correct, self-tested, and independent of the SfM quality
question: `align_sfm_to_survey.py` (orientation-aware Sim(3) with roll disambiguation,
verified against an exactly-collinear adversarial case), `generate_maps.py`,
`export_sfm_points.py`, `filter_sfm_model.py`. If a future SfM is better, these turn it
into a georeferenced, trainer-ready dataset without rework. `align_sfm_to_survey.py` now
also handles both the per-sensor folder layout and the flattened `Sensor_<id>_<basename>`
layout, and preserves image names on output rather than rebuilding them.

Added by the pose-prior work: `prepare_prior_db.py`, which stages a COLMAP database for
`pose_prior_mapper` (copy, keypoint-format repair, known-focal marking, survey priors into
`pose_priors`, geometric verification) and asserts the invariants that make it readable —
including the `CARTESIAN = 1` coordinate-system trap from finding 11. On a DGSfM-written
database it is what made the LoMaB features usable at all; on a COLMAP-written database only
the `priors` and `focal` stages apply.

The prior-ready database and the LoMaB-graph reference run (1352 registered, `georef/0_clean`
with 827 cameras) are kept at `datasets/kolonitzplatz/prior_run_lomab/`. The DGSfM and mpsfm
trees were deleted once these conclusions were recorded.
