#!/usr/bin/env python3
"""
Make the DGSfM kolonitzplatz database consumable by COLMAP's `pose_prior_mapper`.

The route this serves: seed SfM with the survey camera centres as pose priors, so the
reconstruction comes out already in the survey frame instead of relying on a global solver
to find the gauge. DGSfM's database is reused purely as a bag of precomputed features and
matches - the solver that produced it was abandoned, but LoMaB keypoints plus 16,619 pairs
on the 1355-image set are the expensive part and none of it depends on DGSfM's code.

Three things have to be true before `pose_prior_mapper` will read that database. All three
were established by reading COLMAP 4.1.0.dev2's headers and by round-tripping a scratch
database through pycolmap; two of them contradict the obvious guess:

1. KEYPOINTS must be (x, y) - `cols = 2`, 8 bytes each. DGSfM wrote 3-column blobs
   (x, y, score); COLMAP's reader accepts only 0/2/4/6 and throws
   `LOG(FATAL_THROW) "Keypoint format not supported"` for 3, which aborts both the
   geometric verifier and the database cache. Note `cols = 2` and not 4: a 4-column blob
   means (x, y, scale, orientation), and writing it would mean inventing a scale and an
   orientation. `cols = 2` needs nothing invented and is byte-for-byte what COLMAP itself
   writes for an (x, y) keypoint array.

2. `two_view_geometries` must be populated - `DatabaseCache::Load` reads only that table
   and ignores the raw `matches` table. The DGSfM database has 16,619 matches and zero
   geometry rows, so `colmap geometric_verifier` runs over the existing pairs. No
   descriptors are needed for geometric verification and the database has none.

3. `pose_priors` must be populated. The schema is the COLMAP `PosePrior` one, which
   normalises the `data_t` address into three columns (`corr_data_id` = image id,
   `corr_sensor_id` = camera id, `corr_sensor_type` = 0 for CAMERA). The position is the
   camera CENTRE in world coordinates, matching the residual in
   `AbsolutePosePositionPriorCostFunctor`, which evaluates
   `prior_position + R(sensor_from_world)^T t(sensor_from_world)` - i.e. it measures
   `prior_position - (-R^T t)`, the offset from the camera centre. So the value to write is
   `C = -R^T t` from the survey's world-to-camera poses, not the translation `t` itself.

   Two enum traps, both verified rather than assumed:
   - `PosePrior::CoordinateSystem` is `UNDEFINED = -1, WGS84 = 0, CARTESIAN = 1`. CARTESIAN
     is 1, not 0. Getting this wrong produces rows that parse cleanly and are then
     interpreted as lat/lon/alt.
   - `gravity` is written as three NaNs, not NULL, when unset.

   The covariances written here are diagonal with `--prior-std`; `pose_prior_mapper` reads
   them from the database unless `--overwrite_priors_covariance` is passed, so the prior
   weight can be swept from the command line without rewriting the database.

The survey poses are Z-DOWN, deliberately (`scene_origin.txt`). Nothing here flips that:
the priors are written verbatim in the survey frame, so the reconstruction comes out in the
same frame the rest of the pipeline expects.

Usage, from `colmap_pipeline/`:
    python prepare_prior_db.py                     # all stages, skipping ones already done
    python prepare_prior_db.py --stages copy,keypoints
    python prepare_prior_db.py --force             # redo everything
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from colmap_io import quat_to_rotmat

PIPELINE = Path(__file__).resolve().parent
DATASETS = PIPELINE / "datasets"
SCENE = DATASETS / "kolonitzplatz_dgsfm" / "kolonitzplatz"

DEFAULT_SOURCE_DB = SCENE / "databases" / "LoMaB_LoMaB.db"
DEFAULT_WORKSPACE = SCENE / "prior_run"
DEFAULT_SURVEY = DATASETS / "kolonitzplatz" / "sparse" / "0" / "images.txt"

STAGES = ("copy", "keypoints", "focal", "priors", "verify")

# PosePriorCoordinateSystem: UNDEFINED = -1, WGS84 = 0, CARTESIAN = 1.
CARTESIAN = 1
# SensorType: INVALID = -1, CAMERA = 0, IMU = 1.
SENSOR_TYPE_CAMERA = 0


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# Stage 1 - copy
# --------------------------------------------------------------------------------------

def stage_copy(source_db, workspace, force):
    """Copy with sqlite3's backup API rather than a file copy.

    A file copy of a WAL database silently loses whatever is still in the -wal sidecar, and
    the earlier attempt left a stale -wal/-shm pair next to a half-written copy. The backup
    API produces a single consistent file regardless of the source's journal state.
    """
    dst_path = workspace / "database.db"
    if not source_db.exists():
        raise SystemExit(f"source database not found: {source_db}")

    if dst_path.exists() and not force:
        log(f"[copy]      exists, keeping: {dst_path}")
        return dst_path

    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(dst_path) + suffix)
        if stale.exists():
            stale.unlink()
            log(f"[copy]      removed stale {stale.name}")

    workspace.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    log(f"[copy]      {source_db.name} -> {dst_path}  ({dst_path.stat().st_size / 1e6:.1f} MB)")
    return dst_path


# --------------------------------------------------------------------------------------
# Stage 2 - keypoints
# --------------------------------------------------------------------------------------

def stage_keypoints(db_path):
    """Rewrite keypoint blobs 3-column (x, y, score) -> 2-column (x, y).

    Only needed for databases written by DGSfM, which stored a score as a third column that
    COLMAP's reader rejects outright. A database COLMAP wrote itself already carries a
    supported layout (4 or 6 columns), and this stage refuses to touch formats it does not
    recognise rather than guessing - so on a COLMAP-written database, skip this stage.
    """
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("BEGIN")
        badges = con.execute(
            "SELECT image_id, rows, cols, data FROM keypoints").fetchall()

        bad = {}
        rewritten = 0
        for i, (image_id, rows, cols, data) in enumerate(badges):
            if cols == 2 and len(data) == rows * 8:
                continue
            if cols != 3 or len(data) != rows * 12:
                bad[cols] = bad.get(cols, 0) + 1
                continue

            xy = np.frombuffer(data, dtype="<f4").reshape(rows, 3)[:, :2]
            con.execute("UPDATE keypoints SET rows = ?, cols = 2, data = ? WHERE image_id = ?",
                        (rows, np.ascontiguousarray(xy).tobytes(), image_id))
            rewritten += 1
            if rewritten % 250 == 0:
                log(f"[keypoints] {rewritten}/{len(badges)} rewritten")

        if bad:
            raise SystemExit(
                f"unexpected keypoint column counts {bad}; expected 3-column (x, y, score) "
                "blobs, or 2-column ones already done. Refusing to guess.")

        con.commit()
    finally:
        con.close()

    log(f"[keypoints] rewrote {rewritten} images to cols=2")


# --------------------------------------------------------------------------------------
# Stage 3 - intrinsics
# --------------------------------------------------------------------------------------

def stage_focal(db_path):
    """Mark every camera's focal length as known, so bundle adjustment will not refine it.

    The database has one camera per image, so letting `--Mapper.ba_refine_focal_length`
    refine would add one free focal per image to the problem. A single image constrains its
    own focal very weakly, and focal trades against depth along the view direction - the
    exact degeneration that has derailed this scene before. The vendor metadata gives one
    focal, 3565 px at full resolution (1782.5 at half), for all twelve sensors, so it is a
    known quantity rather than something to solve for. The run should also pass
    `--Mapper.ba_refine_focal_length=0` so the intent does not depend on which predicate the
    BA applies.
    """
    con = sqlite3.connect(str(db_path))
    try:
        n = con.execute("UPDATE cameras SET prior_focal_length = 1").rowcount
        con.commit()
    finally:
        con.close()
    log(f"[focal]     marked {n} cameras as having a known focal length")


# --------------------------------------------------------------------------------------
# Stage 4 - pose priors
# --------------------------------------------------------------------------------------

def read_survey_images(path):
    """Parse a COLMAP images.txt into {name: (qvec, tvec, camera_id)}.

    Each image contributes a pose line and a (here empty) POINTS2D line, so lines are keyed
    off shape rather than by alternating blindly: a pose line has exactly ten whitespace
    fields with an integer first and ninth. Comment lines are skipped, and the POINTS2D
    lines are empty in this export.
    """
    out = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            int(parts[0]), int(parts[8])
        except ValueError:
            continue
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
        out[name] = (np.array([float(qw), float(qx), float(qy), float(qz)]),
                     np.array([float(tx), float(ty), float(tz)]),
                     int(camera_id))
    return out


def stage_priors(db_path, survey_path, prior_std, force):
    """Write survey camera centres into `pose_priors`."""
    survey = read_survey_images(survey_path)
    if not survey:
        raise SystemExit(f"no poses parsed from {survey_path}")

    con = sqlite3.connect(str(db_path))
    try:
        existing = con.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
        if existing and not force:
            log(f"[priors]   {existing} rows already present, keeping")
            return
        con.execute("DELETE FROM pose_priors")

        images = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        # Key the lookup from the survey side, because only that direction is unambiguous.
        # Two layouts have to be covered: a database COLMAP wrote keeps the survey's own
        # `Sensor_<id>/frame_<n>.jpg` name, while DGSfM's flattened it to
        # `Sensor_<id>_frame_<n>.jpg`. Register both spellings; a database name can only
        # match one of them.
        by_db_name = {}
        for name, pose in survey.items():
            by_db_name[name] = pose
            by_db_name[name.replace("/", "_")] = pose
        variance = float(prior_std) ** 2
        cov = struct.pack("<9d", variance, 0.0, 0.0,
                          0.0, variance, 0.0,
                          0.0, 0.0, variance)
        nan3 = struct.pack("<3d", float("nan"), float("nan"), float("nan"))

        rows, unmatched = [], []
        for image_id, db_name, camera_id in images:
            hit = by_db_name.get(db_name)
            if hit is None:
                unmatched.append(db_name)
                continue
            qvec, tvec, _ = hit
            # x_cam = R x_world + t, so the camera centre is the world point mapping to 0:
            # 0 = R C + t  ->  C = -R^T t.
            centre = -quat_to_rotmat(qvec).T @ tvec
            rows.append((image_id, image_id, camera_id, SENSOR_TYPE_CAMERA,
                         struct.pack("<3d", *centre), cov, nan3, CARTESIAN))

        if unmatched:
            log(f"[priors]   WARNING {len(unmatched)} database images have no survey pose, "
                f"e.g. {unmatched[:3]}")
        con.executemany(
            "INSERT OR REPLACE INTO pose_priors (pose_prior_id, corr_data_id, corr_sensor_id,"
            " corr_sensor_type, position, position_covariance, gravity, coordinate_system)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.commit()
    finally:
        con.close()

    unmatched_set = set(unmatched)
    survey_in_db = {n.replace("/", "_") for n in survey} - unmatched_set
    log(f"[priors]   wrote {len(rows)} priors (std {prior_std} m) from a survey file with "
        f"{len(survey)} poses, of which {len(survey_in_db)} name a database image")
    if len(rows) != len(survey):
        log(f"[priors]   NOTE coverage is not exact: "
            f"{len(rows)} priors written vs {len(survey)} survey poses")


# --------------------------------------------------------------------------------------
# Stage 5 - geometric verification
# --------------------------------------------------------------------------------------

def stage_verify(db_path, colmap, force):
    """Populate `two_view_geometries` so the database cache has something to load."""
    con = sqlite3.connect(str(db_path))
    try:
        n = con.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        n_pairs = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    finally:
        con.close()

    if n and not force:
        log(f"[verify]    {n}/{n_pairs} pairs already verified, keeping")
        return
    if not n_pairs:
        raise SystemExit("no raw matches in the database; nothing to verify")

    # `--log_path` is a DIRECTORY, not a file: COLMAP mkdirs it and writes a timestamped
    # log inside. Passing a file path fails with "Cannot create a file when that file
    # already exists" and the verifier never starts.
    log_dir = db_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log(f"[verify]    verifying {n_pairs} pairs with {colmap} (this is the slow stage)")
    cmd = [colmap, "geometric_verifier",
           "--database_path", str(db_path),
           "--log_path", str(log_dir)]
    log("[verify]    " + " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"geometric_verifier exited {proc.returncode}; see the newest "
                         f"file in {log_dir}")

    con = sqlite3.connect(str(db_path))
    try:
        n = con.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    finally:
        con.close()
    log(f"[verify]    {n}/{n_pairs} pairs have geometry rows")


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def report(db_path):
    """Read back what was written and assert the invariants that make it readable."""
    con = sqlite3.connect(str(db_path))
    try:
        c = con.cursor()
        counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("images", "cameras", "keypoints", "matches",
                            "two_view_geometries", "pose_priors")}

        # COLMAP reads keypoint blobs of 0/2/4/6 columns, each a float32 per column. This
        # is the general invariant, not "must be 2": a database COLMAP wrote itself
        # carries 4 or 6 columns (x, y, scale, orientation, ...) and is perfectly valid.
        kp_bad = c.execute(
            "SELECT COUNT(*) FROM keypoints WHERE cols NOT IN (0, 2, 4, 6)"
            " OR (cols > 0 AND length(data) != rows * 4 * cols)").fetchone()[0]

        rows = c.execute(
            "SELECT corr_data_id, corr_sensor_id, corr_sensor_type, position,"
            " position_covariance, gravity, coordinate_system FROM pose_priors").fetchall()
        bad_blobs = sum(1 for r in rows
                        if len(r[3]) != 24 or len(r[4]) != 72 or len(r[5]) != 24)
        bad_cs = sum(1 for r in rows if r[6] != CARTESIAN)
        bad_type = sum(1 for r in rows if r[2] != SENSOR_TYPE_CAMERA)
        centres = np.array([struct.unpack("<3d", r[3]) for r in rows]) if rows else np.zeros((0, 3))

        orphan = c.execute(
            "SELECT COUNT(*) FROM pose_priors p LEFT JOIN images i"
            " ON p.corr_data_id = i.image_id WHERE i.image_id IS NULL").fetchone()[0]
        cam_mismatch = c.execute(
            "SELECT COUNT(*) FROM pose_priors p JOIN images i ON p.corr_data_id = i.image_id"
            " WHERE p.corr_sensor_id != i.camera_id").fetchone()[0]
    finally:
        con.close()

    log("")
    log(f"database: {db_path}")
    for t, n in counts.items():
        log(f"  {t:<20} {n}")
    log(f"  keypoints in an unreadable format    {kp_bad}")
    log(f"  priors with wrong blob sizes         {bad_blobs}")
    log(f"  priors not CARTESIAN({CARTESIAN})     {bad_cs}")
    log(f"  priors not CAMERA({SENSOR_TYPE_CAMERA}) type   {bad_type}")
    log(f"  priors with no matching image        {orphan}")
    log(f"  priors whose camera id differs       {cam_mismatch}")
    if len(centres):
        span = centres.max(axis=0) - centres.min(axis=0)
        log(f"  prior centre extent (x, y, z)        {span[0]:.1f} {span[1]:.1f} {span[2]:.1f} m")

    problems = kp_bad + bad_blobs + bad_cs + bad_type + orphan + cam_mismatch
    if problems:
        raise SystemExit(f"{problems} consistency problem(s) - do not run the mapper on this")
    log("  all consistency checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--survey-images", type=Path, default=DEFAULT_SURVEY)
    ap.add_argument("--prior-std", type=float, default=0.5,
                    help="position prior std in metres, written as a diagonal covariance "
                         "(default 0.5)")
    ap.add_argument("--colmap", default="colmap", help="colmap executable (default: PATH)")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--force", action="store_true", help="redo stages even if already done")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {','.join(STAGES)}")

    db_path = args.workspace / "database.db"
    # Copy first so every later stage can assume the workspace exists. Omit the copy stage
    # to operate on a database the workspace already holds (e.g. one just written by
    # `colmap feature_extractor`).
    if "copy" in stages:
        db_path = stage_copy(args.source_db, args.workspace, args.force)
    if not db_path.exists():
        raise SystemExit(f"{db_path} does not exist; run the copy stage, or point "
                         f"--workspace at a directory that already holds a database.db")

    if "keypoints" in stages:
        stage_keypoints(db_path)
    if "focal" in stages:
        stage_focal(db_path)
    if "priors" in stages:
        stage_priors(db_path, args.survey_images, args.prior_std, args.force)
    if "verify" in stages:
        stage_verify(db_path, args.colmap, args.force)

    report(db_path)


if __name__ == "__main__":
    main()
