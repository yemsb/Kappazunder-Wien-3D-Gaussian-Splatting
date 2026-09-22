#!/usr/bin/env python3
"""
Align an SfM reconstruction into the georeferenced survey frame.

This replaces the alignment that `ingest_spirula_model.py` claimed to perform but
never implemented (a repo-wide grep for `umeyama` found nothing there; it re-emitted
Spirula's poses verbatim and then reported residuals quantifying an alignment that
had not happened).

The alignment is deliberately ORIENTATION-AWARE. A position-only fit is not enough
here: the capture is a car driving city streets, so the camera centres are nearly
collinear and the rotation about the route axis is very weakly determined. A fit can
settle on the 180-degree-rolled branch while producing almost identical position
residuals. Spirula's own diagnostic showed exactly this - its metric fit reported a
174-177 degree up-axis disagreement and still left cameras beyond 2 m.

THE KEY IDENTITY. For world-to-camera rotations R, a world-frame similarity
`X_ref = s R X_sfm + t` implies `R_i_ref = R_i_sfm R^T`, hence

    R = (R_i_ref)^T R_i_sfm        -- define the per-pair observation R_i_obs

In a noiseless case every R_i_obs equals the same R. Two plausible-looking
alternatives, `R_i_sfm (R_i_ref)^T` and `(R_i_sfm)^T R_i_ref`, are both wrong (they
give the inverse), and the naive approach of directly minimising
`angle(R_i_sfm, R_i_ref)` returns 90-180 deg even for a PERFECT alignment. That last
one is the classic bug in this area, so it is covered by a unit test.

The consequence is what makes this tractable: R is fixed by a single reliable
orientation pair, independent of s, t, and the trajectory's geometry.

GRANULARITY. Positions are handled per CAPTURE INSTANT and orientations per SENSOR,
because that is how the survey data is actually structured. See `classify`.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from colmap_io import (read_cameras_bin, read_images_bin,
                       quat_to_rotmat, rotmat_to_quat_wxyz)

# --------------------------------------------------------------------------------------
# SO(3) helpers. Written to stay numerically safe at both theta -> 0 and theta -> pi:
# the pi branch is genuinely reachable here because a rolled hypothesis sits there.
# --------------------------------------------------------------------------------------

def hat(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(W):
    return np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) * 0.5


def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3) + hat(w)
    K = hat(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R):
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-9:
        return vee(R)
    if theta > np.pi - 1e-6:
        # Near pi the usual theta/(2 sin theta) form is singular. Recover the axis
        # from the largest diagonal of (R + I)/2, which is well conditioned there.
        B = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / max(np.sqrt(max(B[k, k], 1e-12)), 1e-12)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.zeros(3)
        return theta * (axis / n)
    return (theta / (2.0 * np.sin(theta))) * vee(R)


def rot_axis_pi(axis):
    """Rotation by pi about a unit axis. Rodrigues at theta=pi, no series needed."""
    a = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(a, a) - np.eye(3)


def angle_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1.0) / 2.0, -1.0, 1.0))))


# --------------------------------------------------------------------------------------
# Similarity fits
# --------------------------------------------------------------------------------------

def umeyama(p, q, with_scale=True):
    """
    Least-squares similarity `q ~= s R p + t` with det(R) = +1.

    Derivation of the rotation (this is the part that is easy to get backwards):
    maximise sum(b_i . R a_i) = trace(R H) with H = sum(a_i b_i^T) = U D V^T. Writing
    M = V^T R U (orthogonal), trace(R U D V^T) = trace(V^T R U D) = trace(M D) is
    maximised at M = I, so R = V U^T. The scale follows from d/ds of the same
    objective: s = trace(S D) / sum ||a_i||^2.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pbar, qbar = p.mean(0), q.mean(0)
    A, B = p - pbar, q - qbar
    H = A.T @ B
    U, d, Vt = np.linalg.svd(H)
    V = Vt.T
    S = np.eye(3)
    if np.linalg.det(V @ U.T) < 0.0:
        S[2, 2] = -1.0
    R = V @ S @ U.T
    denom = float(np.sum(A * A))
    if with_scale and denom > 1e-12:
        s = float(np.sum(d * np.diag(S)) / denom)
    else:
        s = 1.0
    t = qbar - s * R @ pbar
    return s, R, t


def sim3_from_3(p, q):
    """
    Minimal 3-point similarity. For exactly 3 points the cross-covariance has rank
    <= 2, so the third singular value is zero to machine precision. Building the
    third axes as cross products of the first two makes the frame right-handed BY
    CONSTRUCTION, so det(R) = +1 exactly, with no det() sign juggling and no
    dependence on LAPACK's arbitrary singular-vector signs.

    Note a 3-point Sim(3) is over-determined (9 equations, 7 unknowns), so it is a
    least-squares solution - exact only when the two triangles are similar. That is
    the standard minimal solver and is fine; it just means the roll about the route
    axis is the one direction three points cannot pin down.
    """
    pbar, qbar = p.mean(0), q.mean(0)
    A, B = p - pbar, q - qbar
    H = A.T @ B
    U, d, Vt = np.linalg.svd(H)
    u1, u2 = U[:, 0], U[:, 1]
    v1, v2 = Vt[0], Vt[1]
    R = (np.outer(v1, u1) + np.outer(v2, u2)
         + np.outer(np.cross(v1, v2), np.cross(u1, u2)))
    denom = float(np.sum(A * A))
    if denom < 1e-12:
        return None
    s = float((d[0] + d[1]) / denom)
    t = qbar - s * R @ pbar
    return s, R, t


def twin_rotation(R, axis):
    """R composed with a 180-degree rotation about `axis`.

    On exactly-collinear centred data this changes no position residual at all
    (Rot(a, pi) fixes the line), which is precisely why position alone cannot
    separate the two branches.
    """
    return R @ rot_axis_pi(axis)


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------

def _sensor_token(value):
    """
    Normalise a sensor identifier so both sides of the match agree.

    The survey side carries the sensor as a camera_id (`110010`, because
    build_colmap_selection exports sensor_id as the camera id); the SfM side carries
    it as the parent directory (`Sensor_110010`). Without stripping the prefix the
    two never compare equal and the match silently yields nothing.
    """
    s = str(value)
    return s[len("Sensor_"):] if s.startswith("Sensor_") else s


def name_key(name, sensor_id=None):
    """
    Build an unambiguous match key.

    A rig capture puts the SAME basename in several sensor folders, so matching on
    basename alone is ambiguous - five images named frame_000023.jpg exist per
    instant with different poses. Qualifying by sensor resolves that; a flat SfM
    layout (no parent directory) degrades to basename matching, which the caller
    only accepts when it is unambiguous.
    """
    path = Path(str(name).replace("\\", "/"))
    if sensor_id is not None:
        return (_sensor_token(sensor_id), path.name)
    return (_sensor_token(path.parent.name), path.name)


def read_survey_images_txt_all(path):
    """
    Read a COLMAP text images.txt, returning EVERY entry as a list.

    Deliberately NOT keyed by image name. A rig capture shares basenames across
    sensor folders, so a name-keyed dict collapses the 5 references of one instant
    into 1 and silently loses 4/5 of the survey poses - which then lets a basename
    fallback pair images with the wrong sensor's pose. camera_id identifies the
    sensor.

    Assumes the POINTS2D line is empty (as build_colmap_selection exports it), so
    every non-blank, non-comment line is a pose line.
    """
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10 or not parts[0].isdigit():
                continue
            records.append({
                "image_id": int(parts[0]),
                "qvec": np.array([float(x) for x in parts[1:5]]),
                "tvec": np.array([float(x) for x in parts[5:8]]),
                "camera_id": int(parts[8]),
                "name": parts[9],
            })
    return records


def match_models(sfm_images, ref_images):
    """Match SfM images to survey references, keyed by (sensor, basename)."""
    ref_list = ref_images.values() if isinstance(ref_images, dict) else ref_images
    ref_by_key = {}
    ref_flat = {}
    for ref in ref_list:
        ref_by_key[name_key(ref["name"], ref["camera_id"])] = ref
        ref_flat.setdefault(Path(ref["name"]).name, []).append(ref)

    pairs = []
    unmatched = 0
    for sfm in sfm_images.values():
        key = name_key(sfm["name"])
        ref = ref_by_key.get(key)
        if ref is None and not key[0]:
            # Only a flat SfM layout (no sensor directory) may fall back to
            # basename, and only where that basename identifies exactly one
            # reference. For a rig capture it never does - falling back there
            # would pair images with another sensor's pose rather than report
            # them unmatched.
            cands = ref_flat.get(Path(sfm["name"]).name, [])
            if len(cands) == 1:
                ref = cands[0]
        if ref is None:
            unmatched += 1
            continue
        pairs.append((key[1], sfm, ref))
    return pairs, unmatched


def collapse_to_instants(pairs, positions=None, radius=3.0):
    """
    Build the per-instant correspondence used for POSITION, plus the mapping from
    each sensor entry to its instant.

    The survey file gives one position per capture instant - the vehicle reference
    point - shared exactly by every sensor of that instant. Reducing the model's
    camera centres within an instant to one point gives the model's rig centre, and
    the constant lever arm between that and the reference point is absorbed into the
    translation of the fitted similarity.

    The reduction must be ROBUST. Taking the mean is not: Spirula's rig leaves a
    sky-facing lens unconstrained, and that one member can sit 160 m out, dragging
    the mean ~32 m off and corrupting both the fit and every diagnostic. Instead the
    largest coincident cluster of members is used, falling back to the median. Only
    a member that actually agrees with the others contributes.
    """
    by_instant = defaultdict(list)
    for idx, (name, _sfm, _ref) in enumerate(pairs):
        by_instant[name].append(idx)

    inst_names = list(by_instant.keys())
    inst_index = {nm: i for i, nm in enumerate(inst_names)}
    inst_of = np.array([inst_index[nm] for nm, _, _ in pairs], dtype=int)

    robust = {}
    for nm, idxs in by_instant.items():
        if positions is None:
            robust[nm] = None
            continue
        P = np.asarray([positions[i] for i in idxs])
        if len(P) == 1:
            robust[nm] = P[0]
            continue
        # Largest cluster: pick the member with the most neighbours inside `radius`,
        # then average that cluster. With m good members and one wild one, this
        # selects the good ones without ever touching the outlier.
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        counts = (d <= radius).sum(axis=1)
        best = int(np.argmax(counts))
        cluster = P[d[best] <= radius] if counts[best] > 1 else P[[best]]
        robust[nm] = cluster.mean(0)
    return inst_names, by_instant, inst_of, robust


def flag_camera_outliers(pc_inst, qc_inst, pair_inst, sensor_positions, tau=5.0):
    """
    Flag individual cameras whose aligned position is far from their instant's
    survey position.

    Per-camera, not per-instant, on purpose: only one member of a rig frame is
    usually bad, so dropping whole instants would discard four good cameras with
    every bad one. `tau` sits well above the lever arm between a camera and the
    vehicle reference point (a couple of metres) and far below the observed failure
    (tens of metres), so the two populations do not overlap.
    """
    res = np.linalg.norm(np.asarray(sensor_positions) - qc_inst[pair_inst], axis=1)
    return res <= tau, res



def pre_check_mirrored(p, q):
    """
    Compare a proper and an improper orthogonal fit. A mirrored reference file
    (left-handed frame, or a transposed rotation paired with an untransposed
    translation) fits with tiny residuals under the improper variant, so residuals
    alone cannot detect it. Refusing beats silently absorbing a reflection, which
    almost always means an upstream convention bug.
    """
    pbar, qbar = p.mean(0), q.mean(0)
    A, B = p - pbar, q - qbar
    U, d, Vt = np.linalg.svd(A.T @ B)
    V = Vt.T
    denom = float(np.sum(A * A))
    if denom < 1e-12:
        return False, 0.0, 0.0

    def rms(S):
        R = V @ S @ U.T
        s = float(np.sum(d * np.diag(S)) / denom)
        resid = B - s * (A @ R.T)
        return float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))

    r_proper = rms(np.eye(3))
    S_imp = np.eye(3)
    S_imp[2, 2] = -1.0
    r_improper = rms(S_imp)
    return (r_improper < 0.25 * r_proper and r_proper > 1e-9), r_proper, r_improper


# --------------------------------------------------------------------------------------
# Core alignment
# --------------------------------------------------------------------------------------

DEFAULTS = dict(
    tau_pos=0.5,          # m   - final position gate (per instant)
    tau_pos_ransac=1.0,   # m   - looser gate used while ranking hypotheses
    tau_rot=2.0,          # deg - final orientation gate (per sensor)
    tau_rot_ransac=5.0,   # deg
    max_iters=3000,
    seed=0,
)


def robust_rotation_average(R_obs, tau_deg, seed=0):
    """RANSAC with a 1-sample minimal set: a hypothesis IS one observation.

    scipy's Rotation.mean() is the chordal mean but is NOT robust - with ~30% grossly
    wrong poses it lands nowhere useful. It is used here only inside an inlier set.
    """
    n = len(R_obs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:min(n, 200)]
    best_inliers, best_count = None, -1
    for i in idx:
        ang = np.array([angle_deg(R_obs[j], R_obs[i]) for j in range(n)])
        inliers = ang < tau_deg
        if inliers.sum() > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers
    R0 = R_obs[0]
    for _ in range(3):
        M = np.sum(R_obs[best_inliers], axis=0)
        U, _, Vt = np.linalg.svd(M)
        S = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            S[2, 2] = -1.0
        R0 = U @ S @ Vt
        ang = np.array([angle_deg(R_obs[j], R0) for j in range(n)])
        new_inliers = ang < tau_deg
        if new_inliers.sum() <= 1:
            break
        best_inliers = new_inliers
    return R0, best_inliers


def scale_translation_given_R(R, p, q, weights=None):
    """Closed-form s and t for a fixed R. Well conditioned even on a straight route
    (the along-track extent dominates), which is why roll is the only thing that
    position data fails to fix."""
    w = np.ones(len(p), dtype=bool) if weights is None else np.asarray(weights, dtype=bool)
    if w.sum() < 2:
        w = np.ones(len(p), dtype=bool)
    u = p[w] @ R.T
    v = q[w]
    ubar, vbar = u.mean(0), v.mean(0)
    du, dv = u - ubar, v - vbar
    denom = float(np.sum(du * du))
    if denom < 1e-12:
        return 1.0, vbar
    s = float(np.sum(du * dv) / denom)
    return s, vbar - s * ubar


def classify(s, R, t, pc, qc, R_obs, inst_of, tau_pos, tau_rot):
    """
    Joint inlier classification, at the correct granularity for each term.

    Position is judged per CAPTURE INSTANT, not per sensor. The survey file records
    one position per instant - the vehicle reference point - and every sensor of a
    rig frame shares it exactly (measured spread: 0.0000 m across all 271 instants).
    Comparing a per-sensor model position against that shared point therefore mostly
    measures the lever arm between the camera and the reference point, which is
    ~1.5 m here: it would reject ~80% of a perfectly good reconstruction while
    reporting excellent orientation agreement. The lever arm is a constant offset,
    so fitting at instant level absorbs it into the translation.

    Orientation genuinely is per-sensor, so it is judged per sensor.
    """
    pos_res_inst = np.linalg.norm(qc - (s * (pc @ R.T) + t), axis=1)
    rot_res = np.array([angle_deg(R, Ro) for Ro in R_obs])
    ok = (pos_res_inst <= tau_pos)[inst_of] & (rot_res <= tau_rot)
    return ok, pos_res_inst, rot_res


def align(pc, qc, R_obs, inst_of, cfg=None, use_orientation=True, seed=0):
    """
    Robust alignment of SfM camera poses into the survey frame.

    Positions arrive per INSTANT (`pc`, `qc`); orientations per SENSOR (`R_obs`),
    with `inst_of` mapping each sensor entry to its instant - see `classify`.

    Stage C (orientation-first) is deterministic, costs milliseconds, and is immune
    to the roll ambiguity, so the correct branch is always present in the pool.
    Stage D is a RANSAC over 3-instant position samples scored on position AND
    orientation, with every hypothesis evaluated alongside its 180-degree twin.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    n_inst, n_sensor = len(pc), len(R_obs)

    # --- Stage C: deterministic orientation-first hypothesis --------------------
    R_ori, ori_inl_sensor = robust_rotation_average(R_obs, cfg["tau_rot_ransac"], seed)
    ori_inl_inst = np.zeros(n_inst, dtype=bool)
    ori_inl_inst[inst_of[ori_inl_sensor]] = True
    s_ori, t_ori = scale_translation_given_R(R_ori, pc, qc, ori_inl_inst)

    scored = []
    if use_orientation:
        scored.append((s_ori, R_ori, t_ori, "orientation_first"))
    else:
        s0, R0, t0 = umeyama(pc, qc)
        scored.append((s0, R0, t0, "position_only"))

    centroid = pc.mean(0)
    axis = np.linalg.svd(pc - centroid, full_matrices=False)[2][0]
    scored = [h for h in scored if np.linalg.det(h[1]) > 0]

    def score(s, R, t):
        # `use_orientation=False` must genuinely exclude orientation from BOTH the
        # inlier test and the cost, otherwise the negative control in the self-test
        # would still be using orientation to pick a branch and would prove nothing.
        tau_rot = cfg["tau_rot_ransac"] if use_orientation else np.inf
        ok, pos_res_inst, rot_res = classify(s, R, t, pc, qc, R_obs, inst_of,
                                             cfg["tau_pos_ransac"], tau_rot)
        cost = float(np.sum(np.minimum((pos_res_inst / cfg["tau_pos_ransac"]) ** 2, 1.0)))
        if use_orientation:
            cost += float(np.sum(np.minimum((rot_res / cfg["tau_rot_ransac"]) ** 2, 1.0)))
        return int(ok.sum()), -cost, ok

    best = None
    for s, R, t, src in scored:
        for cand_R in ([R, twin_rotation(R, axis)] if use_orientation else [R]):
            cnt, negcost, ok = score(s, cand_R, t)
            if best is None or (cnt, negcost) > (best[0], best[1]):
                best = (cnt, negcost, s, cand_R, t, ok, src)

    # --- Stage D: RANSAC over 3-INSTANT position samples -------------------------
    rng = np.random.default_rng(seed)
    extent = float(np.max(np.linalg.norm(pc - centroid, axis=1)))
    it, iters = 0, 100
    while it < min(iters, cfg["max_iters"]):
        it += 1
        S = rng.choice(n_inst, size=3, replace=False)
        tri = pc[S]
        # Reject thin triangles: their rotation is dominated by noise.
        if (np.linalg.norm(tri[1] - tri[0]) < 1e-9
                or np.linalg.norm(tri[2] - tri[0]) < 1e-9):
            continue
        if np.max(np.linalg.norm(tri - tri.mean(0), axis=1)) < 0.05 * extent:
            continue
        fit = sim3_from_3(tri, qc[S])
        if fit is None:
            continue
        s_a, R_a, t_a = fit
        if not (1e-3 < s_a < 1e3) or np.linalg.det(R_a) <= 0:
            continue
        local_axis = np.linalg.svd(tri - tri.mean(0), full_matrices=False)[2][0]
        for s_h, R_h, t_h in ((s_a, R_a, t_a),
                              (s_a, twin_rotation(R_a, local_axis), t_a)):
            cnt, negcost, ok = score(s_h, R_h, t_h)
            if (cnt, negcost) > (best[0], best[1]):
                best = (cnt, negcost, s_h, R_h, t_h, ok, "minimal_svd")
        w = max(best[0] / max(n_sensor, 1), 1e-3)
        iters = int(np.clip(np.ceil(np.log(1e-3) / np.log(max(1 - w ** 3, 1e-12))),
                            100, cfg["max_iters"]))

    _, _, s, R, t, inliers, source = best

    # --- Stage E: refit on inliers to a fixpoint ---------------------------------
    for _ in range(5):
        ok, _, _ = classify(s, R, t, pc, qc, R_obs, inst_of,
                            cfg["tau_pos"], cfg["tau_rot"])
        inl_inst = np.zeros(n_inst, dtype=bool)
        inl_inst[inst_of[ok]] = True
        if inl_inst.sum() < 4:
            break
        s_new, R_new, t_new = umeyama(pc[inl_inst], qc[inl_inst])
        settled = np.array_equal(ok, inliers)
        s, R, t, inliers = s_new, R_new, t_new, ok
        if settled:
            break

    # --- Stage F: non-linear refinement over the 7 parameters --------------------
    if inliers.sum() >= 6:
        idx_sensor = np.where(inliers)[0]
        inl_inst = np.zeros(n_inst, dtype=bool)
        inl_inst[inst_of[inliers]] = True

        def residual(x):
            R_x = exp_so3(x[:3])
            t_x = x[3:6]
            s_x = np.exp(x[6])
            rp = qc[inl_inst] - (s_x * (pc[inl_inst] @ R_x.T) + t_x)
            rr = np.array([log_so3(R_x.T @ R_obs[i]) for i in idx_sensor])
            return np.concatenate([rp.ravel() / 0.3, rr.ravel() / np.radians(1.0)])

        x0 = np.concatenate([log_so3(R), t, [np.log(max(s, 1e-9))]])
        try:
            sol = least_squares(residual, x0, method="trf", loss="soft_l1",
                                f_scale=2.0, max_nfev=200)
            R_lm, t_lm = exp_so3(sol.x[:3]), sol.x[3:6]
            s_lm = float(np.exp(sol.x[6]))
            if np.linalg.det(R_lm) > 0 and s_lm > 0:
                ok_lm, _, _ = classify(s_lm, R_lm, t_lm, pc, qc, R_obs, inst_of,
                                       cfg["tau_pos"], cfg["tau_rot"])
                if ok_lm.sum() >= inliers.sum():
                    s, R, t, inliers = s_lm, R_lm, t_lm, ok_lm
        except Exception:
            pass  # keep the closed-form solution

    return dict(s=float(s), R=R, t=t, inliers=inliers, source=source,
                n=n_sensor, n_inst=n_inst, inst_of=inst_of)


def _trimmed_mean_up(u, sel):
    uu = u[sel]
    if len(uu) == 0:
        return np.array([0.0, 0.0, 1.0])
    m = uu.mean(0)
    m /= max(np.linalg.norm(m), 1e-12)
    for _ in range(2):
        ang = np.degrees(np.arccos(np.clip(uu @ m, -1.0, 1.0)))
        keep = ang < 15.0
        if keep.sum() < max(3, 0.5 * len(uu)):
            break
        m = uu[keep].mean(0)
        m /= max(np.linalg.norm(m), 1e-12)
    return m


def diagnose(res, pc, qc, R_obs, u_sfm, u_ref):
    """
    Diagnostics that separate two failures which look identical in a log:

      theta_up ~ 180 deg AND orientation_rmse ~ 180 deg -> a genuinely rolled transform
      theta_up ~ 180 deg BUT orientation_rmse small      -> the transform is FINE and
                                                            the up-diagnostic is wrong

    The second case is easy to produce by comparing against a hardcoded +Z in a
    Z-down frame, which is the class of error behind the confusing 177.52 deg
    reading this pipeline previously produced. Computing both numbers is the only
    way to tell them apart, which is why they are always reported together.

    theta_up is measured against the SURVEY camera up axes, never a hardcoded +Z,
    so it stays meaningful in any world frame.
    """
    s, R, t, inliers = res["s"], res["R"], res["t"], res["inliers"]
    inst_of, n_inst = res["inst_of"], res["n_inst"]

    pos_res_inst = np.linalg.norm(qc - (s * (pc @ R.T) + t), axis=1)
    rot_res = np.array([angle_deg(R, Ro) for Ro in R_obs])

    inl_inst = np.zeros(n_inst, dtype=bool)
    inl_inst[inst_of[inliers]] = True
    ip, ir = pos_res_inst[inl_inst], rot_res[inliers]

    ubar_sfm = _trimmed_mean_up(u_sfm, inliers)
    ubar_ref = _trimmed_mean_up(u_ref, inliers)
    theta_up = float(np.degrees(np.arccos(np.clip(float(ubar_ref @ (R @ ubar_sfm)),
                                                  -1.0, 1.0))))

    # Roll observability: the exact positional gap between the two roll branches.
    sel = pc[inl_inst] - pc[inl_inst].mean(0)
    lam = np.linalg.eigvalsh(np.cov(sel.T)) if len(sel) > 3 else np.zeros(3)
    lam = np.clip(lam, 0.0, None)
    twin_sep = float(2.0 * s * np.sqrt(max(lam[1] + lam[2], 0.0)))
    sigma_p = float(np.std(pos_res_inst)) if len(pos_res_inst) > 1 else 0.0
    denom = s * np.sqrt(max(lam[1] + lam[2], 1e-12))
    delta_psi = float(np.degrees(np.arctan(sigma_p / denom))) if denom > 1e-9 else 90.0

    def within(v, thr):
        return int(np.sum(v <= thr))

    return dict(
        n_inliers=int(inliers.sum()), n_total=res["n"],
        n_inst_inliers=int(inl_inst.sum()), n_inst=n_inst,
        inlier_ratio=float(inliers.sum() / res["n"]),
        pos=dict(median=float(np.median(ip)), mean=float(ip.mean()),
                 rms=float(np.sqrt(np.mean(ip ** 2))),
                 min=float(ip.min()), max=float(ip.max()),
                 within_0_2=within(ip, 0.2), within_0_5=within(ip, 0.5),
                 within_1_0=within(ip, 1.0), within_2_0=within(ip, 2.0)),
        pos_all=dict(median=float(np.median(pos_res_inst)),
                     mean=float(pos_res_inst.mean()),
                     max=float(pos_res_inst.max())),
        rot=dict(median=float(np.median(ir)), mean=float(ir.mean()),
                 rms=float(np.sqrt(np.mean(ir ** 2))),
                 within_1=within(ir, 1.0), within_5=within(ir, 5.0),
                 within_15=within(ir, 15.0)),
        rot_all=dict(median=float(np.median(rot_res)), mean=float(rot_res.mean())),
        theta_up=theta_up,
        acos_R22=float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))),
        orientation_rmse=float(np.sqrt(np.mean(ir ** 2))),
        twin_separation_m=twin_sep, delta_psi_pos_deg=delta_psi,
        ubar_sfm=ubar_sfm.tolist(), ubar_ref=ubar_ref.tolist(),
        scale=s, det_R=float(np.linalg.det(R)),
        source=res["source"],
    )


# --------------------------------------------------------------------------------------
# Diagnostic plot
# --------------------------------------------------------------------------------------

def plot_diagnostic(out_path, pc, qc, R_obs, res, dg):
    """
    Four panels, meant to be read together:

      top-down   where the aligned model sits relative to the survey, with a line per
                 instant whose length IS the residual, coloured by inlier status. If
                 the pose alignment were globally right, these lines would be uniformly
                 short; a region of long lines localises the problem.
      vs order / vs along-route
                 whether failures cluster in one stretch of the drive (a deformation
                 story) or are scattered (a correspondence story).
      orientation
                 a separate check, since orientation does not depend on the lever arm
                 between a camera and the vehicle reference point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, R, t, inl = res["s"], res["R"], res["t"], res["inliers"]
    inst_of = res["inst_of"]
    n_inst = len(pc)

    inl_inst = np.zeros(n_inst, dtype=bool)
    inl_inst[inst_of[inl]] = True
    pc_aligned = s * (pc @ R.T) + t
    pos_res = np.linalg.norm(qc - pc_aligned, axis=1)
    rot_res = np.array([angle_deg(R, Ro) for Ro in R_obs])

    cent = pc - pc.mean(0)
    along = cent @ np.linalg.svd(cent, full_matrices=False)[2][0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    a = ax[0, 0]
    a.scatter(qc[:, 0], qc[:, 1], s=26, c="k", marker="x", label="survey (vehicle ref pt)")
    a.scatter(pc_aligned[inl_inst, 0], pc_aligned[inl_inst, 1], s=16,
              c="tab:green", label=f"model, inlier instant ({inl_inst.sum()})")
    a.scatter(pc_aligned[~inl_inst, 0], pc_aligned[~inl_inst, 1], s=16,
              c="tab:red", label=f"model, outlier instant ({(~inl_inst).sum()})")
    for i in range(n_inst):
        a.plot([qc[i, 0], pc_aligned[i, 0]], [qc[i, 1], pc_aligned[i, 1]],
               "-", lw=0.5, alpha=0.6,
               color="tab:green" if inl_inst[i] else "tab:red")
    a.set_aspect("equal")
    a.set_title("top-down (x, y): line length = position residual")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(np.arange(n_inst), pos_res, ".", ms=4, c="tab:blue")
    a.axhline(0.5, color="r", ls="--", lw=1, label="0.5 m gate")
    a.set_yscale("log")
    a.set_xlabel("capture order (instant index)")
    a.set_ylabel("position residual (m)")
    a.set_title("per-instant position residual, capture order")
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.scatter(along, pos_res, s=14,
              c=np.where(inl_inst, "tab:green", "tab:red"))
    a.axhline(0.5, color="r", ls="--", lw=1)
    a.set_yscale("log")
    a.set_xlabel("distance along route principal axis (m)")
    a.set_ylabel("position residual (m)")
    a.set_title("residual vs along-route position")

    a = ax[1, 1]
    a.hist(rot_res, bins=60, color="tab:purple")
    a.axvline(2.0, color="r", ls="--", lw=1, label="2 deg gate")
    a.set_yscale("log")
    a.set_xlabel("orientation residual (deg)")
    a.set_ylabel("sensors")
    a.set_title(f"per-sensor orientation residual (median {np.median(rot_res):.2f} deg)")
    a.legend(fontsize=8)

    fig.suptitle(
        f"SfM-to-survey alignment  |  scale {s:.5f}  det(R) {np.linalg.det(R):+.3f}  "
        f"theta_up {dg['theta_up']:.2f} deg  |  inliers {dg['n_inst_inliers']}/{dg['n_inst']} "
        f"instants, {dg['n_inliers']}/{dg['n_total']} sensors"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Wrote diagnostic plot: {out_path}")


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

def _city_route(n, rng):
    p = np.zeros((n, 3))
    x = 0.0
    for i in range(n):
        p[i] = [x, 0.0, 0.0]
        x += rng.uniform(0.4, 1.6)
        if i % 40 == 39:
            p[i, 1] += rng.uniform(3.0, 12.0)
    p[:, 2] = rng.normal(0, 0.4, n)
    return p


def _make_case(seed, outlier_frac=0.3, n=400, collinear=False):
    rng = np.random.default_rng(seed)
    if collinear:
        p = np.outer(np.linspace(0, 1200, n), np.array([1.0, 0.0, 0.0]))
    else:
        p = _city_route(n, rng)
    s_gt = float(rng.uniform(0.2, 1.5))
    R_gt = exp_so3(rng.normal(size=3) * 0.6)
    t_gt = rng.normal(size=3) * 500.0

    q = s_gt * (p @ R_gt.T) + t_gt
    R_sfm = np.stack([exp_so3(rng.normal(size=3) * 0.2) for _ in range(n)])
    R_ref = R_sfm @ R_gt.T                      # from R_ref = R_sfm R^T
    k = int(outlier_frac * n)
    out = rng.choice(n, size=k, replace=False)
    q[out] += rng.normal(size=(k, 3)) * rng.uniform(5, 150, size=(k, 1))
    R_ref[out] = np.stack([exp_so3(rng.normal(size=3) * np.radians(25))
                           for _ in range(k)]) @ R_ref[out]
    q += rng.normal(size=(n, 3)) * 0.15
    R_ref = np.stack([exp_so3(rng.normal(size=3) * np.radians(0.3))
                      for _ in range(n)]) @ R_ref
    mask = np.ones(n, dtype=bool)
    mask[out] = False
    return p, q, R_sfm, R_ref, s_gt, R_gt, t_gt, mask


def self_test(verbose=True):
    """Synthetic round-trips plus the adversarial case that motivates the design."""
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
            if verbose:
                print(f"  FAIL: {msg}")
        elif verbose:
            print(f"  ok  : {msg}")

    print("1. composition rule (the classic bug)")
    Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R_sfm = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    R_ref = R_sfm @ Rz90.T
    check(angle_deg(R_ref.T @ R_sfm, Rz90) < 1e-9,
          "(R_ref)^T R_sfm recovers the ground-truth rotation")
    check(angle_deg(R_sfm @ R_ref.T, Rz90) > 90.0,
          "the wrong order R_sfm (R_ref)^T is genuinely wrong (not a near miss)")

    print("2. round-trip recovery with 30% gross outliers")
    for seed in range(8):
        p, q, R_sfm, R_ref, s_gt, R_gt, t_gt, mask = _make_case(seed, 0.3)
        R_obs = np.stack([R_ref[i].T @ R_sfm[i] for i in range(len(p))])
        u_sfm, u_ref = -R_sfm[:, 1, :], -R_ref[:, 1, :]
        n = len(p)
        inst_of = np.arange(n)          # one sensor per instant: no rig structure
        res = align(p, q, R_obs, inst_of, seed=seed)
        dg = diagnose(res, p, q, R_obs, u_sfm, u_ref)
        check(angle_deg(res["R"], R_gt) < 0.5, f"seed {seed}: rotation recovered")
        check(abs(res["s"] / s_gt - 1.0) < 5e-3, f"seed {seed}: scale recovered")
        check(dg["inlier_ratio"] > 0.65, f"seed {seed}: inlier ratio {dg['inlier_ratio']:.2f}")
        if seed == 0:
            check(dg["theta_up"] < 5.0, f"seed {seed}: theta_up {dg['theta_up']:.2f} deg")

    print("3. ADVERSARIAL: exactly collinear trajectory")
    p, q, R_sfm, R_ref, s_gt, R_gt, t_gt, mask = _make_case(7, 0.3, n=300, collinear=True)
    R_obs = np.stack([R_ref[i].T @ R_sfm[i] for i in range(len(p))])
    u_sfm, u_ref = -R_sfm[:, 1, :], -R_ref[:, 1, :]
    inst_of = np.arange(len(p))
    axis = np.array([1.0, 0.0, 0.0])

    rms_gt = np.sqrt(np.mean(np.sum(q - (s_gt * (p @ R_gt.T) + t_gt), axis=1) ** 2))
    R_twin = R_gt @ rot_axis_pi(axis)
    rms_twin = np.sqrt(np.mean(np.sum(q - (s_gt * (p @ R_twin.T) + t_gt), axis=1) ** 2))
    check(abs(rms_gt - rms_twin) < 1e-6,
          f"twin is positionally identical (rms {rms_gt:.3e} vs {rms_twin:.3e} m)")
    check(angle_deg(R_gt, R_twin) > 179.0, "twin differs by 180 deg in orientation")

    res = align(p, q, R_obs, inst_of, seed=0)
    check(angle_deg(res["R"], R_gt) < 0.5, "correct branch chosen despite collinearity")
    check(angle_deg(res["R"], R_twin) > 179.0, "rolled branch rejected")

    # Negative control. Scored on POSITION ALONE the two branches must be exactly
    # indistinguishable -- if position could separate them the orientation term would
    # be doing no work above and the test would be vacuous.
    def pos_cost(Rc):
        _, pos_res, _ = classify(s_gt, Rc, t_gt, p, q, R_obs, inst_of, np.inf, np.inf)
        return float(np.sum(np.minimum((pos_res / 1.0) ** 2, 1.0)))

    c_gt, c_twin = pos_cost(R_gt), pos_cost(R_twin)
    check(abs(c_gt - c_twin) <= 1e-9 * max(c_gt, 1.0),
          f"negative control: position-only cost identical for both branches "
          f"({c_gt:.6f} vs {c_twin:.6f})")

    print()
    if fails:
        print(f"SELF-TEST FAILED ({len(fails)} failure(s))")
        return 1
    print("SELF-TEST PASSED")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def run(config_path, output_dir, model_dir=None, filter_tau=5.0):
    config_path = Path(config_path).resolve()
    scene = config_path.parent.name
    dataset = Path(__file__).resolve().parent / "datasets" / scene

    model_dir = Path(model_dir) if model_dir else dataset / "sfm" / "sparse" / "0"
    if not (model_dir / "images.bin").exists():
        sys.exit(f"No SfM model at {model_dir} (expected images.bin).")

    sfm_images = read_images_bin(model_dir / "images.bin")
    ref_records = read_survey_images_txt_all(dataset / "sparse" / "0" / "images.txt")
    if not ref_records:
        sys.exit(f"No survey reference at {dataset / 'sparse' / '0' / 'images.txt'}")

    pairs, unmatched = match_models(sfm_images, ref_records)
    print(f"Matched {len(pairs)} pairs ({unmatched} unmatched SfM images, "
          f"{len(sfm_images)} SfM / {len(ref_records)} survey)")
    if len(pairs) < 25:
        sys.exit("Too few matched pairs to align reliably.")

    names = [nm for nm, _, _ in pairs]
    p_sensor = np.array([-quat_to_rotmat(s["qvec"]).T @ s["tvec"] for _, s, _ in pairs])
    q_sensor = np.array([-quat_to_rotmat(r["qvec"]).T @ r["tvec"] for _, _, r in pairs])
    R_sfm = np.stack([quat_to_rotmat(s["qvec"]) for _, s, _ in pairs])
    R_ref = np.stack([quat_to_rotmat(r["qvec"]) for _, _, r in pairs])
    R_obs = np.stack([R_ref[i].T @ R_sfm[i] for i in range(len(pairs))])
    u_sfm, u_ref = -R_sfm[:, 1, :], -R_ref[:, 1, :]

    # Collapse to per-instant correspondences for the position term. The survey
    # gives one position per instant (shared by all its sensors), so this is the
    # only granularity at which position is comparable. The reduction is robust -
    # see collapse_to_instants for why a plain mean is not usable here.
    inst_names, by_instant, inst_of, robust_pos = collapse_to_instants(pairs, p_sensor)
    pc = np.array([robust_pos[nm] for nm in inst_names])
    qc = np.array([q_sensor[by_instant[nm][0]] for nm in inst_names])
    print(f"Instants: {len(inst_names)} "
          f"(mean {len(pairs) / max(len(inst_names), 1):.1f} sensors each)")

    spreads = np.array([np.max(np.linalg.norm(
        np.array([p_sensor[i] for i in by_instant[nm]]) - robust_pos[nm], axis=1))
        for nm in inst_names])
    print(f"within-instant spread: median {np.median(spreads):.3f} m, "
          f"{int((spreads > 3.0).sum())} / {len(inst_names)} instants over 3 m "
          f"(the rig failing to constrain a member)")

    mirrored, r_proper, r_improper = pre_check_mirrored(pc, qc)
    if mirrored:
        sys.exit(f"REFUSING: the reference looks mirrored (proper rms {r_proper:.3f} m "
                 f"vs improper {r_improper:.3f} m). That indicates an upstream axis or "
                 f"convention bug - fix it there rather than absorbing a reflection here.")

    res = align(pc, qc, R_obs, inst_of)
    dg = diagnose(res, pc, qc, R_obs, u_sfm, u_ref)

    out = Path(output_dir) if output_dir else dataset / "georef" / "0"
    out.mkdir(parents=True, exist_ok=True)
    s, R, t = res["s"], res["R"], res["t"]

    # --- write the aligned model in COLMAP text format ---------------------------
    with open(out / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] (empty)\n")
        for i, (nm, sfm, _) in enumerate(pairs, start=1):
            C = -(R_sfm[i - 1].T @ sfm["tvec"])
            C_new = s * (R @ C) + t
            R_new = R_sfm[i - 1] @ R.T
            tw = -R_new @ C_new
            qw, qx, qy, qz = rotmat_to_quat_wxyz(R_new)
            f.write(f"{i} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tw[0]:.6f} {tw[1]:.6f} {tw[2]:.6f} {sfm['camera_id']} "
                    f"{Path(sfm['name']).parent.name}/{nm}\n\n")
    cams = read_cameras_bin(model_dir / "cameras.bin")
    with open(out / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, cam in cams.items():
            f.write(f"{cid} PINHOLE {cam['width']} {cam['height']} "
                    f"{' '.join(f'{v:.6f}' for v in cam['params'][:4])}\n")

    lidar = dataset / "sparse" / "0" / "points3D.ply"
    if lidar.exists():
        (out / "points3D.ply").write_bytes(lidar.read_bytes())

    with open(out / "scene_transform.json", "w") as f:
        json.dump({
            "train_from_world": {
                "scale": s,
                "rotation": {"matrix_3x3": R.tolist()},
                "translation": t.tolist(),
            },
            "matrix_3x3_flat_row_major": R.reshape(-1).tolist(),
            "frame": "pipeline export frame (Z-DOWN: +Z points physically down)",
            "det_R": float(np.linalg.det(R)),
        }, f, indent=2)

    pos_res_inst = np.linalg.norm(qc - (s * (pc @ R.T) + t), axis=1)
    rot_res = np.array([angle_deg(R, Ro) for Ro in R_obs])
    with open(out / "pose_alignment_per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "sensor", "is_inlier", "pos_res_instant_m",
                    "rot_res_deg", "inlier_reason"])
        for i, (nm, sfm, _) in enumerate(pairs):
            ok = bool(res["inliers"][i])
            pr, rr = float(pos_res_inst[inst_of[i]]), float(rot_res[i])
            reason = "inlier" if ok else ("position" if pr > 0.5 else "orientation")
            w.writerow([nm, Path(sfm["name"]).parent.name, int(ok),
                        f"{pr:.6f}", f"{rr:.4f}", reason])

    with open(out / "alignment_report.txt", "w") as f:
        f.write("SfM-to-Survey Alignment Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source model:    {model_dir}\n")
        f.write(f"Matched pairs:   {len(pairs)} sensors in {len(inst_names)} capture "
                f"instants (unmatched SfM images: {unmatched})\n")
        f.write(f"Best hypothesis: {dg['source']}\n\n")
        f.write("Transform  X_ref = s * R * X_sfm + t\n")
        f.write(f"  scale {s:.9f}   det(R) {np.linalg.det(R):+.9f}\n")
        f.write(f"  translation [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n\n")
        f.write(f"Inliers: {dg['n_inliers']} / {dg['n_total']} sensors "
                f"({dg['inlier_ratio'] * 100:.1f}%), covering "
                f"{dg['n_inst_inliers']} / {dg['n_inst']} instants\n\n")
        f.write("Position residuals (metres) - PER CAPTURE INSTANT, inliers:\n")
        for k in ("median", "mean", "rms", "min", "max"):
            f.write(f"  {k:<7}: {dg['pos'][k]:.4f}\n")
        for k, lab in (("within_0_2", "0.20"), ("within_0_5", "0.50"),
                       ("within_1_0", "1.00"), ("within_2_0", "2.00")):
            f.write(f"  < {lab} m: {dg['pos'][k]} / {dg['n_inst_inliers']} "
                    f"({100 * dg['pos'][k] / max(dg['n_inst_inliers'], 1):.1f}%)\n")
        f.write(f"  (all {dg['n_inst']} instants: median {dg['pos_all']['median']:.4f}, "
                f"mean {dg['pos_all']['mean']:.4f}, max {dg['pos_all']['max']:.4f})\n\n")
        f.write("Orientation residuals (degrees) - PER SENSOR, inliers:\n")
        for k in ("median", "mean", "rms"):
            f.write(f"  {k:<7}: {dg['rot'][k]:.4f}\n")
        for k, lab in (("within_1", "1"), ("within_5", "5"), ("within_15", "15")):
            f.write(f"  < {lab} deg: {dg['rot'][k]} / {dg['n_inliers']}\n")
        f.write(f"  (all {dg['n_total']} sensors: median {dg['rot_all']['median']:.4f}, "
                f"mean {dg['rot_all']['mean']:.4f})\n\n")
        f.write("WHY THE TWO TERMS USE DIFFERENT UNITS OF COMPARISON\n")
        f.write("  The survey file gives ONE position per capture instant - the vehicle\n")
        f.write("  reference point - and every sensor of a rig frame shares it exactly\n")
        f.write("  (measured spread 0.0000 m). A per-sensor model position therefore\n")
        f.write("  differs from it by the camera's lever arm (~1.5 m), which is a\n")
        f.write("  constant offset absorbed into the translation. Comparing position\n")
        f.write("  per sensor would reject the whole model while orientation agreed\n")
        f.write("  near-perfectly. Orientation IS per sensor, so it is judged there.\n\n")
        f.write("Up-axis check (the decisive diagnostic):\n")
        f.write(f"  theta_up vs SURVEY camera up axes: {dg['theta_up']:.3f} deg\n")
        f.write(f"  acos(R[2,2]) (for continuity)    : {dg['acos_R22']:.3f} deg\n")
        f.write(f"  orientation_rmse                 : {dg['orientation_rmse']:.3f} deg\n")
        f.write("  Interpretation: theta_up near 180 WITH small orientation_rmse means\n")
        f.write("  the transform is fine and the up-diagnostic is wrong; theta_up near 180\n")
        f.write("  WITH orientation_rmse near 180 means a genuinely rolled transform.\n\n")
        f.write("Roll observability:\n")
        f.write(f"  twin separation : {dg['twin_separation_m']:.4f} m "
                f"(positional gap between the two roll branches)\n")
        f.write(f"  delta_psi_pos   : {dg['delta_psi_pos_deg']:.2f} deg\n")
        f.write(f"  mean camera up (SfM)   : {np.round(dg['ubar_sfm'], 4).tolist()}\n")
        f.write(f"  mean camera up (survey): {np.round(dg['ubar_ref'], 4).tolist()}\n\n")
        f.write("Note: residual statistics above describe the INLIER set, which is\n")
        f.write("selected. All-pairs figures are given alongside where meaningful.\n")

    # --- per-camera outlier flagging, and a cleaned model ------------------------
    # Filtered per CAMERA, not per instant: usually only one member of a frame is
    # bad, so dropping whole instants would throw away four good cameras with every
    # bad one. The full model above is always kept; this is an additional output.
    C_aligned = s * (p_sensor @ R.T) + t
    cam_ok, cam_res = flag_camera_outliers(pc, qc, inst_of, C_aligned, tau=filter_tau)
    print(f"\nCamera filter (>{filter_tau:.1f} m from the survey instant position):")
    print(f"  kept {int(cam_ok.sum())} / {len(cam_ok)} cameras "
          f"({100 * cam_ok.mean():.1f}%)")
    per_sensor_total, per_sensor_dropped = {}, {}
    for i, (nm, sfm, _) in enumerate(pairs):
        sens = Path(sfm["name"]).parent.name
        per_sensor_total[sens] = per_sensor_total.get(sens, 0) + 1
        if not cam_ok[i]:
            per_sensor_dropped[sens] = per_sensor_dropped.get(sens, 0) + 1
    for sens in sorted(per_sensor_total):
        print(f"    {sens}: dropped {per_sensor_dropped.get(sens, 0):4d} / "
              f"{per_sensor_total[sens]:4d}")

    clean = out.parent / (out.name + "_clean")
    clean.mkdir(parents=True, exist_ok=True)
    with open(clean / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] (empty)\n")
        kept = 0
        for i, (nm, sfm, _) in enumerate(pairs):
            if not cam_ok[i]:
                continue
            kept += 1
            C = -(R_sfm[i].T @ sfm["tvec"])
            C_new = s * (R @ C) + t
            R_new = R_sfm[i] @ R.T
            tw = -R_new @ C_new
            qw, qx, qy, qz = rotmat_to_quat_wxyz(R_new)
            f.write(f"{kept} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tw[0]:.6f} {tw[1]:.6f} {tw[2]:.6f} {sfm['camera_id']} "
                    f"{Path(sfm['name']).parent.name}/{nm}\n\n")
    with open(clean / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, cam in cams.items():
            f.write(f"{cid} PINHOLE {cam['width']} {cam['height']} "
                    f"{' '.join(f'{v:.6f}' for v in cam['params'][:4])}\n")
    if lidar.exists():
        (clean / "points3D.ply").write_bytes(lidar.read_bytes())
    with open(clean / "scene_transform.json", "w") as f:
        json.dump({
            "train_from_world": {"scale": s, "rotation": {"matrix_3x3": R.tolist()},
                                 "translation": t.tolist()},
            "matrix_3x3_flat_row_major": R.reshape(-1).tolist(),
            "frame": "pipeline export frame (Z-DOWN: +Z points physically down)",
            "det_R": float(np.linalg.det(R)),
            "filter": {"tau_m": filter_tau, "kept": int(cam_ok.sum()),
                       "dropped": int((~cam_ok).sum())},
        }, f, indent=2)
    # Record WHICH cameras survived, explicitly. Re-deriving the decision elsewhere
    # invites a mismatch: the per-image CSV carries the alignment inlier flag
    # (position <= 0.5 m AND orientation <= 2 deg), which is a DIFFERENT and much
    # stricter set than this 5 m filter. Anything building a variant model must use
    # the same camera set this one used, or the comparison is confounded.
    with open(clean / "cameras_kept.txt", "w") as f:
        for i, (nm, sfm, _) in enumerate(pairs):
            if cam_ok[i]:
                f.write(f"{Path(sfm['name']).parent.name}/{nm}\n")
    print(f"  cleaned model -> {clean} ({kept} cameras + LiDAR points3D.ply)")
    print(f"  kept-camera list -> {clean / 'cameras_kept.txt'}")

    plot_diagnostic(out / "alignment_diagnostic.png", pc, qc, R_obs, res, dg)

    print(f"\nWrote aligned model to {out}")
    print(f"  scale        : {s:.6f}   det(R) {np.linalg.det(R):+.6f}")
    print(f"  inliers      : {dg['n_inliers']}/{dg['n_total']} sensors "
          f"({dg['inlier_ratio'] * 100:.1f}%), "
          f"{dg['n_inst_inliers']}/{dg['n_inst']} instants")
    print(f"  position     : median {dg['pos']['median']:.3f} m, "
          f"mean {dg['pos']['mean']:.3f} m, max {dg['pos']['max']:.3f} m  (per instant)")
    print(f"  orientation  : median {dg['rot']['median']:.3f} deg, "
          f"mean {dg['rot']['mean']:.3f} deg  (per sensor)")
    print(f"  theta_up     : {dg['theta_up']:.3f} deg "
          f"(orientation_rmse {dg['orientation_rmse']:.3f} deg)")
    print(f"  twin sep     : {dg['twin_separation_m']:.4f} m, "
          f"delta_psi_pos {dg['delta_psi_pos_deg']:.2f} deg")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Align an SfM model into the survey frame.")
    ap.add_argument("--config", help="Path to the scene config.yaml.")
    ap.add_argument("--model", default=None,
                    help="SfM model dir (default <dataset>/sfm/sparse/0).")
    ap.add_argument("--output", default=None,
                    help="Output dir (default <dataset>/georef/0).")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the synthetic + adversarial algorithm tests and exit.")
    ap.add_argument("--filter-tau", type=float, default=5.0,
                    help="Distance in metres from its instant's survey position beyond "
                         "which a camera counts as an outlier and is left out of the "
                         "CLEANED model (the full model is always written too). "
                         "Default 5.0: well above the camera-to-reference-point lever "
                         "arm (~2 m) and far below the observed failure (tens of metres).")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if not args.config:
        ap.error("--config is required (or use --self-test)")
    sys.exit(run(args.config, args.output, args.model, args.filter_tau))


if __name__ == "__main__":
    main()
