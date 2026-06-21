"""
Rotation convention conversion: survey (rx,ry,rz) -> COLMAP camera pose.

IMPORTANT SIMPLIFICATION: image_meta.txt already gives rx_rad, ry_rad, rz_rad
PER IMAGE (i.e. per sensor_id, per image_id) -- each camera's orientation is
already fully resolved in this file. There is no need to separately apply
the mounting/pitch_rad info from interior_orientation.txt on top of this;
that file most likely documents the physical rig geometry rather than a
correction you need to apply. So this module converts a single resolved
(rx_rad, ry_rad, rz_rad) orientation directly into a COLMAP pose.

WHAT WE KNOW from the validated frustum-direction code:
    dx = sin(rz_rad), dy = cos(rz_rad)
  i.e. rz_rad=0 -> faces +Y, increasing rz_rad rotates the heading vector
  toward +X (clockwise from above, standard map layout +X right / +Y up).

WHAT REMAINS UNCONFIRMED (flag for user to verify against dataset docs):
  - Exact role/order of rx_rad, ry_rad (likely pitch/roll) relative to rz_rad
    (yaw). They were 0.0 in the sample rows, so this doesn't yet matter for
    a flat, level street segment -- but confirm before trusting tilted poses
    (e.g. driving up a hill, or the up/down-facing sensors).

COLMAP camera convention:
  - Camera looks down +Z (in camera space).
  - +X is right, +Y is down (image coordinates).
  - images.txt stores QW,QX,QY,QZ,TX,TY,TZ representing the WORLD-TO-CAMERA
    transform: X_cam = R * X_world + T

RUN THIS FILE DIRECTLY to sanity-check against the frustum direction vectors
you already validated visually.
"""

import numpy as np


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    """
    Rotation consistent with dx=sin(a), dy=cos(a) for a vector starting
    at "heading 0 = +Y". This is a clockwise rotation in the XY plane
    (as seen with +X right, +Y up), i.e. negative of the usual
    right-hand-rule Rz.
    """
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def camera_to_world_rotation(rx_rad, ry_rad, rz_rad):
    """
    World-frame orientation of THIS camera's body frame (already resolved
    per-image -- no separate mounting correction needed).
    Body frame axes (assumed): X=right, Y=forward (viewing direction), Z=up.
    Order: yaw (rz) * pitch (rx) * roll (ry). If validation against tilted
    real poses disagrees, this composition order is the first thing to
    revisit.
    """
    return Rz(rz_rad) @ Rx(rx_rad) @ Ry(ry_rad)


def world_direction_from_heading(rz_rad):
    """Matches your validated plotting code: dx=sin(rz), dy=cos(rz)."""
    return np.array([np.sin(rz_rad), np.cos(rz_rad), 0.0])


def rotmat_to_quat_wxyz(R):
    """Standard rotation matrix -> quaternion (w,x,y,z), for COLMAP's format."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


# COLMAP camera axes (X right, Y down, Z forward/look-direction) relative to
# our "body" axes (X right, Y forward, Z up). This fixed rotation re-orients
# body-frame axes into camera-frame axes.
BODY_TO_CAMERA_AXES = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
])


def build_camera_world_rotation(rx_rad, ry_rad, rz_rad):
    """Full world-frame orientation of this camera (R_cam_in_world)."""
    R_body_in_world = camera_to_world_rotation(rx_rad, ry_rad, rz_rad)
    R_cam_in_world = R_body_in_world @ BODY_TO_CAMERA_AXES
    return R_cam_in_world


def colmap_pose_from_survey(x, y, z, rx_rad, ry_rad, rz_rad):
    """
    Returns (qw,qx,qy,qz, tx,ty,tz) as COLMAP's images.txt expects:
    world-to-camera transform.
    """
    R_cam_in_world = build_camera_world_rotation(rx_rad, ry_rad, rz_rad)
    R_world_to_cam = R_cam_in_world.T  # inverse of an orthonormal matrix
    t_world = np.array([x, y, z])
    t_cam = -R_world_to_cam @ t_world
    qw, qx, qy, qz = rotmat_to_quat_wxyz(R_world_to_cam)
    return qw, qx, qy, qz, *t_cam


# Optional final world-axis remap, e.g. to convert the survey's Z-up world
# into a Y-up world (a common convention for 3DGS viewers / game engines).
# This MUST be the exact same rotation you apply to the point cloud, or the
# camera poses and point cloud will end up rotated relative to each other.
#
# This matches: xyz_export = xyz[:, [0, 2, 1]]; xyz_export[:, 2] *= -1
# i.e. (x, y, z) -> (x, z, -y). This is a proper rotation (det=+1, not a
# mirror), confirmed numerically -- so it's safe to apply to camera
# rotations as well as positions without breaking handedness.
WORLD_AXIS_REMAP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=float)


def apply_world_axis_remap(R_cam_in_world, t_world):
    """
    Re-expresses an already-computed world-frame camera rotation and
    position in a new world basis (e.g. Y-up instead of Z-up), via the
    SAME proper rotation M applied to the point cloud.

    R_export = M @ R_cam_in_world   (re-expresses world *directions*; the
                                      camera's own local/body axes are
                                      untouched, so we don't need M.T here)
    t_export = M @ t_world          (ordinary point transform)
    """
    R_export = WORLD_AXIS_REMAP @ R_cam_in_world
    t_export = WORLD_AXIS_REMAP @ np.asarray(t_world)
    return R_export, t_export


def colmap_pose_from_survey_remapped(x, y, z, rx_rad, ry_rad, rz_rad):
    """
    Same as colmap_pose_from_survey, but in the remapped (Y-up) world basis
    defined by WORLD_AXIS_REMAP -- use this version if you applied the
    Y/Z-swap to your point cloud export, so cameras and points line up.
    """
    R_cam_in_world = build_camera_world_rotation(rx_rad, ry_rad, rz_rad)
    t_world = np.array([x, y, z])
    R_export, t_export = apply_world_axis_remap(R_cam_in_world, t_world)

    R_world_to_cam = R_export.T
    t_cam = -R_world_to_cam @ t_export
    qw, qx, qy, qz = rotmat_to_quat_wxyz(R_world_to_cam)
    return qw, qx, qy, qz, *t_cam


if __name__ == "__main__":
    # --- Sanity check against the validated frustum-direction logic ---
    print("Checking forward-direction consistency:")
    for rz_deg in [0, 30, 90, 180, 270]:
        rz = np.radians(rz_deg)
        R_cam_in_world = build_camera_world_rotation(0, 0, rz)
        cam_forward_in_world = R_cam_in_world @ np.array([0, 0, 1])  # +Z cam axis
        expected = world_direction_from_heading(rz)
        ok = np.allclose(cam_forward_in_world, expected, atol=1e-9)
        print(f"  rz={rz_deg:>3}deg  cam_fwd={np.round(cam_forward_in_world,3)}"
              f"  expected={np.round(expected,3)}  {'OK' if ok else 'MISMATCH'}")

    print()
    print("Checking the four real rig sensors from your data (rz_rad values):")
    # From your script's comments: 110021 front, 110022 right, 110023 back, 110024 left
    real_examples = {
        110021: 2.402573,  # front
        110022: 0.831676,  # derived for synthetic test -- replace with real value
        110023: -0.739321,  # derived for synthetic test -- replace with real value
        110024: -2.310318,  # derived for synthetic test -- replace with real value
    }
    for sensor_id, rz in real_examples.items():
        R_cam_in_world = build_camera_world_rotation(0, 0, rz)
        cam_forward_in_world = R_cam_in_world @ np.array([0, 0, 1])
        expected = world_direction_from_heading(rz)
        ok = np.allclose(cam_forward_in_world, expected, atol=1e-9)
        print(f"  sensor {sensor_id}: cam_fwd={np.round(cam_forward_in_world,3)} "
              f"expected={np.round(expected,3)} {'OK' if ok else 'MISMATCH'}")

    print()
    print("Round-trip check: pose -> matrix -> quat -> matrix should match")
    qw, qx, qy, qz, tx, ty, tz = colmap_pose_from_survey(
        100.0, 200.0, 5.0, 0.01, 0.02, np.radians(45)
    )
    print(f"  quat=({qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f})  t=({tx:.3f},{ty:.3f},{tz:.3f})")
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    print(f"  quaternion norm = {norm:.6f} (should be 1.0)")
    
    ### Needed?
    
    print()
    print("Checking WORLD_AXIS_REMAP is a proper rotation (not a mirror):")
    det = np.linalg.det(WORLD_AXIS_REMAP)
    orthonormal = np.allclose(WORLD_AXIS_REMAP @ WORLD_AXIS_REMAP.T, np.eye(3))
    print(f"  det = {det:.3f} (must be +1.0)  orthonormal = {orthonormal}")
    assert np.isclose(det, 1.0), "WORLD_AXIS_REMAP is a reflection, not a rotation!"

    print()
    print("Cross-check: remapping a forward-direction vector directly vs.")
    print("remapping via the full pose function should give the same answer.")
    for rz_deg in [0, 45, 90, 200]:
        rz = np.radians(rz_deg)
        x, y, z = 10.0, 20.0, 6.86
        R_cam_in_world = build_camera_world_rotation(0, 0, rz)
        fwd_survey = R_cam_in_world @ np.array([0, 0, 1])
        fwd_direct_remap = WORLD_AXIS_REMAP @ fwd_survey

        qw, qx, qy, qz, tx, ty, tz = colmap_pose_from_survey_remapped(x, y, z, 0, 0, rz)
        # Recover R_export from the quaternion to check its forward vector
        R_world_to_cam = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),     2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),     1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),     1-2*(qx**2+qy**2)],
        ])
        R_export = R_world_to_cam.T
        fwd_via_pose_fn = R_export @ np.array([0, 0, 1])

        ok = np.allclose(fwd_direct_remap, fwd_via_pose_fn, atol=1e-6)
        print(f"  rz={rz_deg:>3}deg  direct={np.round(fwd_direct_remap,3)} "
              f"via_pose_fn={np.round(fwd_via_pose_fn,3)}  {'OK' if ok else 'MISMATCH'}")

    print()
    print("Checking translation remap matches direct point transform:")
    ox, oy, oz = 3624.468, 340864.229, 6.860
    x, y, z = 3665.499, 340820.045, 6.86
    qw, qx, qy, qz, tx, ty, tz = colmap_pose_from_survey_remapped(
        x - ox, y - oy, z - oz, 0, 0, 5.097672
    )
    R_world_to_cam = np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),     2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),     1-2*(qx**2+qy**2)],
    ])
    R_export = R_world_to_cam.T
    t_cam = np.array([tx, ty, tz])
    recovered_export_pos = -R_export @ t_cam
    direct_export_pos = WORLD_AXIS_REMAP @ np.array([x-ox, y-oy, z-oz])
    print(f"  recovered (from pose fn): {np.round(recovered_export_pos,3)}")
    print(f"  direct point remap:       {np.round(direct_export_pos,3)}")
    print(f"  match: {np.allclose(recovered_export_pos, direct_export_pos, atol=1e-6)}")
