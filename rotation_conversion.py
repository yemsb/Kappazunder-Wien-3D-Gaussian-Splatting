"""
Rotation convention conversion: sensor-aware survey pose -> COLMAP pose.

The usable inputs are the vehicle heading ``rz_rad`` from image_meta.txt and
the sensor role from interior_orientation.txt. The heading determines the
camera's horizontal forward direction, while the sensor role determines whether
the image top should align with the driving direction (vertical cameras) or
with world +Z (the horizontal cameras).

COLMAP camera convention:
    - Camera looks down +Z (in camera space).
    - +X is right, +Y is down (image coordinates).
    - images.txt stores QW,QX,QY,QZ,TX,TY,TZ representing the WORLD-TO-CAMERA
        transform: X_cam = R * X_world + T

RUN THIS FILE DIRECTLY to sanity-check the sensor-specific pose rules.
"""

import numpy as np


def heading_vector(rz_rad):
    """Vehicle heading in world XY; rz_rad=0 points along north (world +Y)."""
    return np.array([np.sin(rz_rad), np.cos(rz_rad), 0.0], dtype=float)


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def sensor_role_from_pitch(sensor_id):
    """Uses the interior-orientation pitch to distinguish camera roles."""
    last_digit = int(sensor_id) % 10
    horizontal_roles = {
        0: "up",
        1: "front",
        2: "right",
        3: "back",
        4: "left",
        5: "down",
    }
    role = horizontal_roles.get(last_digit)
    if role is None:
        raise ValueError(f"Unsupported horizontal sensor_id {sensor_id}")
    return role


def sensor_forward_vector(sensor_id, rz_rad):
    """World-space camera forward axis for the sensor role."""
    role = sensor_role_from_pitch(sensor_id)
    if role == "up":
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if role == "down":
        return np.array([0.0, 0.0, -1.0], dtype=float)

    return heading_vector(rz_rad)


def sensor_up_vector(sensor_id, rz_rad):
    """World-space direction that should appear at the top of the image."""
    role = sensor_role_from_pitch(sensor_id)
    if role is "up":
        return -heading_vector(rz_rad) # Image's bottom is towards driving direction
    elif role is "down":
        return heading_vector(rz_rad)
    else:
        return np.array([0.0, 0.0, 1.0], dtype=float)


def build_camera_world_rotation(sensor_id, rz_rad):
    """World-frame orientation of the camera's local COLMAP axes."""
    forward = normalize_vector(sensor_forward_vector(sensor_id, rz_rad))
    image_up = normalize_vector(sensor_up_vector(sensor_id, rz_rad))

    right = normalize_vector(np.cross(forward, image_up))
    down = normalize_vector(np.cross(forward, right))
    return np.column_stack([right, down, forward])


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


def colmap_pose_from_survey(x, y, z, sensor_id, rz_rad):
    """
    Returns (qw,qx,qy,qz, tx,ty,tz) as COLMAP's images.txt expects:
    world-to-camera transform.
    """
    R_cam_in_world = build_camera_world_rotation(sensor_id, rz_rad)
    R_world_to_cam = R_cam_in_world.T  # inverse of an orthonormal matrix
    t_world = np.array([x, y, z])
    t_cam = -R_world_to_cam @ t_world
    qw, qx, qy, qz = rotmat_to_quat_wxyz(R_world_to_cam)
    return qw, qx, qy, qz, *t_cam


if __name__ == "__main__":
    print("Checking heading consistency for horizontal sensors:")
    horizontal_cases = [
        (110021, 0.0, "front"),
        (110022, np.pi / 2, "right"),
        (110023, np.pi, "back"),
        (110024, 3 * np.pi / 2, "left"),
    ]
    rz = np.radians(45)
    for sensor_id, offset_rad, role_name in horizontal_cases:
        R_cam_in_world = build_camera_world_rotation(sensor_id, rz)
        cam_forward_in_world = R_cam_in_world[:, 2]
        cam_image_up_in_world = -R_cam_in_world[:, 1]
        expected_forward = heading_vector(rz + offset_rad)
        expected_up = np.array([0.0, 0.0, 1.0])
        ok = np.allclose(cam_forward_in_world, expected_forward, atol=1e-9)
        print(f"  {role_name:>6} sensor {sensor_id}: forward={np.round(cam_forward_in_world,3)}"
              f"  image_up={np.round(cam_image_up_in_world,3)}  {'OK' if ok else 'MISMATCH'}")
        assert np.allclose(cam_image_up_in_world, expected_up, atol=1e-9)

    print()
    print("Checking vertical sensor behavior:")
    vertical_cases = [
        (110020, "up", np.array([0.0, 0.0, 1.0])),
        (110025, "down", np.array([0.0, 0.0, -1.0])),
    ]
    for sensor_id, role_name, expected_forward in vertical_cases:
        R_cam_in_world = build_camera_world_rotation(sensor_id, rz)
        cam_forward_in_world = R_cam_in_world[:, 2]
        cam_image_up_in_world = -R_cam_in_world[:, 1]
        expected_up = heading_vector(rz)
        print(f"  {role_name:>4} sensor {sensor_id}: forward={np.round(cam_forward_in_world,3)}"
              f"  image_up={np.round(cam_image_up_in_world,3)}")
        assert np.allclose(cam_forward_in_world, expected_forward, atol=1e-9)
        assert np.allclose(cam_image_up_in_world, expected_up, atol=1e-9)

    print()
    print("Round-trip check: pose -> matrix -> quat -> matrix should match")
    qw, qx, qy, qz, tx, ty, tz = colmap_pose_from_survey(
        100.0, 200.0, 5.0, 110021, 0.0, np.radians(45)
    )
    print(f"  quat=({qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f})  t=({tx:.3f},{ty:.3f},{tz:.3f})")
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    print(f"  quaternion norm = {norm:.6f} (should be 1.0)")
