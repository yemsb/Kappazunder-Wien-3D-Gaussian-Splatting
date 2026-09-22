"""
COLMAP model readers and quaternion helpers, shared by the pipeline scripts.

Extracted from `ingest_spirula_model.py`, which was deleted. That script's stated job
was to rectify Spirula's poses into the georeferenced frame; it never did — no SVD, no
alignment, it re-emitted the poses verbatim and then reported residuals quantifying an
alignment that had not happened. It also wrote to the same `sparse_refined/0` as the
COLMAP refiner and silently overwrote a good result.

Its I/O helpers were sound, though, and three scripts depend on them, so they live here
instead. The alignment that script claimed to perform is now done properly in
`align_sfm_to_survey.py`.
"""

import struct

import numpy as np


def read_cameras_bin(path):
    """Read CAMERAS_BIN file from COLMAP/Spirula."""
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = struct.unpack('<iiQQ', f.read(24))
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 8, 6: 12, 7: 10,
                          8: 10, 9: 12, 10: 4}.get(model_id, 4)
            params = struct.unpack(f'<{num_params}d', f.read(num_params * 8))
            cameras[cam_id] = {'model_id': model_id, 'width': width,
                               'height': height, 'params': params}
    return cameras


def read_images_bin(path):
    """
    Read IMAGES_BIN file from COLMAP/Spirula.

    Skips the 2D observations. Fine for diagnostics; use a reader that keeps them if
    the tracks matter (see `filter_sfm_model.read_images_bin_full`).
    """
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))  # qw, qx, qy, qz
            tvec = struct.unpack('<3d', f.read(24))  # tx, ty, tz
            camera_id = struct.unpack('<I', f.read(4))[0]
            name_bytes = []
            while True:
                b = f.read(1)
                if b == b'\x00':
                    break
                name_bytes.append(b)
            name = b''.join(name_bytes).decode('utf-8', errors='ignore')
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.seek(num_points2D * 24, 1)  # (x, y, point3D_id) = double, double, uint64
            images[image_id] = {
                'image_id': image_id,
                'name': name,
                'qvec': np.array(qvec, dtype=float),
                'tvec': np.array(tvec, dtype=float),
                'camera_id': camera_id,
            }
    return images


def quat_to_rotmat(q):
    """Convert quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.dot(q, q)
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=float)


def rotmat_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to a quaternion [qw, qx, qy, qz]."""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=float)
