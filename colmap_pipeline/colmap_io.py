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

One helper was NOT sound and is corrected here: the camera-model parameter-count
table was wrong for five model ids. Since CAMERAS_BIN is a flat record stream whose
only framing is that count, a wrong value desynchronises the whole file - camera 1
reads correctly (no drift yet) and every camera after it is garbage reinterpreted
from the wrong byte offset. It stayed dormant for as long as every model happened to
use PINHOLE (model 1, the one entry that was right); the first OPENCV model (model 4)
turned it into nonsense ids and float64 bit patterns printed as dimensions. See
`CAMERA_MODELS` and the end-of-file assertion that now makes such a desync loud.
"""

import struct
from pathlib import Path

import numpy as np

# COLMAP/Spirula camera model id -> (name, number of parameters).
# Order and counts follow COLMAP's `CameraModelId` enum. The count MUST match the
# binary layout exactly: a mismatch desyncs `read_cameras_bin` (see module docstring).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),        # f, cx, cy
    1: ("PINHOLE", 4),               # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),         # f, cx, cy, k
    3: ("RADIAL", 5),                # f, cx, cy, k1, k2
    4: ("OPENCV", 8),                # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE", 8),        # fx, fy, cx, cy, k1, k2, k3, k4
    6: ("FULL_OPENCV", 12),          # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    7: ("FOV", 5),                   # fx, fy, cx, cy, omega
    8: ("SIMPLE_RADIAL_FISHEYE", 4),  # f, cx, cy, k
    9: ("RADIAL_FISHEYE", 5),        # f, cx, cy, k1, k2
    10: ("THIN_PRISM_FISHEYE", 12),  # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
}


def read_cameras_bin(path):
    """
    Read CAMERAS_BIN file from COLMAP/Spirula.

    Raises if the file is not consumed exactly, which is the only cheap detector for a
    wrong parameter count (the failure is otherwise silent - you get a plausible-looking
    dict of nonsense cameras).
    """
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = struct.unpack('<iiQQ', f.read(24))
            if model_id not in CAMERA_MODELS:
                raise ValueError(
                    f"{path}: camera {cam_id} has unknown model id {model_id}; "
                    f"its parameter count is unknown, so the rest of the file cannot "
                    f"be parsed. Add it to CAMERA_MODELS."
                )
            model_name, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack(f'<{num_params}d', f.read(num_params * 8))
            cameras[cam_id] = {'model_id': model_id, 'model_name': model_name,
                               'width': width, 'height': height, 'params': params}
        consumed, size = f.tell(), Path(path).stat().st_size
    if consumed != size:
        raise ValueError(
            f"{path}: parsed {consumed} of {size} bytes ({len(cameras)} cameras). The "
            f"parameter counts do not match the file's layout, so every camera after "
            f"the first mismatch is garbage. Check CAMERA_MODELS."
        )
    return cameras


def write_cameras_txt(path, cameras):
    """
    Write a COLMAP cameras.txt preserving each camera's true model and ALL parameters.

    Do not be tempted to emit `PINHOLE` with the first four parameters: for an OPENCV
    camera that silently discards the distortion coefficients and relabels the model,
    which is both a wrong intrinsics block and a lie about it.
    """
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, cam in cameras.items():
            params = " ".join(f"{v:.6f}" for v in cam['params'])
            f.write(f"{cid} {cam['model_name']} {cam['width']} {cam['height']} "
                    f"{params}\n")


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
