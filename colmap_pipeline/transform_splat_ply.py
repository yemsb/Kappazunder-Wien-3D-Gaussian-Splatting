#!/usr/bin/env python3
"""
Apply a similarity transform to a 3DGS splat PLY, rewriting it into a new frame.

Why the attributes all move, not just the positions. A Gaussian's stored parameters are
"means in world", "rotation as a world quaternion", "log scales" and "SH coefficients for
view-dependent colour". Under a similarity `p -> s R p + t`:

    means   : s*R*p + t
    rot     : q_R (x) q            (the Gaussian's orientation is composed with the new one)
    scales  : scale + log(s)       (stored in log space, so a uniform scale is a constant shift;
                                    exact -- this is why a similarity is the only scale this can do)
    f_dc    : unchanged            (degree 0 is isotropic)
    f_rest  : rotated per degree   (a vector rotates with the object, so the view-dependent
                                    part must rotate with it)

The SH rotation is the part that is easy to get silently wrong: skipping it leaves the
geometry correct and only the view-dependent shading wrong, which reads as "the render looks
a bit off" rather than as a bug. It is therefore checked end-to-end below -- for random view
directions, the reconstructed colour must be identical before and after.

Conventions are taken from the tool that wrote the file (spirula-studio source), not assumed:
  * `SplatPly.h`      -- raw, un-activated: log scales, logit opacity, f_rest channel-major
                         (red 0..K-1, then green, then blue)
  * `meshing.slang:427` -- quats[0]=w, so rot_0..3 is (w, x, y, z)
  * `sh_quant.slang`  -- the degree 0..3 basis and its coefficient order, reproduced verbatim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

C0 = 0.28209479177387814


def sh_basis(v: np.ndarray) -> np.ndarray:
    """Degree 1..3 real SH basis, in the f_rest ordering (indices 0..14). v: (N,3) unit vectors."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    z2 = z * z
    out = np.empty((len(v), 15), dtype=np.float64)
    # l = 1
    out[:, 0] = 0.48860251190292 * (-y)
    out[:, 1] = 0.48860251190292 * (z)
    out[:, 2] = 0.48860251190292 * (-x)
    # l = 2
    out[:, 3] = 0.5462742152960395 * (2.0 * x * y)
    out[:, 4] = -1.092548430592079 * z * y
    out[:, 5] = 0.9461746957575601 * z2 - 0.3153915652525201
    out[:, 6] = -1.092548430592079 * z * x
    out[:, 7] = 0.5462742152960395 * (x * x - y * y)
    # l = 3
    out[:, 8] = -0.5900435899266435 * (3.0 * x * x * y - y * y * y)
    out[:, 9] = 1.445305721320277 * z * (2.0 * x * y)
    out[:, 10] = (-2.285228997322329 * z2 + 0.4570457994644658) * y
    out[:, 11] = z * (1.865881662950577 * z2 - 1.119528997770346)
    out[:, 12] = (-2.285228997322329 * z2 + 0.4570457994644658) * x
    out[:, 13] = 1.445305721320277 * z * (x * x - y * y)
    out[:, 14] = -0.5900435899266435 * (x * x * x - 3.0 * x * y * y)
    return out


SH_SLICES = {1: slice(0, 3), 2: slice(3, 8), 3: slice(8, 15)}


def sh_rotation_matrices(R: np.ndarray, n: int = 20000, seed: int = 0) -> dict:
    """Per-degree matrices M_l with Y_l(R^T v) = M_l Y_l(v), fitted on sampled directions.

    Fitting rather than a closed form: the basis above is transcribed from the renderer, so
    fitting against that same basis is what guarantees the two agree.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    B = sh_basis(v @ R)                     # R applied to v is R v; (R^T)^T ... see caller
    A = sh_basis(v)
    M = {}
    for l, sl in SH_SLICES.items():
        M[l], *_ = np.linalg.lstsq(A[:, sl], B[:, sl], rcond=None)
    return M


def check_sh(M, R, tol=1e-6):
    """The invariant that matters: colour must be preserved for every view direction.

    With M fitted as `sum_j Y_j(v) M_jk = Y_k(R^T v)`, M IS the coefficient operator, so the
    transformed coefficients are `c' = M c` and the invariant to test is

        sum_k (M c)_k Y_k(v)  ==  sum_j c_j Y_j(R^T v)      for every v

    i.e. A @ (M c) == B @ c. Testing it the other way round asks whether M is its own inverse
    rotation, which it is not for any rotation except 0 or 180 degrees -- that mistake is what
    the first version of this check did.
    """
    rng = np.random.default_rng(1)
    v = rng.normal(size=(5000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    A, B = sh_basis(v), sh_basis(v @ R)      # A = Y(v),  B = Y(R^T v)
    worst_orth, worst_col = 0.0, 0.0
    for l, sl in SH_SLICES.items():
        worst_orth = max(worst_orth, np.abs(M[l] @ M[l].T - np.eye(sl.stop - sl.start)).max())
        c = rng.normal(size=sl.stop - sl.start)
        worst_col = max(worst_col, np.abs(A[:, sl] @ (M[l] @ c) - B[:, sl] @ c).max())
    return worst_orth, worst_col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--transform", required=True, type=Path,
                    help=".npz with s, R, t (workspace: p -> s R p + t)")
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    z = np.load(args.transform)
    s, R, t = float(z["s"]), np.asarray(z["R"], dtype=np.float64), np.asarray(z["t"], dtype=np.float64)
    print(f"transform: scale {s:.6f}  det(R) {np.linalg.det(R):+.6f}  |t| {np.linalg.norm(t):.3f}")

    M = sh_rotation_matrices(R)
    orth, col = check_sh(M, R)
    print(f"SH rotation check: |M M^T - I| max {orth:.2e}   colour preserved to {col:.2e}")
    if orth > 1e-4 or col > 1e-4:
        sys.exit("SH rotation failed its invariant check - refusing to write a wrong file")

    # quaternion of R, w-first
    tr = np.trace(R)
    qR = np.empty(4)
    qR[0] = np.sqrt(max(tr + 1.0, 0.0)) / 2.0
    qR[1] = (R[2, 1] - R[1, 2]) / (4.0 * qR[0])
    qR[2] = (R[0, 2] - R[2, 0]) / (4.0 * qR[0])
    qR[3] = (R[1, 0] - R[0, 1]) / (4.0 * qR[0])
    qR /= np.linalg.norm(qR)
    print(f"Q_R = {np.round(qR, 6)}  (angle {np.degrees(2*np.arccos(np.clip(abs(qR[0]),-1,1))):.3f} deg)")

    with open(args.input, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        props = [h.split()[-1] for h in header if h.startswith("property")]
        n_vert = int([h for h in header if h.startswith("element vertex")][0].split()[-1])
        ix = {n: i for i, n in enumerate(props)}
        stride = len(props)
        need = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2"]
        missing = [n for n in need if n not in ix]
        if missing:
            sys.exit(f"input is missing {missing}")
        n_rest = sum(1 for p in props if p.startswith("f_rest_"))
        if n_rest and n_rest % 3:
            sys.exit(f"f_rest count {n_rest} is not a multiple of 3")
        K = n_rest // 3
        print(f"{n_vert} gaussians, {stride} props, {K} SH coeffs per channel")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as g:
            g.write(("\n".join(header) + "\n").encode("ascii"))

            cols = np.array([ix["x"], ix["y"], ix["z"]])
            cq = np.array([ix["rot_0"], ix["rot_1"], ix["rot_2"], ix["rot_3"]])
            cs = np.array([ix["scale_0"], ix["scale_1"], ix["scale_2"]])
            cre = np.array([ix[f"f_rest_{i}"] for i in range(n_rest)]) if n_rest else None
            cdc = np.array([ix[f"f_dc_{i}"] for i in range(3)])
            cn = np.array([ix[f"n{i}"] for i in ("x", "y", "z")]) if "nx" in ix else None

            done = 0
            while done < n_vert:
                m = min(args.chunk, n_vert - done)
                buf = np.frombuffer(f.read(m * stride * 4), dtype="<f4").reshape(m, stride).copy()

                buf[:, cols] = (s * (buf[:, cols].astype(np.float64) @ R.T) + t).astype(np.float32)

                q = buf[:, cq].astype(np.float64)                     # (m,4) w-first
                w = (qR[0] * q[:, 0] - qR[1] * q[:, 1] - qR[2] * q[:, 2] - qR[3] * q[:, 3])
                x = (qR[0] * q[:, 1] + qR[1] * q[:, 0] + qR[2] * q[:, 3] - qR[3] * q[:, 2])
                y = (qR[0] * q[:, 2] - qR[1] * q[:, 3] + qR[2] * q[:, 0] + qR[3] * q[:, 1])
                zz = (qR[0] * q[:, 3] + qR[1] * q[:, 2] - qR[2] * q[:, 1] + qR[3] * q[:, 0])
                nq = np.stack([w, x, y, zz], axis=1)
                buf[:, cq] = (nq / np.linalg.norm(nq, axis=1, keepdims=True)).astype(np.float32)

                buf[:, cs] += np.float32(np.log(s))

                if cn is not None:
                    buf[:, cn] = (buf[:, cn].astype(np.float64) @ R.T).astype(np.float32)

                if cre is not None:
                    # file is channel-major: [R coeffs][G coeffs][B coeffs], each K coefficients
                    for ch in range(3):
                        blk = cre[ch * K:(ch + 1) * K]
                        v = buf[:, blk].astype(np.float64)
                        out = v.copy()
                        for l, sl in SH_SLICES.items():
                            if sl.stop <= K:
                                out[:, sl] = v[:, sl] @ M[l].T
                        buf[:, blk] = out.astype(np.float32)

                g.write(buf.tobytes())
                done += m
                if done % (args.chunk * 5) == 0:
                    print(f"  {done}/{n_vert}", flush=True)

    print(f"wrote {args.output}  ({args.output.stat().st_size/1e9:.2f} GB)")
    meta = args.output.with_suffix(".transform.json")
    meta.write_text(json.dumps({
        "source": str(args.input), "scale": s, "rotation_row_major": R.tolist(),
        "translation": t.tolist(), "quaternion_wxyz": qR.tolist(),
        "sh_rotation_check_orthogonality": orth, "sh_rotation_check_colour": col,
    }, indent=1))
    print("wrote", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
