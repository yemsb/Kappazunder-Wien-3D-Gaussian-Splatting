#!/usr/bin/env python3
"""
Clean-GS - Whitelist-based 3D Gaussian Splatting pruning

Two-stage approach with optional outlier removal:
1. Whitelist: Keep only Gaussians that project to object regions
2. Color validation: Remove floaters using depth buffering and color matching
3. Outlier removal (optional): Neighbor-based, spatial, or multi-view filtering

Vendored from https://github.com/smlab-niser/clean-gs at commit ba5c2fa, MIT
licensed (see LICENSE alongside this file). The modifications listed below are
ours, not upstream; diffing against that commit gives the exact change set.
This copy is pinned deliberately: the pipeline's recipe depends on behaviour
upstream does not have, so a user who installs upstream instead gets a silently
different result rather than an error.

Local modifications (2026-10-11), all verified to return exactly the original
result on synthetic scenes and the real splat:
  * both per-view passes projected Gaussians one 3-vector at a time in Python,
    ~10^10 interpreter iterations for a 6M-Gaussian scene and a few thousand
    views; now vectorised over the whole splat
  * the 7130^2 masks were converted to float32 RGB whole (a 610 MB allocation per
    view) just to test a threshold and read a few hundred thousand pixels; now the
    mask stays uint8 and only the survivor pixels' colour is converted
  * the splat was re-pickled into the task queue with every chunk; now handed to
    each worker once at pool start (this was waste, not the bottleneck)
  * k-NN used ball_tree, 39x slower than kd_tree for identical output
  * --photos: the colour stage compares against the photograph rather than the
    mask, without which a binary mask makes it a test for whiteness
  * --whitelist all: the original rule (keep if any view sees object) saturates at
    this view density and removes nothing, so the verdict can be inverted to
    removal-if-any-view-sees-sky, with --min_blacklist setting how many views must
    agree
  * --no_color: skips stage 2 for scenes captured across changing light
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
import json
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
except ImportError:  # pip install tqdm to get progress bars
    def tqdm(iterable, **_):
        return iterable

# Populated once per worker by _init_worker, so the multi-hundred-MB arrays travel
# through the task queue once per process rather than once per chunk.
_SHARED = {}


def _init_worker(xyz, rgb_colors):
    _SHARED['xyz'] = xyz
    _SHARED['rgb'] = rgb_colors

@dataclass
class Camera:
    image_name: str
    width: int
    height: int
    R: np.ndarray
    T: np.ndarray
    K: np.ndarray

def load_cameras(cameras_path):
    with open(cameras_path) as f:
        cameras_data = json.load(f)
    cameras = []
    for cam_data in cameras_data:
        cameras.append(Camera(
            image_name=cam_data['image_name'],
            width=cam_data['width'],
            height=cam_data['height'],
            R=np.array(cam_data['R']),
            T=np.array(cam_data['T']),
            K=np.array(cam_data['K'])
        ))
    return cameras

def project_gaussian_with_depth(xyz, camera):
    """Project 3D point to 2D image coordinates with depth"""
    R_w2c = camera.R.T
    T_w2c = -R_w2c @ camera.T
    xyz_cam = R_w2c @ xyz + T_w2c
    depth = xyz_cam[2]
    if depth <= 0:
        return None
    uv_h = camera.K @ xyz_cam
    u = uv_h[0] / uv_h[2]
    v = uv_h[1] / uv_h[2]
    if 0 <= u < camera.width and 0 <= v < camera.height:
        return int(u), int(v), depth
    return None

def project_all_gaussians(xyz, camera):
    """The batch form of project_gaussian_with_depth, and required to agree with it:
    returns (u, v, depth, ok) as arrays, `ok` marking the ones that are in front of
    the camera and inside the frame. u/v are only meaningful where ok is set."""
    R_w2c = camera.R.T
    T_w2c = -R_w2c @ camera.T
    xyz_cam = xyz @ R_w2c.T + T_w2c          # (R @ x).T == x.T @ R.T
    depth = xyz_cam[:, 2]
    uv_h = xyz_cam @ camera.K.T              # (K @ x).T == x.T @ K.T
    z = uv_h[:, 2]
    ok = (depth > 0) & (z != 0)
    u = np.divide(uv_h[:, 0], z, out=np.zeros_like(z), where=ok)
    v = np.divide(uv_h[:, 1], z, out=np.zeros_like(z), where=ok)
    ok &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return u, v, depth, ok

def find_mask_path(cam, masked_images_dir):
    """Find mask file matching the image base name (any extension)"""
    base_name = Path(cam.image_name).stem
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        candidate = os.path.join(masked_images_dir, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

# `masked_rgb > 0.05` in uint8: 13/255 = 0.051 is over, 12/255 = 0.047 is under.
OBJECT_THRESHOLD = 13

def load_masked_image(cam, masked_images_dir):
    """Read a camera's masked image as uint8 BGR, resized to the camera, or None.

    Stays uint8: the old path converted the whole frame to float32 RGB, which on a
    7130^2 mask is a 610 MB allocation for a comparison against 0.05.
    """
    masked_path = find_mask_path(cam, masked_images_dir)
    if masked_path is None:
        return None
    img = cv2.imread(masked_path)
    if img is None:
        return None
    if img.shape[0] != cam.height or img.shape[1] != cam.width:
        img = cv2.resize(img, (cam.width, cam.height))
    return img

def object_mask_of(img):
    """Which pixels are object: any channel over the threshold.

    Per channel, not np.any(img >= 13, axis=2), which is 8x slower here (420 ms vs
    50 ms on a 7130^2 frame): reducing over the last axis strides across channels
    and materialises a 150 MB boolean on the way. Slices are contiguous. Exactly
    the same predicate either way.
    """
    return ((img[:, :, 0] >= OBJECT_THRESHOLD) | (img[:, :, 1] >= OBJECT_THRESHOLD)
            | (img[:, :, 2] >= OBJECT_THRESHOLD))

def sh_to_rgb(sh_features):
    """Convert SH coefficients to RGB"""
    C0 = 0.28209479177387814
    rgb = sh_features[:3] * C0 + 0.5
    return np.clip(rgb, 0, 1)

def color_distance(c1, c2):
    """Euclidean distance between two RGB colors"""
    return np.sqrt(np.sum((c1 - c2) ** 2))

def process_view_whitelist(args):
    """Classify each Gaussian against one view's mask.

    Returns int8: 2 = projects into an object region, 1 = projects in frame but on
    a non-object pixel, 0 = not in this view at all. Both whitelist rules are read
    off this one array -- the union rule wants `> 0` / `== 2`, the consensus rule
    wants `== 1` -- so they cost one pass between them.
    """
    view_idx, cam, masked_images_dir = args
    xyz = _SHARED['xyz']
    img = load_masked_image(cam, masked_images_dir)
    if img is None:
        return None
    object_mask = object_mask_of(img)
    u, v, depth, ok = project_all_gaussians(xyz, cam)
    state = np.zeros(len(xyz), dtype=np.int8)
    hit = np.flatnonzero(ok)
    if len(hit):
        bright = object_mask[v[hit].astype(np.int64), u[hit].astype(np.int64)]
        state[hit] = 1 + bright          # False -> 1 (over non-object), True -> 2
    return state

def process_view_color_validate(args):
    """Render with depth buffering and validate colors for whitelisted Gaussians"""
    view_idx, cam, masked_images_dir, whitelist_mask, color_threshold, photos_dir = args
    xyz = _SHARED['xyz']
    rgb_colors = _SHARED['rgb']
    whitelisted_indices = np.where(whitelist_mask)[0]
    if len(whitelisted_indices) == 0:
        return None
    img = load_masked_image(cam, masked_images_dir)
    if img is None:
        return None
    object_mask = object_mask_of(img)
    # With --photos, the whitelist comes from the mask but the colour to validate
    # against comes from the photograph. Survivors are object pixels by
    # construction, where a masked photograph equals the plain one, so this is the
    # same comparison the original makes when fed masked photographs -- and it is
    # the only thing that works when the masks are binary, since then every object
    # pixel is white and the test collapses to "is this Gaussian white?".
    colour_src = img if photos_dir is None else load_masked_image(cam, photos_dir)
    if colour_src is None:
        return None

    u, v, depth, ok = project_all_gaussians(xyz[whitelisted_indices], cam)
    sel = np.flatnonzero(ok)
    if len(sel):
        sel = sel[object_mask[v[sel].astype(np.int64), u[sel].astype(np.int64)]]

    num_gaussians = len(xyz)
    is_rendered = np.zeros(num_gaussians, dtype=np.bool_)
    color_match = np.zeros(num_gaussians, dtype=np.bool_)
    if len(sel):
        # Nearest whitelisted Gaussian per masked pixel. Sorting by (pixel, depth)
        # makes each pixel's candidates contiguous with the nearest first, so the
        # survivor is the first entry of each run -- the depth buffer and its
        # pixmap in one pass, instead of a dict keyed by the (u, v) it won.
        # lexsort is stable, so an exact depth tie resolves to the lowest Gaussian
        # index, which is what the loop's strict `<` did. Verified against those
        # loops on synthetic scenes, duplicates included: identical.
        ui = u[sel].astype(np.int64)
        vi = v[sel].astype(np.int64)
        flat = vi * cam.width + ui
        order = np.lexsort((depth[sel], flat))
        flat_sorted = flat[order]
        first = np.empty(flat_sorted.shape, dtype=bool)
        first[0] = True
        np.not_equal(flat_sorted[1:], flat_sorted[:-1], out=first[1:])
        survivor = order[first]
        g_idx = whitelisted_indices[sel[survivor]]
        is_rendered[g_idx] = True
        # Expected colour at those pixels only: a handful of hundred thousand
        # lookups, not a 610 MB conversion of the whole frame. `flat` is the
        # raveled pixel index, and the channel flip is cvtColor BGR2RGB.
        expected = colour_src.reshape(-1, 3)[flat[survivor]][:, ::-1].astype(np.float32) / 255.0
        dist = np.linalg.norm(rgb_colors[g_idx] - expected, axis=1)
        color_match[g_idx[dist < color_threshold]] = True
    return (is_rendered, color_match)

def clean_gs(ply_path, cameras_path, masked_images_dir, output_path,
             outlier_mode='neighbor',
             color_threshold=0.40,
             min_views=1,
             spatial_percentile=99,
             neighbor_percentile=95,
             num_workers=None,
             photos_dir=None,
             use_color=True,
             whitelist_mode='any',
             min_blacklist=1):
    """
    Clean-GS with outlier removal options

    photos_dir: optional directory of the unmasked photographs, named like the
      masks. The whitelist is still read from masked_images_dir; only the colour
      comparison uses these. Required in practice whenever the masks are binary,
      i.e. always: see the comment in process_view_color_validate.

    use_color: set False to skip stage 2 entirely. Worth doing when a scene is
      captured across changing light: a Gaussian trained on a surface seen under
      three different suns has one colour that matches no single photograph, so the
      colour test flags it as a floater. The whitelist then passes through and only
      stage 3 filters.

    whitelist_mode: how the per-view verdicts combine.
      'any'  -- the original: kept if ANY view sees it in an object region. With
                thousands of views covering every direction this union saturates and
                stage 1 stops discriminating, because one all-bright mask whitelists
                its whole frustum regardless of what is in front of what.
      'all'  -- the consensus rule: removed if ANY view sees a non-object pixel
                where it projects. There is no depth test in either rule, so
                'projected onto sky' means no solid surface lay along that ray: sky
                floaters go, and so does anything the mask did not classify as scene,
                such as thin wires, poles and a shell off silhouette edges.

    outlier_mode:
      - 'none': Basic cleaning (no extra outlier removal)
      - 'spatial': Remove far Gaussians (beyond spatial_percentile)
      - 'neighbor': Remove isolated Gaussians (beyond neighbor_percentile, default)
      - 'multiview': Require min_views appearances
      - 'combined': All three methods
    """
    if num_workers is None:
        num_workers = cpu_count()

    mode_name = {
        'none': 'Basic (No outlier removal)',
        'spatial': 'Spatial (Remove far Gaussians)',
        'neighbor': 'Neighbor (Remove isolated)',
        'multiview': 'MultiView (Require multi-view)',
        'combined': 'Combined (All methods)'
    }[outlier_mode]

    print("=" * 70)
    print(f"CLEAN-GS: {mode_name}")
    print("=" * 70)
    print(f"Workers: {num_workers} cores")
    print(f"Color validation: " + (f"threshold {color_threshold:.2f}" if use_color
                                  else "SKIPPED (--no_color)"))
    print(f"Whitelist: " + (f"removed if >={max(int(min_blacklist), 1)} views see non-object"
                            f" (consensus)" if whitelist_mode == 'all'
                            else "kept if ANY view sees object (union)"))
    if outlier_mode == 'multiview' or outlier_mode == 'combined':
        print(f"Min masked views: {min_views}")
    if outlier_mode == 'spatial' or outlier_mode == 'combined':
        print(f"Spatial percentile: {spatial_percentile}th")
    if outlier_mode == 'neighbor' or outlier_mode == 'combined':
        print(f"Neighbor percentile: {neighbor_percentile}th")
    print("=" * 70)

    # Load Gaussians
    print(f"\nLoading Gaussians from {ply_path}...")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    num_gaussians = len(xyz)
    sh_features = np.zeros((num_gaussians, 3))
    for i in range(3):
        sh_features[:, i] = vertex[f'f_dc_{i}']
    C0 = 0.28209479177387814
    rgb_colors = np.clip(sh_features * C0 + 0.5, 0, 1)   # sh_to_rgb, over all rows
    print(f"  Loaded {num_gaussians:,} Gaussians")

    # Load cameras
    print(f"\nLoading cameras from {cameras_path}...")
    cameras = load_cameras(cameras_path)

    # Count mask files (any image extension). A set, because on Windows globbing
    # both '*.png' and '*.PNG' returns each file twice and doubled the count.
    mask_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    masked_images = {p for ext in mask_extensions
                     for p in Path(masked_images_dir).glob(ext)}

    print(f"  Loaded {len(cameras)} cameras ({len(masked_images)} with masks)")

    # STEP 1: WHITELIST
    print(f"\n[STEP 1] Building whitelist ({whitelist_mode} over views)...")
    args_list = [(idx, cam, masked_images_dir) for idx, cam in enumerate(cameras)]

    # The three counts are all read off one pass of the same int8 state, so the
    # two whitelist rules cost the same to evaluate.
    object_count = np.zeros(num_gaussians, dtype=np.int32)   # views seeing it in an object region
    blacklist_count = np.zeros(num_gaussians, dtype=np.int32)  # views seeing it over non-object
    seen_count = np.zeros(num_gaussians, dtype=np.int32)     # views seeing it at all
    with Pool(num_workers, initializer=_init_worker, initargs=(xyz, rgb_colors)) as pool:
        results = pool.imap_unordered(process_view_whitelist, args_list, chunksize=2)
        for state in tqdm(results, total=len(args_list), desc="  whitelist",
                          unit="view", miniters=5):
            if state is not None:
                object_count += (state == 2)
                blacklist_count += (state == 1)
                seen_count += (state > 0)

    if whitelist_mode == 'all':
        # Removed iff at least --min_blacklist views saw a non-object pixel where it
        # projects: what the camera saw there was sky, so nothing solid was along
        # that ray. K=1 is the strict consensus rule; raising it discards the votes
        # that rest on a single view, which on the secession set is ~45% of the
        # removals and mostly sits at silhouette edges.
        k = max(int(min_blacklist), 1)
        whitelist_mask = blacklist_count < k
        removed = int(num_gaussians - np.sum(whitelist_mask))
        never = int((seen_count == 0).sum())
        print(f"  Removed: {removed:,} ({100*removed/num_gaussians:.1f}%) -- "
              f"projected onto a non-object pixel in >="
              f"{'' if k > 1 else ' '}{k} of {len(cameras):,} views")
        print(f"  Kept: {int(np.sum(whitelist_mask)):,} "
              f"({100*np.sum(whitelist_mask)/num_gaussians:.1f}%)")
        if k > 1 and removed:
            votes = blacklist_count[~whitelist_mask]
            print(f"  votes among the removed: median {int(np.median(votes))}, "
                  f"90th {int(np.percentile(votes, 90))}")
        if never:
            print(f"  note: {never:,} Gaussians ({100*never/num_gaussians:.1f}%) are in no "
                  f"view's frustum at all, so nothing here has judged them. --mode "
                  f"spatial/neighbor or --mode combined would cut those too.")
    elif outlier_mode == 'multiview' or outlier_mode == 'combined':
        whitelist_mask = object_count >= min_views
        print(f"  Whitelisted: {np.sum(whitelist_mask):,} (appear in >={min_views} views)")
    else:
        # Original: appear in at least 1 view
        whitelist_mask = object_count > 0
        print(f"  Whitelisted: {np.sum(whitelist_mask):,} ({100*np.sum(whitelist_mask)/num_gaussians:.1f}%)")
        if whitelist_mask.mean() > 0.99 and len(cameras) > 20:
            print(f"  note: a Gaussian is whitelisted if ANY of the {len(cameras):,} views "
                  f"sees it in an object region, so at this view density the union covers "
                  f"everything and stage 1 removes nothing. Expect stages 2 and 3 to do the "
                  f"work; --whitelist all or --mode multiview are the levers for stage 1.")

    # STEP 2: COLOR VALIDATION
    if not use_color:
        # Keeping only (~has_renders | has_matches) needs a colour test to be
        # meaningful: without one, has_matches is never set and the mask would keep
        # only the Gaussians that are never frontmost anywhere. So skipping the
        # stage means the whitelist passes through untouched, and only stage 3
        # filters further.
        keep_mask = whitelist_mask.copy()
        print(f"\n[STEP 2] Color validation... skipped (--no_color)")
        print(f"  Kept: {int(np.sum(keep_mask)):,} "
              f"({100*np.sum(keep_mask)/num_gaussians:.1f}%)")
    else:
        print(f"\n[STEP 2] Color validation...")
        args_list = [
            (idx, cam, masked_images_dir, whitelist_mask, color_threshold, photos_dir)
            for idx, cam in enumerate(cameras)
        ]
        rendered_count = np.zeros(num_gaussians, dtype=np.int32)
        match_count = np.zeros(num_gaussians, dtype=np.int32)
        with Pool(num_workers, initializer=_init_worker, initargs=(xyz, rgb_colors)) as pool:
            results = pool.imap_unordered(process_view_color_validate, args_list, chunksize=2)
            for result in tqdm(results, total=len(args_list), desc="  colour   ",
                               unit="view", miniters=5):
                if result is not None:
                    is_rendered, color_match = result
                    rendered_count += is_rendered.astype(np.int32)
                    match_count += color_match.astype(np.int32)

        has_renders = rendered_count > 0
        has_matches = match_count > 0
        keep_mask = whitelist_mask & ((~has_renders) | has_matches)
        color_validated = int(np.sum(keep_mask))
        color_removed = int(np.sum(whitelist_mask)) - color_validated
        print(f"  Removed: {color_removed:,} floaters ({100*color_removed/num_gaussians:.1f}%)")
        print(f"  Kept: {color_validated:,} ({100*color_validated/num_gaussians:.1f}%)")

    # STEP 3: OUTLIER REMOVAL
    if outlier_mode in ['spatial', 'neighbor', 'combined']:
        print(f"\n[STEP 3] Outlier removal...")
        initial_kept = np.sum(keep_mask)
        
        # Spatial outlier removal
        if outlier_mode in ['spatial', 'combined']:
            center = np.mean(xyz[keep_mask], axis=0)
            distances = np.linalg.norm(xyz - center, axis=1)
            threshold = np.percentile(distances[keep_mask], spatial_percentile)
            spatial_outliers = (distances > threshold) & keep_mask
            keep_mask = keep_mask & ~spatial_outliers
            print(f"  Spatial: Removed {np.sum(spatial_outliers):,} far Gaussians (>{threshold:.2f})")
        
        # Neighbor-based outlier removal
        if outlier_mode in ['neighbor', 'combined']:
            try:
                from sklearn.neighbors import NearestNeighbors
                kept_indices = np.where(keep_mask)[0]
                kept_xyz = xyz[keep_mask]
                # 'auto', not 'ball_tree': for 3-D points sklearn picks kd_tree, which
                # returns the same distances 38x faster (7.8 s vs 296 s at 1M points).
                nbrs = NearestNeighbors(n_neighbors=min(10, len(kept_xyz)), algorithm='auto').fit(kept_xyz)
                distances_knn, _ = nbrs.kneighbors(kept_xyz)
                avg_neighbor_dist = np.mean(distances_knn, axis=1)
                threshold = np.percentile(avg_neighbor_dist, neighbor_percentile)
                isolated_mask = avg_neighbor_dist > threshold
                isolated_indices = kept_indices[isolated_mask]
                keep_mask[isolated_indices] = False
                print(f"  Neighbor: Removed {len(isolated_indices):,} isolated Gaussians")
            except ImportError:
                print("  Warning: scikit-learn not installed, skipping neighbor filtering")
        
        total_removed = initial_kept - np.sum(keep_mask)
        print(f"  Total outliers removed: {total_removed:,}")

    # Final results
    num_keep = np.sum(keep_mask)
    num_remove = num_gaussians - num_keep
    print(f"\nFinal Results:")
    print(f"  Kept: {num_keep:,} ({100*num_keep/num_gaussians:.1f}%)")
    print(f"  Removed: {num_remove:,} ({100*num_remove/num_gaussians:.1f}%)")

    # Save
    print(f"\nSaving to {output_path}...")
    vertex_kept = vertex[keep_mask]
    el = PlyElement.describe(vertex_kept, 'vertex')
    PlyData([el]).write(output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    # ASCII only: a check mark is not in cp1252, so printing one kills the run
    # with a UnicodeEncodeError whenever stdout is a pipe or a log file.
    print(f"  Saved {num_keep:,} Gaussians")
    print(f"  File size: {file_size:.1f} MB")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--masked_images", required=True)
    parser.add_argument("--photos", default=None,
                        help="unmasked photographs, named like the masks; the colour "
                             "validation compares against these instead of the masks "
                             "(which are binary and so have no colour to compare)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=['none', 'spatial', 'neighbor', 'multiview', 'combined'], default='neighbor')
    parser.add_argument("--color_threshold", type=float, default=0.40)
    parser.add_argument("--no_color", action="store_true",
                        help="skip colour validation (stage 2); useful when the scene "
                             "is captured across changing light")
    parser.add_argument("--whitelist", choices=['any', 'all'], default='any',
                        help="'any' (default) keeps a Gaussian if any view sees it in "
                             "an object region; 'all' removes it if --min_blacklist "
                             "views see a non-object pixel where it projects")
    parser.add_argument("--min_blacklist", type=int, default=1,
                        help="votes needed to remove under --whitelist all (default 1)")
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--spatial_percentile", type=int, default=99)
    parser.add_argument("--neighbor_percentile", type=int, default=95)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    
    clean_gs(
        args.ply, args.cameras, args.masked_images, args.output,
        args.mode, args.color_threshold, args.min_views,
        args.spatial_percentile, args.neighbor_percentile, args.workers,
        args.photos, not args.no_color, args.whitelist, args.min_blacklist
    )
