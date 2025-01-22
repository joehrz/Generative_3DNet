# File: src/data_processing/data_processing.py

"""
Point Cloud Preprocessing:
  1) Reads .ply (or other open3d-supported) files from input_dir
  2) Voxel downsampling
  3) Adjust total point count (with optional farthest-point sampling)
  4) Normalize to unit sphere
  5) Saves the final arrays in .npz with 'points' and optional 'colors'
"""

import os
import numpy as np
import open3d as o3d
from typing import Tuple


def voxel_down_sample_with_indices(pcd: o3d.geometry.PointCloud, voxel_size: float):
    min_bound = pcd.get_min_bound() - voxel_size * 0.5
    max_bound = pcd.get_max_bound() + voxel_size * 0.5

    downsampled_pcd, _, point_indices = pcd.voxel_down_sample_and_trace(
        voxel_size, min_bound, max_bound, False
    )

    indices = []
    for idx_list in point_indices:
        if len(idx_list) > 0:
            indices.append(idx_list[0])
    indices = np.array(indices, dtype=int)
    return downsampled_pcd, indices


def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    N = points.shape[0]
    if k >= N:
        return np.arange(N)
    
    sampled_indices = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf, dtype=np.float32)

    # random start
    sampled_indices[0] = np.random.randint(N)
    current = sampled_indices[0]

    for i in range(1, k):
        current_pt = points[current]
        diff = points - current_pt
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        dist = np.minimum(dist, dist_sq)
        current = np.argmax(dist)
        sampled_indices[i] = current

    return sampled_indices


def adjust_point_count(pcd: o3d.geometry.PointCloud, num_points: int, use_fps: bool = True):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    N = points.shape[0]

    if N == num_points:
        return pcd
    elif N < num_points:
        # Upsample
        extra = num_points - N
        repeat_idx = np.random.randint(0, N, extra)
        final_indices = np.concatenate([np.arange(N), repeat_idx])
        np.random.shuffle(final_indices)
    else:
        # Downsample
        if use_fps:
            sampled_indices = farthest_point_sampling(points, num_points)
        else:
            sampled_indices = np.random.choice(N, num_points, replace=False)
        final_indices = sampled_indices

    final_points = points[final_indices]
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(final_points)

    if colors is not None:
        final_colors = colors[final_indices]
        new_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    return new_pcd


def normalize_point_cloud(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    centroid = points.mean(axis=0)
    points -= centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if max_dist > 1e-9:
        points /= max_dist

    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def preprocess_point_clouds(
    input_dir: str,
    output_dir: str,
    voxel_size: float,
    num_points: int,
    use_fps: bool = True,
    skip_downsample: bool = False
) -> None:
    """
    Now handles .ply, .npz, or .npy files:
      - loads them
      - voxel downsampling if .ply or if you want a single pipeline
      - unify to 'num_points'
      - normalize
      - saves final as .npz
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_exts = {".ply", ".npz", ".npy"}
    all_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    if not all_files:
        print(f"No .ply/.npz/.npy files found in {input_dir}.")
        return

    for fname in all_files:
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.join(input_dir, fname)

        pcd = None  # We'll build or load into Open3D pcd

        # 1) LOAD
        if ext == ".ply":
            # load with open3d
            pcd_o3d = o3d.io.read_point_cloud(fpath)
            if len(pcd_o3d.points) == 0:
                print(f"Empty or invalid .ply: {fpath}")
                continue
            pcd = pcd_o3d

        elif ext == ".npz":
            data = np.load(fpath)
            keys = list(data.keys())
            if "points" in data:
                pts = data["points"]
                cols = data["colors"] if "colors" in data else None
            elif "pc" in data:
                pts = data["pc"]
                cols = data["colors"] if "colors" in data else None
            else:
                # fallback to the first array in keys
                if not keys:
                    print(f"No arrays in {fpath}")
                    continue
                pts = data[keys[0]]
                cols = None

            if pts.ndim != 2 or pts.shape[1] != 3:
                print(f"Skipping {fpath}, points not Nx3")
                continue

            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pts)
            if cols is not None and cols.shape == pts.shape:
                pcd_o3d.colors = o3d.utility.Vector3dVector(cols)
            pcd = pcd_o3d

        elif ext == ".npy":
            pts = np.load(fpath)
            if pts.ndim != 2 or pts.shape[1] != 3:
                print(f"Skipping {fpath}, array not Nx3")
                continue
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pts)
            pcd = pcd_o3d

        else:
            print(f"Skipping unrecognized extension {fname}")
            continue

        if pcd is None or len(pcd.points) == 0:
            print(f"Empty or invalid data in {fname}")
            continue

        # 2) VOXEL DOWNSAMPLE (if you want to apply the same pipeline to all)
        if (not skip_downsample) and (voxel_size > 1e-9):
            pcd_down, _ = voxel_down_sample_with_indices(pcd, voxel_size)
        else:
            pcd_down = pcd

        # 3) Adjust total point count
        pcd_adj = adjust_point_count(pcd_down, num_points=num_points, use_fps=use_fps)

        # 4) Normalize
        pcd_norm = normalize_point_cloud(pcd_adj)

        # Convert to arrays
        final_points = np.asarray(pcd_norm.points)
        final_colors = None
        if pcd_norm.has_colors():
            final_colors = np.asarray(pcd_norm.colors)

        # 5) Save .npz
        out_name = os.path.splitext(fname)[0] + ".npz"
        out_path = os.path.join(output_dir, out_name)

        if final_colors is not None:
            np.savez(out_path, points=final_points, colors=final_colors)
        else:
            np.savez(out_path, points=final_points)

        print(f"Preprocessed => {out_path}  [points={len(final_points)}]")