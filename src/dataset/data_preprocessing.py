# File: src/dataset/data_processing.py

"""
Point Cloud Preprocessing:
  1) Reads .ply (or other open3d-supported) files from input_dir
  2) Voxel downsampling
  3) Adjust total point count (with optional farthest-point sampling)
  4) Normalize to [0,1] (per axis)
  5) Saves the final arrays in .npz with 'points' and optional 'colors'
"""

import os
import numpy as np
import open3d as o3d
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_plant_point_cloud(file_path: str) -> np.ndarray:
    """
    Loads a real plant point cloud from a .txt file, where each line contains
    three coordinates in the format "x y z". It extracts these coordinates and
    returns them as a NumPy array of shape (N, 3).
    """
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) < 3:
                continue
            x, y, z = vals[0], vals[1], vals[2]
            points.append([float(x), float(y), float(z)])
    return np.array(points)


def voxel_down_sample_with_indices(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """
    Downsamples a point cloud using voxelization and also returns the indices of
    the points selected from the original point cloud.
    """
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
    """
    Performs farthest-point sampling (FPS) on a set of points and returns the
    indices of the 'k' most distant points.
    """
    N = points.shape[0]
    if k >= N:
        return np.arange(N)

    sampled_indices = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf, dtype=np.float32)

    # Random initial pick
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
    """
    Resamples a point cloud to exactly 'num_points' points.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    N = points.shape[0]

    if N == num_points:
        return pcd
    elif N < num_points:
        extra = num_points - N
        repeat_idx = np.random.randint(0, N, extra)
        final_indices = np.concatenate([np.arange(N), repeat_idx])
        np.random.shuffle(final_indices)
    else:
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
    """
    Normalizes a point cloud to the [0, 1] range along each axis (x, y, z).
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-9] = 1.0

    normalized_points = (points - min_vals) / range_vals
    pcd.points = o3d.utility.Vector3dVector(normalized_points)
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
    Preprocesses a set of point cloud files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_exts = {".ply", ".txt", ".npz", ".npy"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    if not files:
        logger.warning(f"No valid files found in {input_dir}. Supported: {valid_exts}")
        return

    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.join(input_dir, fname)
        pcd = None

        try:
            # 1) Load step
            if ext == ".ply":
                pcd = o3d.io.read_point_cloud(fpath)
            elif ext == ".txt":
                pts = load_plant_point_cloud(fpath)
                if pts.size > 0 and pts.shape[1] == 3:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
            elif ext == ".npz":
                data = np.load(fpath)
                keys = list(data.keys())
                if "points" in data:
                    pts = data["points"]
                    cols = data.get("colors")
                elif "pc" in data:
                    pts = data["pc"]
                    cols = data.get("colors")
                elif keys:
                    pts = data[keys[0]]
                    cols = None
                else:
                    logger.warning(f"No arrays found in {fpath}")
                    continue

                if pts.ndim == 2 and pts.shape[1] == 3:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    if cols is not None and cols.shape == pts.shape:
                        pcd.colors = o3d.utility.Vector3dVector(cols)
                else:
                    logger.warning(f"Skipping {fpath}, array is not Nx3.")
                    continue
            elif ext == ".npy":
                pts = np.load(fpath)
                if pts.ndim == 2 and pts.shape[1] == 3:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                else:
                    logger.warning(f"Skipping {fpath}, array is not Nx3.")
                    continue
            else:
                logger.warning(f"Unrecognized file extension: {fname}")
                continue

            if pcd is None or not pcd.has_points():
                logger.warning(f"Could not load valid data or empty point cloud from {fname}")
                continue

            # 2) Voxel downsample if enabled
            if not skip_downsample and voxel_size > 1e-9:
                pcd_down, _ = voxel_down_sample_with_indices(pcd, voxel_size)
            else:
                pcd_down = pcd

            # 3) Adjust total point count
            pcd_adj = adjust_point_count(pcd_down, num_points=num_points, use_fps=use_fps)

            # 4) Normalize
            pcd_norm = normalize_point_cloud(pcd_adj)

            # 5) Save final .npz
            final_points = np.asarray(pcd_norm.points)
            final_colors = np.asarray(pcd_norm.colors) if pcd_norm.has_colors() else None

            out_name = os.path.splitext(fname)[0] + ".npz"
            out_path = os.path.join(output_dir, out_name)

            if final_colors is not None:
                np.savez_compressed(out_path, points=final_points, colors=final_colors)
            else:
                np.savez_compressed(out_path, points=final_points)

            logger.info(f"Preprocessed => {out_path} [points={len(final_points)}]")

        except Exception as e:
            logger.error(f"Failed to process {fname}: {e}")