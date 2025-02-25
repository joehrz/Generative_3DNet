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
from typing import Tuple


def load_plant_point_cloud(file_path: str) -> np.ndarray:
    """
    Loads a real plant point cloud from a .txt file, where each line contains
    three coordinates in the format "x y z". It extracts these coordinates and
    returns them as a NumPy array of shape (N, 3).

    Args:
        file_path (str): The path to the .txt file containing point cloud data.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3) representing the point cloud.
                    Each row corresponds to one (x, y, z) point.

    Example:
        >>> import numpy as np
        >>> pc_array = load_plant_point_cloud("plant_points.txt")
        >>> print(pc_array.shape)
        (N, 3)
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

    This function uses Open3D’s `voxel_down_sample_and_trace` to reduce the
    point density based on a specified voxel size. In addition to returning the
    downsampled point cloud, it gathers the indices of which points from the
    original cloud were retained.

    Args:
        pcd (o3d.geometry.PointCloud): The input Open3D point cloud.
        voxel_size (float): The voxel size in the same units as the point cloud’s coordinates.

    Returns:
        tuple:
            - downsampled_pcd (o3d.geometry.PointCloud):
              The resulting voxel-downsampled point cloud.
            - indices (np.ndarray of shape (M,)):
              The indices of the original point cloud that correspond to the
              M points in the downsampled cloud.

    Example:
        >>> import open3d as o3d
        >>> pcd = o3d.io.read_point_cloud("some_cloud.ply")
        >>> down_pcd, idx = voxel_down_sample_with_indices(pcd, 0.02)
        >>> print("Downsampled cloud points:", len(down_pcd.points))
        >>> print("Indices in the original cloud:", idx)
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

    FPS iteratively selects the point that is farthest from the previously
    selected points, ensuring a diverse spread of sampled points.

    Args:
        points (np.ndarray): The input points of shape (N, 3).
        k (int): The number of points to sample.

    Returns:
        np.ndarray: A 1D array of indices of length 'k', indicating which points
                    in the original array were selected.

    Example:
        >>> import numpy as np
        >>> pts = np.random.rand(100, 3)
        >>> sampled_indices = farthest_point_sampling(pts, 10)
        >>> print(sampled_indices)  # e.g. [ 5  73  22 ...]
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
    Resamples a point cloud to exactly 'num_points' points. If the current number
    of points is greater than 'num_points', the function downsamples using
    farthest-point sampling (if 'use_fps' is True) or random selection. If the
    current number of points is less than 'num_points', the function upsamples by
    randomly duplicating points.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud to be resampled.
        num_points (int): The target number of points.
        use_fps (bool, optional): Whether to use farthest-point sampling for
                                  downsampling. If False, random downsampling
                                  is used. Defaults to True.

    Returns:
        o3d.geometry.PointCloud: A new Open3D point cloud with exactly
                                 'num_points' points.

    Example:
        >>> import open3d as o3d
        >>> pcd = o3d.io.read_point_cloud("large_cloud.ply")
        >>> # Downsample to 2048 points using FPS
        >>> pcd_2048 = adjust_point_count(pcd, 2048, use_fps=True)
        >>> print("New cloud points:", len(pcd_2048.points))
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    N = points.shape[0]

    # Case 1: Already the target size
    if N == num_points:
        return pcd
    # Case 2: Need to upsample (fewer points than needed)
    elif N < num_points:
        extra = num_points - N
        repeat_idx = np.random.randint(0, N, extra)
        final_indices = np.concatenate([np.arange(N), repeat_idx])
        np.random.shuffle(final_indices)
    # Case 3: Need to downsample
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

    It shifts the point cloud so that its minimum coordinate is (0,0,0), and
    scales it so that its maximum coordinate is (1,1,1). If the range along any
    axis is extremely small, this function prevents division by zero by capping
    the denominator at 1e-9.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud to be normalized.

    Returns:
        o3d.geometry.PointCloud: A point cloud with points normalized to [0,1].
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals
    # Avoid division-by-zero issues
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
    Preprocesses a set of point cloud files by:
      1) Loading from various formats (.ply, .txt, .npz, .npy).
      2) (Optionally) voxel downsampling to reduce point density.
      3) Adjusting the total number of points to 'num_points' (using either
         farthest-point sampling or random selection).
      4) Normalizing each point cloud to the [0,1] cube.
      5) Saving the results as compressed .npz files, preserving colors if present.

    Args:
        input_dir (str): Directory containing the input point cloud files. Supported
                         file types are .ply, .txt, .npz, and .npy.
        output_dir (str): Directory where the preprocessed .npz files will be saved.
        voxel_size (float): The voxel size for downsampling. If skip_downsample
                            is True or voxel_size is very small (<=1e-9), the
                            step is skipped.
        num_points (int): The target number of points in each preprocessed cloud.
        use_fps (bool, optional): Whether to use farthest-point sampling when
                                  downsampling to 'num_points'. Defaults to True.
        skip_downsample (bool, optional): If True, skips voxel downsampling entirely.
                                          Defaults to False.

    Returns:
        None. The function saves preprocessed .npz files in 'output_dir'.

    Example:
        >>> preprocess_point_clouds(
        ...     input_dir="data/raw",
        ...     output_dir="data/processed",
        ...     voxel_size=0.02,
        ...     num_points=2048,
        ...     use_fps=True,
        ...     skip_downsample=False
        ... )
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_exts = {".ply", ".txt", ".npz", ".npy"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    if not files:
        print(f"No valid files found in {input_dir}. Supported: {valid_exts}")
        return

    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.join(input_dir, fname)
        pcd = None

        # 1) Load step
        if ext == ".ply":
            pcd_o3d = o3d.io.read_point_cloud(fpath)
            if not pcd_o3d.points:
                print(f"[Warning] Empty or invalid .ply: {fpath}")
                continue
            pcd = pcd_o3d
        elif ext == ".txt":
            pts = load_plant_point_cloud(fpath)
            if pts.size == 0 or pts.shape[1] != 3:
                print(f"[Warning] Invalid or empty .txt: {fpath}")
                continue
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pts)
            pcd = pcd_o3d
        elif ext == ".npz":
            data = np.load(fpath)
            keys = list(data.keys())
            if "points" in data:
                pts = data["points"]
                cols = data["colors"] if "colors" in data else None
            elif "pc" in data:  # Some alternative naming
                pts = data["pc"]
                cols = data["colors"] if "colors" in data else None
            else:
                if not keys:
                    print(f"[Warning] No arrays in {fpath}")
                    continue
                pts = data[keys[0]]
                cols = None

            if pts.ndim != 2 or pts.shape[1] != 3:
                print(f"[Warning] Skipping {fpath}, array not Nx3.")
                continue

            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pts)
            if cols is not None and cols.shape == pts.shape:
                pcd_o3d.colors = o3d.utility.Vector3dVector(cols)
            pcd = pcd_o3d
        elif ext == ".npy":
            pts = np.load(fpath)
            if pts.ndim != 2 or pts.shape[1] != 3:
                print(f"[Warning] Skipping {fpath}, array not Nx3.")
                continue
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pts)
            pcd = pcd_o3d
        else:
            print(f"[Warning] Unrecognized file extension: {fname}")
            continue

        if pcd is None or len(pcd.points) == 0:
            print(f"[Warning] Could not load valid data from {fname}")
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
        final_colors = None
        if pcd_norm.has_colors():
            final_colors = np.asarray(pcd_norm.colors)

        out_name = os.path.splitext(fname)[0] + ".npz"
        out_path = os.path.join(output_dir, out_name)

        if final_colors is not None:
            np.savez_compressed(out_path, points=final_points, colors=final_colors)
        else:
            np.savez_compressed(out_path, points=final_points)

        print(f"[INFO] Preprocessed => {out_path} [points={len(final_points)}]")

