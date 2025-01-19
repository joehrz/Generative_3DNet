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


def voxel_down_sample_with_indices(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Downsamples the point cloud using a voxel grid and returns the indices
    of the selected points in the original cloud.

    Args:
        pcd (o3d.geometry.PointCloud)
        voxel_size (float)

    Returns:
        downsampled_pcd (o3d.geometry.PointCloud)
        indices (np.ndarray): indices of selected points in the original cloud
    """
    min_bound = pcd.get_min_bound() - voxel_size * 0.5
    max_bound = pcd.get_max_bound() + voxel_size * 0.5

    downsampled_pcd, _, point_indices = pcd.voxel_down_sample_and_trace(
        voxel_size, min_bound, max_bound, False
    )

    # We pick the first point in each voxel
    indices = []
    for idx_list in point_indices:
        if len(idx_list) > 0:
            indices.append(idx_list[0])
    indices = np.array(indices, dtype=int)


    return downsampled_pcd, indices

def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """
    Farthest point sampling (FPS) to pick k points from a larger set.
    This helps preserve geometry structure better than random sampling.

    Args:
        points: shape (N, 3)
        k: number of points to sample

    Returns:
        sampled_indices: shape (k,) of chosen indices
    """
    N = points.shape[0]
    if k >= N:
        return np.arrange(N)
    
    sampled_indices = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf, dtype=np.float32)

    #  Start with a random index
    sampled_indices[0] = np.random.randint(N)
    current =  sampled_indices[0]

    for i in range(1, k):
        current_pt = points[current]
        diff = points - current_pt
        dist_sq = np.einsum('ij,ij->i', diff, diff)  # squared dist
        dist = np.minimum(dist, dist_sq)
        current = np.argmax(dist)
        sampled_indices[i] = current

    return sampled_indices


def adjust_point_count(
    pcd: o3d.geometry.PointCloud, 
    num_points: int,
    use_fps: bool = True
) -> o3d.geometry.PointCloud:
    """
    Adjusts the total number of points to 'num_points'.
    If N < num_points, we upsample by repeating.
    If N > num_points, we downsample (FPS or random).

    Args:
        pcd (o3d.geometry.PointCloud)
        num_points (int)
        use_fps (bool)

    Returns:
        adjusted_pcd (o3d.geometry.PointCloud)
    """
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
    if colors is not None:
        final_colors = colors[final_indices]
    else:
        final_colors = None

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(final_points)
    if final_colors is not None:
        new_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    return new_pcd

def normalize_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Centers at origin and scales to fit in a unit sphere.
    """
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
    file_ext: str = ".ply",
    use_fps: bool = True
) -> None:
    """
    Reads .ply from input_dir => [voxel downsample] => [unify point count] => [normalize]
    => saves to .npz in output_dir with 'points' + optional 'colors'.

    Args:
        input_dir (str): directory with .ply files
        output_dir (str): directory to save .npz
        voxel_size (float): for voxel-based downsampling
        num_points (int): unify each cloud to num_points
        use_fps (bool): if True, use farthest point sampling
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(file_ext)]
    if not files:
        print(f"No .ply files found in {input_dir}.")
        return

    for fname in files:
        pcd_path = os.path.join(input_dir, fname)
        # 1) Load
        pcd = o3d.io.read_point_cloud(pcd_path)
        if len(pcd.points) == 0:
            print(f"Empty or invalid cloud: {pcd_path}")
            continue

        # 2) Voxel downsample
        pcd_down, idx_down = voxel_down_sample_with_indices(pcd, voxel_size)

        # 3) Adjust point count
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