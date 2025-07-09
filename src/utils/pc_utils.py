# File: src/utils/pc_utils.py

import torch
import math
import random
import numpy as np
from scipy.spatial.distance import cdist
def random_rotate(pc):
    """
    Rotate the entire cloud around Z-axis by a random angle in [0, 2π].
    pc: (N, 3) or (B, N, 3) torch.Tensor
    """
    if pc.dim() == 2:
        # shape (N, 3)
        angle = random.random() * 2 * math.pi
        cosval = math.cos(angle)
        sinval = math.sin(angle)
        R = torch.tensor([[cosval, -sinval, 0],
                          [sinval,  cosval, 0],
                          [0,       0,      1]],
                          dtype=pc.dtype, device=pc.device)
        rotated = pc @ R.T
        return rotated
    else:
        # shape (B, N, 3)
        B, N, _ = pc.shape
        out_list = []
        for b in range(B):
            angle = random.random() * 2 * math.pi
            cosval = math.cos(angle)
            sinval = math.sin(angle)
            R = torch.tensor([[cosval, -sinval, 0],
                              [sinval,  cosval, 0],
                              [0,       0,      1]],
                              dtype=pc.dtype, device=pc.device)
            rotated = pc[b] @ R.T  # (N, 3)
            out_list.append(rotated.unsqueeze(0))
        return torch.cat(out_list, dim=0)

def normalize_to_unit_sphere(pc):
    """
    pc: (N, 3) or (B, N, 3) torch Tensor
    Shift & scale so max distance from origin is 1.
    """
    if pc.dim() == 3:
        mean = pc.mean(dim=1, keepdim=True)
        pc = pc - mean
        max_dist = pc.norm(dim=2).max(dim=1, keepdim=True)[0]
        pc = pc / (max_dist.unsqueeze(-1) + 1e-9)
        return pc
    else:
        mean = pc.mean(dim=0, keepdim=True)
        pc = pc - mean
        max_dist = pc.norm(dim=1).max()
        pc = pc / (max_dist + 1e-9)
        return pc

def compute_bounding_box(pc):
    """
    Return the min & max corners.
    """
    if pc.dim() == 2:
        min_vals = pc.min(dim=0)[0]
        max_vals = pc.max(dim=0)[0]
        return min_vals, max_vals
    else:
        min_vals = pc.min(dim=1)[0]
        max_vals = pc.max(dim=1)[0]
        return min_vals, max_vals

def center_and_scale(pc, min_val=None, max_val=None):
    """
    Maps pc into [0,1]^3 -> shift by -0.5 => [-0.5,0.5]^3.
    """
    if pc.dim() == 2:
        if min_val is None or max_val is None:
            min_val, max_val = compute_bounding_box(pc)
        pc_range = (max_val - min_val).clamp(min=1e-9)
        pc = (pc - min_val) / pc_range
        pc = pc - 0.5
        return pc
    else:
        B = pc.size(0)
        if min_val is None or max_val is None:
            min_vals, max_vals = compute_bounding_box(pc)
        for b in range(B):
            pc_range = (max_vals[b] - min_vals[b]).clamp(min=1e-9)
            pc[b] = (pc[b] - min_vals[b]) / pc_range
            pc[b] = pc[b] - 0.5
        return pc

def sample_points(pc, num_samples=1024):
    """
    Randomly sample 'num_samples' points. If fewer points, repeat some.
    """
    N = pc.size(0)
    if N > num_samples:
        idx = torch.randperm(N)[:num_samples]
        return pc[idx]
    else:
        extra = num_samples - N
        repeated = pc[torch.randint(0, N, (extra,))]
        return torch.cat([pc, repeated], dim=0)

def random_flip(pc, p_x=0.5, p_y=0.5, p_z=0.0):
    """
    Randomly flips the point cloud along specified axes.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    p_x, p_y, p_z: probability of flipping along x, y, z axis respectively.
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0) # Treat as a batch of 1

    B, N, C = pc.shape
    flipped_pc = pc.clone()

    # Flip X
    if random.random() < p_x:
        flipped_pc[:, :, 0] = -flipped_pc[:, :, 0]
    # Flip Y
    if random.random() < p_y:
        flipped_pc[:, :, 1] = -flipped_pc[:, :, 1]
    # Flip Z
    if random.random() < p_z: # Often Z-flips are not desired for upright objects
        flipped_pc[:, :, 2] = -flipped_pc[:, :, 2]

    return flipped_pc.squeeze(0) if single_pc else flipped_pc

def random_noise(pc, mean=0.0, std=0.01):
    """
    Adds Gaussian noise to the point cloud.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    mean: mean of the Gaussian noise.
    std: standard deviation of the Gaussian noise.
    """
    if pc.dim() == 2: # (N,3)
        noise = torch.normal(mean, std, size=pc.shape, device=pc.device, dtype=pc.dtype)
    else: # (B,N,3)
        noise = torch.normal(mean, std, size=pc.shape, device=pc.device, dtype=pc.dtype)
    return pc + noise

def random_scale(pc, min_scale=0.8, max_scale=1.2):
    """
    Randomly scales the point cloud uniformly.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    min_scale, max_scale: range for the random scaling factor.
    """
    if pc.dim() == 2: # (N,3)
        scale_factor = random.uniform(min_scale, max_scale)
        return pc * scale_factor
    else: # (B,N,3)
        B, N, C = pc.shape
        # Apply a different random scale to each item in the batch, but same scale for all points in one item
        scale_factors = torch.rand(B, 1, 1, device=pc.device, dtype=pc.dtype) * (max_scale - min_scale) + min_scale
        return pc * scale_factors

def jitter_points(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. Jittering is per point.
    This function now handles both single (N, 3) and batched (B, N, 3) tensors.
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0) # Temporarily add a batch dimension

    B, N, C = pc.shape
    assert(clip > 0)
    jittered_data = torch.clamp(sigma * torch.randn(B, N, C, device=pc.device, dtype=pc.dtype), -1*clip, clip)
    jittered_data += pc
    
    # Remove the batch dimension if we added it
    return jittered_data.squeeze(0) if single_pc else jittered_data

def random_perspective_transform(pc, strength=0.1):
    """
    Apply random perspective transformation to point cloud.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    strength: controls the strength of perspective distortion
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0)
    
    B, N, C = pc.shape
    transformed_pc = pc.clone()
    
    for b in range(B):
        # Create random perspective matrix
        perspective_factor = random.uniform(-strength, strength)
        
        # Apply perspective transformation (simple z-dependent scaling)
        z_vals = transformed_pc[b, :, 2]
        scale_factor = 1.0 + perspective_factor * z_vals
        scale_factor = scale_factor.unsqueeze(1).expand(-1, 3)
        
        transformed_pc[b] = transformed_pc[b] * scale_factor
    
    return transformed_pc.squeeze(0) if single_pc else transformed_pc

def random_elastic_deformation(pc, strength=0.1, num_control_points=4):
    """
    Apply random elastic deformation using radial basis functions.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    strength: controls the magnitude of deformation
    num_control_points: number of control points for deformation
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0)
    
    B, N, C = pc.shape
    deformed_pc = pc.clone()
    
    for b in range(B):
        # Generate random control points within the point cloud bounds (GPU-based)
        pc_batch = pc[b]
        min_vals = torch.min(pc_batch, dim=0)[0]
        max_vals = torch.max(pc_batch, dim=0)[0]
        
        # Generate control points directly on GPU
        control_points = torch.rand(num_control_points, 3, device=pc.device)
        control_points = control_points * (max_vals - min_vals) + min_vals
        
        # Generate random displacement vectors on GPU
        displacements = torch.rand(num_control_points, 3, device=pc.device) * 2 - 1
        displacements = displacements * strength
        
        # Compute distances using torch.cdist (GPU-optimized)
        distances = torch.cdist(pc_batch.unsqueeze(0), control_points.unsqueeze(0)).squeeze(0)
        
        # Apply RBF-based deformation (all on GPU)
        weights = torch.exp(-distances**2 / (2 * strength**2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Apply weighted displacements
        displacement_field = torch.mm(weights, displacements)
        
        deformed_pc[b] = pc[b] + displacement_field
    
    return deformed_pc.squeeze(0) if single_pc else deformed_pc

def random_occlusion(pc, occlusion_ratio=0.1):
    """
    Randomly occlude (remove) points from the point cloud.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    occlusion_ratio: fraction of points to remove
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0)
    
    B, N, C = pc.shape
    occluded_pc = pc.clone()
    
    for b in range(B):
        num_to_remove = int(N * occlusion_ratio)
        if num_to_remove > 0:
            # Randomly select points to remove
            indices_to_remove = torch.randperm(N)[:num_to_remove]
            
            # Create mask for points to keep
            mask = torch.ones(N, dtype=torch.bool, device=pc.device)
            mask[indices_to_remove] = False
            
            # Keep only non-occluded points, pad with duplicates to maintain size
            kept_points = occluded_pc[b][mask]
            num_kept = kept_points.shape[0]
            
            if num_kept < N:
                # Pad with random duplicates to maintain point count
                num_to_pad = N - num_kept
                pad_indices = torch.randint(0, num_kept, (num_to_pad,), device=pc.device)
                padding = kept_points[pad_indices]
                occluded_pc[b] = torch.cat([kept_points, padding], dim=0)
    
    return occluded_pc.squeeze(0) if single_pc else occluded_pc

def random_dropout(pc, dropout_ratio=0.1):
    """
    Randomly dropout points and replace with noise.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    dropout_ratio: fraction of points to replace with noise
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0)
    
    B, N, C = pc.shape
    dropped_pc = pc.clone()
    
    for b in range(B):
        num_to_drop = int(N * dropout_ratio)
        if num_to_drop > 0:
            # Randomly select points to replace
            indices_to_drop = torch.randperm(N, device=pc.device)[:num_to_drop]
            
            # Generate random noise points within bounding box
            min_vals = pc[b].min(dim=0)[0]
            max_vals = pc[b].max(dim=0)[0]
            noise_points = torch.rand(num_to_drop, 3, device=pc.device, dtype=pc.dtype)
            noise_points = noise_points * (max_vals - min_vals) + min_vals
            
            # Replace selected points with noise
            dropped_pc[b][indices_to_drop] = noise_points
    
    return dropped_pc.squeeze(0) if single_pc else dropped_pc

def point_cutmix(pc1, pc2, alpha=0.3):
    """
    Apply CutMix augmentation to point clouds.
    pc1, pc2: (N, 3) torch.Tensor - two point clouds to mix
    alpha: mixing parameter
    """
    N = pc1.shape[0]
    
    # Generate random mixing ratio
    lam = np.random.beta(alpha, alpha)
    
    # Randomly select points from pc1 and pc2
    num_from_pc1 = int(N * lam)
    num_from_pc2 = N - num_from_pc1
    
    if num_from_pc1 > 0:
        indices_pc1 = torch.randperm(N, device=pc1.device)[:num_from_pc1]
        selected_pc1 = pc1[indices_pc1]
    else:
        selected_pc1 = torch.empty(0, 3, device=pc1.device, dtype=pc1.dtype)
    
    if num_from_pc2 > 0:
        indices_pc2 = torch.randperm(N, device=pc2.device)[:num_from_pc2]
        selected_pc2 = pc2[indices_pc2]
    else:
        selected_pc2 = torch.empty(0, 3, device=pc2.device, dtype=pc2.dtype)
    
    # Combine selected points
    mixed_pc = torch.cat([selected_pc1, selected_pc2], dim=0)
    
    # Shuffle to avoid ordering bias
    shuffle_indices = torch.randperm(N, device=mixed_pc.device)
    mixed_pc = mixed_pc[shuffle_indices]
    
    return mixed_pc, lam

def random_rotate_3d(pc, max_angle=0.5):
    """
    Apply random 3D rotation (more general than Z-axis only).
    pc: (N, 3) or (B, N, 3) torch.Tensor
    max_angle: maximum rotation angle in radians
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0)
    
    B, N, C = pc.shape
    
    # Vectorized rotation matrix generation for better GPU utilization
    # Generate random rotation angles for all batches at once
    angles = torch.rand(B, 3, device=pc.device, dtype=pc.dtype) * 2 * max_angle - max_angle
    
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    cos_x, sin_x = cos_angles[:, 0], sin_angles[:, 0]
    cos_y, sin_y = cos_angles[:, 1], sin_angles[:, 1]
    cos_z, sin_z = cos_angles[:, 2], sin_angles[:, 2]
    
    # Create rotation matrices for all batches at once
    zeros = torch.zeros(B, device=pc.device, dtype=pc.dtype)
    ones = torch.ones(B, device=pc.device, dtype=pc.dtype)
    
    # Rotation around X-axis (B, 3, 3)
    R_x = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, cos_x, -sin_x], dim=1),
        torch.stack([zeros, sin_x, cos_x], dim=1)
    ], dim=1)
    
    # Rotation around Y-axis (B, 3, 3)
    R_y = torch.stack([
        torch.stack([cos_y, zeros, sin_y], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sin_y, zeros, cos_y], dim=1)
    ], dim=1)
    
    # Rotation around Z-axis (B, 3, 3)
    R_z = torch.stack([
        torch.stack([cos_z, -sin_z, zeros], dim=1),
        torch.stack([sin_z, cos_z, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)
    
    # Combined rotation matrices (B, 3, 3)
    R = torch.bmm(torch.bmm(R_z, R_y), R_x)
    
    # Apply rotations to all batches at once
    rotated_pc = torch.bmm(pc, R.transpose(1, 2))
    
    return rotated_pc.squeeze(0) if single_pc else rotated_pc

def random_point_resampling(pc, resample_ratio=0.2):
    """
    Randomly resample points using different strategies.
    pc: (N, 3) or (B, N, 3) torch.Tensor
    resample_ratio: fraction of points to resample
    """
    single_pc = pc.dim() == 2
    if single_pc:
        pc = pc.unsqueeze(0)
    
    B, N, C = pc.shape
    resampled_pc = pc.clone()
    
    for b in range(B):
        num_to_resample = int(N * resample_ratio)
        if num_to_resample > 0:
            # Randomly select points to resample
            indices_to_resample = torch.randperm(N, device=pc.device)[:num_to_resample]
            
            # Strategy 1: Interpolate between nearby points
            for idx in indices_to_resample:
                # Find k nearest neighbors
                k = min(5, N-1)
                distances = torch.norm(pc[b] - pc[b][idx], dim=1)
                _, nearest_indices = torch.topk(distances, k+1, largest=False)
                nearest_indices = nearest_indices[1:]  # Exclude the point itself
                
                # Interpolate
                weights = torch.rand(k, device=pc.device, dtype=pc.dtype)
                weights = weights / weights.sum()
                
                new_point = torch.sum(pc[b][nearest_indices] * weights.unsqueeze(1), dim=0)
                resampled_pc[b][idx] = new_point
    
    return resampled_pc.squeeze(0) if single_pc else resampled_pc

class PointCloudAugmentation:
    """
    A class to compose a series of augmentation functions.
    """
    def __init__(self, augmentations):
        """
        augmentations: a list of callable augmentation functions.
        """
        self.augmentations = augmentations

    def __call__(self, pc):
        for aug in self.augmentations:
            pc = aug(pc)
        return pc