# File: src/utils/pc_utils.py

import torch
import math
import random
import numpy as np
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