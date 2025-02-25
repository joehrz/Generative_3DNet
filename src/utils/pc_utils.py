# File: src/utils/pc_utils.py

import torch
import math
import random

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

