import torch

def emd_loss(pred_points, gt_points):
    """
    Placeholder for Earth Mover's Distance or Chamfer Distance.

    For demonstration, let's do naive MSE:
    """
    return torch.mean((pred_points - gt_points) ** 2)


def chamfer_distance(pc1, pc2):
    """Compute Chamfer Distance between two point clouds."""
    B, N, _ = pc1.size()
    M = pc2.size(1)

    pc1_expand = pc1.unsqueeze(2).expand(B, N, M, 3)
    pc2_expand = pc2.unsqueeze(1).expand(B, N, M, 3)

    dist = torch.norm(pc1_expand - pc2_expand, dim=3)  # [B, N, M]

    min_dist_pc1, _ = torch.min(dist, dim=2)  # [B, N]
    min_dist_pc2, _ = torch.min(dist, dim=1)  # [B, M]

    loss = torch.mean(min_dist_pc1) + torch.mean(min_dist_pc2)
    return loss