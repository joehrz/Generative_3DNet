# File: src/utils/losses.py

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # (can be removed if no longer used)
from src.utils.emd.emd_module import emdModule

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

# Optionally, older emd_loss function based on the Hungarian algorithm:
def emd_loss(pc1, pc2):
    """
    Compute the Earth Mover's Distance (EMD) between two point clouds
    pc1 and pc2 of shape [B, N, D].
    Uses SciPy's linear_sum_assignment for each batch sample.
    Returns a scalar tensor (float) with the average EMD over the batch.
    """
    assert pc1.dim() == 3 and pc2.dim() == 3, "Point clouds must be [B, N, D]"
    B, N, D = pc1.shape
    emd_total = 0.0

    for b in range(B):
        dist_matrix = torch.cdist(pc1[b].unsqueeze(0), pc2[b].unsqueeze(0), p=2)[0] ** 2
        cost_matrix = dist_matrix.detach().cpu().numpy()

        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        emd_b = cost_matrix[row_idx, col_idx].sum() / N
        emd_total += emd_b

    emd_avg = emd_total / B
    return torch.tensor(emd_avg, dtype=torch.float32, device=pc1.device)

def evaluate_on_loader_emd_chamfer(model, data_loader, device, emd_eps=0.002, emd_iters=50):
    """
    Evaluate the model reconstruction using both the new EMD loss (via the EMD module)
    and the Chamfer Distance.
    This function loops over the data loader, computes the reconstruction for each batch,
    and then calculates the average EMD and Chamfer losses.
    """
    model.eval()
    total_emd = 0.0
    total_chamfer = 0.0
    count = 0

    # Instantiate the new EMD loss module once.
    emd_loss_fn = emdModule()

    with torch.no_grad():
        for real_points in data_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)
            latent = model.encode(real_points)
            rec = model.decode(latent)

            # Compute new EMD loss for the batch.
            emd_loss_tensor, _ = emd_loss_fn(rec, real_points, emd_eps, emd_iters)
            emd_val = emd_loss_tensor.mean().item()

            # Compute Chamfer Distance loss for the batch.
            chamfer_val = chamfer_distance(rec, real_points).item()

            total_emd += emd_val * B
            total_chamfer += chamfer_val * B
            count += B

    avg_emd = total_emd / max(count, 1)
    avg_chamfer = total_chamfer / max(count, 1)
    return avg_emd, avg_chamfer

def gradient_penalty(discriminator, real_points, fake_points, device='cuda'):
    """
    Computes the gradient penalty for WGAN-GP.
    
    WGAN-GP introduces a gradient penalty term that penalizes the deviation of the
    gradient norm of the discriminator from 1. This helps stabilize training and
    enforce the Lipschitz constraint required by the Wasserstein distance.
    
    Args:
        discriminator (nn.Module): The discriminator (critic) network used in WGAN-GP.
        real_points (torch.Tensor): A batch of real data points of shape (B, N, 3) 
            or (B, 3, N), depending on how the points are fed into the network.
        fake_points (torch.Tensor): A batch of generated (fake) data points 
            of the same shape as `real_points`.
        device (str, optional): The device on which the computations will be performed.
            Defaults to 'cuda'.
    
    Returns:
        torch.Tensor: A scalar tensor representing the gradient penalty term.
    
    Example:
        gp = gradient_penalty(discriminator, real_batch, fake_batch, device='cuda')
        d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    """
    alpha = torch.rand(real_points.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real_points)
    interpolates = alpha * real_points + (1 - alpha) * fake_points
    interpolates.requires_grad_(True)

    disc_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(disc_interpolates)

    grads = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.contiguous().view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def nnme_loss(point_clouds, sample_fraction=0.1):
    """
    Compute a loss that encourages uniformity in the point cloud by evaluating
    the variance of the minimum pairwise distances from a random subset.
    """
    device = point_clouds.device
    B, N, _ = point_clouds.shape
    subset_size = int(N * sample_fraction)

    losses = []
    for b in range(B):
        shape = point_clouds[b]
        idx = torch.randperm(N)[:subset_size]
        sub_shape = shape[idx]  # (subset_size, 3)
        
        dist_matrix = torch.cdist(sub_shape, sub_shape, p=2)  # (subset_size, subset_size)
        dist_clone = dist_matrix.clone()
        dist_clone.fill_diagonal_(1e6)
        min_dists = torch.min(dist_clone, dim=1)[0]
        variance = torch.var(min_dists)
        losses.append(variance)

    if losses:
        return torch.mean(torch.stack(losses))
    else:
        return torch.tensor(0.0, device=device)
