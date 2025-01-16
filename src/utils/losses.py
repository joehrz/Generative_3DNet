import torch
import torch.nn.functional as f


def emd_loss(pred_points, gt_points):
    """
    Placeholder for real Earth Mover's Distance or Chamfer Distance.
    For demonstration, use MSE.
    """
    return torch.mean((pred_points - gt_points)**2)



def gradient_penalty(discriminator, real_points, fake_points, device='cuda'):
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
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1)**2).mean()
    
    return gp

def nnme_loss(point_clouds, sample_fraction=0.1):
    """
    A partial/optimized approach: random subset for NN queries, 
    to reduce computational cost.
    """
    device = point_clouds.device
    B, N, _ = point_clouds.shape
    subset_size = int(N * sample_fraction)

    losses = []
    for b in range(B):
        shape = point_clouds[b]
        # random subset
        idx = torch.randperm(N)[:subset_size]
        sub_shape = shape[idx]  # (subset_size, 3)
        
        dist_matrix = torch.cdist(sub_shape, sub_shape, p=2)  # (subset_size, subset_size)
        # We only consider the minimum nonzero distance for each point
        dist_matrix.fill_diagonal_(1e6)
        min_dists = torch.min(dist_matrix, dim=1)[0]
        
        # variance of min distances
        variance = torch.var(min_dists)
        losses.append(variance)
    if losses:
        return torch.mean(torch.stack(losses))
    else:
        return torch.tensor(0.0, device=device)