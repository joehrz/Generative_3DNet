# File: src/utils/train_utils.py

import torch
import os
import open3d as o3d
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math

# Import the binet model
from src.models.bi_net import BiNet
# Import losses and supporting functions
from src.utils.losses import gradient_penalty, nnme_loss
from src.utils.emd.emd_module import emdModule  

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    return math.sqrt(total_norm)

def validate_binet(model, val_loader, device='cuda', emd_eps=0.002, emd_iters=50):
    """
    Evaluate AE reconstruction using EMD
    """
    model.eval()
    emd_fn = emdModule()
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for real_points in val_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            latent = model.encode(real_points)
            rec_points = model.decode(latent)

            emd_val, _ = emd_fn(rec_points, real_points, emd_eps, emd_iters)
            emd_mean = emd_val.mean().item()

            total_loss += emd_mean * B
            total_samples += B

    model.train()
    return total_loss / max(total_samples, 1)

def log_training_config(logger, config):
    """
    Logs the training config in a neat block to either logger or stdout.
    """
    lines = [
        "===== TRAINING CONFIG =====",
        f"device           = {config.training.get('device', 'cuda')}",
        f"epochs           = {config.training.epochs}",
        f"latent_dim       = {config.model.latent_dim}",
        f"lambda_gp        = {config.model.lambda_gp}",
        f"lambda_nnme      = {config.model.lambda_nnme}",
        f"lambda_rec       = {config.training.get('lambda_rec', 1.0)}",
        f"lr_enc           = {config.training.get('lr_enc', 1e-4)}",
        f"lr_dec           = {config.training.get('lr_dec', 1e-4)}",
        f"lr_disc          = {config.training.get('lr_disc', 1e-4)}",
        f"betas            = {config.training.betas}",
        f"d_iters          = {config.training.get('d_iters', 1)}",
        f"g_iters          = {config.training.get('g_iters', 1)}",
        f"val_interval     = {config.training.get('val_interval', 1)}",
        f"emd_eps          = {config.training.get('emd_eps', 0.002)}",
        f"emd_iters        = {config.training.get('emd_iters', 50)}",
        "=========================="
    ]

    for line in lines:
        logger.info(line)

def save_point_cloud_open3d(points, filename):
    """
    Saves a Nx3 point cloud to a .ply file using Open3D.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def train_binet(
    model,
    train_loader,
    config,
    val_loader=None,
    logger=None,
):
    """
    A typical collaborative AE+GAN training loop for BI-Net.
    """
    device = config.training.get('device', 'cuda')
    log_training_config(logger=logger, config=config)

    model = model.to(device)
    emd_fn = emdModule()
    writer = SummaryWriter(log_dir="runs/binet_train")

    # Get hyperparameters from config
    epochs = config.training.epochs
    latent_dim = config.model.latent_dim
    lambda_gp = config.model.lambda_gp
    lambda_nnme = config.model.lambda_nnme
    lambda_rec = config.training.get('lambda_rec', 1.0)
    lr_enc = config.training.get('lr_enc', 1e-4)
    lr_dec = config.training.get('lr_dec', 1e-4)
    lr_disc = config.training.get('lr_disc', 1e-4)
    betas = tuple(config.training.betas)
    d_iters = config.training.get('d_iters', 1)
    g_iters = config.training.get('g_iters', 1)
    val_interval = config.training.get('val_interval', 1)
    emd_eps = config.training.get('emd_eps', 0.002)
    emd_iters = config.training.get('emd_iters', 50)
    log_interval = config.training.get('log_interval', 10)
    warmup_epochs = config.training.get('warmup_epochs', 5)
    mesh_log_interval = config.training.get('tensorboard_mesh_log_interval', 500)

    # 1) Get parameter groups from the model
    enc_fc_params, shared_conv_params = model.get_encoder_params()
    gen_params = model.get_generator_params()
    disc_fc_params, _ = model.get_discriminator_params() # Shared params are the same

    # (A) AE step optimizer
    optimizer_ae = optim.Adam(enc_fc_params + shared_conv_params + gen_params, lr=lr_enc, betas=betas)
    # (B) Discriminator step optimizer
    optimizer_d = optim.Adam(disc_fc_params + shared_conv_params, lr=lr_disc, betas=betas)
    # (C) Generator step optimizer
    optimizer_g = optim.Adam(gen_params, lr=lr_dec, betas=betas)

    logger.info(f"Optimizer AE: Adam, lr={lr_enc}, betas={betas}")
    logger.info(f"Optimizer D: Adam, lr={lr_disc}, betas={betas}")
    logger.info(f"Optimizer G: Adam, lr={lr_dec}, betas={betas}")

    global_step = 0

    for epoch in range(epochs):
        model.train()
        for real_points in train_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            # ----- (1) AE step (always) -----
            optimizer_ae.zero_grad()
            latent = model.encode(real_points)
            rec_points = model.decode(latent)
            emd_val, _ = emd_fn(rec_points, real_points, emd_eps, emd_iters)
            ae_loss = lambda_rec * emd_val.mean()
            ae_loss.backward()
            ae_grad_norm = compute_gradient_norm(model)
            optimizer_ae.step()

            d_loss = torch.tensor(0.0, device=device)
            g_loss = torch.tensor(0.0, device=device)
            d_grad_norm = 0.0
            g_grad_norm = 0.0
            
            if epoch >= warmup_epochs:
                # ----- (2) Discriminator step -----
                for _ in range(d_iters):
                    optimizer_d.zero_grad()
                    z = torch.randn(B, latent_dim, device=device)
                    with torch.no_grad():
                        fake_points = model.generate(z)
                    d_real = model.discriminate(real_points)
                    d_fake = model.discriminate(fake_points)
                    gp = gradient_penalty(model.discriminate, real_points, fake_points, device=device)
                    d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
                    d_loss.backward()
                    d_grad_norm = compute_gradient_norm(model)
                    optimizer_d.step()

                # ----- (3) Generator step -----
                for _ in range(g_iters):
                    optimizer_g.zero_grad()
                    z = torch.randn(B, latent_dim, device=device)
                    fake_points = model.generate(z)
                    d_fake_for_g = model.discriminate(fake_points)
                    adv_loss = -d_fake_for_g.mean()
                    unif_loss = nnme_loss(fake_points, sample_fraction=0.1)
                    g_loss = adv_loss + lambda_nnme * unif_loss
                    g_loss.backward()
                    g_grad_norm = compute_gradient_norm(model)
                    optimizer_g.step()

            global_step += 1

            if global_step % log_interval == 0:
                msg = (f"[Epoch {epoch}/{epochs}] Step {global_step} | "
                       f"AE_loss: {ae_loss.item():.4f} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | "
                       f"AE_Grad: {ae_grad_norm:.4f} | D_Grad: {d_grad_norm:.4f} | G_Grad: {g_grad_norm:.4f}")
                logger.info(msg)
                writer.add_scalar("Loss/AE_loss", ae_loss.item(), global_step)
                writer.add_scalar("Loss/D_loss", d_loss.item(), global_step)
                writer.add_scalar("Loss/G_loss", g_loss.item(), global_step)
                writer.add_scalar("Gradients/AE_GradNorm", ae_grad_norm, global_step)
                writer.add_scalar("Gradients/D_GradNorm", d_grad_norm, global_step)
                writer.add_scalar("Gradients/G_GradNorm", g_grad_norm, global_step)

            if global_step % mesh_log_interval == 0:
                model.eval()
                with torch.no_grad():
                    writer.add_mesh("Real_Points_Batch", vertices=real_points.cpu(), global_step=global_step)
                    writer.add_mesh("Reconstructed_Points_Batch", vertices=rec_points.cpu(), global_step=global_step)
                    z_sample = torch.randn(min(B, 4), latent_dim, device=device)
                    generated_sample = model.generate(z_sample)
                    writer.add_mesh("Generated_Points_Sample_Batch", vertices=generated_sample.cpu(), global_step=global_step)
                model.train()

        if val_loader and (epoch + 1) % val_interval == 0:
            val_loss = validate_binet(model, val_loader, device, emd_eps, emd_iters)
            logger.info(f"Validation AE_loss at epoch {epoch}: {val_loss:.4f}")
            writer.add_scalar("Loss/Val_AE_loss", val_loss, epoch)

    writer.close()
    return model
