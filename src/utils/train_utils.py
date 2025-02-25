# File: src/utils/train_utils.py

import torch
import os
import open3d as o3d
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import your model
from src.models.bi_net import BiNet
# Import losses and supporting functions
from src.utils.losses import gradient_penalty, nnme_loss
from src.utils.emd.emd_module import emdModule  
import math

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

def log_training_config(
    logger,
    device,
    epochs,
    latent_dim,
    lambda_gp,
    lambda_nnme,
    lambda_rec,
    lr_enc,
    lr_dec,
    lr_disc,
    betas_enc,
    betas_dec,
    betas_disc,
    d_iters,
    g_iters,
    val_interval,
    emd_eps,
    emd_iters
):
    """
    Logs the training config in a neat block to either logger or stdout.
    """
    lines = [
        "===== TRAINING CONFIG =====",
        f"device           = {device}",
        f"epochs           = {epochs}",
        f"latent_dim       = {latent_dim}",
        f"lambda_gp        = {lambda_gp}",
        f"lambda_nnme      = {lambda_nnme}",
        f"lambda_rec       = {lambda_rec}",
        f"lr_enc           = {lr_enc}",
        f"lr_dec           = {lr_dec}",
        f"lr_disc          = {lr_disc}",
        f"betas_enc        = {betas_enc}",
        f"betas_dec        = {betas_dec}",
        f"betas_disc       = {betas_disc}",
        f"d_iters          = {d_iters}",
        f"g_iters          = {g_iters}",
        f"val_interval     = {val_interval}",
        f"emd_eps          = {emd_eps}",
        f"emd_iters        = {emd_iters}",
        "=========================="
    ]

    if logger is not None:
        for line in lines:
            logger.info(line)
    else:
        for line in lines:
            print(line)

def save_point_cloud_open3d(points, filename):
    """
    Saves a Nx3 point cloud to a .ply file using Open3D.
    
    Args:
        points: can be a Nx3 NumPy array or a PyTorch tensor
        filename (str): path for saving the .ply file
    """
    # If points is a torch tensor, convert it to NumPy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    # Construct an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Finally write the .ply file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to: {filename}")

def train_binet(
    model,
    train_loader,
    val_loader=None,
    device='cuda',
    epochs=50,
    latent_dim=128,
    lambda_gp=10.0,
    lambda_nnme=0.1,
    lambda_rec=1.0,
    lr_enc=1e-4,
    lr_dec=1e-4,
    lr_disc=1e-4,
    betas=(0.9, 0.99),
    d_iters=1,
    g_iters=1,
    logger=None,
    val_interval=1,
    emd_eps=0.002,
    emd_iters=50,
    log_interval=10,
    warmup_epochs=5
):
    """
    A typical collaborative AE+GAN training loop for BI-Net:
      - AE step => EMD reconstruction
      - Discriminator step => WGAN-GP
      - Generator step => WGAN + NNME
    """
    # >>> LOG TRAINING CONFIG <<<
    log_training_config(
        logger=logger,
        device=device,
        epochs=epochs,
        latent_dim=latent_dim,
        lambda_gp=lambda_gp,
        lambda_nnme=lambda_nnme,
        lambda_rec=lambda_rec,
        lr_enc=lr_enc,
        lr_dec=lr_dec,
        lr_disc=lr_disc,
        betas_enc=betas,
        betas_dec=betas,
        betas_disc=betas,
        d_iters=d_iters,
        g_iters=g_iters,
        val_interval=val_interval,
        emd_eps=emd_eps,
        emd_iters=emd_iters
    )

    model = model.to(device)
    emd_fn = emdModule()
    writer = SummaryWriter(log_dir="runs/binet_train")

    # 1) Param grouping by name-based approach
    encoder_params = []
    disc_params    = []
    shared_conv_params = []

    for name, p in model.backbone.named_parameters():
        if "disc_fc" in name:
            disc_params.append(p)
        elif "enc_fc" in name:
            encoder_params.append(p)
        else:
            # conv layers => share
            shared_conv_params.append(p)

    generator_params = list(model.generator.parameters())

    # (A) AE step updates: enc + gen + shared conv => AE
    ae_params = encoder_params + generator_params + shared_conv_params
    # (B) Disc step updates: disc fc + possibly shared conv => D
    d_params  = disc_params + shared_conv_params
    # (C) Gen step: gen + possibly shared conv => G
    g_params  = generator_params + shared_conv_params

    optimizer_ae = optim.Adam(ae_params, lr=lr_enc, betas=betas)
    optimizer_d  = optim.Adam(d_params,  lr=lr_disc, betas=betas)
    optimizer_g  = optim.Adam(g_params,  lr=lr_dec,  betas=betas)

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

            # ----- During warmup, skip disc & gen steps -----
            if epoch >= warmup_epochs:
                # Only run the following steps if we're past warmup

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

            else:
                # If we haven't passed warmup_epochs yet, define placeholders for logging
                # so code doesn't break
                d_loss = torch.tensor(0.0, device=device)
                g_loss = torch.tensor(0.0, device=device)
                d_grad_norm = 0.0
                g_grad_norm = 0.0

            global_step += 1

            # Example: saving shapes periodically
            if global_step % 500 == 0:
                # Generate a shape from random noise
                z = torch.randn(1, latent_dim, device=device)
                gen_points = model.generate(z)   # (1, 2048, 3)
                
                # Save them to disk
                output_dir = "samples_epoch"
                os.makedirs(output_dir, exist_ok=True)
                shape_fname = os.path.join(output_dir, f"epoch{epoch}_step{global_step}.ply")

                pc = gen_points[0]  # shape (2048,3)
                save_point_cloud_open3d(pc, shape_fname)
                print(f"Saved sample at {shape_fname}")

            # Logging
            if global_step % log_interval == 0:
                rec_min = rec_points.min().item()
                rec_max = rec_points.max().item()
                rec_var = rec_points.var().item()

                msg = (f"[Epoch {epoch}/{epochs}] Step {global_step} "
                       f"AE_loss: {ae_loss.item():.4f} | "
                       f"D_loss: {d_loss.item():.4f} | "
                       f"G_loss: {g_loss.item():.4f} || "
                       f"AE_Grad: {ae_grad_norm:.4f} | D_Grad: {d_grad_norm:.4f} | G_Grad: {g_grad_norm:.4f} || "
                       f"rec_min={rec_min:.3f}, rec_max={rec_max:.3f}, rec_var={rec_var:.3f}")
                if logger:
                    logger.info(msg)
                else:
                    print(msg)

                writer.add_scalar("Loss/AE_loss", ae_loss.item(), global_step)
                writer.add_scalar("Loss/D_loss", d_loss.item(), global_step)
                writer.add_scalar("Loss/G_loss", g_loss.item(), global_step)

        # Validation at epoch end
        if val_loader and (epoch + 1) % val_interval == 0:
            val_loss = validate_binet(model, val_loader, device, emd_eps, emd_iters)
            msg_val = f"Validation AE_loss at epoch {epoch}: {val_loss:.4f}"
            if logger:
                logger.info(msg_val)
            else:
                print(msg_val)
            writer.add_scalar("Loss/Val_AE_loss", val_loss, epoch)

    writer.close()
    return model



