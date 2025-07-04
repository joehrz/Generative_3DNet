# File: src/utils/train_utils.py

import torch
import os
import open3d as o3d
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math

from src.models.bi_net import BiNet
from src.utils.losses import gradient_penalty, nnme_loss
from src.utils.emd.emd_module import emdModule

def compute_gradient_norm(parameters):
    """
    Computes the L2 norm of the gradients for a given iterable of parameters.
    
    This is useful for monitoring training stability, specifically to detect
    exploding or vanishing gradients for a specific optimizer's parameters.
    
    Args:
        parameters (iterable): An iterable of torch.Tensor objects, typically from
                               an optimizer's parameter groups.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)

def validate_binet(model, val_loader, device, emd_eps, emd_iters):
    """
    Evaluates the model's autoencoder reconstruction performance on a validation set.
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
            total_loss += emd_val.mean().item() * B
            total_samples += B

    model.train()
    return total_loss / max(total_samples, 1)

def log_training_config(logger, config):
    """Logs the key training parameters for experiment tracking."""
    lines = [
        "===== TRAINING CONFIG =====",
        f"device           = {config.training.device}",
        f"epochs           = {config.training.epochs}",
        f"latent_dim       = {config.model.latent_dim}",
        f"lambda_gp        = {config.model.lambda_gp}",
        f"lambda_nnme      = {config.model.lambda_nnme}",
        f"lambda_rec       = {config.training.lambda_rec}",
        f"lr_enc/dec/disc  = {config.training.lr_enc}",
        f"betas            = {config.training.betas}",
        "=========================="
    ]
    for line in lines:
        logger.info(line)

def train_binet(model, train_loader, config, val_loader, logger):
    """
    The core training loop for BI-Net, implementing the collaborative
    Autoencoder and GAN training strategy.
    """
    device = config.training.device
    log_training_config(logger=logger, config=config)

    model = model.to(device)
    emd_fn = emdModule()
    writer = SummaryWriter(log_dir="runs/binet_train")

    # --- Unpack Hyperparameters ---
    epochs = config.training.epochs
    latent_dim = config.model.latent_dim
    lambda_gp, lambda_nnme, lambda_rec = config.model.lambda_gp, config.model.lambda_nnme, config.training.lambda_rec
    lr_enc, lr_dec, lr_disc = config.training.lr_enc, config.training.lr_dec, config.training.lr_disc
    betas = tuple(config.training.betas)
    d_iters, g_iters = config.training.d_iters, config.training.g_iters
    val_interval, log_interval, warmup_epochs = config.training.val_interval, config.training.log_interval, config.training.warmup_epochs
    emd_eps, emd_iters = config.training.emd_eps, config.training.emd_iters
    mesh_log_interval = config.training.tensorboard_mesh_log_interval

    # --- Setup Optimizers ---
    enc_fc_params, shared_conv_params = model.get_encoder_params()
    gen_params = model.get_generator_params()
    disc_fc_params, _ = model.get_discriminator_params()

    optimizer_ae = optim.Adam(enc_fc_params + shared_conv_params + gen_params, lr=lr_enc, betas=betas)
    optimizer_d = optim.Adam(disc_fc_params + shared_conv_params, lr=lr_disc, betas=betas)
    optimizer_g = optim.Adam(gen_params, lr=lr_dec, betas=betas)

    logger.info(f"Optimizers configured. AE lr: {lr_enc}, D lr: {lr_disc}, G lr: {lr_dec}")

    global_step = 0
    for epoch in range(epochs):
        model.train()
        for i, real_points in enumerate(train_loader):
            real_points = real_points.to(device)
            B = real_points.size(0)

            # === Step 1: Autoencoder Training (Reconstruction) ===
            optimizer_ae.zero_grad()
            latent = model.encode(real_points)
            rec_points = model.decode(latent)
            emd_val, _ = emd_fn(rec_points, real_points, emd_eps, emd_iters)
            ae_loss = lambda_rec * emd_val.mean()
            ae_loss.backward()
            optimizer_ae.step()

            d_loss, g_loss = torch.tensor(0.0), torch.tensor(0.0)
            
            # === Steps 2 & 3: GAN Training (after warmup) ===
            if epoch >= warmup_epochs:
                # --- Step 2: Discriminator Training ---
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
                    optimizer_d.step()

                # --- Step 3: Generator Adversarial Training ---
                for _ in range(g_iters):
                    optimizer_g.zero_grad()
                    z = torch.randn(B, latent_dim, device=device)
                    fake_points = model.generate(z)
                    d_fake_for_g = model.discriminate(fake_points)
                    
                    adv_loss = -d_fake_for_g.mean()
                    unif_loss = nnme_loss(fake_points, sample_fraction=0.1)
                    g_loss = adv_loss + lambda_nnme * unif_loss
                    g_loss.backward()
                    optimizer_g.step()

            global_step += 1

            # --- Logging ---
            if global_step % log_interval == 0:
                ae_grad_norm = compute_gradient_norm(optimizer_ae.param_groups[0]['params'])
                d_grad_norm = compute_gradient_norm(optimizer_d.param_groups[0]['params']) if epoch >= warmup_epochs else 0
                g_grad_norm = compute_gradient_norm(optimizer_g.param_groups[0]['params']) if epoch >= warmup_epochs else 0
                
                logger.info(f"[Epoch {epoch+1}/{epochs}, Step {global_step}] "
                            f"AE Loss: {ae_loss.item():.4f} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | "
                            f"Grads(AE/D/G): {ae_grad_norm:.2f}/{d_grad_norm:.2f}/{g_grad_norm:.2f}")

                writer.add_scalar("Loss/AE_Reconstruction", ae_loss.item(), global_step)
                writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
                writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
                writer.add_scalar("Gradients/Autoencoder", ae_grad_norm, global_step)
                writer.add_scalar("Gradients/Discriminator", d_grad_norm, global_step)
                writer.add_scalar("Gradients/Generator", g_grad_norm, global_step)

            # Log point cloud meshes to TensorBoard periodically.
            if global_step % mesh_log_interval == 0 and B > 0:
                model.eval()
                with torch.no_grad():
                    writer.add_mesh("Batch/Real_Points", vertices=real_points.cpu(), global_step=global_step)
                    writer.add_mesh("Batch/Reconstructed_Points", vertices=rec_points.cpu(), global_step=global_step)
                    z_sample = torch.randn(min(B, 4), latent_dim, device=device)
                    generated_sample = model.generate(z_sample)
                    writer.add_mesh("Batch/Generated_Points", vertices=generated_sample.cpu(), global_step=global_step)
                model.train()

        # --- End of Epoch Validation ---
        if val_loader and (epoch + 1) % val_interval == 0:
            val_loss = validate_binet(model, val_loader, device, emd_eps, emd_iters)
            logger.info(f"--- Validation at Epoch {epoch+1}/{epochs} -> EMD Loss: {val_loss:.4f} ---")
            writer.add_scalar("Loss/Validation_EMD", val_loss, epoch)

    writer.close()
    return model

