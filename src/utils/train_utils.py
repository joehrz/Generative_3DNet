# File: src/utils/train_utils.py

import torch
import torch.optim as optim
# Keep emd_loss imported if you still want to measure it in evaluate_on_loader
from .losses import chamfer_distance, emd_loss, gradient_penalty, nnme_loss

def evaluate_on_loader_chamfer(model, data_loader, device):
    """
    Evaluate model reconstruction using only Chamfer distance.
    Returns avg Chamfer.
    """
    model.eval()
    total_chamfer = 0.0
    count = 0

    with torch.no_grad():
        for real_points in data_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            # Forward pass
            latent_code = model.encode(real_points)
            rec_points  = model.decode(latent_code)

            # Only Chamfer in validation
            chamfer_val = chamfer_distance(rec_points, real_points).item()
            total_chamfer += chamfer_val * B

            count += B

    avg_chamfer = total_chamfer / max(count, 1)
    return avg_chamfer

def evaluate_on_loader_emd_chamfer(model, data_loader, device):
    """
    Evaluate model reconstruction with both EMD and Chamfer.
    Typically used only on test set or final evaluation due to EMD cost.
    """
    model.eval()
    total_emd = 0.0
    total_chamfer = 0.0
    count = 0

    with torch.no_grad():
        for real_points in data_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            latent_code = model.encode(real_points)
            rec_points  = model.decode(latent_code)

            emd_val = emd_loss(rec_points, real_points).item()
            chamfer_val = chamfer_distance(rec_points, real_points).item()

            total_emd += emd_val * B
            total_chamfer += chamfer_val * B

            count += B

    avg_emd = total_emd / max(count, 1)
    avg_chamfer = total_chamfer / max(count, 1)
    return avg_emd, avg_chamfer

def train_binet(
    binet,
    data_loader,
    val_loader=None,
    device='cuda',
    epochs=50,
    latent_dim=96,
    lambda_gp=10.0,
    lambda_nnme=0.1,
    lr_enc=1e-4,
    lr_dec=1e-4,
    lr_disc=5e-5,
    betas_enc=(0.9, 0.999),
    betas_dec=(0.9, 0.999),
    betas_disc=(0.5, 0.9),
    d_iters=1,
    g_iters=1,
    logger=None,
    val_interval=1
):
    """
    Trains BI-Net using a combined auto-encoder + WGAN-GP approach,
    but uses Chamfer distance for the auto-encoder reconstruction loss.

    If `val_loader` is provided, we do an inline validation pass at the end
    of each epoch (or every 'val_interval' epochs).
    """

    binet.to(device)

    # Separate parameters for encoder/discriminator (EnDi) vs. decoder/generator (DeGe)
    enc_params = list(binet.EnDi.parameters()) 
    dec_params = list(binet.DeGe.parameters())  

    # Create separate optimizers
    optimizer_enc = optim.Adam(enc_params, lr=lr_enc, betas=betas_enc)
    optimizer_dec = optim.Adam(dec_params, lr=lr_dec, betas=betas_dec)
    optimizer_disc = optim.Adam(enc_params, lr=lr_disc, betas=betas_disc)

    global_step = 0
    for epoch in range(epochs):
        binet.train()  # put model in training mode

        for real_points in data_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            ########################################################
            # 1) AE direction (Encoder + Decoder) using Chamfer Loss
            ########################################################
            latent = binet.encode(real_points)
            rec_points = binet.decode(latent)

            # Use chamfer_distance instead of emd_loss
            ae_loss = chamfer_distance(rec_points, real_points)

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            ae_loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            ########################################################
            # 2) GAN direction (Discriminator + Generator)
            ########################################################

            # 2.1) Train Discriminator
            for _ in range(d_iters):
                optimizer_disc.zero_grad()

                # sample noise z
                z = torch.randn(B, latent_dim, device=device)
                with torch.no_grad():
                    fake_points = binet.generate(z)

                d_real = binet.discriminate(real_points)
                d_fake = binet.discriminate(fake_points)

                gp = gradient_penalty(
                    binet.discriminate,
                    real_points,
                    fake_points,
                    device=device
                )
                disc_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp

                disc_loss.backward()
                optimizer_disc.step()

            # 2.2) Train Generator (Decoder)
            for _ in range(g_iters):
                optimizer_dec.zero_grad()

                z = torch.randn(B, latent_dim, device=device)
                fake_points = binet.generate(z)
                d_fake_for_g = binet.discriminate(fake_points)
                wgan_loss = -d_fake_for_g.mean()

                # NNME for uniform distribution
                unif_loss = nnme_loss(fake_points)
                g_loss = wgan_loss + lambda_nnme * unif_loss

                g_loss.backward()
                optimizer_dec.step()

            global_step += 1

            # Logging or printing
            if global_step % 10 == 0:
                msg = (f"[Epoch {epoch}/{epochs}] [Step {global_step}] "
                       f"AE_loss(Chamfer): {ae_loss.item():.4f} | "
                       f"D_loss: {disc_loss.item():.4f} | "
                       f"G_loss: {g_loss.item():.4f} | "
                       f"Unif: {unif_loss.item():.4f}")


                print(msg)

        # ---------------------------
        # End of an epoch - do validation
        # ---------------------------
        # Validation with CHAMFER ONLY
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            binet.eval()
            val_chamfer = evaluate_on_loader_chamfer(binet, val_loader, device)
            msg_val = f"[Epoch {epoch}/{epochs}] Validation => Chamfer: {val_chamfer:.4f}"
            print(msg_val)

    return binet

