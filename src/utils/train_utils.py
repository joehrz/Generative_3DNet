# File: src/utils/train_utils.py

import torch
import torch.optim as optim
from .losses import chamfer_distance, gradient_penalty, nnme_loss  # your existing code
# We'll assume chamfer_distance still exist if needed, but we use EMD for AE

def train_binet(
    binet,
    data_loader,
    val_loader=None,
    device='cuda',
    epochs=50,
    latent_dim=128,          # match paper
    lambda_gp=10.0,
    lambda_nnme=0.1,         # or bigger e.g. 1000 if the paper says so
    lr_enc=2e-4,             # per paper
    lr_dec=2e-4,
    lr_disc=2e-4,
    betas_enc=(0.0, 0.99),   # as suggested by the paper
    betas_dec=(0.0, 0.99),
    betas_disc=(0.0, 0.99),
    d_iters=1,
    g_iters=1,
    logger=None,
    val_interval=1
):
    """
    BI-Net training:
      - EMD for AE (forward direction)
      - WGAN-GP for reverse direction
      - NNME for uniform
    """
    binet.to(device)

    # Because we split up, we have distinct param sets
    ae_enc_params = list(binet.encoder_ae.parameters())      # AE encoder
    dec_gen_params = list(binet.decoder_gen.parameters())    # G
    disc_params    = list(binet.discriminator_mlp.parameters())

    # Create separate optimizers
    optimizer_enc = optim.Adam(ae_enc_params, lr=lr_enc, betas=betas_enc)
    optimizer_dec = optim.Adam(dec_gen_params, lr=lr_dec, betas=betas_dec)
    optimizer_disc = optim.Adam(disc_params, lr=lr_disc, betas=betas_disc)

    global_step = 0
    for epoch in range(epochs):
        binet.train()

        for real_points in data_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            # ============ 1) AE direction =============
            # real -> encode -> decode -> EMD
            latent = binet.encode(real_points)
            rec_points = binet.decode(latent)  # shape (B,N,3)

            #ae_loss = emd_loss(rec_points, real_points)
            ae_loss = chamfer_distance(rec_points, real_points)
            

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            ae_loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            # ============ 2) GAN direction =============
            # 2.1) Train Discriminator
            for _ in range(d_iters):
                optimizer_disc.zero_grad()
                z = torch.randn(B, latent_dim, device=device)
                with torch.no_grad():
                    fake_points = binet.generate(z)

                d_real = binet.discriminate(real_points)
                d_fake = binet.discriminate(fake_points)

                gp = gradient_penalty(binet.discriminate, real_points, fake_points, device=device)
                disc_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp

                disc_loss.backward()
                optimizer_disc.step()

            # 2.2) Train Generator
            for _ in range(g_iters):
                optimizer_dec.zero_grad()

                z = torch.randn(B, latent_dim, device=device)
                fake_points = binet.generate(z)
                d_fake_for_g = binet.discriminate(fake_points)
                wgan_loss = -d_fake_for_g.mean()

                unif = nnme_loss(fake_points)  # if you have it
                g_loss = wgan_loss + lambda_nnme * unif

                g_loss.backward()
                optimizer_dec.step()

            global_step += 1
            if global_step % 10 == 0:
                msg = (f"[Epoch {epoch}/{epochs}] [Step {global_step}] "
                       f"AE_loss(EMD): {ae_loss.item():.4f} | "
                       f"D_loss: {disc_loss.item():.4f} | "
                       f"G_loss: {g_loss.item():.4f} | "
                       f"Unif: {unif.item():.4f}")
                print(msg)

        # ============ Validation =============
        # We'll just do EMD on a smaller subset or do a quick check
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            binet.eval()
            total_emd = 0.0
            total_count = 0
            with torch.no_grad():
                for val_points in val_loader:
                    val_points = val_points.to(device)
                    val_B = val_points.size(0)

                    lat = binet.encode(val_points)
                    rec = binet.decode(lat)
                    e = chamfer_distance(rec, val_points).item()
                    total_emd += e * val_B
                    total_count += val_B
            avg_emd = total_emd / max(total_count,1)
            print(f"[Epoch {epoch}/{epochs}] Validation => EMD: {avg_emd:.4f}")

    return binet


