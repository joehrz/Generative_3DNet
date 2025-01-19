# File: src/utils/train_utils.py

import torch
import torch.optim as optim
from .losses import emd_loss, gradient_penalty, nnme_loss

def train_binet(
    binet,
    data_loader,
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
    logger=None
):
    """
    Trains BI-Net using a combined auto-encoder + WGAN-GP approach.

    Steps:
      1) AE direction (encoder + decoder):
         - Reconstruct real_points via binet.encode() & binet.decode()
         - Update encoder (enc) & decoder (dec) optimizers.
      2) GAN direction (discriminator + generator):
         - Train discriminator (disc) for 'd_iters' steps:
            * Use real_points & generated fake_points
            * Add gradient penalty
         - Train generator/decoder for 'g_iters' steps:
            * Minimizes WGAN loss + optional NNME (uniformity)
    
    Args:
        binet (nn.Module): BI-Net model with EnDi & DeGe modules
        data_loader (DataLoader): yields batches of real_points
        device (str): 'cuda' or 'cpu'
        epochs (int): total epochs
        latent_dim (int): dimension of latent code for random z
        lambda_gp (float): gradient penalty weight
        lambda_nnme (float): nearest neighbor mutual exclusion weight
        lr_enc/lr_dec/lr_disc (float): learning rates for enc/dec/disc
        betas_enc/betas_dec/betas_disc (tuple): Adam betas for enc/dec/disc
        d_iters, g_iters (int): how many discriminator/generator steps per iteration
        logger: optional logging object (logger.info(...)); if None, fallback to print

    Returns:
        binet (nn.Module): trained BI-Net model
    """

    binet.to(device)

    # Separate parameters for encoder vs. decoder vs. disc
    # In BI-Net, "EnDi" is shared for encoder + disc, so we do a simplistic approach
    enc_params = list(binet.EnDi.parameters())  # encoder or discriminator
    dec_params = list(binet.DeGe.parameters())  # decoder/generator

    optimizer_enc = optim.Adam(enc_params, lr=lr_enc, betas=betas_enc)
    optimizer_dec = optim.Adam(dec_params, lr=lr_dec, betas=betas_dec)
    optimizer_disc = optim.Adam(enc_params, lr=lr_disc, betas=betas_disc)

    global_step = 0
    for epoch in range(epoch):
        for real_points in data_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            ########################################################
            # 1) AE direction (Encoder + Decoder)
            ########################################################
            latent = binet.encode(real_points)
            rec_points = binet.decode(latent)

            ae_loss = emd_loss(rec_points, real_points)

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
                    discriminator=binet.discriminate,
                    real_data=real_points,
                    fake_data=fake_points,
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
                       f"AE_loss: {ae_loss.item():.4f} | "
                       f"D_loss: {disc_loss.item():.4f} | "
                       f"G_loss: {g_loss.item():.4f} | "
                       f"Unif: {unif_loss.item():.4f}")

                logger.info(msg)


    return binet