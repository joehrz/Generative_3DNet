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
    g_iters=1
):
    binet.to(device)

    # Separate parameters
    enc_params = list(binet.EnDi.parameters())  # In practice, might separate encoder vs. disc 
    dec_params = list(binet.DeGe.parameters())
    # For clarity, let's differentiate between using EnDi as encoder vs. disc:
    #   In a more advanced setup, you'd define separate param groups 
    #   or even separate copies of the same network. This is simplified.

    # We will treat them separately:
    optimizer_enc = optim.Adam(enc_params, lr=lr_enc, betas=betas_enc)
    optimizer_dec = optim.Adam(dec_params, lr=lr_dec, betas=betas_dec)
    optimizer_disc = optim.Adam(enc_params, lr=lr_disc, betas=betas_disc)

    step = 0
    for epoch in range(epochs):
        for real_points in data_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            ##################################################################
            # 1) AE direction
            ##################################################################
            # Encode -> decode -> reconstruct
            latent = binet.encode(real_points)
            rec_points = binet.decode(latent)

            ae_loss = emd_loss(rec_points, real_points)

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            ae_loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            ##################################################################
            # 2) GAN direction
            ##################################################################
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

            # 2.2) Train Generator (Decoder)
            for _ in range(g_iters):
                optimizer_dec.zero_grad()

                z = torch.randn(B, latent_dim, device=device)
                fake_points = binet.generate(z)
                d_fake_for_g = binet.discriminate(fake_points)
                wgan_loss = -d_fake_for_g.mean()

                # NNME
                unif_loss = nnme_loss(fake_points)

                g_loss = wgan_loss + lambda_nnme * unif_loss

                g_loss.backward()
                optimizer_dec.step()

            step += 1
            if step % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Step {step}] "
                      f"AE_loss: {ae_loss.item():.4f} | "
                      f"D_loss: {disc_loss.item():.4f} | "
                      f"G_loss: {g_loss.item():.4f} | "
                      f"Unif: {unif_loss.item():.4f}")

    return binet

