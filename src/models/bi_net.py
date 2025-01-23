# File: src/models/bi_net.py

import torch
import torch.nn as nn

from .encoder_conv_ae import EncoderConvAE
from .discriminator_mlp import DiscriminatorMLP
from .TreeGCN_decoder_generator import TreeGCNDecoderGenerator

class BiNet(nn.Module):
    """
    Bidirectional Network (BI-Net):
      - AE direction uses: EncoderConvAE + TreeGCNDecoderGenerator
      - GAN direction uses: TreeGCNDecoderGenerator (generator) + DiscriminatorMLP
    """
    def __init__(
        self,
        batch_size=16,
        ae_enc_feat=[3, 64, 128, 256, 512],
        latent_dim=128,
        disc_hidden=[256, 128],
        features_g=[128, 256, 256, 256, 128, 128, 128, 3],
        degrees=[1, 2, 2, 2, 2, 2, 64],
        support=10
    ):
        super(BiNet, self).__init__()
        self.batch_size = batch_size

        # AE encoder (conv-based)
        self.encoder_ae = EncoderConvAE(
            latent_dim=latent_dim,
            d_feat=ae_enc_feat
        )

        # Generator / Decoder (TreeGCN)
        self.decoder_gen = TreeGCNDecoderGenerator(
            batch_size=self.batch_size,  # pass explicitly
            features=features_g,
            degrees=degrees,
            support=support
        )

        # Discriminator MLP
        self.discriminator_mlp = DiscriminatorMLP(
            hidden_dims=disc_hidden,
            pooling='max'
        )

    # ============== AE direction ==============
    def encode(self, real_points):
        # real_points => shape (B, N, 3)
        return self.encoder_ae(real_points)

    def decode(self, latent_code):
        # latent_code => shape (B, latent_dim)
        # or (B, 1, latent_dim). We'll unify to (B, 1, latent_dim).
        if latent_code.dim() == 2:
            latent_code = latent_code.unsqueeze(1)  # => (B, 1, latent_dim)

        tree = [latent_code]
        rec_points = self.decoder_gen(tree)  # => (B, final_num_points, 3)
        return rec_points

    # ============== GAN direction ==============
    def generate(self, noise):
        # noise => (B, latent_dim)
        if noise.dim() == 2:
            noise = noise.unsqueeze(1)

        tree = [noise]
        fake_points = self.decoder_gen(tree)
        return fake_points

    def discriminate(self, points):
        # points => (B, N, 3)
        return self.discriminator_mlp(points)
        
