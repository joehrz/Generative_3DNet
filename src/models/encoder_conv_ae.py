# File: src/models/encoder_conv_ae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderConvAE(nn.Module):
    """
    The AE encoder: multiple conv layers + global pooling -> latent_dim.
    Used only in the forward direction to reconstruct real shapes (with EMD).
    """
    def __init__(self, latent_dim=128, d_feat=[3, 64, 128, 256, 512]):
        super(EncoderConvAE, self).__init__()
        self.latent_dim = latent_dim

        layers = []
        for i in range(len(d_feat) - 1):
            in_ch = d_feat[i]
            out_ch = d_feat[i + 1]
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1)
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2))

        self.shared_conv = nn.Sequential(*layers)
        self.fc_latent = nn.Linear(d_feat[-1], latent_dim)

    def forward(self, x):
        """
        x: (B, N, 3)
        Returns latent code: (B, latent_dim)
        """
        # (B,3,N)
        x = x.transpose(1, 2)
        feat = self.shared_conv(x)  # (B, 512, N) if d_feat[-1]=512
        feat = F.max_pool1d(feat, kernel_size=feat.shape[-1]).squeeze(-1)
        latent = self.fc_latent(feat)  # (B, latent_dim)
        return latent