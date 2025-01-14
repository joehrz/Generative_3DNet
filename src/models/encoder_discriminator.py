## src/models/encoder_discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDiscriminator(nn.Module):
    """
    Acts as Encoder (AE direction) or Discriminator (Gan direction)
    """
    def __init__(self, latent_dim=96, d_feat=[3, 64, 128, 256, 512]):
        super(EncoderDiscriminator, self).__init__()
        self.latent_dim = latent_dim

        layers = []

        for i in range(len(d_feat) - 1):
            in_ch = d_feat[i]
            out_ch = d_feat[i+1]
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1)
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2))

            self.shared_conv = nn.Sequential(*layers)

            # separate heads
            self.fc_latent = nn.Linear(d_feat[-1], latent_dim)
            self.fc_disc = nn.Linear(latent_dim, 1)


    

    def forward(self, x, mode='encoder'):
        """
        x: (B, N, 3)
        """
        x = x.transpose(1, 2) # (B, 3, N)
        feat = self.shared_conv(x) # e.g. (B, 512, N)

        # global pooling
        feat = F.max_pool1d(feat, kernel_size=feat.shape[-1]).squeeze(-1) # (B, 512)

        latent = self.fc_latent(feat) # (B, latent_dim)
        
        if mode == 'encoder':
            return latent
        elif mode == 'discriminator':
            return self.fc_disc(latent)  # (B, 1)
        else:
            raise ValueError("Mode must be 'encoder' or 'discriminator'")