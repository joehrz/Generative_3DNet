
import torch.nn as nn
from .encoder_discriminator import EncoderDiscriminator
from .decoder_generator import TreeGCNDecoderGenerator

class BiNet(nn.Module):
    """
    Bidirectional network (BI-Net): 
      - EnDi: shared encoder & discriminator
      - DeGe: shared decoder & generator
    """
    def __init__(self,
                 batch_size,
                 features_g,        # e.g. [96, 128, 64, 3]
                 degrees,           # e.g. [4, 4, 4]
                 enc_disc_feat,     # e.g. [3, 64, 128, 256, 512]
                 latent_dim=96,
                 support=10):
        super(BiNet, self).__init__()
        self.EnDi = EncoderDiscriminator(latent_dim=latent_dim, d_feat=enc_disc_feat)
        self.DeGe = TreeGCNDecoderGenerator(batch_size, features_g, degrees, support)

    # AE direction
    def encode(self, real_points):
        return self.EnDi(real_points, mode='encoder')

    def decode(self, latent_code):
        if latent_code.dim() == 2:  # (B, latent_dim)
            latent_code = latent_code.unsqueeze(1)  # (B, 1, latent_dim)
        tree = [latent_code]
        rec_points = self.DeGe(tree)
        return rec_points

    # GAN direction
    def generate(self, noise):
        if noise.dim() == 2:
            noise = noise.unsqueeze(1)
        tree = [noise]
        return self.DeGe(tree)

    def discriminate(self, points):
        return self.EnDi(points, mode='discriminator')

        
