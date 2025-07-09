# File: src/models/bi_net.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# =======================================================
# == Shared Encoder/Discriminator Architecture =======
# =======================================================
class EncoderDiscriminatorBackbone(nn.Module):
    """
    The paper implements a single 5-convolution + max-pool + 3-FC backbone.
    - If 'mode=encoder', we return a 128-D latent code
    - If 'mode=discriminator', we apply an extra final FC to get a scalar.
    No BatchNorm is used. Activation = LeakyReLU.

    Conv layers: 3 -> 64 -> 128 -> 256 -> 512 -> 1024
    Global max pool => shape (B, 1024)
    Then 3 FC layers: 
      for the encoder  : 1024 -> 512 -> 256 -> 128
      for the discrim. : 1024 -> 512 -> 256 -> 1
    """
    def __init__(self, latent_dim=128, dropout_rate=0.1, use_spectral_norm=False):
        super().__init__()
        # 1) 5 conv1d layers, kernel_size=1, stride=1 => pointwise transformations
        self.conv1 = nn.Conv1d(3,   64, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv1d(64,  128, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=True)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=True)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, stride=1, bias=True)
        
        # 2) For the final pooled feature => we have a 3-FC "head"
        # Shared layers for encoder:
        enc_fc1 = nn.Linear(1024, 512)
        enc_fc2 = nn.Linear(512, 256)
        enc_fc3 = nn.Linear(256, latent_dim)
        
        # For the discriminator:
        disc_fc1 = nn.Linear(1024, 512)
        disc_fc2 = nn.Linear(512, 256)
        disc_fc3 = nn.Linear(256, 1)
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            self.enc_fc1 = spectral_norm(enc_fc1)
            self.enc_fc2 = spectral_norm(enc_fc2)
            self.enc_fc3 = spectral_norm(enc_fc3)
            self.disc_fc1 = spectral_norm(disc_fc1)
            self.disc_fc2 = spectral_norm(disc_fc2)
            self.disc_fc3 = spectral_norm(disc_fc3)
        else:
            self.enc_fc1 = enc_fc1
            self.enc_fc2 = enc_fc2
            self.enc_fc3 = enc_fc3
            self.disc_fc1 = disc_fc1
            self.disc_fc2 = disc_fc2
            self.disc_fc3 = disc_fc3
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, mode='encoder'):
        """
        x: shape (B, N, 3)
        mode: 'encoder' => return 128-D latent
              'discriminator' => return scalar score
        """
        B, N, C = x.shape
        # 1) transpose => (B, 3, N)
        x = x.transpose(1, 2)  # shape (B, 3, N)

        # 2) pass conv layers with LeakyReLU
        x = self.leaky_relu(self.conv1(x))   # (B, 64, N)
        x = self.leaky_relu(self.conv2(x))   # (B, 128, N)
        x = self.leaky_relu(self.conv3(x))   # (B, 256, N)
        x = self.leaky_relu(self.conv4(x))   # (B, 512, N)
        x = self.leaky_relu(self.conv5(x))   # (B, 1024, N)

        # 3) global max pool => (B, 1024)
        x = F.max_pool1d(x, kernel_size=x.shape[-1])  # (B, 1024, 1)
        x = x.squeeze(-1)  # shape (B, 1024)

        if mode == 'encoder':
            # pass 3 FC layers => latent code
            x = self.leaky_relu(self.enc_fc1(x))   # (B, 512)
            x = self.dropout(x)                    # Apply dropout
            x = self.leaky_relu(self.enc_fc2(x))   # (B, 256)
            x = self.dropout(x)                    # Apply dropout
            x = self.enc_fc3(x)                    # (B, latent_dim)
            return x
        elif mode == 'discriminator':
            # pass 3 FC layers => (B, 1)
            x = self.leaky_relu(self.disc_fc1(x))  # (B, 512)
            x = self.dropout(x)                    # Apply dropout
            x = self.leaky_relu(self.disc_fc2(x))  # (B, 256)
            x = self.dropout(x)                    # Apply dropout
            x = self.disc_fc3(x)                   # (B, 1)
            return x
        else:
            raise ValueError("Invalid mode. Must be 'encoder' or 'discriminator'.")

# =======================================================
# ==========   4. TreeGCN Generator/Decoder   ===========
# =======================================================
class TreeGCNLayer(nn.Module):
    """
    One TreeGCN layer that upsamples from 'old_num' nodes -> 'old_num * degree' nodes
    or simply transforms them if degree=1. 
    """
    def __init__(self, in_feat, out_feat, degree, support=10, upsample=True, activation=True):
        super().__init__()
        self.in_feat     = in_feat
        self.out_feat    = out_feat
        self.degree      = degree
        self.upsample    = upsample
        self.activation  = activation
        self.leaky_relu  = nn.LeakyReLU(0.2, inplace=True)
        
        self.W_root = nn.Linear(in_feat, out_feat, bias=False)

        if self.upsample and degree > 1:
            self.W_branch = nn.Linear(in_feat, in_feat * degree, bias=False)
        
        self.W_loop = nn.Sequential(
            nn.Linear(in_feat, in_feat * support, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_feat * support, out_feat, bias=False)
        )

        self.bias = nn.Parameter(torch.FloatTensor(1, out_feat))
        nn.init.uniform_(self.bias, -1.0/math.sqrt(out_feat), 1.0/math.sqrt(out_feat))

    def forward(self, x):
        B, old_num, in_feat = x.shape
        root = self.W_root(x.view(-1, in_feat)).view(B, old_num, self.out_feat)
        loop = self.W_loop(x)

        if self.upsample and self.degree > 1:
            branch = self.W_branch(x).view(B, old_num * self.degree, in_feat)
            branch = self.W_loop(branch)
            
            root = root.unsqueeze(2).expand(B, old_num, self.degree, self.out_feat)
            root = root.contiguous().view(B, old_num * self.degree, self.out_feat) 

            combined = root + branch
        else:
            combined = root + loop
        
        out = combined + self.bias
        if self.activation:
            out = self.leaky_relu(out)
        
        return out

class TreeGCNGenerator(nn.Module):
    """
    7-layer TreeGCN that expands a latent vector to a point cloud.
    """
    def __init__(self, features, degrees, support=10):
        super().__init__()
        self.layer_num = len(features) - 1
        self.layers = nn.ModuleList()

        for i in range(self.layer_num):
            in_feat  = features[i]
            out_feat = features[i+1]
            degree   = degrees[i]
            upsample = (degree > 1)
            activation = (i != self.layer_num-1)
            self.layers.append(TreeGCNLayer(in_feat, out_feat, degree, support, upsample, activation))

    def forward(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x

# =======================================================
# ============   5. Full BI-Net Implementation  =========
# =======================================================
class BiNet(nn.Module):
    """
    The main container that does:
      - Shared Encoder+Discriminator backbone (EncoderDiscriminatorBackbone)
      - TreeGCN Generator/Decoder
      - AE direction => encode real -> decode to reconstruct
      - GAN direction => generate from noise -> discriminate real vs fake
    """
    def __init__(
        self,
        latent_dim=128,
        features_g = [128, 256, 256, 256, 128, 128, 128, 3],
        degrees_g  = [1,   1,   2,   2,   2,   2,   2,  64],
        support    = 10,
        dropout_rate=0.1,
        use_spectral_norm=False
    ):
        super().__init__()
        self.backbone = EncoderDiscriminatorBackbone(
            latent_dim=latent_dim, 
            dropout_rate=dropout_rate, 
            use_spectral_norm=use_spectral_norm
        )
        self.generator = TreeGCNGenerator(features=features_g, degrees=degrees_g, support=support)
        self.latent_dim = latent_dim
    
    def encode(self, real_points):
        return self.backbone(real_points, mode='encoder')
    
    def decode(self, latent_code):
        if latent_code.dim() == 2:
            latent_code = latent_code.unsqueeze(1)
        return self.generator(latent_code)

    def generate(self, noise):
        if noise.dim() == 2:
            noise = noise.unsqueeze(1)
        return self.generator(noise)
    
    def discriminate(self, points):
        return self.backbone(points, mode='discriminator')

    def get_encoder_params(self):
        """Returns parameters for the encoder path."""
        enc_params = []
        shared_params = []
        for name, p in self.backbone.named_parameters():
            if "enc_fc" in name:
                enc_params.append(p)
            elif "disc_fc" not in name:
                shared_params.append(p)
        return enc_params, shared_params

    def get_generator_params(self):
        """Returns parameters of the TreeGCN generator."""
        return list(self.generator.parameters())

    def get_discriminator_params(self):
        """Returns parameters for the discriminator path."""
        disc_params = []
        shared_params = []
        for name, p in self.backbone.named_parameters():
            if "disc_fc" in name:
                disc_params.append(p)
            elif "enc_fc" not in name:
                shared_params.append(p)
        return disc_params, shared_params
        
