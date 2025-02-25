# File: src/models/bi_net.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# =======================================================
# == 3. Shared Encoder/Discriminator Architecture =======
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
    def __init__(self, latent_dim=128):
        super().__init__()
        # 1) 5 conv1d layers, kernel_size=1, stride=1 => pointwise transformations
        self.conv1 = nn.Conv1d(3,   64, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv1d(64,  128, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=True)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=True)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, stride=1, bias=True)
        
        # 2) For the final pooled feature => we have a 3-FC "head"
        # We'll build them so we can route them differently for encoder or discriminator
        # Shared layers for encoder:
        self.enc_fc1 = nn.Linear(1024, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_fc3 = nn.Linear(256, latent_dim)
        
        # For the discriminator:
        self.disc_fc1 = nn.Linear(1024, 512)
        self.disc_fc2 = nn.Linear(512, 256)
        self.disc_fc3 = nn.Linear(256, 1)
        
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
            x = self.leaky_relu(self.enc_fc2(x))   # (B, 256)
            x = self.enc_fc3(x)                    # (B, latent_dim)
            return x
        elif mode == 'discriminator':
            # pass 3 FC layers => (B, 1)
            x = self.leaky_relu(self.disc_fc1(x))  # (B, 512)
            x = self.leaky_relu(self.disc_fc2(x))  # (B, 256)
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
    We'll replicate the structure from the user snippet & from the TreeGAN approach.

    Steps:
      1) Root transform: linear(in_feat -> out_feat)
      2) Branch transform if upsample: 
         linear(in_feat -> in_feat * degree) => reshape => pass "support" => out_feat
      3) Sum root + branch, add bias, apply LeakyReLU
    """
    def __init__(self, in_feat, out_feat, degree, support=10, upsample=True, activation=True):
        super().__init__()
        self.in_feat     = in_feat
        self.out_feat    = out_feat
        self.degree      = degree
        self.upsample    = upsample
        self.activation  = activation
        self.leaky_relu  = nn.LeakyReLU(0.2, inplace=True)
        
        # Root transform:
        self.W_root = nn.Linear(in_feat, out_feat, bias=False)

        # Branch transform if upsample
        if self.upsample and degree > 1:
            self.W_branch = nn.Linear(in_feat, in_feat * degree, bias=False)
        
        # "Loop" transform (like a small MLP)
        self.W_loop = nn.Sequential(
            nn.Linear(in_feat, in_feat * support, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_feat * support, out_feat, bias=False)
        )

        # bias for combination
        self.bias = nn.Parameter(torch.FloatTensor(1, out_feat))
        # The original snippet had shape (1, degree, out_feat), but we can broadcast
        nn.init.uniform_(self.bias, -1.0/math.sqrt(out_feat), 1.0/math.sqrt(out_feat))

    def forward(self, x):
        """
        x: shape (B, old_num, in_feat)
        Returns: shape (B, new_num, out_feat) 
                 new_num = old_num * degree if upsample, else old_num
        """
        B, old_num, in_feat = x.shape

        # Root transform
        root = self.W_root(x.view(-1, in_feat))        # (B*old_num, out_feat)
        root = root.view(B, old_num, self.out_feat)    # (B, old_num, out_feat)

        # "Loop" transform on x
        loop = self.W_loop(x)  # shape (B, old_num, out_feat)

        if self.upsample and self.degree > 1:
            # Branch transform => upsample
            branch = self.W_branch(x)  # (B, old_num, in_feat*degree)
            branch = branch.view(B, old_num * self.degree, in_feat)
            branch = self.W_loop(branch)  # => (B, old_num*degree, out_feat)
            
            # Repeat the root features to match upsample dimension 
            # (since root has shape (B, old_num, out_feat))
            # we replicate it 'degree' times across node dimension
            root = root.unsqueeze(2).expand(B, old_num, self.degree, self.out_feat)
            root = root.contiguous().view(B, old_num*self.degree, self.out_feat) 

            combined = root + branch
        else:
            # No upsample
            combined = root + loop
        
        out = combined + self.bias  # (B, new_num, out_feat), broadcast bias
        if self.activation:
            out = self.leaky_relu(out)
        
        return out


class TreeGCNGenerator(nn.Module):
    """
    7-layer TreeGCN that expands a (B,1,128) latent to (B,2048,3).
    We'll replicate the structure from the snippet: 
      features = [128,256,256,256,128,128,128,3]
      degrees  = [1, 2, 2, 2, 2, 2, 64]  (this multiplies up to 2048)
      support  = 10
    """
    def __init__(self, features, degrees, support=10):
        super().__init__()
        self.layer_num = len(features) - 1  # 7 layers if len(features)=8
        self.layers = nn.ModuleList()

        # Build each layer
        for i in range(self.layer_num):
            in_feat  = features[i]
            out_feat = features[i+1]
            degree   = degrees[i]  # index offset if first is 1
            upsample = (degree > 1)  # if degree=1 => no upsample
            activation = (i != self.layer_num-1)  # last layer => no activation
            layer = TreeGCNLayer(
                in_feat  = in_feat,
                out_feat = out_feat,
                degree   = degree,
                support  = support,
                upsample = upsample,
                activation = activation
            )
            self.layers.append(layer)

    def forward(self, z):
        """
        z: shape (B, 1, 128)  (the root node of the tree)
        returns: shape (B, 2048, 3) final points
        """
        x = z
        for layer in self.layers:
            x = layer(x)
        return x  # (B, final_num_points, 3)


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
        # for TreeGCN:
        features_g = [128, 256, 256, 256, 128, 128, 128, 3],
        degrees_g  = [1,   1,   2,   2,   2,   2,   2,  64],
        support    = 10
    ):
        super().__init__()
        # Shared backbone for encoder/discriminator
        self.backbone = EncoderDiscriminatorBackbone(latent_dim=latent_dim)

        # TreeGCN generator (used as AE decoder & GAN generator)
        self.generator = TreeGCNGenerator(
            features = features_g,
            degrees  = degrees_g,
            support  = support
        )

        self.latent_dim = latent_dim
    
    # ----------- AE direction -----------
    def encode(self, real_points):
        """
        real_points: (B, N, 3)
        returns: (B, latent_dim)
        """
        return self.backbone(real_points, mode='encoder')
    
    def decode(self, latent_code):
        """
        latent_code: (B, latent_dim)
        => unsqueeze => (B,1,latent_dim) => pass to TreeGCN => (B, 2048, 3)
        """
        if latent_code.dim() == 2:
            latent_code = latent_code.unsqueeze(1)  # (B,1,latent_dim)
        return self.generator(latent_code)

    # ----------- GAN direction -----------
    def generate(self, noise):
        """
        noise: (B, latent_dim)
        => unsqueeze => (B,1,latent_dim) => pass generator => (B, 2048, 3)
        """
        if noise.dim() == 2:
            noise = noise.unsqueeze(1)
        return self.generator(noise)
    
    def discriminate(self, points):
        """
        points: (B, N, 3)
        returns: (B,1) score
        """
        return self.backbone(points, mode='discriminator')
        
