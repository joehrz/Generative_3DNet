# File: src/models/discriminator_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorMLP(nn.Module):
    """
    A simpler MLP-based discriminator for the GAN direction.
    We'll do:
      1) Flatten or quick pooling (B, N, 3) -> (B, 3)
         or (B, some_small_feature) 
      2) A few linear layers -> final scalar
    """

    def __init__(self, hidden_dims=[256, 128], pooling='max'):
        super(DiscriminatorMLP, self).__init__()
        self.pooling = pooling
        # We’ll flatten the point cloud dimension: from (B, N, 3) -> (B, N*3)
        # Or do a quick global pooling if you prefer:
        #   e.g. max over N => shape (B, 3)
        # The paper suggests "only a few MLPs", so let's do flatten for demonstration.

        # Let’s do flatten approach:
        # input_dim = N*3 if we just flatten. That can be large for N=2048...
        # Alternatively, we can do a quick global max or mean for shape (B,3).
        # We'll demonstrate a "pool to 3" approach for simplicity:

        self.net = nn.ModuleList()
        in_dim = 3  # after pooling to shape (B, 3)

        # Build MLP layers:
        for hdim in hidden_dims:
            self.net.append(nn.Linear(in_dim, hdim))
            self.net.append(nn.LeakyReLU(0.2))
            in_dim = hdim

        # Final output => 1 scalar
        self.final_fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        """
        x: (B, N, 3)
        We do a global pooling => shape (B, 3), then pass MLP => shape (B, 1).
        """
        B, N, C = x.shape
        if self.pooling == 'max':
            # max over points => (B, 3)
            x = x.max(dim=1)[0]  # shape (B, 3)
        else:
            # average
            x = x.mean(dim=1)    # shape (B, 3)
        
        # Pass MLP
        for layer in self.net:
            x = layer(x)
        out = self.final_fc(x)
        return out
