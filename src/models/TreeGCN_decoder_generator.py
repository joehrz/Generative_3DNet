# File: src/models/TreeGCN_decoder_generator.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeGCNLayer(nn.Module):
    """
    Single TreeGCN layer that can upsample from node -> node * degree.

    Key fix: Use nn.Linear(in_feat -> degree * in_feat) for branching 
    and reshape only once.
    """
    def __init__(
        self, 
        batch_size, 
        depth, 
        features, 
        degrees,
        support=10, 
        node=1, 
        upsample=False, 
        activation=True
    ):
        super().__init__()
        self.batch_size   = batch_size
        self.depth        = depth
        self.in_feature   = features[depth]
        self.out_feature  = features[depth + 1]
        self.node         = node
        self.degree       = degrees[depth]
        self.upsample     = upsample
        self.activation   = activation

        # 1) Root transform
        self.W_root = nn.Linear(self.in_feature, self.out_feature, bias=False)

        # 2) Branch transform if upsample
        if self.upsample:
            # Instead of an nn.Parameter(...) + manual matmul,
            # we define a proper linear layer: (in_feat -> degree * in_feat).
            self.W_branch = nn.Linear(self.in_feature, self.degree * self.in_feature, bias=False)

        # 3) "Loop" transform
        self.W_loop = nn.Sequential(
            nn.Linear(self.in_feature, self.in_feature * support, bias=False),
            nn.Linear(self.in_feature * support, self.out_feature, bias=False)
        )

        # 4) Bias for after combination
        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))
        nn.init.uniform_(
            self.bias,
            -1.0 / math.sqrt(self.out_feature),
            1.0 / math.sqrt(self.out_feature)
        )

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, tree_list):
        """
        Operate on the last entry: tree_list[-1] has shape (B, old_num, in_feat).
        """
        x = tree_list[-1]                        # (B, old_num, in_feat)
        B, old_num, in_feat = x.shape

        # 1) Root transform
        x_flat = x.view(B * old_num, in_feat)     # flatten across B & old_num
        tmp    = self.W_root(x_flat)             # => (B*old_num, out_feature)
        tmp    = tmp.view(B, old_num, self.out_feature)

        # We will "repeat" to match self.node
        repeat_num = self.node // old_num         # e.g. if node=1 at first layer, old_num=1
        root_sum   = tmp.repeat(1, repeat_num, 1) # => shape (B, self.node, out_feature)

        if self.upsample:
            # 2) Branch transform via W_branch
            #    x has shape (B, old_num, in_feat).
            #    W_branch => (in_feat -> degree*in_feat).
            branch = self.W_branch(x)  # => (B, old_num, degree*in_feat)

            #    Reshape so we "group" the degree dimension
            branch = branch.view(B, old_num * self.degree, in_feat)

            # 3) Pass through W_loop => (B, old_num*degree, out_feature)
            branch = self.W_loop(branch)

            # 4) Expand root_sum to match (B, old_num*degree, out_feature)
            root_sum = root_sum.repeat(1, self.degree, 1)

            combined = root_sum + branch
        else:
            # No branching => pass x directly through W_loop
            branch   = self.W_loop(x)       # => (B, old_num, out_feature)
            combined = root_sum + branch    # => same shape: (B, old_num, out_feature)

        # Optional activation + bias
        if self.activation:
            final_nodes = combined.size(1)               # e.g. old_num * degree
            times       = final_nodes // self.degree     # how many times to repeat the bias
            bias_rep    = self.bias.repeat(1, times, 1)  # => shape (1, final_nodes, out_feature)
            combined    = self.leaky_relu(combined + bias_rep)

        tree_list.append(combined)
        return tree_list


class TreeGCNDecoderGenerator(nn.Module):
    """
    7-layer TreeGCN for:
      input latent dim = 128 -> final shape (B, 2048, 3) if degrees multiply up correctly.
    """
    def __init__(self, 
                 batch_size, 
                 features, 
                 degrees, 
                 support=10):
        super().__init__()
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        self.layers = nn.ModuleList()

        node_count = 1
        for i in range(self.layer_num):
            layer = TreeGCNLayer(
                batch_size   = self.batch_size,
                depth        = i,
                features     = features,
                degrees      = degrees,
                support      = support,
                node         = node_count,
                upsample     = True,
                activation   = (i != self.layer_num - 1)
            )
            self.layers.append(layer)
            node_count *= degrees[i]  # update node for next layer

    def forward(self, tree):
        """
        'tree' is a list of node embeddings, typically just [latent_code].
        Each layer updates 'tree' by appending the next level's output.
        """
        for layer in self.layers:
            tree = layer(tree)
        # Final pointcloud is tree[-1]
        pointcloud = tree[-1]  # shape (B, final_num_points, 3)
        return pointcloud

    
