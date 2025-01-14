import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeGCNLayer(nn.Module):
    """
    Single TreeGCN layer that can upsample from node -> node*degree.
    """
    def __init__(self, batch_size, depth, features, degrees, support=10, node=1, upsample=False, activation=True):
        super(TreeGCNLayer, self).__init__()
        self.batch = batch_size
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth + 1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation

        # W_root for summation
        self.W_root = nn.ModuleList([
            nn.Linear(features[i], self.out_feature, bias=False) for i in range(self.depth + 1)
        ])

        # optional branching
        if self.upsample:
            self.W_branch = nn.Parameter(torch.FloatTensor(self.node, self.in_feature, self.degree * self.in_feature))
            nn.init.xavier_uniform_(self.W_branch)

        # "loop" block
        self.W_loop = nn.Sequential(
            nn.Linear(self.in_feature, self.in_feature * support, bias=False),
            nn.Linear(self.in_feature * support, self.out_feature, bias=False)
        )

        # bias for final activation
        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))
        nn.init.uniform_(self.bias, -1.0 / math.sqrt(self.out_feature), 1.0 / math.sqrt(self.out_feature))
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, tree_list):
        # sum up root transforms
        root_sum = 0
        for i in range(self.depth + 1):
            root_num = tree_list[i].size(1)
            repeat_num = int(self.node / root_num)
            tmp = self.W_root[i](tree_list[i])  # (B, root_num, out_feature)
            tmp = tmp.repeat(1, 1, repeat_num).view(self.batch, -1, self.out_feature)
            root_sum = root_sum + tmp

        if self.upsample:
            # branching
            branch = tree_list[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(self.batch, self.node * self.degree, self.in_feature)
            branch = self.W_loop(branch)
            root_sum = root_sum.repeat(1, 1, self.degree).view(self.batch, -1, self.out_feature)
            combined = root_sum + branch
        else:
            branch = self.W_loop(tree_list[-1])
            combined = root_sum + branch

        if self.activation:
            combined = self.leaky_relu(combined + self.bias.repeat(1, self.node, 1))

        tree_list.append(combined)
        return tree_list

class TreeGCNDecoderGenerator(nn.Module):
    """
    Stacks multiple TreeGCN layers to decode from latent code -> 3D points
    or generate from noise -> 3D points
    """
    def __init__(self, batch_size, features, degrees, support=10):
        super(TreeGCNDecoderGenerator, self).__init__()
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        self.layers = nn.ModuleList()

        node_count = 1
        for i in range(self.layer_num):
            activation = True
            if i == self.layer_num - 1:
                activation = False  # final layer outputs raw coords
            self.layers.append(TreeGCNLayer(batch_size, i, features, degrees, support, node_count, True, activation))
            node_count *= degrees[i]

        self.pointcloud = None

    def forward(self, tree):
        # tree is a list, e.g., [z], shape (B, 1, latent_dim)
        for layer in self.layers:
            tree = layer(tree)
        self.pointcloud = tree[-1]
        return self.pointcloud

    def get_pointcloud(self):
        return self.pointcloud
