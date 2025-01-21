import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeGCNLayer(nn.Module):
    """
    Single TreeGCN layer that can upsample from node -> node * degree.

    Quick fix: after 'branch = x.unsqueeze(2) @ self.W_branch',
    we call 'branch = branch.squeeze(2)' to remove the extra dimension.
    """
    def __init__(self, batch_size, depth, features, degrees,
                 support=10, node=1, upsample=False, activation=True):
        super(TreeGCNLayer, self).__init__()
        self.batch = batch_size
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth + 1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation

        # Single linear transform for the last node set
        self.W_root = nn.Linear(self.in_feature, self.out_feature, bias=False)

        # Optional branching parameters
        if self.upsample:
            # shape: (node, in_feature, degree * in_feature)
            # NOTE: if you want simpler matmul shapes, you could
            # define (in_feature, degree * in_feature) instead.
            self.W_branch = nn.Parameter(torch.FloatTensor(
                self.node, self.in_feature, self.degree * self.in_feature
            ))
            nn.init.xavier_uniform_(self.W_branch)

        # "loop" block
        self.W_loop = nn.Sequential(
            nn.Linear(self.in_feature, self.in_feature * support, bias=False),
            nn.Linear(self.in_feature * support, self.out_feature, bias=False)
        )

        # Bias for final activation
        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))
        nn.init.uniform_(
            self.bias,
            -1.0 / math.sqrt(self.out_feature),
            1.0 / math.sqrt(self.out_feature)
        )

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, tree_list):
        """
        Only operate on the last entry tree_list[-1].
        Debug prints illustrate each step's shape.
        """
        x = tree_list[-1]  # shape (B, old_num, in_feat)
        B, old_num, in_feat = x.shape
        #print(f"[DEBUG] Input x => {x.shape}")

        # 1) Flatten + linear
        x_flat = x.view(B * old_num, in_feat)
        #print(f"[DEBUG] x_flat => {x_flat.shape}")
    
        tmp = self.W_root(x_flat)  # => (B*old_num, out_feature)
        tmp = tmp.view(B, old_num, self.out_feature)
        #print(f"[DEBUG] tmp after linear => {tmp.shape}")

        # 2) Upsample
        repeat_num = self.node // old_num
        tmp = tmp.repeat(1, repeat_num, 1)  # => (B, self.node, out_feature)
        #print(f"[DEBUG] root_sum => {tmp.shape}")
        root_sum = tmp

        if self.upsample:
            # Branching
            branch = x.unsqueeze(2) @ self.W_branch
            #print(f"[DEBUG] branch after matmul => {branch.shape}")
            # Possibly squeeze dim 2, or reshape
            branch = branch.squeeze(2)
            #print(f"[DEBUG] branch after squeeze => {branch.shape}")

            # Now parse shape
            B2, old_num2, total_in = branch.shape
            branch = branch.view(B2, old_num2, self.degree, in_feat)
            #print(f"[DEBUG] branch reshaped => {branch.shape}")
        
            branch = branch.view(B2, old_num2 * self.degree, in_feat)
            #print(f"[DEBUG] branch => {branch.shape} before W_loop")
        
            branch = self.W_loop(branch)  # => (B2, old_num2*degree, out_feature)
            #print(f"[DEBUG] branch after W_loop => {branch.shape}")
        
            # Also repeat root_sum
            root_sum = root_sum.repeat(1, self.degree, 1)  
            #print(f"[DEBUG] root_sum repeated => {root_sum.shape}")

            combined = root_sum + branch
            #print(f"[DEBUG] combined => {combined.shape}")
        else:
            branch = self.W_loop(x)
            #print(f"[DEBUG] branch (no upsample) => {branch.shape}")
            combined = root_sum + branch
            #print(f"[DEBUG] combined => {combined.shape}")

        # Optional activation
        if self.activation:
            final_nodes = combined.size(1)             # 16
            times = final_nodes // self.degree         # = 16 // 4 = 4
            bias_rep = self.bias.repeat(1, times, 1)   # => (1, 16, 128)
            #bias_rep = self.bias.repeat(1, final_nodes, 1)
            #print(f"[DEBUG] bias => {self.bias.shape}, bias_rep => {bias_rep.shape}")
            #print(f"[DEBUG] combined => {combined.shape} before adding bias")
        
            combined = self.leaky_relu(combined + bias_rep)
            #print(f"[DEBUG] combined => {combined.shape} after activation")

        tree_list.append(combined)
        
        return tree_list


class TreeGCNDecoderGenerator(nn.Module):
    """
    Stacks multiple TreeGCN layers to decode from latent code -> 3D points
    or generate from noise -> 3D points.
    """
    def __init__(self, batch_size, features, degrees, support=10):
        super(TreeGCNDecoderGenerator, self).__init__()
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        self.layers = nn.ModuleList()
        

        node_count = 1
        for i in range(self.layer_num):
            #print(f"[DEBUG INIT] Building layer {i}, node_count={node_count}, using degrees[{i}]={degrees[i]}")
            start_nodes = node_count              # old_num for this layer
            degree_i    = degrees[i]             # 2, 2, 2, 2, 2, then 64
            # The layer output => start_nodes * degree_i
            # So next node_count is updated AFTER the layer
            layer = TreeGCNLayer(
                batch_size=batch_size,
                depth=i,
                features=features,
                degrees=degrees,
                support=support,
                node=start_nodes,  # <--- Instead of node_count*degree_i
                upsample=True,
                activation=(i != self.layer_num - 1)
            )
            self.layers.append(layer)
            # Now we accumulate for next layer
            node_count *= degree_i




        # node_count = 1
        # for i in range(self.layer_num):
            
        #     activation = (i != self.layer_num - 1)  # last layer no activation
        #     layer = TreeGCNLayer(
        #         batch_size=batch_size,
        #         depth=i,
        #         features=features,
        #         degrees=degrees,
        #         support=support,
        #         node=node_count * degrees[i],
        #         upsample=True,
        #         activation=activation
        #     )
        #     self.layers.append(layer)
        #     node_count *= degrees[i]

        self.pointcloud = None

    def forward(self, tree):
        # tree is a list, e.g. [z], shape (B, 1, latent_dim)
        for layer in self.layers:
            tree = layer(tree)
        self.pointcloud = tree[-1]
        return self.pointcloud

    def get_pointcloud(self):
        return self.pointcloud

