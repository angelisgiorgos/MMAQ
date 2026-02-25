import torch
import torch.nn as nn
import torch.nn.functional as F

class MMAQProjector(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        out = self.projector(x)
        return out


class AQRProjector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.linear1 = nn.Sequential(
            nn.Linear(self.args.imaging_embedding, self.args.tabular_net_features),
            nn.BatchNorm1d(args.tabular_net_features),
            nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(
            nn.Linear(args.tabular_net_features, args.tabular_net_features, bias=False),
            nn.BatchNorm1d(args.tabular_net_features),
            nn.ReLU(inplace=True),
            nn.Linear(args.tabular_net_features, args.tabular_net_features, bias=False),
            nn.BatchNorm1d(args.tabular_net_features),
            nn.ReLU(inplace=True)
        )

        # IMPORTANT: these must multiply to tabular_net_features
        self.seq_len = 32
        self.embed_dim = args.tabular_net_features // self.seq_len

        assert self.seq_len * self.embed_dim == args.tabular_net_features, \
            "tabular_net_features must be divisible by 32"

    def forward(self, features1, features2):
        B = features1.size(0)

        # Project first modality
        features1 = self.linear1(features1)

        # Reshape to (B, seq_len, embed_dim)
        query = features1.view(B, self.seq_len, self.embed_dim)
        key   = features2.view(B, self.seq_len, self.embed_dim)
        value = features2.view(B, self.seq_len, self.embed_dim)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value
        )  # (B, seq_len, embed_dim)

        # Flatten back to (B, tabular_net_features)
        lct_out = attn_output.reshape(B, -1)

        # MLP
        out = self.mlp(lct_out)

        return out
