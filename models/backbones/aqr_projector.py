import torch
import torch.nn as nn
from .lct import LCT
from .dot_product import DotProductAttention

class AQRProjector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = nn.Sequential(*[
            nn.Linear(self.args.imaging_embedding, self.args.tabular_net_features),
            nn.BatchNorm1d(args.tabular_net_features),
            nn.ReLU(inplace=True)])
        self.dproduct = DotProductAttention(32)
        self.mlp = nn.Sequential(*[
            nn.Linear(args.tabular_net_features*2, args.tabular_net_features, bias=False),
            nn.BatchNorm1d(args.tabular_net_features),
            nn.ReLU(inplace=True),
            nn.Linear(args.tabular_net_features, args.tabular_net_features, bias=False),
            nn.BatchNorm1d(args.tabular_net_features),
            nn.ReLU(inplace=True)

        ])

    def forward(self, features1, features2):
        features1 = self.linear1(features1)
        # merged_features = torch.cat([features1, features2], axis=1)
        _, lct_out = self.dproduct(features1, features2)
        out = self.mlp(lct_out)
        return out