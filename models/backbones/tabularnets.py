import torch
import torch.nn as nn
from .attention import Attention_Layer
from .layer_utils import BasicBlock


class TabularAttention(nn.Module):
    def __init__(self, args):
        super(TabularAttention, self).__init__()
        self.linear1 = nn.Linear(args.tabular_input, args.tabular_net_features // 2)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(args.tabular_net_features // 2)
        self.attention = Attention_Layer(args.tabular_net_features // 2)
        self.layer_norm2 = nn.LayerNorm(args.tabular_net_features // 2)
        self.linear = nn.Linear(args.tabular_net_features // 2, args.tabular_net_features // 2)
        self.relu2 = nn.ReLU()
        self.layernorm3 = nn.LayerNorm(args.tabular_net_features // 2)
        self.output = nn.Linear(args.tabular_net_features // 2, args.tabular_net_features)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.layer_norm2(x)
        x = self.linear(x)
        x = self.relu2(x)
        x = self.layernorm3(x)
        out = self.output(x)
        return out

class DANet(nn.Module):
    def __init__(self, input_dim=8, layer_num=4, base_outdim=256, output_dim=512, k=2, virtual_batch_size=64, drop_rate=0.1):
        super(DANet, self).__init__()
        params = {'base_outdim': base_outdim, 'k': k, 'virtual_batch_size': virtual_batch_size,
                  'fix_input_dim': input_dim, 'drop_rate': drop_rate}
        self.init_layer = BasicBlock(input_dim, **params)
        self.lay_num = layer_num
        self.layer = nn.ModuleList()
        for i in range((layer_num // 2) - 1):
            self.layer.append(BasicBlock(base_outdim, **params))
        self.drop = nn.Dropout(drop_rate)

        self.fc = nn.Sequential(nn.Linear(base_outdim, base_outdim),
                                nn.ReLU(inplace=True),
                                nn.Linear(base_outdim, output_dim))

    def forward(self, x):
        out = self.init_layer(x)
        for i in range(len(self.layer)):
            out = self.layer[i](x, out)
        out = self.drop(out)
        out = self.fc(out)
        return out


class TabularInitial(nn.Module):
    def __init__(self, args):
        super(TabularInitial, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(args.tabular_input, args.tabular_net_features // 2),
            nn.ReLU(),
            nn.Linear(args.tabular_net_features // 2, args.tabular_net_features),
            nn.ReLU(),
            nn.Linear(args.tabular_net_features, args.tabular_net_features),
            nn.ReLU(),
            nn.Linear(args.tabular_net_features, args.tabular_net_features)
        )
    
    def forward(self, x):
        return self.net(x)


class TabularNet(nn.Module):
    def __init__(self, args):
        super(TabularNet, self).__init__()
        self.args = args
        if self.args.tabular_net == "initial":
            self.backbone = TabularInitial(args)
        elif self.args.tabular_net == "attention":
            self.backbone = TabularAttention(args)
        elif self.args.tabular_net == "danet":
            # using DANet from tabularnets
            self.backbone = DANet(
                input_dim=args.tabular_input,
                base_outdim=args.tabular_net_features // 2,
                output_dim=args.tabular_net_features
            )
        else:
            raise ValueError(f"Unknown tabular_net value: {self.args.tabular_net}")
            
    def forward(self, x):
        return self.backbone(x)