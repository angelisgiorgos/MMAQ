import torch
import torch.nn as nn
from models.backbones.vit import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from models.backbones.resnet import get_resnet_model
from torchvision.models import mobilenet_v3_small
from models.backbones.tabularnets import TabularNet


def s2backbone_selection(args):
    if "resnet" in args.network:
        net = get_resnet_model(args.network, channels=args.channels)
        net.fc = nn.Identity()
    elif args.network == "aqnet":
        net = mobilenet_v3_small(pretrained=None, num_classes=1000)
        net.features[0][0] = nn.Conv2d(args.channels, 16, 3, 1, 1)
    elif args.network == 'vit_b16':
        net = vit_b_16(channels=args.channels)
        net.mlp_head = nn.Linear(768, args.s2_net_features)
    elif args.network == 'vit_b32':
        net = vit_b_32(channels=args.channels)
        net.mlp_head = nn.Linear(768, args.s2_net_features)
    elif args.network == 'vit_l16':
        net = vit_l_16(channels=args.channels)
        net.mlp_head = nn.Linear(1024, args.s2_net_features)
    elif args.network == 'vit_l32':
        net = vit_l_32(channels=args.channels)
        net.mlp_head = nn.Linear(1024, args.s2_net_features)
    else:
        raise ValueError(f"Unknown network: {args.network}")
    return net


def s5backbone_selection(args):
    if args.s5pnet == 'initial':
        net = nn.Sequential(nn.Conv2d(1, 10, 3),
                            nn.ReLU(),
                            nn.MaxPool2d(3),
                            nn.Conv2d(10, 15, 5),
                            nn.ReLU(),
                            nn.MaxPool2d(3),
                            nn.Flatten(),
                            nn.Linear(7935, args.s5p_net_features),
                            )
    elif args.s5pnet == "resnet50":
        net = get_resnet_model(args.s5pnet, channels=1)
        net.fc = nn.Identity()
    elif args.s5pnet == 'vit_b16':
        net = vit_b_16(channels=1)
    elif args.s5pnet == 'vit_b32':
        net = vit_b_32(channels=1)
    elif args.s5pnet == 'vit_l16':
        net = vit_l_16(channels=1)
    elif args.s5pnet == 'vit_l32':
        net = vit_l_32(channels=1)
    else:
        raise ValueError(f"Unknown network: {args.s5pnet}")
    return net


def tabular_selection(args):
    """ Returns a tabular backbone wrapper configured by args """
    return TabularNet(args)


class S2Backbone(nn.Module):
    def __init__(self, args):
        super(S2Backbone, self).__init__()
        self.args = args
        self.initialize_dims()
        self.backbone_S2 = s2backbone_selection(args=self.args)

    def initialize_dims(self):
        if self.args.network == "resnet18":
            self.args.imaging_embedding = 512
        elif self.args.network == "vit_b16":
            self.args.imaging_embedding = 768
        else:
            self.args.imaging_embedding = 2048
        self.args.s5p_net_features = self.args.imaging_embedding

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            img = x
        else:
            img = x.get("img")
        img_out = self.backbone_S2(img)
        return img_out


class S5Backbone(nn.Module):
    def __init__(self, args):
        super(S5Backbone, self).__init__()
        self.args = args
        self.initialize_dims()
        self.backbone_S5 = s5backbone_selection(args=self.args)

    def initialize_dims(self):
        if self.args.network == "resnet18":
            self.args.imaging_embedding = 512
        elif self.args.network == "vit_b16":
            self.args.imaging_embedding = 768
        else:
            self.args.imaging_embedding = 2048
        self.args.s5p_net_features = self.args.imaging_embedding

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            s5p = x
        else:
            s5p = x.get("s5p")
        s5p_out = self.backbone_S5(s5p)
        return s5p_out


class ImagingNet(nn.Module):
    def __init__(self, args):
        super(ImagingNet, self).__init__()
        self.args = args
        self.initialize_dims()
        
        if args.fusion_type == "late":
            self.backbone_S2 = S2Backbone(args)
            self.backbone_S5P = S5Backbone(args)
            self.head = nn.Linear(self.args.imaging_embedding + self.args.s5p_net_features, self.args.imaging_embedding)
        elif args.fusion_type == "early":
            self.args.channels = 13
            self.backbone = S2Backbone(args)
    
    def initialize_dims(self):
        if self.args.network == "resnet18":
            self.args.imaging_embedding = 512
        elif self.args.network == "vit_b16":
            self.args.imaging_embedding = 768
        else:
            self.args.imaging_embedding = 2048
        self.args.s5p_net_features = self.args.imaging_embedding
     
    def forward(self, x):
        if self.args.fusion_type  == "late":
            img_out = self.backbone_S2(x)
            s5p_out = self.backbone_S5P(x)
            x_cat = torch.cat((img_out, s5p_out), dim=1)
            out = self.head(x_cat)
        elif self.args.fusion_type == "early":
            img = x.get("img")
            s5p = x.get("s5p")
            input_img = torch.cat((img, s5p), dim=1)
            out = self.backbone(input_img)
        else:
            raise ValueError(f"Unknown fusion type: {self.args.fusion_type}")
        return out
