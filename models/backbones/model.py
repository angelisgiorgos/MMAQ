import torch
import torch.nn as nn
from models.backbones.vit import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from models.backbones.resnet import get_resnet_model, ResnetRegressionHead
from torchvision.models import mobilenet_v3_small

def get_model(sources, args, heteroscedastic=False):
    """ Returns a model suitable for the given sources """
    if not args.tabular:
        if sources == "S2":
            return get_S2_no2_model(args, heteroscedastic)

        elif sources == "S2S5P":
            return get_S2S5P_no2_model(args, heteroscedastic)
    else:
        return getS2PS5P_tab_model(args=args, heteroscedastic=heteroscedastic)


def get_S2_no2_model(args, heteroscedastic=False):
    """ Returns a ResNet for Sentinel-2 data with a regression head """
    backbone = s2backbone_selection(args=args)
    backbone.fc = nn.Identity()

    if args.dropout is not None:
        head = Head(2048, 512, args, heteroscedastic)
        head.turn_dropout_on()
    else:
        head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 1))

    regression_model = ResnetRegressionHead(backbone, head)

    return regression_model


def get_S2S5P_no2_model(args, heteroscedastic=False):
    """ Returns a model with two input streams
    (one for S2, one for S5P) followed by a dense
    regression head """
    backbone_S2 = s2backbone_selection(args=args)

    backbone_S5P = s5backbone_selection(args=args)

    if args.s5pnet != 'initial':
        dims = 2048
    else:
        dims = 128

    head = Head(2048 + dims, 544, args)
    if args.dropout is not None:
        # add dropout to linear layers of regression head
        head.turn_dropout_on()

    regression_model = MultiBackboneRegressionHead(backbone_S2, backbone_S5P, head)

    return regression_model


def getS2PS5P_tab_model(args, heteroscedastic=False):
    backbone_S2 = s2backbone_selection(args=args)

    backbone_S5P = s5backbone_selection(args=args)

    backbone_tabular = tabular_selection(args=args)

    head = HeadMultimodal(544, args)

    mixer = nn.Sequential(
        nn.Linear(args.head_features + args.tabular_net_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 1))

    regression_model = MultiModal(
        backbone_S2=backbone_S2,
        backbone_S5P=backbone_S5P,
        backbone_tabular=backbone_tabular,
        head=head,
        mixer=mixer
        )
    return regression_model


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
    elif args.network == 'vit_l16':
        net = vit_l_32(channels=args.channels)
        net.mlp_head = nn.Linear(1024, args.s2_net_features)
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
    elif args.network == 'vit_b32':
        net = vit_b_32(channels=1)
    elif args.network == 'vit_l16':
        net = vit_l_16(channels=1)
    elif args.network == 'vit_l32':
        net = vit_l_32(channels=1)
    return net


def tabular_selection(args):
    backbone_tabular = nn.Sequential(
        nn.Linear(args.tabular_input, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, args.tabular_net_features))
    return backbone_tabular

class HeadMultimodal(nn.Module):
    def __init__(self, s2_features, s5_features, intermediate_dim, args):
        super(HeadMultimodal, self).__init__()
        self.fc1 = nn.Linear(s2_features + s5_features, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, intermediate_dim // 2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(intermediate_dim // 2, args.head_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.output(x)
        return out


class Head(nn.Module):
    def __init__(self, input_dim, intermediate_dim, args):
        super(Head, self).__init__()
        self.dropout1_p = args.dropout_p_second_to_last_layer
        self.dropout2_p = args.dropout_p_last_layer
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if args.heteroscedastic:
            # split the output layer into [mean, sigma2]
            self.fc2 = nn.Linear(intermediate_dim, 2)
        else:
            self.fc2 = nn.Linear(intermediate_dim, 1)
        self.dropout_on = True

    def forward(self, x):
        x = nn.functional.dropout(x, p=self.dropout1_p, training=self.dropout_on)
        x = self.fc1(x)
        x = self.relu(x)
        x = nn.functional.dropout(x, p=self.dropout2_p, training=self.dropout_on)
        x = self.fc2(x)

        return x

    def turn_dropout_on(self, use=True):
        self.dropout_on = use


class MultiBackboneRegressionHead(nn.Module):
    """ Wrapper class that combines features extracted
    from two inputs (S2 and S5P) with a regression head """

    def __init__(self, backbone_S2, backbone_S5P, head):
        super(MultiBackboneRegressionHead, self).__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_S5P = backbone_S5P
        self.head = head
        self.use_dropout = True

    def forward(self, x):
        s5p = x.get("s5p")
        x = x.get("img")

        x = self.backbone_S2(x)
        s5p = self.backbone_S5P(s5p)
        x = torch.cat((x, s5p), dim=1)
        x = self.head(x)

        return x


class ImagingNet(nn.Module):
    def __init__(self, args):
        super(ImagingNet, self).__init__()
        self.args = args
        self.initialize_dims()
        
        if args.fusion_type == "late":
            self.backbone_S2 = s2backbone_selection(args=self.args)
            self.backbone_S5P = s5backbone_selection(args=self.args)
            self.head = nn.Linear(self.args.imaging_embedding + self.args.s5p_net_features, self.args.imaging_embedding)
        elif args.fusion_type == "early":
            self.args.channels = 13
            self.backbone = s2backbone_selection(args)
    
    def initialize_dims(self):
        if self.args.network == "resnet18":
            self.args.imaging_embedding = 512
        elif self.args.network == "vit_b16":
            self.args.imaging_embedding = 768
        else:
            self.args.imaging_embedding = 2048
        self.args.s5p_net_features = self.args.imaging_embedding
     
    def forward(self, x):
        img = x.get("img")
        s5p = x.get("s5p")
        
        if self.args.fusion_type  == "late":
            img_out = self.backbone_S2(img)
            s5p_out = self.backbone_S5P(s5p)
            x = torch.cat((img_out, s5p_out), dim=1)
            out = self.head(x)
        elif self.args.fusion_type == "early":
            img = x.get("img")
            s5p = x.get("s5p")
            input_img = torch.cat((img, s5p), dim=1)
            out = self.backbone(input_img)
        return out



class TabularNet(nn.Module):
    def __init__(self, args):
        super(TabularNet, self).__init__()
        self.args = args
        if self.args.tabular_net == "initial":
            self.linear1 = nn.Linear(args.tabular_input, args.tabular_net_features  // 2)
            self.relu = nn.ReLU()
            self.linear = nn.Linear(args.tabular_net_features  // 2, args.tabular_net_features )
            self.linear2 = nn.Linear(args.tabular_net_features, args.tabular_net_features)
            self.output = nn.Linear(args.tabular_net_features  , args.tabular_net_features* 2)
        elif self.args.tabular_net == "danet":
            from pytorch_tabular.models.danet import DANetBackbone
            self.backbone = DANetBackbone(
                n_continuous_features=2,
                cat_embedding_dims=256,
                n_layers=4,
                abstlay_dim_1=32,
                k=5,
                dropout_rate=0.1,
                block_activation ="ReLU"
            )
            self._embedding_layer = self.backbone._build_embedding_layer()
    
    def forward(self, x):
        if self.args.tabular_net == "initial":
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            out = self.output(x)
        elif self.args.tabular_net == "danet":
            x = self.backbone(x)
            out = self._embedding_layer(x)
        return out



class MultiModal(nn.Module):
    def __init__(self, args):
        super(MultiModal, self).__init__()
        self.args = args
        self.imaging_model = ImagingNet(args)
        self.tabular_model  = TabularNet(args)
        self.mixer = nn.Sequential(
                            nn.Linear(args.head_features + args.tabular_net_features, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 16),
                            nn.ReLU(),
                            nn.Linear(16, 1))

    def forward(self, x):
        # get
        tabular = x.get("tabular")

        img = self.imaging_model(x)

        x_tab = self.tabular_model(tabular)
        x = torch.cat([img, x_tab], dim=1)
        out = self.mixer(x)
        return out
    

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
        s5p = x.get("s5p")
        s5p_out = self.backbone_S5(s5p)
        return s5p_out

    

