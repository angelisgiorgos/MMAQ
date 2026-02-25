import torch
from torch import nn, Tensor
from models.backbones.model import ImagingNet, S2Backbone, S5Backbone
from models.backbones.tabularnets import TabularNet
from models.backbones.resnet import ResnetRegressionHead


def get_model(sources, args, heteroscedastic=False):
    """ Returns a model suitable for the given sources """
    if not args.tabular:
        if sources == "S2":
            return get_S2_no2_model(args, heteroscedastic)

        elif sources == "S2S5P":
            return get_S2S5P_no2_model(args, heteroscedastic)
    else:
        return getS2PS5P_tab_model(args=args, heteroscedastic=heteroscedastic)


def get_S2_no2_model(args):
    """ Returns a ResNet for Sentinel-2 data with a regression head """
    backbone_S2 = S2Backbone(args)

    if args.dropout is not None:
        head = Head(2048, 512, args)
        head.turn_dropout_on()
    else:
        head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 1))

    regression_model = ResnetRegressionHead(backbone_S2, head)
    return regression_model


def get_S2S5P_no2_model(args):
    """ Returns a model with two input streams
    (one for S2, one for S5P) followed by a dense
    regression head """
    backbone_S2 = S2Backbone(args)
    backbone_S5P = S5Backbone(args)

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
    """ Returns a MultiModal model that uses S2, S5P, and Tabular networks. """
    return MultiModal(args)


class HeadMultimodal(nn.Module):
    def __init__(self, input_dim, intermediate_dim, args):
        super(HeadMultimodal, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
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
        x_concat = torch.cat([img, x_tab], dim=1)
        out = self.mixer(x_concat)
        return out


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
        # We can pass x directly since S2Backbone and S5Backbone 
        # both handle the dictionary format inherently inside forward().
        img_out = self.backbone_S2(x)
        s5p_out = self.backbone_S5P(x)
        
        features = torch.cat((img_out, s5p_out), dim=1)
        out = self.head(features)
        return out