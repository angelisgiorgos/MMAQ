import torch
import torchvision
import torch.nn as nn


def get_resnet_model(network, channels=12):
    """
    create a resnet50 model, optionally load pretrained checkpoint
    and pass it to the device
    """
    model = torchvision.models.__dict__[network](weights=None, num_classes=19)
    model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), bias=False)
    return model

class ResnetRegressionHead(nn.Module):
    """ Wrapper class to put a regression head on
    a resnet model """
    def __init__(self, backbone, head):
        super(ResnetRegressionHead, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x
