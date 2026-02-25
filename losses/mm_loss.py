import torch
import torch.nn as nn


class Multimodal_Loss(nn.Module):
    def __init__(self, args, imaging_loss, tabular_loss=None):
        super(Multimodal_Loss, self).__init__()
        self.args = args
        self.imaging_loss = imaging_loss
        self.tabular_loss = tabular_loss

    def forward(self, im_0, im_1, t1=None, t2=None):
        if self.tabular_loss is not None:
            loss = self.args.loss_weight * self.imaging_loss(im_0, im_1) + (
                1 - self.args.loss_weight
            ) * self.tabular_loss(t1, t2)
        else:
            loss = self.args.loss_weight * self.imaging_loss(im_0, im_1)
        return loss