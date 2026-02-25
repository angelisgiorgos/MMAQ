import torch
import torch.nn as nn
import numpy as np


class MixUPLoss(nn.Module):
    def __init__(self, args, bn, scale, lambd):
        super().__init__()
        self.args = args
        self.bn = bn
        self.lambd = lambd
        self.scale = scale

    def forward(self, z1, z2, z2m, zm, alpha):
        cm1 = self.bn(zm).T @ self.bn(z1)

        cm1.div_(self.args.batch_size)


        c1 = alpha*self.bn(z1).T @ self.bn(z1)
        c1.div_(self.args.batch_size)

        c2 = (1-alpha)*self.bn(z2m).T @ self.bn(z1)
        c2.div_(self.args.batch_size)

        c12 = c1+c2

        cm2 = self.bn(zm).T @ self.bn(z2)

        cm2.div_(self.args.batch_size)

        loss_mix = self.scale *self.lambd*((cm1-c12).pow_(2).sum() + (cm2-c12).pow_(2).sum())
        return loss_mix



