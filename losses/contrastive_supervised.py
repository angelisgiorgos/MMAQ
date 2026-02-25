import torch
from torch import nn


class ContrastiveRegressionLoss(nn.Module):
    def __init__(self, args, w=1, weights=1, t=0.07, e=0.01):
        super().__init__()
        self.args = args
        self.w = w
        self.weights = weights
        self.t = t
        self.e = e
        self.bn = nn.BatchNorm1d(2048*2, affine=False)

    def forward(self, features, targets, preds):
        q = self.bn(features)
        k = self.bn(features)


        l_k = targets.flatten()[None, :]
        l_q = targets

        p_k = preds.flatten()[None, :]
        p_q = preds

        l_dist = torch.abs(l_q - l_k)
        p_dist = torch.abs(p_q - p_k)

        pos_i = l_dist.le(self.w)
        neg_i = ((~ (l_dist.le(self.w))) * (p_dist.le(self.w)))

        for i in range(pos_i.shape[0]):
            pos_i[i][i] = 0

        prod = torch.einsum("nc,kc->nk", [q, k]) / self.t
        pos = prod * pos_i
        neg = prod * neg_i

        for i in range(pos_i.shape[0]):
            pos_i[i][i] = 0

        prod = torch.einsum("nc,kc->nk", [q, k]) / self.t
        pos = prod * pos_i
        neg = prod * neg_i

        pushing_w = self.weights * torch.exp(l_dist * self.e)
        neg_exp_dot = (pushing_w * (torch.exp(neg)) * neg_i).sum(1)

        # For each query sample, if there is no negative pair, zero-out the loss.
        no_neg_flag = (neg_i).sum(1).bool()

        # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
        denom = pos_i.sum(1)

        loss = ((-torch.log(
            torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (
                    pos_i)).sum(1) / denom)

        loss = (self.weights * (loss * no_neg_flag).unsqueeze(-1)).mean()

        return loss