import torch
from torch import nn
import torch.nn.functional as F
from losses.utils import euclidean_dist, up_triu


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


class RandomLinearProjection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.criterion = nn.MSELoss()

    def forward(self, images, labels, output):
        B = images.size(0)
        batch_X = images.view(B, -1)
        # c = torch.linalg.pinv((batch_X.T @ batch_X)) @ (batch_X.T @ batch_y)
        # c = torch.linalg.pinv((batch_X.T @ batch_X)) @ (batch_X.T @ outputs)
        c = torch.linalg.lstsq(batch_X, labels).solution
        c_pred = torch.linalg.lstsq(batch_X, output).solution
        loss = self.criterion(batch_X @ c_pred, batch_X @ c) # RLP Loss
        return loss


class OrdinalEntropy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args= args
        self.mse = nn.MSELoss()

    def ordinalentropy(self, features, gt,  mask=None):
        """
        Features: a certain layer's features
        gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
        mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
        """
        f_n, f_c = features.size()

        u_value, u_index, u_counts = torch.unique(gt, return_inverse=True, return_counts=True)
        # center_f = torch.zeros([len(u_value), f_c]).cuda()
        # for idx in range(len(u_value)):
        #     center_f[idx, :] = torch.mean(_features[u_index==idx, :], dim=0)

        center_f = torch.zeros([len(u_value), f_c]).cuda()
        u_index = u_index.squeeze()
        center_f.index_add_(0, u_index, features)
        u_counts = u_counts.unsqueeze(1)
        center_f = center_f / u_counts

        p = F.normalize(center_f, dim=1)
        _distance = euclidean_dist(p, p)
        _distance = up_triu(_distance)

        u_value = u_value.unsqueeze(1)
        _weight = euclidean_dist(u_value, u_value)
        _weight = up_triu(_weight)
        _max = torch.max(_weight)
        _min = torch.min(_weight)
        _weight = ((_weight - _min) / _max)

        _distance = _distance * _weight
        _entropy = torch.mean(_distance)      

        _features_center = p[u_index, :]
        _features = features - _features_center
        _features = _features.pow(2)
        _tightness = torch.sum(_features, dim=1)
        _mask = _tightness > 0
        _tightness = _tightness[_mask]
        _tightness = torch.sqrt(_tightness)
        _tightness = torch.mean(_tightness)

        return _tightness - _entropy


    def forward(self, features, labels, preds):
        mse_loss = self.mse(labels, preds)
        ord_entropy = self.ordinalentropy(features, labels)
        loss = mse_loss + 1e-3* ord_entropy
        return loss