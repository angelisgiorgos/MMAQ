import torch
import torch.nn as nn
from .utils import off_diagonal
from .dcl import DCL
from .uncertainty import UncertaintyRevised


class DeCURLoss(nn.Module):
    def __init__(self, args, bn):
        super().__init__()
        self.args = args
        self.bn = bn

    def _cross_corr(self, z1, z2):
        z1_bn = self.bn(z1)
        z2_bn = self.bn(z2)
        return (z1_bn.T @ z2_bn) / self.args.batch_size

    def bt_loss_single(self, z1, z2):
        c = self._cross_corr(z1, z2)

        diag = torch.diagonal(c)
        off = off_diagonal(c)

        loss = ((diag - 1) ** 2).sum() + self.args.lambd * (off ** 2).sum()
        return loss, diag.sum(), (off ** 2).sum()

    def bt_loss_cross(self, z1, z2):
        c = self._cross_corr(z1, z2)

        dim_c = self.args.dim_common
        lambd = self.args.lambd

        c_c = c[:dim_c, :dim_c]
        c_u = c[dim_c:, dim_c:]

        diag_c = torch.diagonal(c_c)
        off_c = off_diagonal(c_c)

        diag_u = torch.diagonal(c_u)
        off_u = off_diagonal(c_u)

        loss_c = ((diag_c - 1) ** 2).sum() + lambd * (off_c ** 2).sum()
        loss_u = (diag_u ** 2).sum() + lambd * (off_u ** 2).sum()

        return loss_c, loss_u

    def forward(self, z1_1, z1_2, z2_1, z2_2):
        loss1, _, _ = self.bt_loss_single(z1_1, z1_2)
        loss2, _, _ = self.bt_loss_single(z2_1, z2_2)

        loss_c, loss_u = self.bt_loss_cross(z1_1, z2_1)
        loss_cross = 0.5 * (loss_c + loss_u)

        total_loss = (loss1 + loss2 + loss_cross) / 3
        return total_loss


# ============================================================
# Multi DeCUR Loss
# ============================================================

class MultiDeCURLoss(nn.Module):
    def __init__(self, args, bn, uncertainty=False):
        super().__init__()
        self.args = args
        self.bn = bn
        self.uncertainty = uncertainty
        self.dcl_single = DCL()

        if args.correlation == "pearson":
            from .correlations import PearsonCorrelation
            self.corr = PearsonCorrelation()
        else:
            self.corr = None

        if uncertainty:
            self.uncert = UncertaintyRevised()

    # -----------------------------
    # Correlation
    # -----------------------------

    def _cross_corr(self, z1, z2):
        if self.corr is not None:
            return self.corr(z1, z2)

        z1_bn = self.bn(z1)
        z2_bn = self.bn(z2)
        return (z1_bn.T @ z2_bn) / self.args.batch_size

    # -----------------------------
    # Barlow-style losses
    # -----------------------------

    def bt_loss_single(self, z1, z2):
        c = self._cross_corr(z1, z2)

        diag = torch.diagonal(c)
        off = off_diagonal(c)

        loss = ((diag - 1) ** 2).sum() + self.args.lambd * (off ** 2).sum()
        return loss

    def bt_loss_cross(self, z1, z2):
        c = self._cross_corr(z1, z2)

        dim_c = self.args.dim_common
        lambd = self.args.lambd

        c_c = c[:dim_c, :dim_c]
        c_u = c[dim_c:, dim_c:]

        diag_c = torch.diagonal(c_c)
        off_c = off_diagonal(c_c)

        diag_u = torch.diagonal(c_u)
        off_u = off_diagonal(c_u)

        loss_c = ((diag_c - 1) ** 2).sum() + lambd * (off_c ** 2).sum()
        loss_u = (diag_u ** 2).sum() + lambd * (off_u ** 2).sum()

        return loss_c + loss_u

    # -----------------------------
    # Forward
    # -----------------------------

    def forward(self, z1_1, z1_2, z2_1, z2_2, z3_1, z3_2):

        loss1 = self.bt_loss_single(z1_1, z1_2)
        loss2 = self.bt_loss_single(z2_1, z2_2)
        loss3 = self.bt_loss_single(z3_1, z3_2)

        if self.args.decoupling:
            loss12 = self.bt_loss_cross(z1_1, z2_1)
            loss13 = self.bt_loss_cross(z1_1, z3_1)
        else:
            loss12 = self.bt_loss_single(z1_1, z2_1)
            loss13 = self.bt_loss_single(z1_1, z3_1)

        loss_cross = 0.5 * (loss12 + loss13)

        if not self.uncertainty:
            total_loss = (loss1 + loss2 + loss3 + loss_cross) / 4
        else:
            total_loss = self.uncert(loss1, loss2, loss3, loss_cross) / 4

        return total_loss