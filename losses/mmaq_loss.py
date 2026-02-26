import torch
from torch import nn
from losses.uncertainty import UncertaintyRevised
from losses.utils import off_diagonal


class MMAQLoss(nn.Module):
    def __init__(self, args, bn, uncertainty=False):
        super().__init__()
        self.args = args
        self.bn = bn
        self.uncertainty = uncertainty

        if self.uncertainty:
            self.uncert = UncertaintyRevised()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _cross_correlation(self, z1, z2):
        z1_bn = self.bn(z1)
        z2_bn = self.bn(z2)
        c = z1_bn.T @ z2_bn
        return c / self.args.batch_size

    def _identity(self, dim, device):
        return torch.eye(dim, device=device)

    # --------------------------------------------------
    # Intermodal Loss
    # --------------------------------------------------

    def intermodal_loss(self, z1, z2, z3):
        """
        Computes:
        1) Common-space invariance loss
        2) Unique-space decorrelation loss
        """

        # Cross-correlations
        c12 = self._cross_correlation(z1, z2)
        c13 = self._cross_correlation(z1, z3)
        c23 = self._cross_correlation(z2, z3)

        dim_c = self.args.dim_common
        lambd = self.args.lambd

        # --------------------------------------------------
        # 1️⃣ Common space loss
        # --------------------------------------------------

        c12_c = c12[:dim_c, :dim_c]
        c13_c = c13[:dim_c, :dim_c]
        c23_c = c23[:dim_c, :dim_c]

        I_c = self._identity(dim_c, z1.device)

        # Invariance term (maximize diagonal correlation)
        loss_inv = -torch.diagonal(c12_c).sum()

        # Decorrelation toward identity
        diff_13 = (I_c - c13_c) ** 2
        diff_23 = (I_c - c23_c) ** 2

        off_diag_mask = ~torch.eye(dim_c, dtype=torch.bool, device=z1.device)
        diff_13[off_diag_mask] *= lambd
        diff_23[off_diag_mask] *= lambd

        loss_dec = 0.025 * (diff_13.sum() + diff_23.sum()) / 2
        loss_common = loss_inv + loss_dec

        # --------------------------------------------------
        # 2️⃣ Unique space loss
        # --------------------------------------------------

        c12_u = c12[dim_c:, dim_c:]

        diag_u = torch.diagonal(c12_u)
        off_diag_u = off_diagonal(c12_u)

        loss_unique = (diag_u ** 2).sum() + lambd * (off_diag_u ** 2).sum()

        # --------------------------------------------------
        # Final intermodal loss
        # --------------------------------------------------

        loss_inter = (loss_common + loss_unique) / 2

        return loss_inter, (off_diag_u ** 2).sum()

    # --------------------------------------------------
    # Intramodal Loss
    # --------------------------------------------------

    def intramodal_loss(self, z, z_hat):
        c = self._cross_correlation(z, z_hat)

        diag = torch.diagonal(c)
        off_diag = off_diagonal(c)

        loss = ((diag - 1) ** 2).sum() + self.args.lambd * (off_diag ** 2).sum()

        return loss, diag.sum(), (off_diag ** 2).sum()

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, z1, z2, z3, z1_hat, z2_hat, z3_hat):

        loss_inter, off_diag = self.intermodal_loss(z1, z2, z3)

        loss_intra1, _, _ = self.intramodal_loss(z1, z1_hat)
        loss_intra2, _, _ = self.intramodal_loss(z2, z2_hat)
        loss_intra3, _, _ = self.intramodal_loss(z3, z3_hat)

        if not self.uncertainty:
            total_loss = (
                loss_intra1
                + loss_intra2
                + loss_intra3
                + loss_inter
            )
        else:
            total_loss = self.uncert(
                loss_intra1,
                loss_intra2,
                loss_intra3,
                loss_inter
            )

        return total_loss / 4, off_diag
