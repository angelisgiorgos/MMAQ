import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from .utils import make_idx, tuple2tensor
from .dcl import DCL
import torch.distributed as dist
from .uncertainty import UncertaintyRevised, UncertaintyOrig


def wmse_loss_func(z1: torch.Tensor, z2: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Computes W-MSE's loss given two batches of whitened features z1 and z2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing whitened features from view 1.
        z2 (torch.Tensor): NxD Tensor containing whitened features from view 2.
        simplified (bool): faster computation, but with same result.

    Returns:
        torch.Tensor: W-MSE loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(z1, z2.detach(), dim=-1).mean()

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    return 2 - 2 * (z1 * z2).sum(dim=-1).mean()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class DeCURLoss(nn.Module):
    def __init__(self, args, bn):
        super().__init__()
        self.args = args
        self.bn = bn
        
    
    def bt_loss_cross(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)

        dim_c = self.args.dim_common
        c_c = c[:dim_c,:dim_c]
        c_u = c[dim_c:,dim_c:]

        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()
        
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        
        loss_c = on_diag_c + self.args.lambd * off_diag_c
        loss_u = on_diag_u + self.args.lambd * off_diag_u
        
        return loss_c,on_diag_c,off_diag_c,loss_u,on_diag_u,off_diag_u   


    def bt_loss_single(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss,on_diag,off_diag


    def forward(self, z1_1, z1_2, z2_1, z2_2):      

        loss1, on_diag1, off_diag1 = self.bt_loss_single(z1_1,z1_2)
        loss2, on_diag2, off_diag2 = self.bt_loss_single(z2_1,z2_2)        
        loss12_c, on_diag12_c, off_diag12_c, loss12_u, on_diag12_u, off_diag12_u = self.bt_loss_cross(z1_1,z2_1)
        loss12 = (loss12_c + loss12_u) / 2.0

        loss = (loss1 + loss2 + loss12) / 3

        return loss, on_diag12_c





class MultiDeCURLoss(nn.Module):
    def __init__(self, args, bn, uncertainty = True):
        super().__init__()
        self.args = args
        self.bn = bn
        self.uncertainty = uncertainty
        self.dcl_single = DCL()

        if self.args.correlation == "pearson":
            from .correlations import PearsonCorrelation
            self.pearson = PearsonCorrelation()


        if self.uncertainty:

            self.uncert = UncertaintyRevised()

    
    def canonical_correlation(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        return c


    def autocorr_asym(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        assert z1.dim() == 3, "not batch"
        assert z2.dim() == 3, "not batch"
        fz1 = fft.rfft(z1)  # B x N x Df
        fz2 = fft.rfft(z2)  # B x N x Df
        fz1_conj = fz1.conj()  # B x N x Df
        fz_prod = fz1_conj * fz2  # B x N x Df
        fc = torch.sum(fz_prod, dim=1)  # B x Df
        corr_vec = fft.irfft(fc)  # B x D
        assert z1.shape[2] == corr_vec.shape[1]
        return corr_vec  # B x D

    
    def bt_loss_cross(self, z1, z2):
        if self.args.correlation == "pearson":
             c = self.pearson(z1, z2)
        else:
            c = self.canonical_correlation(z1, z2)

        dim_c = self.args.dim_common
        c_c = c[:dim_c,:dim_c]
        c_u = c[dim_c:,dim_c:]

        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()
        
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        
        loss_c = on_diag_c + self.args.lambd * off_diag_c
        loss_u = on_diag_u + self.args.lambd * off_diag_u
        
        return loss_c,on_diag_c,off_diag_c,loss_u,on_diag_u,off_diag_u  

    

    def bt_loss_single(self, z1, z2):
        if self.args.correlation == "pearson":
             c = self.pearson(z1, z2)
        else:
            c = self.canonical_correlation(z1, z2)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss,on_diag,off_diag


    def sbarlow_loss_func(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        lamb: float = 5e-3,
        scale_loss: float = 0.025,
        exponent: int = 2,
        group_size: int = 2048,
        rand_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computes Barlow Twins style loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.

        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
                Defaults to 5e-3.
            scale_loss (float, optional): final scaling factor of the loss. Defaults to 0.025.

        Returns:
            torch.Tensor: loss.
        """

        N, D = z1.size()

        # to match the original code
        bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
        z1 = bn(z1)  # N x D
        z2 = bn(z2)  # N x D

        # diagonal loss
        var = torch.sum(z1 * z2, dim=0) / N  # diagonal elements in C: D
        on_diag = F.mse_loss(var, torch.ones_like(var), reduction="sum")

        # feature permutation
        if rand_idx != None:
            z1 = z1[:, rand_idx]  # N x D
            z2 = z2[:, rand_idx]  # N x D

        # grouping
        group_z1 = z1.split(group_size, dim=1)  # tuple: num_group x N x group_size
        group_z2 = z2.split(group_size, dim=1)  # tuple: num_group x N x group_size
        group_size_last = group_z1[-1].shape[1]
        Z1 = tuple2tensor(
            group_z1, group_size, group_size_last
        )  # num_groups x N x group_size
        Z2 = tuple2tensor(
            group_z2, group_size, group_size_last
        )  # num_groups x N x group_size

        # off-diagonal loss in diagonal sub matrices
        # circular correlation
        corr_vec = self.autocorr_asym(Z1, Z2) / N  # num_groups x group_size
        # exclude 0-th value
        if exponent == 1:
            off_diag = torch.sum(corr_vec[:, 1:].abs())  # 1
        elif exponent == 2:
            off_diag = torch.sum(corr_vec[:, 1:].pow(2))  # 1

        # off-diagonal loss in off-diagonal sub-matrices
        if Z1.shape[0] > 1:
            # make indexes to compute the combination of sub-matrices
            idx_z1, idx_z2, idx_last = make_idx(Z1.shape[0])
            Z1_off = Z1[idx_z1, :, :]  # B_off x N x group_size
            Z2_off = Z2[idx_z2, :, :]  # B_off x N x group_size
            # circualr correlation
            corr_vec = self.autocorr_asym(Z1_off, Z2_off) / N  # B_off x group_size
            # include 0-th value
            if exponent == 1:
                off_diag += torch.sum(corr_vec.abs())  # 1
            elif exponent == 2:
                off_diag += torch.sum(corr_vec.pow(2))  # 1

        loss = scale_loss * (on_diag + lamb * off_diag)

        return loss, Z1_off, off_diag


    def forward(self, z1_1, z1_2, z2_1, z2_2, z3_1, z3_2):      

        loss1, on_diag1, off_diag1 = self.bt_loss_single(z1_1,z1_2)
        loss2, on_diag2, off_diag2 = self.bt_loss_single(z2_1,z2_2)
        loss3, on_diag3, off_diag3 = self.bt_loss_single(z3_1,z3_2)

        # loss1 = self.dcl_single(z1_1,z1_2)
        # loss2 = self.dcl_single(z2_1,z2_2)
        # loss3 = self.dcl_single(z3_1,z3_2)


        if not self.args.decoupling:
            loss12, on_diag12_c, off_diag12_c =  self.sbarlow_loss_func(z1_1,z2_1)
            loss13, on_diag13_c, off_diag13_c =  self.sbarlow_loss_func(z1_1,z3_1)
        else:
            loss12_c, on_diag12_c, off_diag12_c, loss12_u, on_diag12_u, off_diag12_u = self.bt_loss_cross(z1_1,z2_1)
            loss13_c, on_diag13_c, off_diag13_c, loss13_u, on_diag13_u, off_diag13_u = self.bt_loss_cross(z1_1,z3_1)
            loss12 = (loss12_c + loss12_u) 

            loss13 = (loss13_c + loss13_u) 

        # loss_cross = (loss12+loss13) / 2.0

        if not self.uncertainty:
            loss = (loss1 + loss2 + loss3 + loss_cross) / 4
        else:
            # loss = self.uncert(loss1, loss2, loss3, loss_cross) / 4
            loss = self.uncert(loss1, loss2, loss3) / 4

        return loss, on_diag12_c


class MultiModalDecur(nn.Module):
    def __init__(self, args, bn, uncertainty=True):
        super().__init__()
        self.args = args
        self.bn = bn
        self.uncertainty = uncertainty

        if self.uncertainty:
            from .uncertainty import UncertaintyRevised

            self.uncert = UncertaintyRevised()

    def canonical_correlation(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size * 2)
        return c

    
    def intermodal_loss(self, z1, z2, z3):
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c1 = torch.mm(self.bn(z1).T, self.bn(z3))
        c2 = torch.mm(self.bn(z2).T, self.bn(z3))

        c.div_(self.args.batch_size)
        c1.div_(self.args.batch_size)
        c2.div_(self.args.batch_size)

        dim_c = self.args.dim_common
        c_c = c[:dim_c,:dim_c]
        c_u = c[dim_c:,dim_c:]

        c1_c = c1[:dim_c,:dim_c]
        c1_u = c1[dim_c:,dim_c:]
        
        c2_c = c2[:dim_c,:dim_c]
        c2_u = c2[dim_c:,dim_c:]

        N, D = z1.size()
        # on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        # off_diag_c = off_diagonal(c_c).pow_(2).sum()
        
        # on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        # off_diag_u = off_diagonal(c_u).pow_(2).sum()
        
        # loss_c = on_diag_c + self.args.lambd * off_diag_c
        # loss_u = on_diag_u + self.args.lambd * off_diag_u


        loss_inv = -torch.diagonal(c_c).sum()


        iden_c = torch.tensor(torch.eye(c_c.shape[0])).to(z1.device)
        cdif_dec1_c = (iden_c - c1_c).pow(2)
        cdif_dec2_c = (iden_c - c2_c).pow(2)

        cdif_dec1_c[~iden_c.bool()] *= self.args.lambd
        loss_dec1_c = 0.025 * cdif_dec1_c.sum()

        cdif_dec2_c[~iden_c.bool()] *= self.args.lambd
        loss_dec2_c = 0.025 * cdif_dec2_c.sum()


        loss_c = loss_inv + (loss_dec1_c + loss_dec2_c) / 2

        # loss_inv_u = -torch.diagonal(c_u).sum()
        # iden_u = torch.tensor(torch.eye(c_u.shape[0])).to(z1.device)
        # loss_dec1_u = (iden_u - c1_u).pow(2).sum()
        # loss_dec2_u = (iden_u - c2_u).pow(2).sum()
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        loss_u = on_diag_u + self.args.lambd *off_diag_u
        loss = (loss_c+loss_u) / 2
        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(torch.eye(c.shape[0])).to(z1.device)

        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + self.args.lambd * (loss_dec1 + loss_dec2)
        return loss, off_diag_u


    def intramodal_loss(self, z, z_hat):
        c = self.canonical_correlation(z, z_hat)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss,on_diag,off_diag


    def forward(self, z1, z2, z3, z1_hat, z2_hat, z3_hat):
        loss_inter, off_diag = self.intermodal_loss(z1, z2, z3)
        # loss_intra1, _, off_diag = self.intramodal_loss(z1, z1_hat)
        # loss_intra2, _, _ = self.intramodal_loss(z2, z2_hat)
        # loss_intra3, _, _ = self.intramodal_loss(z3, z3_hat)

        # if not self.uncertainty:
        #     loss = (loss_intra1 + loss_intra2 + loss_intra3 + loss_inter)
        # else:
        #     # loss = self.uncert(loss_intra1, loss_intra2, loss_intra3, loss_inter)
        #     loss = self.uncert(loss_intra1, loss_intra2, loss_intra3)

        loss = loss_inter
        return loss / 4, off_diag