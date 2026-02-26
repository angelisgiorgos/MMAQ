import torch 
from torch import nn
import torch.nn.functional as F
import torch.fft as fft


class PearsonCorrelation(nn.Module):
    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def partial_correlation_matrix(self, x, y, z, regularization=1e-6):
        data = torch.cat([x, y, z], dim=-1)
        # Compute the covariance matrix manually
        cov_matrix = torch.matmul(data.t(), data) / (data.size(0) - 1)
        cov_matrix += regularization * torch.ones_like(cov_matrix)
        # Compute the inverse square roots of the diagonal matrices
        epsilon = 1e-8
        diag_cxx = torch.diag(cov_matrix[:x.size(-1)]).float()
        diag_cyy = torch.diag(cov_matrix[x.size(-1):x.size(-1) + y.size(-1)]).float()

        diag_czz = torch.diag(cov_matrix[-z.size(-1):]).float()

        inv_sqrt_cxx = 1 / torch.sqrt(diag_cxx+ epsilon)
        inv_sqrt_cyy = 1 / torch.sqrt(torch.abs(diag_cyy)+ epsilon).float()
        inv_sqrt_czz = 1 / torch.sqrt(torch.abs(diag_czz)+ epsilon).float()
        # Extract relevant submatrices
        cov_xy = cov_matrix[:x.size(-1), x.size(-1):x.size(-1) + y.size(-1)]
        cov_yz = cov_matrix[x.size(-1):x.size(-1) + y.size(-1), -z.size(-1):]


        # Compute the partial correlation matrix
        pc_matrix = -inv_sqrt_cxx.view(-1, 1) * cov_xy * inv_sqrt_cyy.view(1, -1)
        pc_matrix = pc_matrix @ cov_yz * inv_sqrt_czz.view(-1, 1)
        return pc_matrix

    def forward(self, x, y, z):
        # Mean-center the inputs
        correlation = self.partial_correlation_matrix(x, y, z)

        return correlation


def autocorr(z: torch.Tensor) -> torch.Tensor:
    assert z.dim() == 3, "not batch"
    fz = fft.rfft(z)  # B x N x Df (= 1 + D / 2)
    fz_conj = fz.conj()  # B x N x Df
    fz_prod = fz_conj * fz  # B x N x Df
    fc = torch.sum(fz_prod, dim=1)  # B x Df
    corr_vec = fft.irfft(fc)  # B x D (= 2 * (Df - 1))
    assert z.shape[2] == corr_vec.shape[1]
    return corr_vec  # B x D


def autocorr_asym(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
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