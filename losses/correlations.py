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
        # data = pc_matrix.cat([x, y, z], dim=-1)
        # # Compute the covariance matrix manually
        # data_centered = data - torch.mean(data, dim=0, keepdim=True)
        # cov_matrix = torch.matmul(data_centered.t(), data_centered) / (data.size(0) - 1)
        # # correlation_matrix = torch.nn.functional.correlation_matrix(data, data)

        # diag_xx = torch.diag_embed(torch.sqrt(torch.diag(cov_matrix[:x.size(-1)])))
        # diag_yy = torch.diag_embed(torch.sqrt(torch.diag(cov_matrix[x.size(-1):x.size(-1) + y.size(-1)])))
        # diag_zz = torch.diag_embed(torch.sqrt(torch.diag(cov_matrix[-z.size(-1):])))

        # # Compute the inverse square roots of the diagonal matrices
        # inv_sqrt_cxx = torch.inverse(diag_xx.double())
        # inv_sqrt_cyy = torch.inverse(diag_yy.double())
        # inv_sqrt_czz = torch.inverse(diag_zz.double())

        # print(inv_sqrt_cxx.shape, inv_sqrt_cyy.shape, inv_sqrt_czz.shape)

        # # Compute the partial correlation matrix
        # pc_matrix = -inv_sqrt_cxx.contiguous().view(-1, 1) @ cov_matrix[:x.size(-1), x.size(-1):x.size(-1) + y.size(-1)] @ inv_sqrt_cyy.contiguous().view(1, -1)
        # pc_matrix = pc_matrix.squeeze()  # Remove extra singleton dimension
        # pc_matrix = pc_matrix @ cov_matrix[x.size(-1):x.size(-1) + y.size(-1), -z.size(-1):] @ inv_sqrt_czz.contiguous().view(-1, 1)

        # print(pc_matrix.shape)
        
        # # Ensure the resulting pc_matrix has size [512, 512]
        # pc_matrix = pc_matrix[:x.size(-1), -z.size(-1):]

        # return pc_matrix

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