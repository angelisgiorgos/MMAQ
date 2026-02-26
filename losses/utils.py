import torch
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[_tmp]


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


def tuple2tensor(group_z: list, group_size: int, group_size_last: int) -> torch.Tensor:
    # pad zero if the last group has smaller dimension size
    if group_size > group_size_last:
        pad_dim = group_size - group_size_last
        pad_z = F.pad(group_z[-1], (0, pad_dim), "constant", 0)  # N x group_size
        # tuple2list to replace the last group
        group_z = list(group_z)
        group_z[-1] = pad_z
    Z = torch.stack(group_z, dim=0)  # num_groups x N x group_size
    return Z


def make_idx(num_groups: int) -> (list, list, torch.Tensor):
    z1 = []
    z2 = []
    last = []
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                z1.append(i)
                z2.append(j)
                if i == num_groups - 1 or j == num_groups - 1:
                    last.append(1)
                else:
                    last.append(0)
    last = torch.Tensor(last)
    return z1, z2, last