import torch
import torch.nn.functional as F


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