import torch
import torch.nn as nn

class UncertaintyRevised(nn.Module):
    def __init__(self, num_losses: int = 4):
        super().__init__()

        if num_losses == 4:
            init_vals = [0.8, 0.8, 0.8, 1.0]
        else:
            init_vals = [0.8] * (num_losses - 1) + [1.0]

        self.log_vars = nn.Parameter(torch.FloatTensor(init_vals))

    def forward(self, *losses):

        total_loss = 0.0

        for i, loss in enumerate(losses):
            total_loss += 1 / (self.log_vars[i] ** 2) * loss + torch.log(1 + self.log_vars[i] ** 2)

        sum_abs_log_vars = torch.sum(torch.abs(self.log_vars))
        loss_weight = torch.abs(2 - sum_abs_log_vars)

        return total_loss + loss_weight



class UncertaintyOrig(nn.Module):
    def __init__(self, num_losses: int = 4, eps: float = 1e-6, init_lambda: float = 0.8):
        super().__init__()
        self.num_losses = num_losses
        init_log_var = -torch.log(torch.tensor(init_lambda) + eps)
        self.log_vars = nn.Parameter(torch.full((num_losses,), init_log_var.item()))

    def forward(self, *losses):
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} losses, got {len(losses)}")
            
        loss_stack = torch.stack(losses)
        weighted_losses = loss_stack * torch.exp(-self.log_vars) + self.log_vars
        
        return torch.sum(weighted_losses)

