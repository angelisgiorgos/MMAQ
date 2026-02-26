import torch
import torch.nn as nn

class UncertaintyRevised(nn.Module):
    def __init__(self):
        super().__init__()

        self.log_vars = nn.Parameter(torch.FloatTensor([0.8, 0.8, 0.8, 1]))

    def forward(self, loss1, loss2, loss3, loss4):

        loss1 = 1 / (self.log_vars[0] ** 2) * loss1 + torch.log(1 + self.log_vars[0] ** 2)
        loss2 = 1 / (self.log_vars[1] ** 2) * loss2 + torch.log(1 + self.log_vars[1] ** 2)
        loss3 = 1 / (self.log_vars[2] ** 2) * loss3 + torch.log(1 + self.log_vars[2] ** 2)
        loss4 = 1 / (self.log_vars[3] ** 2) * loss4 + torch.log(1 + self.log_vars[3] ** 2)


        loss_weight = torch.abs(2 - torch.abs(self.log_vars[0]) - torch.abs(self.log_vars[1]) - torch.abs(self.log_vars[2]) - torch.abs(self.log_vars[3]))

        return loss1 + loss2 + loss3 + loss4 + loss_weight



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

