import torch
import torch.nn as nn

class UncertaintyRevised(nn.Module):
    def __init__(self):
        super().__init__()

        self.log_vars = nn.Parameter(torch.FloatTensor([0.8, 0.8, 0.8]))

    # def forward(self, loss1, loss2, loss3, loss4):
    def forward(self, loss1, loss2, loss3):

        loss1 = 1 / (self.log_vars[0] ** 2) * loss1 + torch.log(1 + self.log_vars[0] ** 2)
        loss2 = 1 / (self.log_vars[1] ** 2) * loss2 + torch.log(1 + self.log_vars[1] ** 2)
        loss3 = 1 / (self.log_vars[2] ** 2) * loss3 + torch.log(1 + self.log_vars[2] ** 2)
        # loss4 = 1 / (self.log_vars[3] ** 2) * loss4 + torch.log(1 + self.log_vars[3] ** 2)


        # loss_weight = torch.abs(2 - torch.abs(self.log_vars[0]) - torch.abs(self.log_vars[1]) - torch.abs(self.log_vars[2]) - torch.abs(self.log_vars[3]))
        loss_weight = torch.abs(2 - torch.abs(self.log_vars[0]) - torch.abs(self.log_vars[1]) - torch.abs(self.log_vars[2]))

        return loss1 + loss2 + loss3 + loss_weight
    


class UncertaintyOrig(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        lambda_loss1 = 0.8
        init_loss_1_log_var = -torch.log(torch.tensor(lambda_loss1) + eps)

        lambda_loss2 = 0.8
        init_loss_2_log_var = -torch.log(torch.tensor(lambda_loss2) + eps)

        lambda_loss3 = 0.8
        init_loss_3_log_var = -torch.log(torch.tensor(lambda_loss3) + eps)

        lambda_loss4 = 0.8
        init_loss_4_log_var = -torch.log(torch.tensor(lambda_loss4) + eps)


        self.loss_1_log_var = nn.Parameter(init_loss_1_log_var.float())

        self.loss_2_log_var = nn.Parameter(init_loss_2_log_var.float())

        self.loss_3_log_var = nn.Parameter(init_loss_3_log_var.float())

        self.loss_4_log_var = nn.Parameter(init_loss_4_log_var.float())

    def forward(self, loss1, loss2, loss3, loss4):

        loss1 = loss1 * torch.exp(-self.loss_1_log_var) + self.loss_1_log_var

        loss2 = loss2 * torch.exp(-self.loss_2_log_var) + self.loss_2_log_var

        loss3 = loss3 * torch.exp(-self.loss_3_log_var) + self.loss_3_log_var

        loss4 = loss4 * torch.exp(-self.loss_4_log_var) + self.loss_4_log_var

        return loss1 + loss2 + loss3 + loss4

