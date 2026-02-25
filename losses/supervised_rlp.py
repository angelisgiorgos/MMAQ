import torch
import torch.nn as nn

class RandomLinearProjection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.criterion = nn.MSELoss()

    def forward(self, images, labels, output):
        B = images.size(0)
        batch_X = images.view(B, -1)
        # c = torch.linalg.pinv((batch_X.T @ batch_X)) @ (batch_X.T @ batch_y)
        # c = torch.linalg.pinv((batch_X.T @ batch_X)) @ (batch_X.T @ outputs)
        c = torch.linalg.lstsq(batch_X, labels).solution
        c_pred = torch.linalg.lstsq(batch_X, output).solution
        loss = self.criterion(batch_X @ c_pred, batch_X @ c) # RLP Loss
        return loss