from typing import Dict, Tuple
import torch, lightning.pytorch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import MSELoss, Linear
import torchmetrics
from utils.utils import undo_normalization


class OnlineLinearRegressor(LightningModule):
    def __init__(
        self,
        datastats,
        feature_dim: int = 2048) -> None:
        super().__init__()
        self.datastats = datastats
        self.feature_dim = feature_dim
        self.regression_head = Linear(feature_dim, 1, bias=False)
        self.criterion = MSELoss()

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.r2 = torchmetrics.R2Score()


    def forward(self, x: Tensor) -> Tensor:
        return self.regression_head(x.detach().flatten(start_dim=1))


    def shared_step(self, batch, batch_idx, train=False) -> Tuple[Tensor, Dict[int, Tensor]]:
        features, targets = batch[0], batch[1]
        predictions = self.forward(features)
        targets = targets.float().unsqueeze(1)
        loss = self.criterion(predictions, targets)
        if not train:
            predictions, targets = undo_normalization(predictions.detach(), targets, self.datastats)
        return loss, predictions, targets

    
    def training_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, predictions, targets = self.shared_step(batch=batch, batch_idx=batch_idx, train=True)
        log_dict = {"train_online_rgr_loss": loss}
        metrics = {"mae" : self.mae(predictions, targets), "mape": self.mape(predictions, targets), "r2": self.r2(predictions, targets)}
        log_dict.update({f"train_online_{k}": acc for k, acc in metrics.items()})
        return loss, log_dict

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, predictions, targets = self.shared_step(batch=batch, batch_idx=batch_idx)
        return loss, predictions, targets