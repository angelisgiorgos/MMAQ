import copy
from typing import List, Tuple

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import Identity
from models.backbones.model import S2Backbone
import torchmetrics
from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler


class BarlowTwins(LightningModule):
    def __init__(self, args, data_stats) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.data_stats = data_stats

        self.backbone = S2Backbone(args)
        self.projection_head = BarlowTwinsProjectionHead()
        self.criterion = BarlowTwinsLoss(lambda_param=5e-3, gather_distributed=True)

        self.online_regressor = OnlineLinearRegressor(self.data_stats)

        self.val_preds = []
        self.val_targets = []

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.r2 = torchmetrics.R2Score()
        self.mse = torchmetrics.MeanSquaredError()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        im_views, tab_views, targets, _ = batch
        views = [im_views[0]["img"].float(), im_views[1]["img"].float()]
        input_view = torch.cat(views)
        features = self.forward(input_view).flatten(start_dim=1)
        z = self.projection_head(features)
        z0, z1 = z.chunk(len(views))
        loss = self.criterion(z0, z1)

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # Online linear evaluation.
        regr_loss, regr_log = self.online_regressor.training_step(
            (features.detach(), targets.repeat(len(views))), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-2 * regr_loss
    
    
    def validation_step(
        self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int
    ) -> Tensor:
        im_views, tab_views, targets, _ = batch
        features = self.forward(im_views[0]["img"].float()).flatten(start_dim=1)
        regr_loss, preds, targets = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.val_preds.append(preds.clone().detach())
        self.val_targets.append(targets)
        return regr_loss

    
    def on_validation_epoch_end(self):
        val_loss = torch.stack(outputs).mean()
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        metrics = {
            "mae": self.mae(preds, targets),
            "mape": self.mape(preds, targets),
            "r2": self.r2(preds, targets),
            "mse": self.mse(preds, targets)
        }

        self.log("val_epoch_loss", val_loss, prog_bar=True)
        self.log_dict({f"val_{k}": acc for k, acc in metrics.items()}, prog_bar=True)

    def configure_optimizers(self):
        lr_factor = self.args.batch_size * self.trainer.world_size / 256

        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = LARS(
            [
                {"name": "barlowtwins", "params": params},
                {
                    "name": "barlowtwins_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                    "lr": 0.0048 * lr_factor,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.2 * lr_factor,
            momentum=0.9,
            weight_decay=1.5e-6,
        )

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]