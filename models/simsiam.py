import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity, ModuleList
from torch.nn import functional as F
from models.backbones.model import S2Backbone

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.scheduler import CosineWarmupScheduler


class SimSiam(LightningModule):
    def __init__(self, args, data_stats):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.data_stats = data_stats

        self.backbone = S2Backbone(args)
        self.projection_head = SimSiamProjectionHead(2048, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity()

        self.online_regressor = OnlineLinearRegressor(self.data_stats)


    def forward(self, x: Tensor, tab: Tensor=None) -> Tensor:
        return self.backbone(x)


    def forward_simsiam(self, x: Tensor) -> Tensor:
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p, f


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        im_views, tab_views, targets, _ = batch
        views = [im_views[0]["img"].float(), im_views[1]["img"].float()]
        z0, p0, features = self.forward_simsiam(views[0])
        z1, p1, _ = self.forward_simsiam(views[1])
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        regr_loss, regr_log = self.online_regressor.training_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-2*regr_loss


    def validation_step(
        self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int
    ) -> Tensor:
        im_views, tab_views, targets, orig_img = batch
        features = self.forward(orig_img["img"].float()).flatten(start_dim=1)
        regr_loss, regr_log = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(regr_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return regr_loss


    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = torch.optim.Adam(
            [
                {"name": "simsiam", "params": params},
                {
                    "name": "simsiam_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]