import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity, ModuleList
from torch.nn import functional as F
from models.backbones.model import S2Backbone

from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler


class VICReg(LightningModule):
    def __init__(self, args, data_stats) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.data_stats = data_stats

        self.backbone = S2Backbone(args)
        self.projection_head = VICRegProjectionHead(num_layers=2)
        self.criterion = VICRegLoss()
        self.online_regressor = OnlineLinearRegressor(self.data_stats)
    
    def forward(self, x: Tensor, tab: Tensor=None) -> Tensor:
        return self.backbone(x)


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        im_views, tab_views, targets, _ = batch
        views = [im_views[0]["img"].float(), im_views[1]["img"].float()]
        input_view = torch.cat(views)
        features = self.forward(input_view).flatten(start_dim=1)
        z = self.projection_head(features)

        z_a, z_b = z.chunk(len(views))
        loss = self.criterion(z_a=z_a, z_b=z_b)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        regr_loss, regr_log = self.online_regressor.training_step(
            (features.detach(), targets.repeat(len(views))), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-3*regr_loss


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
        global_batch_size = self.args.batch_size * self.trainer.world_size
        base_lr = _get_base_learning_rate(global_batch_size=global_batch_size)
        optimizer = LARS(
            [
                {"name": "vicreg", "params": params},
                {
                    "name": "vicreg_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Linear learning rate scaling with a base learning rate of 0.2.
            # See https://arxiv.org/pdf/2105.04906.pdf for details.
            lr=base_lr * global_batch_size / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
                end_value=0.01,  # Scale base learning rate from 0.2 to 0.002.
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]



def _get_base_learning_rate(global_batch_size: int) -> float:
    """Returns the base learning rate for training 100 epochs with a given batch size.

    This follows section C.4 in https://arxiv.org/pdf/2105.04906.pdf.

    """
    if global_batch_size == 128:
        return 0.8
    elif global_batch_size == 256:
        return 0.5
    elif global_batch_size == 512:
        return 0.4
    else:
        return 0.3

