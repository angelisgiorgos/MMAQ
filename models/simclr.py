import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from models.backbones.model import S2Backbone

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from utils.benchmarking.online_regressor import OnlineLinearRegressor


class SimCLR(LightningModule):
    def __init__(self, args, data_stats) -> None:
        super().__init__()
        self.args = args

        self.data_stats = data_stats
        self.save_hyperparameters()

        self.backbone = S2Backbone(args)
        self.projection_head = SimCLRProjectionHead()
        self.criterion = NTXentLoss(temperature=0.1, gather_distributed=True)

        self.online_regressor = OnlineLinearRegressor(self.data_stats)

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
        regr_loss, regr_log = self.online_regressor.training_step(
            (features.detach(), targets.repeat(len(views))), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-3*regr_loss

    def validation_step(
        self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int
    ) -> Tensor:
        im_views, tab_views, targets, _ = batch
        features = self.forward(im_views[0]["img"].float()).flatten(start_dim=1)
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
        optimizer = LARS(
            [
                {"name": "simclr", "params": params},
                {
                    "name": "simclr_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            #   lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            lr=0.075 * math.sqrt(self.args.batch_size * self.trainer.world_size),
            momentum=0.9,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
            weight_decay=1e-6,
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