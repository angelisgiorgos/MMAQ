import copy
from typing import List, Tuple

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from models.backbones.model import S2Backbone
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    get_weight_decay_parameters,
    update_momentum,
)
from lightly.utils.scheduler import CosineWarmupScheduler
from utils.benchmarking.online_regressor import OnlineLinearRegressor


class MoCoV2(LightningModule):
    def __init__(self, args, data_stats) -> None:
        super().__init__()
        self.data_stats = data_stats
        self.args = args
        self.save_hyperparameters()
        self.backbone = S2Backbone(args)

        self.projection_head = MoCoProjectionHead()
        self.query_backbone = copy.deepcopy(self.backbone)
        self.query_projection_head = MoCoProjectionHead()
        self.criterion = NTXentLoss(
            temperature=0.2,
            memory_bank_size=(65536, 128),
            gather_distributed=True,
        )

        self.online_regressor = OnlineLinearRegressor()

    
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


    @torch.no_grad()
    def forward_key_encoder(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x, shuffle = batch_shuffle(batch=x, distributed=self.trainer.num_devices > 1)
        features = self.forward(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        features = batch_unshuffle(
            batch=features,
            shuffle=shuffle,
            distributed=self.trainer.num_devices > 1,
        )
        projections = batch_unshuffle(
            batch=projections,
            shuffle=shuffle,
            distributed=self.trainer.num_devices > 1,
        )
        return features, projections


    def forward_query_encoder(self, x: Tensor) -> Tensor:
        features = self.query_backbone(x).flatten(start_dim=1)
        projections = self.query_projection_head(features)
        return projections


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        im_views, tab_views, targets, _ = batch
        views = [im_views[0]["img"].float(), im_views[1]["img"].float()]

        # Encode queries.
        query_projections = self.forward_query_encoder(views[1])

        # Momentum update. This happens between query and key encoding, following the
        # original implementation from the authors:
        # https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/moco/builder.py#L142
        update_momentum(self.query_backbone, self.backbone, m=0.999)
        update_momentum(self.query_projection_head, self.projection_head, m=0.999)

        # Encode keys.
        key_features, key_projections = self.forward_key_encoder(views[0])
        loss = self.criterion(query_projections, key_projections)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        regr_loss, regr_log = self.online_regressor.training_step(
            (key_features.detach(), targets), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-2*regr_loss

    
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
        # NOTE: The original implementation from the authors uses weight decay for all
        # parameters.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.query_backbone, self.query_projection_head]
        )
        optimizer = SGD(
            [
                {"name": "mocov2", "params": params},
                {
                    "name": "mocov2_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=1e-3 * self.args.batch_size * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-6,
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
