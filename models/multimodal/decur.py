# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
from typing import List, Tuple, Dict, Any
from torch import Tensor
import torch
from torch import nn
from models.backbones.model import S2Backbone, S5Backbone, TabularNet
from models.backbones.decur_projector import DeCURProjector
from models.backbones.aqr_projector import AQRProjector
from losses import DeCURLoss
from flash.core.optimizers import LARS
from pytorch_lightning import LightningModule
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models.utils import get_weight_decay_parameters


class DeCUR(LightningModule):
    def __init__(self, args, data_stats):
        super().__init__()
        self.args = args
        self.encoder1 = S2Backbone(args)
        self.encoder2 = S5Backbone(args)
        sizes = [self.args.imaging_embedding] + list(map(int, '8192-8192-8192'.split('-')))
        self.projector1 = DeCURProjector(sizes)
        self.projector2 = DeCURProjector(sizes)
        self.pooled_dim = 2048          

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        self.criterion = DeCURLoss(args, self.bn)

        self.online_regressor = OnlineLinearRegressor(feature_dim =self.pooled_dim*2, datastats=data_stats)


    def forward(self, x: Tensor) -> Tensor:
        features1 = self.encoder1(x)
        embedding = features1
        # features2 = self.encoder2(x)
        # embedding = torch.cat([features1, features2], axis=1)
        return embedding


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx) -> torch.Tensor:
        im_views, tab_views, targets, _ = batch

        features1 = self.encoder1(im_views[0])
        features1_hat = self.encoder1(im_views[1])
        features2 = self.encoder2(im_views[0])
        features2_hat = self.encoder2(im_views[1])
        z_1 = self.projector1(features1)
        z_1_1 = self.projector1(features1_hat)

        z_2 = self.projector2(features2)
        z_2_2 = self.projector2(features2_hat)


        loss, on_diag12_c = self.criterion(z_1, z_1_1, z_2, z_2_2)
        emb = torch.cat([features1.detach(), features2.detach()], axis=1)

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )
        regr_loss, regr_log = self.online_regressor.training_step(
            (emb, targets), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-3*regr_loss
    
    
    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx) -> torch.Tensor:

        """
        Validate contrastive model
        """
        im_views, tab_views, targets, original_im = batch

        features = self.forward(im_views[0])

        regr_loss, regr_log = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )


        self.log_dict(regr_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return regr_loss
    

    def configure_optimizers(self):
        lr_factor = self.args.batch_size * self.trainer.world_size / 256

        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.encoder1, self.encoder2,  self.projector1, self.projector2]
        )
        optimizer = LARS(
            [
                {"name": "decur", "params": params},
                {
                    "name": "decur_no_weight_decay",
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