import copy
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from models.backbones.model import S2Backbone
import torchmetrics
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from utils.benchmarking.online_regressor import OnlineLinearRegressor


class BYOL(LightningModule):
    def __init__(self, args, data_stats) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.data_stats = data_stats

        self.backbone = S2Backbone(args)
        self.projection_head = BYOLProjectionHead()
        self.student_backbone = copy.deepcopy(self.backbone)
        self.student_projection_head = BYOLProjectionHead()
        self.student_prediction_head = BYOLPredictionHead()
        self.criterion = NegativeCosineSimilarity()

        self.online_regressor = OnlineLinearRegressor(self.data_stats)

        self.val_preds = []
        self.val_targets = []

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.r2 = torchmetrics.R2Score()

    
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        return features, projections


    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        projections = self.student_projection_head(features)
        predictions = self.student_prediction_head(projections)
        return predictions


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int) -> Tensor:
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.99,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)

        im_views, tab_views, targets, _ = batch
        teacher_features_0, teacher_projections_0 = self.forward_teacher(im_views[0]["img"].float())
        _, teacher_projections_1 = self.forward_teacher(im_views[1]["img"].float())
        student_predictions_0 = self.forward_student(im_views[0]["img"].float())
        student_predictions_1 = self.forward_student(im_views[1]["img"].float())

        # NOTE: Factor 2 because: L2(norm(x), norm(y)) = 2 - 2 * cossim(x, y)
        loss_0 = 2 * self.criterion(teacher_projections_0, student_predictions_1)
        loss_1 = 2 * self.criterion(teacher_projections_1, student_predictions_0)
        # NOTE: No mean because original code only takes mean over batch dimension, not
        # views.
        loss = loss_0 + loss_1
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # Online linear evaluation.
        regr_loss, regr_log = self.online_regressor.training_step(
            (teacher_features_0.detach(), targets), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-3 * regr_loss


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
    

    def validation_epoch_end(self, outputs):
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
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [
                self.student_backbone,
                self.student_projection_head,
                self.student_prediction_head,
            ]
        )
        optimizer = LARS(
            [
                {"name": "byol", "params": params},
                {
                    "name": "byol_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Settings follow original code for 100 epochs which are slightly different
            # from the paper, see:
            # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
            lr=0.45 * self.args.batch_size * self.trainer.world_size / 256,
            momentum=0.9,
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