import copy
from typing import List, Tuple

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from torch.nn import Identity
from torch.optim import SGD
from torchvision.models import resnet50
from lightly.utils.lars import LARS
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import (
    activate_requires_grad,
    deactivate_requires_grad,
    get_weight_decay_parameters,
    update_momentum,
)
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from utils.benchmarking.online_regressor import OnlineLinearRegressor

class DINO(LightningModule):
    def __init__(self, args, data_stats) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # resnet = resnet50()
        # resnet.fc = Identity()  # Ignore classification head
        # self.backbone = resnet

        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        input_dim = self.backbone.embed_dim

        self.projection_head = DINOProjectionHead(input_dim=input_dim, freeze_last_layer=1)
        self.student_backbone = copy.deepcopy(self.backbone)
        self.student_projection_head = DINOProjectionHead(input_dim=input_dim, )
        self.criterion = DINOLoss(center_momentum=0.4)

        self.data_stats = data_stats

        self.online_regressor = OnlineLinearRegressor(self.data_stats, input_dim)

        self.val_preds = []
        self.val_targets = []

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.r2 = torchmetrics.R2Score()
        self.mse = torchmetrics.MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        projections = self.student_projection_head(features)
        return projections

    def on_train_start(self) -> None:
        deactivate_requires_grad(self.backbone)
        deactivate_requires_grad(self.projection_head)

    def on_train_end(self) -> None:
        activate_requires_grad(self.backbone)
        activate_requires_grad(self.projection_head)

    def training_step(
        self, batch: Tuple[List[torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)

        views, targets = batch[0], batch[1]
        
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        teacher_features = self.forward(global_views).flatten(start_dim=1)
        teacher_projections = self.projection_head(teacher_features)
        student_projections = torch.cat(
            [self.forward_student(global_views), self.forward_student(local_views)]
        )

        loss = self.criterion(
            teacher_out=teacher_projections.chunk(2),
            student_out=student_projections.chunk(len(views)),
            epoch=self.current_epoch,
        )
        self.log_dict(
            {"train_loss": loss, "ema_momentum": momentum},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        regr_loss, regr_log = self.online_regressor.training_step(
            (teacher_features.chunk(2)[0].detach(), targets), batch_idx
        )
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        return loss + 1e-3*regr_loss
    

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        regr_loss, preds, targets = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.val_preds.append(preds.detach())
        self.val_targets.append(targets)
        return regr_loss
    
    def on_validation_epoch_end(self):
        val_loss = torch.stack(outputs).mean()
        preds = torch.cat([pred.to(self.mae.device) for pred in self.val_preds], dim=0)
        targets = torch.cat([target.to(self.mae.device) for target in self.val_targets], dim=0)

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
            [self.student_backbone, self.student_projection_head]
        )
        # For ResNet50 we use SGD instead of AdamW/LARS as recommended by the authors:
        # https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
        optimizer = SGD(
            [
                {"name": "dino", "params": params},
                {
                    "name": "dino_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_regressor",
                    "params": self.online_regressor.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.03 * self.args.batch_size * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
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
    
    # def on_after_backward(self):
    #     self.student_projection_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)


    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=3.0, gradient_clip_algorithm="norm"):
        # if optimizer_idx == 0:
            # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
        )

        self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)

    # def configure_gradient_clipping(
    #     self,
    #     optimizer: Optimizer,

    # ) -> None:
    #     self.clip_gradients(
    #         optimizer=optimizer,
    #         gradient_clip_val=3.0,
    #         gradient_clip_algorithm="norm",
    #     )
    #     self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)