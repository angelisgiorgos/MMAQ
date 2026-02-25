import torch
import torchmetrics
from torch import nn, Tensor
from models.backbones.model import S2Backbone, S5Backbone
from models.backbones.projector.mmaq_projector import MMAQProjector, AQRProjector
from models.backbones.tabularnets import TabularAttention, DANet
from losses.decur_loss import MultiModalDecur
from losses.mixup import MixUPLoss
from flash.core.optimizers import LARS
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models.utils import get_weight_decay_parameters

from models.multimodal.base import BaseMultimodalModel


class MMAQ(BaseMultimodalModel):
    """
    MMAQ pretraining with Online Regression 

    Multi-Modal Self-Supervised Approach For Estimating Air Quality From Remote Sensing Data
    """

    def __init__(self, args, data_stats):
        super().__init__(args, data_stats)

        self.online_regressor = OnlineLinearRegressor(
            self.data_stats,
            feature_dim=self.pooled_dim * 2
        )

        self.val_preds = []
        self.val_targets = []

    # ============================================================
    # Model Builders
    # ============================================================

    def _build_backbones(self):
        self.encoder1 = S2Backbone(self.args)
        self.encoder2 = S5Backbone(self.args)

        self.pooled_dim = 2048

        if self.args.tabular_net == "initial":
            self.tabular_encoder = TabularAttention(self.args)
        elif self.args.tabular_net == "danet":
            self.tabular_encoder = DANet()
        else:
            raise ValueError(f"Unknown tabular_net: {self.args.tabular_net}")

    def _build_projectors(self):
        imaging_sizes = [
            self.args.imaging_embedding,
            512, 512, 512
        ]

        tab_sizes = [
            self.args.tabular_net_features,
            512, 512, 512
        ]

        self.projector_type = getattr(self.args, "projector", "mmaq")

        if self.projector_type == "aqr":
            self.projector1 = AQRProjector(self.args)
            self.projector2 = AQRProjector(self.args)
        else:
            self.projector1 = MMAQProjector(imaging_sizes)
            self.projector2 = MMAQProjector(imaging_sizes)

        self.projector_tab = MMAQProjector(tab_sizes)

        self.bn = nn.BatchNorm1d(imaging_sizes[-1], affine=False)

    def _build_losses(self):
        self.criterion = MultiModalDecur(self.args, self.bn)
        self.mixup = MixUPLoss(self.args, self.bn, 5.0, 0.005)

    def _build_metrics(self):
        self.mae = torchmetrics.MeanAbsoluteError(dist_sync_on_step=True)
        self.mape = torchmetrics.MeanAbsolutePercentageError(dist_sync_on_step=True)
        self.r2 = torchmetrics.R2Score(dist_sync_on_step=True)
        self.mse = torchmetrics.MeanSquaredError(dist_sync_on_step=True)

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, image: Tensor, tabular: Tensor = None, mode="img_both"):
        f1 = self.encoder1(image)

        if mode == "img1":
            return f1

        f2 = self.encoder2(image)

        if mode == "img_both":
            return torch.cat([f1, f2], dim=1)

        if mode == "tab":
            f_tab = self.tabular_encoder(tabular)
            return torch.cat([f1, f_tab], dim=1)

        # full multimodal
        f_tab = self.tabular_encoder(tabular)
        return torch.cat([f1, f2, f_tab], dim=1)

    # ============================================================
    # Training
    # ============================================================

    def training_step(self, batch, batch_idx):

        im_views, tab_views, targets, _ = batch

        ims = torch.cat(im_views[:2], dim=0)
        tabs = torch.cat(tab_views[:2], dim=0)

        f1_all = self.encoder1(ims)
        f2_all = self.encoder2(ims)
        f_tab_all = self.tabular_encoder(tabs)

        if self.projector_type == "aqr":
            z1_all = self.projector1(f1_all, f_tab_all)
            z2_all = self.projector2(f2_all, f_tab_all)
        else:
            z1_all = self.projector1(f1_all)
            z2_all = self.projector2(f2_all)

        z_tab_all = self.projector_tab(f_tab_all)

        z1, z1_hat = z1_all.chunk(2)
        z2, z2_hat = z2_all.chunk(2)
        z_tab, z_tab_hat = z_tab_all.chunk(2)

        contrastive_loss, _ = self.criterion(
            z1, z1_hat,
            z2, z2_hat,
            z_tab, z_tab_hat
        )

        # ---------- Online Regression ----------
        f1, _ = f1_all.chunk(2)
        f2, _ = f2_all.chunk(2)

        emb = torch.cat([f1.detach(), f2.detach()], dim=1)

        regr_loss, regr_log = self.online_regressor.training_step(
            (emb, targets), batch_idx
        )

        total_loss = contrastive_loss + 1e-4 * regr_loss

        self.log("train_loss", total_loss,
                 prog_bar=True,
                 sync_dist=True,
                 batch_size=len(targets))

        self.log_dict(regr_log,
                      sync_dist=True,
                      batch_size=len(targets))

        return total_loss

    # ============================================================
    # Validation
    # ============================================================

    def validation_step(self, batch, batch_idx):
        im_views, tab_views, targets, _ = batch

        features = self.forward(im_views[0], tab_views[0])
        regr_loss, preds, targets = self.online_regressor.validation_step(
            (features.detach(), targets), batch_idx
        )

        self.val_preds.append(preds.detach())
        self.val_targets.append(targets.detach())

        return regr_loss

    def on_validation_epoch_start(self):
        self.val_preds.clear()
        self.val_targets.clear()

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack(outputs).mean()

        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        metrics = {
            "val_mae": self.mae(preds, targets),
            "val_mape": self.mape(preds, targets),
            "val_r2": self.r2(preds, targets),
            "val_mse": self.mse(preds, targets),
        }

        self.log("val_epoch_loss", val_loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

        # reset metrics
        for m in [self.mae, self.mape, self.r2, self.mse]:
            m.reset()

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):

        lr_factor = self.args.batch_size * self.trainer.world_size / 256

        params, params_no_wd = get_weight_decay_parameters(
            [
                self.encoder1,
                self.encoder2,
                self.projector1,
                self.projector2,
                self.tabular_encoder,
                self.projector_tab,
            ]
        )

        optimizer = LARS(
            [
                {"name": "main", "params": params},
                {
                    "name": "no_weight_decay",
                    "params": params_no_wd,
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

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=int(
                self.trainer.estimated_stepping_batches
                / self.trainer.max_epochs * 10
            ),
            max_epochs=int(self.trainer.estimated_stepping_batches),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }
