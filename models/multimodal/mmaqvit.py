import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel
import torchmetrics
from flash.core.optimizers import LARS
from utils.benchmarking.online_regressor import OnlineLinearRegressor
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models.utils import get_weight_decay_parameters
from losses.clip_loss import MultimodalJointLoss


# ============================================================
# Cross-Modal Transformer Fusion
# ============================================================
class CrossModalTransformer(nn.Module):
    def __init__(self, dim=768, depth=4, heads=8):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, tokens):
        return self.transformer(tokens)


# ============================================================
# Projection Head
# ============================================================

class MLPProjector(nn.Module):
    def __init__(self, dim=768, out_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# MMAQ-ViT
# ============================================================

class MMAQViT(nn.Module):
    def __init__(self, args, data_stats):
        super().__init__()
        self.args = args
        self.embed_dim = 768
        # --------------------------------------------------
        # Vision Transformers
        # --------------------------------------------------
        self.data_stats = data_stats

        self.val_preds = []
        self.val_targets = []

        # --------------------------------------------------
        # Text Encoder
        # --------------------------------------------------
        self.text_encoder = AutoModel.from_pretrained(
            "bert-base-uncased"
        )
        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------
        self.fusion = CrossModalTransformer(dim=self.embed_dim)

        # --------------------------------------------------
        # Online Regressor
        # --------------------------------------------------
        self.online_regressor = OnlineLinearRegressor(
            self.data_stats,
            feature_dim=self.embed_dim * 2
        )

        self._build_backbones()
        self._build_projectors()
        self._build_losses()
        self._build_metrics()

    def _build_backbones(self):
        self.vit_s2 = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            channels=12
        )

        self.vit_s5 = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            channels=1
        )

    # --------------------------------------------------
    # Projectors
    # --------------------------------------------------
    def _build_projectors(self):
        self.proj_s2 = MLPProjector(self.embed_dim)
        self.proj_s5 = MLPProjector(self.embed_dim)
        self.proj_txt = MLPProjector(self.embed_dim)
        self.proj_fused = MLPProjector(self.embed_dim)
    
    def _build_losses(self):
        self.ssl_loss = MultimodalJointLoss()

    def _build_metrics(self):
        self.val_mae = torchmetrics.MeanAbsoluteError(dist_sync_on_step=True)
        self.val_mape = torchmetrics.MeanAbsolutePercentageError(dist_sync_on_step=True)
        self.val_r2 = torchmetrics.R2Score(dist_sync_on_step=True)
        self.val_mse = torchmetrics.MeanSquaredError(dist_sync_on_step=True)


    # ============================================================
    # Forward
    # ============================================================
    def forward(self, img_s2, img_s5, input_ids):

        f_s2 = self.vit_s2(img_s2)
        f_s5 = self.vit_s5(img_s5)

        text_out = self.text_encoder(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"]
        )

        f_txt = text_out.last_hidden_state[:, 0]  # CLS token

        # Stack tokens (B, 3, 768)
        tokens = torch.stack([f_s2, f_s5, f_txt], dim=1)

        fused_tokens = self.fusion(tokens)

        # Use mean pooling across modalities
        fused = fused_tokens.mean(dim=1)

        return {
            "s2": f_s2,
            "s5": f_s5,
            "txt": f_txt,
            "fused": fused
        }

    def training_step(self, batch, batch_idx):
        im1, _, s51, _, tab1, _, targets = batch

        out1 = self.forward(im1, s51, tab1)
        # out2 = self.forward(im2, s52, tab2)

        f_s2_1 = self.proj_s2(out1["s2"])
        f_s5_1 = self.proj_s5(out1["s5"])
        f_txt_1 = self.proj_txt(out1["txt"])
        f_fused_1 = self.proj_fused(out1["fused"])

        fs = {
            "s2": f_s2_1,
            "s5": f_s5_1,
            "txt": f_txt_1,
            "fused": f_fused_1
        }

        loss_ssl = self.criterion(fs)

        preds = self.online_regressor(out1["fused"])
        loss_reg = F.mse_loss(preds, targets)

        loss = loss_ssl + 1e-4 * loss_reg

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        imgs, s5, tab, targets = batch
        _, _, _, fused = self.forward(imgs, s5, tab)
        preds = self.online_regressor(fused)

        loss = F.mse_loss(preds, targets)

        # Update metrics (TorchMetrics handles accumulation internally)
        self.val_mae.update(preds, targets)
        self.val_mape.update(preds, targets)
        self.val_r2.update(preds, targets)
        self.val_mse.update(preds, targets)

        # Log step loss (proper DDP sync)
        self.log(
            "val_step_loss",
            regr_loss,
            prog_bar=False,
            sync_dist=True,
            batch_size=targets.size(0)
        )

    def on_validation_epoch_start(self):
        self.val_preds.clear()
        self.val_targets.clear()

    def on_validation_epoch_end(self):
        # Compute metrics (aggregated across devices)
        metrics = {
            "val_mae": self.val_mae.compute(),
            "val_mape": self.val_mape.compute(),
            "val_r2": self.val_r2.compute(),
            "val_mse": self.val_mse.compute(),
        }

        # Log once (Lightning handles sync)
        self.log_dict(
            metrics,
            prog_bar=True,
            sync_dist=True
        )

        # Reset metrics for next epoch
        self.val_mae.reset()
        self.val_mape.reset()
        self.val_r2.reset()
        self.val_mse.reset()

    # ============================================================
    # Optimizer
    # ============================================================
    def configure_optimizers(self):

        lr_factor = self.args.batch_size * self.trainer.world_size / 256

        params, params_no_wd = get_weight_decay_parameters(
            [
                self.vit_s2,
                self.vit_s5,
                self.text_encoder,
                self.fusion,
                self.proj_s2,
                self.proj_s5,
                self.proj_txt,
                self.proj_fused,
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