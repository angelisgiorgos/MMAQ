from typing import Tuple
import torch
import torchmetrics
from torch import Tensor
from lightly.models.modules import SimCLRProjectionHead

from models.backbones.model import TabularNet, ImagingNet
from losses import DCL


from models.multimodal.base import BaseMultimodalModel
from utils.benchmarking.online_regressor import OnlineLinearRegressor

class MultimodalContrastiveSimCLR(BaseMultimodalModel):
    """
    Lightning module for multimodal SimCLR.
    """
    def __init__(self, args, data_stats):
        super().__init__(args, data_stats)
        
        self.online_regressor = OnlineLinearRegressor(
            self.data_stats,
            feature_dim=self.pooled_dim + self.args.embedding_dim
        )

    # ============================================================
    # Model Builders
    # ============================================================

    def _build_backbones(self):
        self.encoder_imaging = ImagingNet(self.args)
        self.pooled_dim = self.args.imaging_embedding
        self.encoder_tabular = TabularNet(self.args)

    def _build_projectors(self):
        self.projector_imaging = SimCLRProjectionHead(
            self.pooled_dim, 
            self.args.embedding_dim, 
            self.args.projection_dim
        )
        self.projector_tabular = SimCLRProjectionHead(
            self.args.embedding_dim, 
            self.args.embedding_dim, 
            self.args.projection_dim
        )

    def _build_losses(self):
        self.criterion_train = DCL()
        self.criterion_val = DCL()

    def _build_metrics(self):
        self.val_mae = torchmetrics.MeanAbsoluteError(dist_sync_on_step=True)
        self.val_mape = torchmetrics.MeanAbsolutePercentageError(dist_sync_on_step=True)
        self.val_r2 = torchmetrics.R2Score(dist_sync_on_step=True)
        self.val_mse = torchmetrics.MeanSquaredError(dist_sync_on_step=True)

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x: Tensor) -> Tensor:
        y = self.encoder_imaging(x)
        return y

    def forward_imaging(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.encoder_imaging(x)
        z = self.projector_imaging(y)
        return z, y

    def forward_tabular(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.encoder_tabular(x).flatten(start_dim=1)
        z = self.projector_tabular(y)
        return z, y

    # ============================================================
    # Training
    # ============================================================

    def training_step(self, batch, batch_idx):
        im_views, tab_views, y, _ = batch

        # Augmented views
        z0, embeddings_0 = self.forward_imaging(im_views[1])
        z1, embeddings_1 = self.forward_tabular(tab_views[1])

        loss = self.criterion_train(z0, z1)
        
        embeddings_cat = torch.cat([embeddings_0.detach(), embeddings_1.detach()], dim=1)
        regr_loss, regr_log = self.online_regressor.training_step((embeddings_cat, y), batch_idx)

        total_loss = loss + 1e-4 * regr_loss

        self.log("train_loss", total_loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=len(y))
        self.log_dict(regr_log, sync_dist=True, batch_size=len(y))

        return total_loss

    # ============================================================
    # Validation
    # ============================================================

    def validation_step(self, batch, batch_idx):
        im_views, tab_views, y, original_im = batch
    
        # Unaugmented views
        z0, embeddings_0 = self.forward_imaging(original_im)
        z1, embeddings_1 = self.forward_tabular(tab_views[0])
        
        loss = self.criterion_val(z0, z1)

        embeddings_cat = torch.cat([embeddings_0.detach(), embeddings_1.detach()], dim=1)
        regr_loss, preds, targets = self.online_regressor.validation_step((embeddings_cat, y), batch_idx)

        self.val_mae.update(preds, targets)
        self.val_mape.update(preds, targets)
        self.val_r2.update(preds, targets)
        self.val_mse.update(preds, targets)

        self.log("val_step_loss", regr_loss, prog_bar=False, sync_dist=True, batch_size=targets.size(0))

        if getattr(self.args, 'log_images', False) and batch_idx == 0:
            self.logger.log_image(key="Image Example", images=[im_views[1]])

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        metrics = {
            "val_mae": self.val_mae.compute(),
            "val_mape": self.val_mape.compute(),
            "val_r2": self.val_r2.compute(),
            "val_mse": self.val_mse.compute(),
        }
        
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

        self.val_mae.reset()
        self.val_mape.reset()
        self.val_r2.reset()
        self.val_mse.reset()

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {'params': self.encoder_imaging.parameters()},
                {'params': self.projector_imaging.parameters()},
                {'params': self.encoder_tabular.parameters()},
                {'params': self.projector_tabular.parameters()},
                {'params': self.online_regressor.parameters(), 'weight_decay': 0.0}
            ], 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )
        
        if getattr(self.args, 'scheduler', 'cosine') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=int(self.args.dataset_length * getattr(self.args, 'cosine_anneal_mult', 1.0)), 
                eta_min=0, 
                last_epoch=-1
            )
        else:
            raise ValueError('Valid schedulers are "cosine"')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }