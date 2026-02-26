""" BARLOW TWINS Model """
from typing import List, Tuple, Any


import warnings
import torch
import torchmetrics

from utils.benchmarking.online_regressor import OnlineLinearRegressor
from models.multimodal.base import BaseMultimodalModel
from lightly.models.modules import BarlowTwinsProjectionHead
from models.backbones.model import TabularNet, ImagingNet
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from losses import select_loss_imaging, select_loss_tabular, Multimodal_Loss


class MM_BarlowTwins(BaseMultimodalModel):
    """Implementation of the BarlowTwins architecture."""

    def __init__(
        self,
        args,
        data_stats,
        hidden_dim: int = 4096,
        out_dim: int = 256,
        m: float = 0.9,
    ):
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.m = m
        super().__init__(args, data_stats)

        self.online_regressor = OnlineLinearRegressor(
            self.data_stats,
            feature_dim=self.pooled_dim + self.args.embedding_dim
        )

        warnings.warn(
            Warning(
                "The high-level building block BYOL will be deprecated in version 1.3.0. "
                + "Use low-level building blocks instead. "
                + "See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information"
            ),
            DeprecationWarning,
        )

    def _build_backbones(self):
        self.encoder_imaging = ImagingNet(self.args)
        self.pooled_dim = 2048
        self.encoder_tabular = TabularNet(self.args)

    def _build_projectors(self):
        self.projector_imaging = BarlowTwinsProjectionHead(self.pooled_dim, self.hidden_dim, self.pooled_dim)
        self.projector_tabular = BarlowTwinsProjectionHead(self.args.embedding_dim, self.args.embedding_dim*2, self.args.projection_dim)

    def _build_losses(self):
        self.initialize_training_losses()

    def _build_metrics(self):
        self.val_mae = torchmetrics.MeanAbsoluteError(dist_sync_on_step=True)
        self.val_mape = torchmetrics.MeanAbsolutePercentageError(dist_sync_on_step=True)
        self.val_r2 = torchmetrics.R2Score(dist_sync_on_step=True)
        self.val_mse = torchmetrics.MeanSquaredError(dist_sync_on_step=True)
        
    def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates projection and encoding of imaging data.
        """
        y = self.encoder_imaging(x)
        z = self.projector_imaging(y)
        return z, y

    def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates projection and encoding of tabular data.
        """
        y = self.encoder_tabular(x).flatten(start_dim=1)
        z = self.projector_tabular(y)
        return z, y
    
    def initialize_training_losses(self):
        imaging_loss = select_loss_imaging(args=self.args)
        tabular_loss = select_loss_tabular(args=self.args)
        
        self.criterion_train = Multimodal_Loss(args=self.args,
                                               imaging_loss=imaging_loss,
                                               tabular_loss=tabular_loss)
        
        self.criterion_val = self.criterion_train

        
    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor] | None], batch_idx) -> Any:
        im_views, tab_views, y = batch[0], batch[1], batch[2]
        
        z0, embeddings_0 = self.forward_imaging(im_views[0])
        z1, _ = self.forward_imaging(im_views[1])
        
        z_t0, temb_0 = self.forward_tabular(tab_views[0])
        z_t1, _ = self.forward_tabular(tab_views[1])
        loss = self.criterion_train(z0, z1, z_t0, z_t1)
        
        emb = torch.cat([embeddings_0.detach(), temb_0.detach()], dim=1)
        regr_loss, regr_log = self.online_regressor.training_step((emb, y), batch_idx)
        
        total_loss = loss + 1e-4 * regr_loss
        self.log(f"train_loss", total_loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=len(y))
        self.log_dict(regr_log, sync_dist=True, batch_size=len(y))
        
        return total_loss


    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor] | None], batch_idx) -> Any:
        # Validate contrastive model
        im_views, tab_views, y = batch[0], batch[1], batch[2]
        z0, embeddings_0 = self.forward_imaging(im_views[0])
        z1, _ = self.forward_imaging(im_views[1])
        z_t0, temb_0 = self.forward_tabular(tab_views[0])
        z_t1, _ = self.forward_tabular(tab_views[1])
        loss = self.criterion_val(z0, z1, z_t0, z_t1)
        
        emb = torch.cat([embeddings_0.detach(), temb_0.detach()], dim=1)
        regr_loss, preds, targets = self.online_regressor.validation_step((emb, y), batch_idx)

        self.val_mae.update(preds, targets)
        self.val_mape.update(preds, targets)
        self.val_r2.update(preds, targets)
        self.val_mse.update(preds, targets)

        self.log("val_step_loss", regr_loss, prog_bar=False, sync_dist=True, batch_size=targets.size(0))

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


    def initialize_scheduler(self, optimizer: Any):
        if self.args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   T_max=self.args.max_epochs, 
                                                                   )
        elif self.args.scheduler == 'anneal':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
                                                      warmup_epochs=self.args.warmup_epochs, 
                                                      max_epochs = self.args.max_epochs)
        else:
            raise ValueError('Valid schedulers are "cosine" and "anneal"')
        
        return scheduler
    
    
    def configure_optimizers(self) -> Any:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        optimizer = torch.optim.Adam(
        [
            {'params': self.encoder_imaging.parameters()}, 
            {'params': self.projector_imaging.parameters()},
            {'params': self.encoder_tabular.parameters()},
            {'params': self.projector_tabular.parameters()},
            {'params': self.online_regressor.parameters(), 'weight_decay': 0.0}
        ], 
        lr=self.args.lr,
        weight_decay=self.args.weight_decay)
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )