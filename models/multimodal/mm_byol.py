""" BYOL Model """
from typing import List, Tuple, Dict, Any

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import warnings
import copy
import torch
import torch.nn as nn
import torchmetrics
import torch
import torch.nn as nn
import torchmetrics
import lightning.pytorch as pl
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from utils.benchmarking.online_regressor import OnlineLinearRegressor

from models.multimodal.base import BaseMultimodalModel
from lightly.models.modules import BYOLProjectionHead
from models.backbones.model import TabularNet, ImagingNet

from losses import select_loss_imaging, select_loss_tabular, Multimodal_Loss


def _get_byol_mlp(num_ftrs: int, hidden_dim: int, out_dim: int):
    """Returns a 2-layer MLP with batch norm on the hidden layer.

    Reference (12.03.2021)
    https://arxiv.org/abs/2006.07733

    """
    modules = [
        nn.Linear(num_ftrs, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    return nn.Sequential(*modules)



def _deactivate_requires_grad(params):
    """Deactivates the requires_grad flag for all parameters."""
    for param in params:
        param.requires_grad = False


def _do_momentum_update(prev_params, params, m):
    """Updates the weights of the previous parameters."""
    for prev_param, param in zip(prev_params, params):
        prev_param.data = prev_param.data * m + param.data * (1.0 - m)


class MM_BYOL(BaseMultimodalModel):
    """Implementation of the BYOL architecture."""

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
        
        # In this implementation, the multimodal regression is concatenated Imaging + Tabular
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

        self.encoder_imaging_momentum = copy.deepcopy(self.encoder_imaging)
        self.encoder_tabular_momentum = copy.deepcopy(self.encoder_tabular)
        _deactivate_requires_grad(self.encoder_imaging_momentum.parameters())
        _deactivate_requires_grad(self.encoder_tabular_momentum.parameters())

    def _build_projectors(self):
        self.projector_imaging = BYOLProjectionHead(self.pooled_dim, self.hidden_dim, self.out_dim)
        self.prediction_imaging = BYOLProjectionHead(self.out_dim, self.hidden_dim, self.out_dim)
        
        self.projector_tabular = BYOLProjectionHead(self.args.embedding_dim, self.args.embedding_dim*2, self.args.projection_dim)
        self.predictor_tabular = BYOLProjectionHead(self.args.projection_dim, self.args.embedding_dim*2, self.args.projection_dim)

        self.projector_imaging_momentum = copy.deepcopy(self.projector_imaging)
        self.projector_tabular_momentum = copy.deepcopy(self.projector_tabular)
        _deactivate_requires_grad(self.projector_imaging_momentum.parameters())
        _deactivate_requires_grad(self.projector_tabular_momentum.parameters())

    def _build_losses(self):
        self.initialize_training_losses()

    def _build_metrics(self):
        self.val_mae = torchmetrics.MeanAbsoluteError(dist_sync_on_step=True)
        self.val_mape = torchmetrics.MeanAbsolutePercentageError(dist_sync_on_step=True)
        self.val_r2 = torchmetrics.R2Score(dist_sync_on_step=True)
        self.val_mse = torchmetrics.MeanSquaredError(dist_sync_on_step=True)
    
    
    @torch.no_grad()
    def _momentum_update(self, m: float = 0.999):
        """Performs the momentum update for the backbone and projection head."""
        _do_momentum_update(
            self.encoder_imaging_momentum.parameters(),
            self.encoder_imaging.parameters(),
            m=m,
        )
        _do_momentum_update(
            self.projector_imaging_momentum.parameters(),
            self.projector_imaging.parameters(),
            m=m,
        )
        
        # TABULAR NETWORK
        _do_momentum_update(
            self.encoder_tabular_momentum.parameters(),
            self.encoder_tabular.parameters(),
            m=m,
        )
        _do_momentum_update(
            self.projector_tabular_momentum.parameters(),
            self.projector_tabular.parameters(),
            m=m,
        )
        
    
    @torch.no_grad()
    def _batch_shuffle(self, batch: torch.Tensor):
        """Returns the shuffled batch and the indices to undo."""
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle

    
    @torch.no_grad()
    def _batch_unshuffle(self, batch: torch.Tensor, shuffle: torch.Tensor):
        """Returns the unshuffled batch."""
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]

    
    def _forward(self, x0: torch.Tensor, tx0: torch.Tensor, x1: torch.Tensor | None = None, tx1: torch.Tensor | None = None):
        """Forward pass through the encoder and the momentum encoder.

        Performs the momentum update, extracts features with the backbone and
        applies the projection (and prediciton) head to the output space. If
        x1 is None, only x0 will be processed otherwise, x0 is processed with
        the encoder and x1 with the momentum encoder.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns:
            The output proejction of x0 and (if x1 is not None) the output
            projection of x1.

        Examples:
            >>> # single input, single output
            >>> out = model._forward(x)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model._forward(x0, x1)

        """
        # forward pass of first input x0
        self._momentum_update(self.m)
        f0 = self.encoder_imaging(x0)
        z0 = self.projector_imaging(f0)
        out0 = self.prediction_imaging(z0)

        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():
            f1 = self.encoder_imaging_momentum(x1)
            out1 = self.projector_imaging_momentum(f1)
            
        tf0 = self.encoder_tabular(tx0)
        tz0 = self.projector_tabular(tf0)
        tout0 = self.predictor_tabular(tz0)
        
        if tx1 is None:
            return tout0
        
        # forward pass of second input x1
        with torch.no_grad():
            tf1 = self.encoder_tabular_momentum(tx1)
            tout1 = self.projector_tabular_momentum(tf1)
            
        return out0, tout0, out1, tout1, [f0, tf0]

    
    def initialize_training_losses(self):
        imaging_loss = select_loss_imaging(args=self.args)
        tabular_loss = select_loss_tabular(args=self.args)
        
        self.criterion_train = Multimodal_Loss(args=self.args,
                                               imaging_loss=imaging_loss,
                                               tabular_loss=tabular_loss)
        
        self.criterion_val = self.criterion_train
    
    
    def training_step(self, batch: Tuple[List[torch.Tensor], 
                                         List[torch.Tensor], 
                                         torch.Tensor, 
                                         torch.Tensor, 
                                         List[torch.Tensor] | None], batch_idx) -> Any:
        im_views, tab_views, targets = batch[0], batch[1], batch[2]
        z0, tz0, z1, tz1, emb = self._forward(im_views[0], tab_views[0], im_views[1], tab_views[1])
        loss = self.criterion_train(z0, z1, tz0, tz1)
        
        emb_cat = torch.cat([emb[0].detach(), emb[1].detach()], dim=1)
        
        regr_loss, regr_log = self.online_regressor.training_step(
            (emb_cat, targets), batch_idx
        )

        total_loss = loss + 1e-4 * regr_loss

        self.log("train_loss", total_loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=len(targets))
        self.log_dict(regr_log, sync_dist=True, batch_size=len(targets))
        
        return total_loss
    
    
    def validation_step(self, batch: Tuple[List[torch.Tensor], 
                                           List[torch.Tensor], 
                                           torch.Tensor, 
                                           torch.Tensor, 
                                           List[torch.Tensor] | None], batch_idx) -> Any:
        # Validate contrastive model and regressor
        im_views, tab_views, targets = batch[0], batch[1], batch[2]
        z0, tz0, z1, tz1, emb = self._forward(im_views[0], tab_views[0], im_views[1], tab_views[1])
        loss = self.criterion_val(z0, z1, tz0, tz1)
        
        emb_cat = torch.cat([emb[0].detach(), emb[1].detach()], dim=1)
        
        regr_loss, preds, targets = self.online_regressor.validation_step(
            (emb_cat, targets), batch_idx
        )

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
        optimizer = torch.optim.SGD(
        [
            {'params': self.encoder_imaging.parameters()}, 
            {'params': self.projector_imaging.parameters()},
            {'params': self.encoder_tabular.parameters()},
            {'params': self.projector_tabular.parameters()},
            {'params': self.online_regressor.parameters(), 'weight_decay': 0.0}
        ], 
        lr=self.args.lr,
        momentum=0.9,
        weight_decay=self.args.weight_decay)
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )