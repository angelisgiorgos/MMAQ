""" BARLOW TWINS Model """
from typing import List, Tuple, Dict, Any


import warnings
import copy
import torch
import torch.nn as nn
from models.multimodal.base import BaseMultimodalModel
from lightly.models.modules import BarlowTwinsProjectionHead
from models.backbones.model import TabularNet, ImagingNet

from losses import select_loss_imaging, select_loss_tabular, Multimodal_Loss


class MM_BarlowTwins(BaseMultimodalModel):
    """Implementation of the BarlowTwins architecture."""

    def __init__(
        self,
        args,
        hidden_dim: int = 4096,
        out_dim: int = 256,
        m: float = 0.9,
    ):
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.m = m
        super().__init__(args)

        # ---------- Evaluation Collect ----------
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []

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
        self.initialize_regressor_and_metrics()
        
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

        
    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor] | None], _) -> Any:
        im_views, tab_views, y = batch[0], batch[1], batch[2]
        
        z0, embeddings_0 = self.forward_imaging(im_views[0])
        z1, _ = self.forward_imaging(im_views[1])
        
        
        z_t0, temb_0 = self.forward_tabular(tab_views[0])
        z_t1, _ = self.forward_tabular(tab_views[1])
        loss = self.criterion_train(z0, z1, z_t0, z_t1)
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False)
        
        emb = torch.cat([embeddings_0, temb_0], dim=1)
        self.train_embeddings.append(emb.detach())
        self.train_labels.append(y.detach())
        return {'loss': loss, 'embeddings': emb, 'labels': y}


    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor] | None], _) -> Any:
        """
        Validate contrastive model
        """
        im_views, tab_views, y = batch[0], batch[1], batch[2]
        z0, embeddings_0 = self.forward_imaging(im_views[0])
        z1, _ = self.forward_imaging(im_views[1])
        z_t0, temb_0 = self.forward_tabular(tab_views[0])
        z_t1, _ = self.forward_tabular(tab_views[1])
        loss = self.criterion_val(z0, z1, z_t0, z_t1)
        self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)
        
        emb = torch.cat([embeddings_0, temb_0], dim=1)
        self.val_embeddings.append(emb.detach())
        self.val_labels.append(y.detach())
        return {'sample_augmentation': im_views[1], 'embeddings': emb, 'labels': y}



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


    def initialize_regressor_and_metrics(self):
        """
        Initializes classifier and metrics. Takes care to set correct number of classes for embedding similarity metric depending on loss.
        """
        # Regressor
        self.estimator = None
        # RMSE calculated against all others in batch of same view except for self (i.e. -1) and all of the other view
        self.mae_train = torchmetrics.MeanAbsoluteError()
        self.mae_val = torchmetrics.MeanAbsoluteError()

        self.pears_cor_train = torchmetrics.PearsonCorrCoef()
        self.pears_cor_val = torchmetrics.PearsonCorrCoef()

        self.mse_train = torchmetrics.MeanSquaredError()
        self.mse_val = torchmetrics.MeanSquaredError()
        
        self.mape_train = torchmetrics.MeanAbsolutePercentageError()
        self.mape_val = torchmetrics.MeanAbsolutePercentageError()
        
        self.r2_train = torchmetrics.R2Score()
        self.r2_val = torchmetrics.R2Score()


    def calc_and_log_train_embedding_metrics(self, logits, labels, modality: str) -> None:
        self.mae_train(logits, labels)
        self.pears_cor_train(logits, labels)
        self.mse_train(logits, labels)
        self.mape_train(logits, labels)
        self.r2_train(logits, labels)

        self.log(f"{modality}.train.mae",self.mae_train, on_epoch=True, on_step=False)
        self.log(f"{modality}.train.pears_cor", self.pears_cor_train, on_epoch=True, on_step=False)
        self.log(f"{modality}.train.mse", self.mse_train, on_epoch=True, on_step=False)
        self.log(f"{modality}.train.mape",self.mape_train, on_epoch=True, on_step=False)
        self.log(f"{modality}.train.r2",self.r2_train, on_epoch=True, on_step=False)


    def calc_and_log_val_embedding_metrics(self, logits, labels, modality: str) -> None:
        self.mae_val(logits, labels)
        self.pears_cor_val(logits, labels)
        self.mse_val(logits, labels)
        self.mape_val(logits, labels)
        self.r2_val(logits, labels)
        

        self.log(f"{modality}.val.mae",self.mae_val, on_epoch=True, on_step=False)
        self.log(f"{modality}.val.pears_cor", self.pears_cor_val, on_epoch=True, on_step=False)
        self.log(f"{modality}.val.mse", self.mse_val, on_epoch=True, on_step=False)
        self.log(f"{modality}.val.mape", self.mape_val, on_epoch=True, on_step=False)
        self.log(f"{modality}.val.r2", self.r2_val, on_epoch=True, on_step=False)


    def on_train_epoch_end(self) -> None:
        if self.current_epoch != 0 and self.current_epoch % getattr(self.args, 'regressor_freq', 1) == 0:
            if not self.train_embeddings:
                return
            embeddings = torch.cat(self.train_embeddings, dim=0).cpu()
            labels = torch.cat(self.train_labels, dim=0).cpu()
            
            self.estimator = LinearRegression().fit(embeddings, labels)
            preds = self.predict_live_estimator(embeddings)

            self.mae_train(preds, labels)
            self.mse_train(preds, labels)
            self.mape_train(preds, labels)
            self.r2_train(preds, labels)

            self.log('regressor.train.mae', self.mae_train, on_epoch=True, on_step=False)
            self.log('regressor.train.mse', self.mse_train, on_epoch=True, on_step=False)
            self.log('regressor.train.mape', self.mape_train, on_epoch=True, on_step=False)
            self.log('regressor.train.r2', self.r2_train, on_epoch=True, on_step=False)
            
        self.train_embeddings.clear()
        self.train_labels.clear()


    def on_validation_epoch_end(self) -> None:
        if self.args.log_images and hasattr(self, 'logger') and self.logger is not None:
            pass

        # Validate regressor
        if self.estimator is not None and getattr(self, 'current_epoch', 1) % getattr(self.args, 'regressor_freq', 1) == 0:
            if not self.val_embeddings:
                return
            embeddings = torch.cat(self.val_embeddings, dim=0).cpu()
            labels = torch.cat(self.val_labels, dim=0).cpu()

            preds = self.predict_live_estimator(embeddings)
        
            self.mae_val(preds, labels)
            self.mse_val(preds, labels)
            self.mape_val(preds, labels)
            self.r2_val(preds, labels)

            self.log('regressor.val.mae', self.mae_val, on_epoch=True, on_step=False)
            self.log('regressor.val.mse', self.mse_val, on_epoch=True, on_step=False)
            self.log('regressor.val.mape', self.mape_val, on_epoch=True, on_step=False)
            self.log('regressor.val.r2', self.r2_val, on_epoch=True, on_step=False)
            
        self.val_embeddings.clear()
        self.val_labels.clear()

    
    def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack outputs from multiple steps
        """
        labels = outputs[0]['labels']
        embeddings = outputs[0]['embeddings']
        for i in range(1, len(outputs)):
            labels = torch.cat((labels, outputs[i]['labels']), dim=0)
            embeddings = torch.cat((embeddings, outputs[i]['embeddings']), dim=0)

        embeddings = embeddings.detach().cpu()
        labels = labels.cpu()

        return embeddings, labels

    
    def predict_live_estimator(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict using live estimator
        """
        preds_numpy = self.estimator.predict(embeddings)
        preds = torch.tensor(preds_numpy)
        return preds
    
    
    
    def configure_optimizers(self) -> Any:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        optimizer = torch.optim.Adam(
        [
            {'params': self.encoder_imaging.parameters()}, 
            {'params': self.projector_imaging.parameters()},
            {'params': self.encoder_tabular.parameters()},
            {'params': self.projector_tabular.parameters()}
        ], 
        lr=self.args.lr,
        # momentum=0.9,
        weight_decay=self.args.weight_decay)
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )