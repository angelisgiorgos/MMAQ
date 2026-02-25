from typing import Tuple
import torch
import torchmetrics
from torch import Tensor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from lightly.models.modules import SimCLRProjectionHead
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from models.backbones.model import TabularNet, ImagingNet
from losses import DCL


from models.multimodal.base import BaseMultimodalModel

class MultimodalContrastiveSimCLR(BaseMultimodalModel):
    """
    Lightning module for multimodal SSL.
    Cleaned & structured implementation.
    """
    def __init__(self, args):
        super().__init__(args)

        # ---------- Evaluation Collect ----------
        self.train_embeddings = []
        self.train_labels = []

        self.val_embeddings = []
        self.val_labels = []

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

    def _build_regressors(self):
        self.estimator = None
        self.knn_estimator = None

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
        
        self.log("multimodal.train.loss", loss, on_epoch=True, on_step=False)

        embeddings_cat = torch.cat([embeddings_0, embeddings_1], dim=1)

        self.train_embeddings.append(embeddings_cat.detach())
        self.train_labels.append(y.detach())

        return loss

    def on_train_epoch_start(self):
        self.train_embeddings.clear()
        self.train_labels.clear()

    def training_epoch_end(self, outputs):
        if self.current_epoch != 0 and self.current_epoch % getattr(self.args, 'regressor_freq', 1) == 0:
            if not self.train_embeddings:
                return
            
            embeddings = torch.cat(self.train_embeddings, dim=0).cpu()
            labels = torch.cat(self.train_labels, dim=0).cpu()
            
            self.estimator = LinearRegression().fit(embeddings, labels)
            self.knn_estimator = KNeighborsRegressor().fit(embeddings, labels)
            
            preds = torch.tensor(self.estimator.predict(embeddings)).to(self.device)
            knn_preds = torch.tensor(self.knn_estimator.predict(embeddings)).to(self.device)
            labels = labels.to(self.device)
            
            metrics = {
                'regressor.train.mae': self.mae_train(preds, labels),
                'regressor.train.mse': self.mse_train(preds, labels),
                'regressor.train.mape': self.mape_train(preds, labels),
                'regressor.train.r2': self.r2_train(preds, labels),
                
                'regressor.train.knn_mae': self.mae_train(knn_preds, labels),
                'regressor.train.knn_mse': self.mse_train(knn_preds, labels),
                'regressor.train.knn_mape': self.mape_train(knn_preds, labels),
                'regressor.train.knn_r2': self.r2_train(knn_preds, labels),
            }
            
            self.log_dict(metrics, on_epoch=True, on_step=False)
            
            for m in [self.mae_train, self.mse_train, self.mape_train, self.r2_train]:
                m.reset()

    # ============================================================
    # Validation
    # ============================================================

    def validation_step(self, batch, batch_idx):
        im_views, tab_views, y, original_im = batch
    
        # Unaugmented views
        z0, embeddings_0 = self.forward_imaging(original_im)
        z1, embeddings_1 = self.forward_tabular(tab_views[0])
        
        loss = self.criterion_val(z0, z1)

        self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)

        embeddings_cat = torch.cat([embeddings_0, embeddings_1], dim=1)

        self.val_embeddings.append(embeddings_cat.detach())
        self.val_labels.append(y.detach())

        if getattr(self.args, 'log_images', False) and batch_idx == 0:
            self.logger.log_image(key="Image Example", images=[im_views[1]])

        return loss

    def on_validation_epoch_start(self):
        self.val_embeddings.clear()
        self.val_labels.clear()

    def validation_epoch_end(self, outputs):
        # Validate regressor
        if self.estimator is not None and self.current_epoch % getattr(self.args, 'regressor_freq', 1) == 0:
            if not self.val_embeddings:
                return

            embeddings = torch.cat(self.val_embeddings, dim=0).cpu()
            labels = torch.cat(self.val_labels, dim=0).cpu()

            preds = torch.tensor(self.estimator.predict(embeddings)).to(self.device)
            knn_preds = torch.tensor(self.knn_estimator.predict(embeddings)).to(self.device)
            labels = labels.to(self.device)
            
            metrics = {
                'regressor.val.mae': self.mae_val(preds, labels),
                'regressor.val.mse': self.mse_val(preds, labels),
                'regressor.val.mape': self.mape_val(preds, labels),
                'regressor.val.r2': self.r2_val(preds, labels),
                
                'regressor.val.knn_mae': self.mae_val(knn_preds, labels),
                'regressor.val.knn_mse': self.mse_val(knn_preds, labels),
                'regressor.val.knn_mape': self.mape_val(knn_preds, labels),
                'regressor.val.knn_r2': self.r2_val(knn_preds, labels),
            }

            self.log_dict(metrics, on_epoch=True, on_step=False)

            for m in [self.mae_val, self.mse_val, self.mape_val, self.r2_val]:
                m.reset()

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {'params': self.encoder_imaging.parameters()},
                {'params': self.projector_imaging.parameters()},
                {'params': self.encoder_tabular.parameters()},
                {'params': self.projector_tabular.parameters()}
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
        elif self.args.scheduler == 'anneal':
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, 
                warmup_epochs=self.args.warmup_epochs, 
                max_epochs=self.args.max_epochs
            )
        else:
            raise ValueError('Valid schedulers are "cosine" and "anneal"')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }