from typing import List, Tuple, Dict, Any

import torch
import pytorch_lightning as pl
import torchmetrics
from sklearn.linear_model import LinearRegression
from lightly.models.modules import SimCLRProjectionHead
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from models.backbones.model import TabularNet, MultiModal, ImagingNet
from sklearn.neighbors import KNeighborsRegressor


class Pretraining(pl.LightningModule):
    def __init__(self, args) -> None:
        super(Pretraining, self).__init__()
        self.args = args

    def initialize_imaging_encoder_and_projector(self) -> None:
        """
        Selects appropriate encoder
        """
        self.encoder_imaging = ImagingNet(self.args)
        self.pooled_dim = self.args.imaging_embedding
        self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.args.embedding_dim, self.args.projection_dim)

    def initialize_tabular_encoder_and_projector(self) -> None:
        self.encoder_tabular = TabularNet(self.args)
        self.projector_tabular = SimCLRProjectionHead(self.args.embedding_dim, self.args.embedding_dim, self.args.projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates encoding of imaging data.
        """
        z, y = self.forward_imaging(x)
        return y

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

    def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.args.dataset_length*self.args.cosine_anneal_mult), eta_min=0, last_epoch=-1)
        elif self.args.scheduler == 'anneal':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.warmup_epochs, max_epochs = self.args.max_epochs)
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

        self.pears_cor_train = torchmetrics.regression.PearsonCorrCoef()
        self.pears_val_train = torchmetrics.regression.PearsonCorrCoef()

        self.mse_train = torchmetrics.regression.MeanSquaredError()
        self.mse_val = torchmetrics.regression.MeanSquaredError()
        
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
        

    def training_epoch_end(self, train_step_outputs: List[Any]) -> None:
        """
        Train and log regressor
        """
        if self.current_epoch != 0 and self.current_epoch % self.args.regressor_freq == 0:
            embeddings, labels = self.stack_outputs(train_step_outputs)
            
            self.estimator = LinearRegression().fit(embeddings, labels)
            self.knn_estimator = KNeighborsRegressor().fit(embeddings, labels)
            
            preds, knn_preds = self.predict_live_estimator(embeddings)

            lr_mae = self.mae_train(preds, labels)
            lr_mse = self.mse_train(preds, labels)
            lr_mape = self.mape_train(preds, labels)
            lr_r2 = self.r2_train(preds, labels)

            self.log('regressor.train.mae', lr_mae, on_epoch=True, on_step=False)
            self.log('regressor.train.mse', lr_mse, on_epoch=True, on_step=False)
            self.log('regressor.train.mape', lr_mape, on_epoch=True, on_step=False)
            self.log('regressor.train.r2', lr_r2, on_epoch=True, on_step=False)
            
            knn_mae = self.mae_train(knn_preds, labels)
            knn_mse = self.mse_train(knn_preds, labels)
            knn_mape = self.mape_train(knn_preds, labels)
            knn_r2 = self.r2_train(knn_preds, labels)

            self.log('regressor.train.knn_mae', knn_mae, on_epoch=True, on_step=False)
            self.log('regressor.train.knn_mse', knn_mse, on_epoch=True, on_step=False)
            self.log('regressor.train.knn_mape', knn_mape, on_epoch=True, on_step=False)
            self.log('regressor.train.knn_r2', knn_r2, on_epoch=True, on_step=False)


    def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
        """
        Log an image from each validation step and calc validation classifier performance
        """
        if self.args.log_images:
            self.logger.log_image(key="Image Example", images=[validation_step_outputs[0]['sample_augmentation']])

        # Validate regressor
        if not self.estimator is None and self.current_epoch % self.args.regressor_freq == 0:
            embeddings, labels = self.stack_outputs(validation_step_outputs)

            preds, knn_preds = self.predict_live_estimator(embeddings)
            
            lr_mae = self.mae_val(preds, labels)
            lr_mse = self.mse_val(preds, labels)
            lr_mape = self.mape_val(preds, labels)
            lr_r2 = self.r2_val(preds, labels)

            self.log('regressor.val.mae', lr_mae, on_epoch=True, on_step=False)
            self.log('regressor.val.mse', lr_mse, on_epoch=True, on_step=False)
            self.log('regressor.val.mape', lr_mape, on_epoch=True, on_step=False)
            self.log('regressor.val.r2', lr_r2, on_epoch=True, on_step=False)
            
            knn_mae = self.mae_val(knn_preds, labels)
            knn_mse = self.mse_val(knn_preds, labels)
            knn_mape = self.mape_val(knn_preds, labels)
            knn_r2 = self.r2_val(knn_preds, labels)

            self.log('regressor.val.knn_mae', knn_mae, on_epoch=True, on_step=False)
            self.log('regressor.val.knn_mse', knn_mse, on_epoch=True, on_step=False)
            self.log('regressor.val.knn_mape', knn_mape, on_epoch=True, on_step=False)
            self.log('regressor.val.knn_r2', knn_r2, on_epoch=True, on_step=False)

    
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
        preds = self.estimator.predict(embeddings)

        preds = torch.tensor(preds)
        
        preds_knn = self.knn_estimator.predict(embeddings)
        
        preds_knn = torch.tensor(preds_knn)

        return preds, preds_knn






