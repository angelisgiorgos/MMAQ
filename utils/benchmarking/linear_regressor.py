from typing import Any, Dict, List, Tuple, Union
import os
import sys
import torch
from lightning.pytorch import LightningModule
from torch import Tensor
import torch.nn as nn
from torch.optim import SGD, Adam, Optimizer
from utils import create_logdir
import torchmetrics
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch import Trainer
from models import build_ssl_model
from utils.utils import undo_normalization
from losses.supervised_losses import ContrastiveRegressionLoss, RandomLinearProjection, OrdinalEntropy


class LinearRegressor(LightningModule):
    def __init__(
        self,
        args,
        data_stats,
        model: nn.Module,
        regression_head: nn.Module,
        batch_size_per_device: int,
        feature_dim: int = 2048,
        freeze_model: bool = False,
    ) -> None:
        """Linear regressor for benchmarking.

        Args:
            model:
                Model used for feature extraction. Must define a forward(images) method
                that returns a feature tensor.
            batch_size_per_device:
                Batch size per device.
            feature_dim:
                Dimension of features returned by forward method of model.
            freeze_model:
                If True, the model is frozen and only the regression_head head is
                trained. This corresponds to the linear eval setting. Set to False for
                finetuning.

        Examples:

            >>> from lightning.pytorch import Trainer
            >>> from torch import nn
            >>> import torchvision
            >>> from lightly.models import LinearClassifier
            >>> from lightly.modles.modules import SimCLRProjectionHead
            >>>
            >>> class SimCLR(nn.Module):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.backbone = torchvision.models.resnet18()
            >>>         self.backbone.fc = nn.Identity() # Ignore regression_head layer
            >>>         self.projection_head = SimCLRProjectionHead(512, 512, 128)
            >>>
            >>>     def forward(self, x):
            >>>         # Forward must return image features.
            >>>         features = self.backbone(x).flatten(start_dim=1)
            >>>         return features
            >>>
            >>> # Initialize a model.
            >>> model = SimCLR()
            >>>
            >>> # Wrap it with a LinearClassifier.
            >>> linear_classifier = LinearClassifier(
            >>>     model,
            >>>     batch_size=256,
            >>>     num_classes=10,
            >>>     freeze_model=True, # linear evaluation, set to False for finetune
            >>> )
            >>>
            >>> # Train the linear classifier.
            >>> trainer = Trainer(max_epochs=90)
            >>> trainer.fit(linear_classifier, train_dataloader, val_dataloader)

        """
        super().__init__()
        self.args = args
        self.data_stats = data_stats
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.batch_size_per_device = batch_size_per_device
        if args.model == "decur":
            feature_dim = feature_dim*2
        self.feature_dim = feature_dim
        self.freeze_model = freeze_model

        self.regression_head = regression_head
        self.init_criterion()

        self.test_preds = []
        self.test_targets = []

        self.val_preds = []
        self.val_targets = []

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.r2 = torchmetrics.R2Score()
        self.mse = torchmetrics.MeanSquaredError()

    def init_criterion(self):
        if self.args.finetune_loss == "mse":
            self.criterion = nn.MSELoss()
        elif self.args.finetune_loss == "rlp":
            self.criterion = RandomLinearProjection(self.args)
        elif self.args.finetune_loss == "contrastive_regression":
            self.criterion = ContrastiveRegressionLoss(self.args)
        elif self.args.finetune_loss == "ordinal":
            self.criterion = OrdinalEntropy(self.args)

    def forward(self, images: Tensor, tabular: Tensor = None) -> Tensor:
        if self.freeze_model:
            with torch.no_grad():
                if self.args.model in ["mmcl", "mmaq", "decur"]:
                    features = self.model.forward(images, tabular).flatten(start_dim=1)
                else:
                    features = self.model.forward(images).flatten(start_dim=1)
        else:
            if self.args.model in ["mmcl", "mmaq", "decur"]:
                features = self.model.forward(images, tabular).flatten(start_dim=1)
            else:
                features = self.model.forward(images).flatten(start_dim=1)
        output: Tensor = self.regression_head(features)
        return output, features


    def calculate_loss(self, features, targets, labels):
        if not self.args.finetune_loss == "mse":
            loss = self.criterion(features, targets, labels)
        else:
            loss = self.criterion(targets, labels)
        return loss


    def shared_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ):

        if self.args.datatype == "rgb_unimodal":
            image, targets = batch
            predictions, features = self.forward(image)
        else:
            im_views, tab_views, targets = batch
            predictions, features = self.forward(im_views, tab_views.float())
        targets = targets.float().unsqueeze(1)
        loss = self.calculate_loss(features, targets, predictions)
        predictions, targets = undo_normalization(predictions.detach(), targets, self.data_stats)
        return 1e-3 * loss, predictions, targets


    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, predictions, targets = self.shared_step(batch=batch, batch_idx=batch_idx)
        metrics = {
            "mae" : self.mae(predictions, targets),
            "mse": self.mse(predictions, targets), 
            "mape": self.mape(predictions, targets), 
            "r2": self.r2(predictions, targets)}
        batch_size = len(batch[1])
        log_dict = {f"train_{k}": acc for k, acc in metrics.items()}
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, predictions, targets = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        # log_dict.update({f"val_online_{k}": acc for k, acc in metrics.items()})
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        self.val_preds.append(predictions.clone().detach())
        self.val_targets.append(targets)
        return loss
    

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        metrics = {
            "mae": self.mae(preds, targets),
            "mape": self.mape(preds, targets),
            "r2": self.r2(preds, targets),
            "mse": self.mse(preds, targets)
        }

        self.log_dict({f"val_{k}": acc for k, acc in metrics.items()}, prog_bar=True)


    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, predictions, targets = self.shared_step(batch=batch, batch_idx=batch_idx)
       
        self.test_preds.append(predictions.detach())
        self.test_targets.append(targets)
        return loss
    
    
    def on_test_epoch_end(self):

        preds = torch.cat(self.test_preds, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        preds, targets = undo_normalization(preds, targets, self.data_stats)

        metrics = {
            "mae": self.mae(preds, targets),
            "mape": self.mape(preds, targets),
            "r2": self.r2(preds, targets),
            "mse": self.mse(preds, targets)
        }

        self.log_dict({f"test_{k}": acc for k, acc in metrics.items()}, prog_bar=True)


    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.regression_head.parameters())
        if not self.freeze_model:
            print(self.model.parameters())
            parameters += self.model.parameters()
        optimizer = SGD(
            parameters,
            lr=self.args.lr * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=self.args.momentum,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def on_train_epoch_start(self) -> None:
        if self.freeze_model:
            # Set model to eval mode to disable norm layer updates.
            self.model.eval()


class FinetuneLinearRegressor(LinearRegressor):
    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.regression_head.parameters())
        parameters += self.model.parameters()
        optimizer = Adam(
            parameters,
            lr=self.args.lr * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=self.args.weight_decay,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

def run_evaluation(
    args,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    data_stats,
    is_finetune: bool = False
) -> None:
    print(f"Running {'fine-tune' if is_finetune else 'linear'} evaluation...")

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        save_dir=base_dir,
        offline=args.offline,
        config=args
    )
    
    # Create logdir based on WandB run name
    logdir = create_logdir(args.datatype, wandb_logger)

    # Pretrained model for linear regression.
    model = build_ssl_model(args, data_stats)
    
    if args.ckpt_path is None:
        ckpt_path = os.path.join("./checkpoints", args.model + ".ckpt")
    else:
        ckpt_path = args.ckpt_path
        
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=False)["state_dict"], 
        strict=True
    )

    if is_finetune:
        setattr(model, "online_regressor", nn.Identity())
        model = model.train()

    if not is_finetune and hasattr(torch, "compile"):
        # Compile model if PyTorch supports it.
        model = torch.compile(model)

    model_checkpoint = ModelCheckpoint(
            filename="checkpoint_last_epoch_{epoch:02d}",
            dirpath=logdir,
            monitor="val_mae",
            mode="min",
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        )
    callbacks=[
            LearningRateMonitor(),
            model_checkpoint,
            ]

    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpu_ids,
        precision=getattr(args, "precision", 32),
        deterministic=getattr(args, "deterministic", True),
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=50 if is_finetune else args.max_num_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=args.enable_progress_bar,
    )

    feature_dim = 2048
    if is_finetune:
        if args.model in ["decur", "mmaq", "mmcl"]:
            feature_dim = 4096
        elif args.model in ["barlow_twins", "byol", "simclr"]:
            feature_dim = 2048 + getattr(args, "embedding_dim", 13)
        elif args.model == "dino":
            feature_dim = 768
    else:
        if args.model == "decur":
            feature_dim = feature_dim*2
        elif args.model == "mmaq":
            feature_dim = feature_dim*2
        elif args.model == "dino":
            feature_dim = 768
    
    regression_head = nn.Linear(feature_dim, 1)

    if is_finetune:
        regressor = FinetuneLinearRegressor(
            args=args,
            data_stats=data_stats,
            model=model,
            regression_head=regression_head,
            batch_size_per_device=args.batch_size,
            feature_dim=feature_dim,
            freeze_model=False,
        )
    else:
        regressor = LinearRegressor(
            args=args,
            data_stats=data_stats,
            model=model,
            regression_head=regression_head,
            batch_size_per_device=args.batch_size,
            feature_dim=feature_dim,
            freeze_model=True,
        )

    trainer.fit(
        model=regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    CKPT_PATH = model_checkpoint.best_model_path
    print(CKPT_PATH)
    checkpoint = torch.load(CKPT_PATH, weights_only=False)
    regressor.load_state_dict(checkpoint["state_dict"])
    trainer.test(model=regressor, dataloaders=val_dataloader)


def linear_eval(args, train_dataloader, val_dataloader, data_stats):
    run_evaluation(args, train_dataloader, val_dataloader, data_stats, is_finetune=False)


def finetune_eval(args, train_dataloader, val_dataloader, data_stats):
    run_evaluation(args, train_dataloader, val_dataloader, data_stats, is_finetune=True)