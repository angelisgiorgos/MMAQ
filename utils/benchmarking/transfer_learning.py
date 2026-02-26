from typing import Any, Dict, List, Tuple, Union
import os
import sys
import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from torch.optim import Optimizer, Adam

from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from models import build_ssl_model
from utils import create_logdir
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from torchmetrics import F1Score


class TransferClassifier(LightningModule):
    def __init__(
        self,
        args,
        model: nn.Module,
        classification_head: nn.Module,
        batch_size_per_device: int,
        topk: Tuple[int, ...] = (1, 1),
        feature_dim: int = 2048,
        freeze_model: bool = False) -> None:
        super().__init__()

        self.args = args
        self.freeze_model = freeze_model
        self.classification_head = classification_head
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.batch_size_per_device = batch_size_per_device
        if args.model == "decur":
            feature_dim = feature_dim*2
        elif args.model == "dino":
            feature_dim = 768
        self.feature_dim = feature_dim

        self.criterion = nn.CrossEntropyLoss()

        self.topk = topk

        self.f1 = F1Score(task="multiclass", num_classes=4)

    def forward(self, images: Tensor) -> Tensor:
        if self.freeze_model:
            with torch.no_grad():
                features = self.model.forward(images).flatten(start_dim=1)
        else:
            features = self.model.forward(images).flatten(start_dim=1)
        output: Tensor = self.classification_head(features)
        return output, features


    def calculate_loss(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss


    def shared_step(self, batch):
        images, targets = batch["img"], batch["labels"]
        predictions, features = self.forward(images)
        loss = self.calculate_loss(predictions, targets.long())
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
        f1 = self.f1(predicted_labels, targets.long())
        return loss, topk, f1

    
    def training_step(self, batch, batch_idx):
        loss, topk, _ = self.shared_step(batch=batch)
        batch_size = len(batch["img"])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, topk, f1 = self.shared_step(batch=batch)
        batch_size = len(batch["img"])
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        log_dict.update({"val_f1_score": f1})
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

    
    def test_step(self, batch, batch_idx):
        loss, topk, f1 = self.shared_step(batch=batch)
        batch_size = len(batch["img"])
        log_dict = {f"test_top{k}": acc for k, acc in topk.items()}
        log_dict.update({"test_f1_score": f1})
        self.log("test_loss", loss)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

    
    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.classification_head.parameters())
        if not self.freeze_model:
            parameters += self.model.parameters()
        optimizer = Adam(
            parameters,
            lr=self.args.lr * self.batch_size_per_device * self.trainer.world_size / 256,
            # momentum=0.9,
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





def tf_classification(args,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    data_stats,
    num_classes) -> None:
    print("Running Transfer Learning Classification...")

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        save_dir=base_dir,
        offline=args.offline,
        config=args
    )
    

    # Create logdir based on WandB run name
    logdir = create_logdir(args.datatype, wandb_logger)

    # Train linear classifier.

    model = build_ssl_model(args, data_stats)
    if args.ckpt_path is None:
        ckpt_path = os.path.join("./checkpoints", args.model + ".ckpt")
    else:
        ckpt_path = args.ckpt_path
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"], strict=True)

    if hasattr(torch, "compile"):
        # Compile model if PyTorch supports it.
        model = torch.compile(model)


    model_checkpoint = ModelCheckpoint(
            filename="checkpoint_last_epoch_{epoch:02d}",
            dirpath=logdir,
            monitor="val_top1",
            mode="max",
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        )
    callbacks=[
            LearningRateMonitor(),
            model_checkpoint,
            ]

    trainer = Trainer(
        accelerator="gpu",
        devices=[1],
        precision=32,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=100,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=args.enable_progress_bar,
    )

    feature_dim = 2048
    if args.model == "dino":
        feature_dim = 768

    
    classification_head = nn.Linear(feature_dim, num_classes)
    

    classifier = TransferClassifier(
        args=args,
        model=model,
        classification_head=classification_head,
        batch_size_per_device=args.batch_size,
        freeze_model=True
    )

    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    CKPT_PATH = model_checkpoint.best_model_path
    print(CKPT_PATH)
    checkpoint = torch.load(CKPT_PATH, weights_only=False)
    classifier.load_state_dict(checkpoint["state_dict"])
    trainer.test(model=classifier, dataloaders=val_dataloader)
    