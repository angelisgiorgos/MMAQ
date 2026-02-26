from typing import Any, Dict, List, Tuple, Union
import os
import sys
import torch
from lightning.pytorch import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim import Optimizer, Adam

from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import F1Score

from models import build_ssl_model
from utils import create_logdir


class TransferClassifier(LightningModule):
    def __init__(
        self,
        args,
        model: nn.Module,
        classification_head: nn.Module,
        batch_size_per_device: int,
        num_classes: int,
        feature_dim: int = 2048,
        freeze_model: bool = False,
    ) -> None:
        super().__init__()

        self.args = args
        self.freeze_model = freeze_model
        self.classification_head = classification_head
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.batch_size_per_device = batch_size_per_device

        if args.model == "decur":
            feature_dim *= 2
        elif args.model == "dino":
            feature_dim = 768
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        # ✅ F1Score for multiclass classification, top_k removed
        self.f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            top_k=1
        )

    def forward(self, images: Tensor) -> Tensor:
        if self.freeze_model:
            with torch.no_grad():
                features = self.model.forward(images).flatten(start_dim=1)
        else:
            features = self.model.forward(images).flatten(start_dim=1)

        output: Tensor = self.classification_head(features)
        return output, features

    def calculate_loss(self, preds, labels):
        return self.criterion(preds, labels)

    def shared_step(self, batch):
        images, targets = batch["img"], batch["labels"]

        predictions, _ = self.forward(images)
        loss = self.calculate_loss(predictions, targets.long())

        # ✅ Top-k accuracy for logging (optional, does not affect F1)
        _, predicted_topk = predictions.topk(1, dim=1)
        topk_dict = mean_topk_accuracy(predicted_topk, targets, k=(1,))
        top1_accuracy = topk_dict[1]   # extract tensor

        # ✅ F1Score expects [B] class indices
        pred_labels = torch.argmax(predictions, dim=1)
        f1 = self.f1(pred_labels, targets.long())

        return loss, {"top1": top1_accuracy}, f1

    def training_step(self, batch, batch_idx):
        loss, topk, _ = self.shared_step(batch)
        batch_size = len(batch["img"].shape)

        log_dict = {f"train_{k}": acc for k, acc in topk.items()}

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, topk, f1 = self.shared_step(batch)
        batch_size = len(batch["img"])

        log_dict = {f"val_{k}": acc for k, acc in topk.items()}
        log_dict.update({"val_f1_score": f1})

        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        loss, topk, f1 = self.shared_step(batch)
        batch_size = len(batch["img"])

        log_dict = {f"test_{k}": acc for k, acc in topk.items()}
        log_dict.update({"test_f1_score": f1})

        self.log("test_loss", loss)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.classification_head.parameters())

        if not self.freeze_model:
            parameters += list(self.model.parameters())

        optimizer = Adam(
            parameters,
            lr=self.args.lr * self.batch_size_per_device * self.trainer.world_size / 256,
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


def tf_classification(
    args,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    data_stats,
    num_classes: int,
) -> None:

    print("Running Transfer Learning Classification...")

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        save_dir=base_dir,
        offline=args.offline,
        config=args
    )

    logdir = create_logdir(args.datatype, wandb_logger)

    model = build_ssl_model(args, data_stats)

    if args.ckpt_path is None:
        ckpt_path = os.path.join("./checkpoints", args.model + ".ckpt")
    else:
        ckpt_path = args.ckpt_path

    model.load_state_dict(
        torch.load(ckpt_path, weights_only=False)["state_dict"],
        strict=True
    )

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    model_checkpoint = ModelCheckpoint(
        filename="checkpoint_last_epoch_{epoch:02d}",
        dirpath=logdir,
        monitor="val_top1",
        mode="max",
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
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
        max_epochs=args.max_num_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=args.enable_progress_bar,
    )

    feature_dim = 2048
    if args.model == "dino":
        feature_dim = 768
    elif args.model == "decur":
        feature_dim *= 2

    classification_head = nn.Linear(feature_dim, num_classes)

    classifier = TransferClassifier(
        args=args,
        model=model,
        classification_head=classification_head,
        batch_size_per_device=args.batch_size,
        num_classes=num_classes,
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
