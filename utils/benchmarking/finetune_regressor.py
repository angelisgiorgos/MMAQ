import os, sys
from pathlib import Path
from torch.nn import Module
import torch
import torch.nn as nn
from .linear_regressor import LinearRegressor
from utils import create_logdir
from torch.optim import SGD, Adam
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning.pytorch.loggers import WandbLogger
from lightly.utils.benchmarking import MetricCallback
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch import Trainer
from models import build_ssl_model


class FinetuneLinearRegressor(LinearRegressor):
    def configure_optimizers(self):
        parameters = list(self.regression_head.parameters())
        parameters += self.model.parameters()
        optimizer = SGD(
            parameters,
            lr=0.05 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


def finetune_eval(
    args,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    data_stats) -> None:
    print("Running fine-tune evaluation...")

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
    metric_callback = MetricCallback()

    model = build_ssl_model(args, data_stats)
    model.online_regressor = nn.Identity()
    if args.ckpt_path is None:
        ckpt_path = os.path.join("./checkpoints", args.model + ".ckpt")
    else:
        ckpt_path = args.ckpt_path
    model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)

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
            metric_callback,
            model_checkpoint,
            ]

    trainer = Trainer(
        gpus=[1],
        precision=32,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=50,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=args.enable_progress_bar,
    )

    feature_dim = 2048
    if args.model == "decur":
        feature_dim = feature_dim*2
    elif args.model == "mmaq":
        feature_dim = feature_dim*2
    elif args.model == "dino":
        feature_dim = 768


    regression_head = nn.Linear(feature_dim, 1)

    # regression_head = nn.Sequential(
    # nn.Linear(feature_dim, feature_dim // 4),
    # nn.ReLU(),
    # nn.BatchNorm1d(feature_dim // 4),
    # nn.Linear(feature_dim // 4, 1))



    regressor = FinetuneLinearRegressor(
        args=args,
        data_stats=data_stats,
        model=model,
        regression_head=regression_head,
        batch_size_per_device=args.batch_size,
        feature_dim=feature_dim,
        freeze_model=False,
    )

    trainer.fit(
        model=regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    CKPT_PATH = model_checkpoint.best_model_path
    checkpoint = torch.load(CKPT_PATH)
    regressor.load_state_dict(checkpoint["state_dict"])
    trainer.test(ckpt_path="best", dataloaders=val_dataloader)

