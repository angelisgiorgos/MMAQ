import os
import sys
import warnings

# Ignore all warnings to keep the output clean
warnings.filterwarnings("ignore")


import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from lightly.utils.dist import rank

# Project-specific imports
from dataset.data_loaders import load_datasets, load_rgb_unimodal_datasets
from models import build_ssl_model
from opts import TrainOptions
from utils import create_logdir


def pretrain(args, wandb_logger):
    """
    Main training routine for self-supervised pretraining.

    Args:
        args: Parsed command line hyperparameters.
        wandb_logger: Instantiated weights and biases logger for tracking.
    """
    pl.seed_everything(args.seed)

    # Load appropriate datasets based on datatype
    if args.datatype == "rgb_unimodal":
        train_dataset, val_dataset, data_stats = load_rgb_unimodal_datasets(args)
    else:
        train_dataset, val_dataset, data_stats = load_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")

    # Create log directory correctly on rank 0
    logdir = create_logdir(args.datatype, wandb_logger) if rank() == 0 else ""

    model = build_ssl_model(args, data_stats)

    callbacks = [
        EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
        DeviceStatsMonitor(),
        ModelCheckpoint(
            filename="checkpoint_last_epoch_{epoch:02d}",
            dirpath=logdir if logdir else None,
            monitor="val_mae",
            mode="min",
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    trainer = Trainer.from_argparse_args(
        args,
        devices=[0, 1],
        accelerator="gpu",
        sync_batchnorm=True,
        precision="32",
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=args.enable_progress_bar,
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    # Performance and reproducibility settings
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    
    opts = TrainOptions()
    args = opts.parse()
    
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        save_dir=base_dir,
        offline=args.offline,
        experiment=args.experiment,
        config=args
    )
    
    pretrain(args, wandb_logger)
