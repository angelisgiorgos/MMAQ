import os
import sys
import warnings

# Ignore all warnings to keep the output clean
warnings.filterwarnings("ignore")

import torch
import torch.multiprocessing
import argparse

try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except AttributeError:
    pass  # For older PyTorch versions

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor, 
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from lightly.utils.dist import rank

# Project-specific imports
from dataset.data_loaders import (
    load_datasets,
    load_rgb_unimodal_datasets,
    load_segmentation_data,
    load_transfer_data,
)
from models import build_ssl_model
from opts.run_opts import RunOptions
from utils import create_logdir
from utils.benchmarking.linear_regressor import linear_eval, finetune_eval
from utils.benchmarking.transfer_learning import tf_classification
from utils.benchmarking.transfer_segmentation import tf_segmentation


def pretrain(args, wandb_logger):
    """
    Main training routine for self-supervised pretraining.
    """
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

    trainer = Trainer(
        devices=args.gpu_ids,
        accelerator="gpu",
        sync_batchnorm=True,
        precision=32,
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


def run_linear_eval(args):
    """
    Main training routine for linear evaluation.
    """
    args.linear_eval = True

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
    linear_eval(args, train_loader, val_loader, data_stats)


def run_fine_tune(args):
    """
    Main training routine for supervised fine-tuning.
    """
    args.linear_eval = True

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
    finetune_eval(args, train_loader, val_loader, data_stats)


def run_transfer_learning(args):
    """
    Main routine for transfer learning classification evaluation.
    """
    args.linear_eval = True
    train_dataloader, validation_dataloader = load_transfer_data(args, "labels.csv")

    tf_classification(
        args,
        train_dataloader,
        validation_dataloader,
        None,
        4
    )


def run_transfer_segmentation(args):
    """
    Main routine for transfer learning segmentation evaluation.
    """
    args.linear_eval = True
    train_dataloader, validation_dataloader = load_segmentation_data(args)

    tf_segmentation(
        args=args,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader
    )


if __name__ == "__main__":
    # Performance and reproducibility settings
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    
    opts = RunOptions()
    args = opts.parse()
    
    pl.seed_everything(args.seed)

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    # Task routing
    if args.task == "pretrain":
        # Initialize Weights & Biases logger only for operations that used it traditionally
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            save_dir=base_dir,
            offline=args.offline,
            experiment=args.experiment,
            config=args
        )
        pretrain(args, wandb_logger)
    
    elif args.task == "linear_eval":
        run_linear_eval(args)
        
    elif args.task == "fine_tune":
        args.freeze = False
        run_fine_tune(args)
        
    elif args.task == "transfer_learning":
        args.freeze = False
        run_transfer_learning(args)
        
    elif args.task == "transfer_segmentation":
        run_transfer_segmentation(args)
