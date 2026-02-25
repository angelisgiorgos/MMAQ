import os
import sys

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset.data_loaders import load_datasets, load_rgb_unimodal_datasets
from opts import LinearEvalOptions
from utils.benchmarking.linear_regressor import linear_eval


def main_linear_eval(args):
    """
    Main training routine for linear evaluation.

    Args:
        args: Parsed command line hyperparameters.
    """
    pl.seed_everything(args.seed)
    args.linear_eval = True

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
    
    # Run linear evaluation
    linear_eval(args, train_loader, val_loader, data_stats)


if __name__ == "__main__":
    # Performance and reproducibility settings
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    
    opts = LinearEvalOptions()
    args = opts.parse()

    main_linear_eval(args)
