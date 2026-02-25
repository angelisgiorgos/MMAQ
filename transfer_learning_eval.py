import os
import warnings

# Ignore all warnings to keep logs clean
warnings.filterwarnings('ignore')

import pandas as pd
import pytorch_lightning as pl
import torch
from dataset.data_loaders import load_transfer_data
from opts import TransferLearningOptions
from utils.benchmarking.transfer_learning import tf_classification


def transfer_learning_cls(args):
    """
    Main routine for transfer learning classification evaluation.
    
    Args:
        args: Command line arguments.
    """
    pl.seed_everything(args.seed)
    args.linear_eval = True

    train_dataloader, validation_dataloader = load_transfer_data(args, "labels.csv")

    tf_classification(
        args,
        train_dataloader,
        validation_dataloader,
        None,
        4
    )


if __name__ == "__main__":
    # Performance and reproducibility settings
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    
    opts = TransferLearningOptions()
    args = opts.parse()
    
    args.freeze = False
    args.wandb_project = "AQ Multimodal SSL - Transfer Learning"
    
    transfer_learning_cls(args)