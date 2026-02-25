import pytorch_lightning as pl
import torch

from dataset.data_loaders import load_segmentation_data
from opts import TransferLearningOptions
from utils.benchmarking.transfer_segmentation import tf_segmentation


def transfer_learning_segm(args):
    """
    Main routine for transfer learning segmentation evaluation.
    
    Args:
        args: Command line arguments.
    """
    pl.seed_everything(args.seed)
    args.linear_eval = True

    train_dataloader, validation_dataloader = load_segmentation_data(args)

    tf_segmentation(
        args=args,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader
    )


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    
    opts = TransferLearningOptions()
    args = opts.parse()
    
    args.wandb_project = "AQ Multimodal SSL - Transfer Learning Segm"
    transfer_learning_segm(args)