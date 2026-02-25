import os

import numpy as np
import pandas as pd
import torchvision.transforms as T
from lightly.transforms import SimCLRTransform
from lightly.transforms.byol_transform import BYOLTransform
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

# Project-specific imports
from dataset.plant_dataset import create_dataset
from dataset.SatelliteConstrativeDataset import SatelliteContrastiveDataset
from dataset.segmentation_dataset import init_segmentation_datasets
from dataset.transforms import (
    ChangeBandOrder,
    CustomColorJitter,
    DatasetStatistics,
    GaussianBlur,
    Normalize,
    RandomChannelDrop,
    RandomHorizontalFlip,
    Randomize,
    RandomResizeCrop,
    RandomVerticalFlip,
    Resize,
    ToGray,
    ToTensor,
    random_rotation_transform,
)
from dataset.UniModalDataset import UniModalRGBDataset
from utils import split_samples_df


def load_rgb_unimodal_datasets(args):
    """
    Loads and prepares the training and validation datasets for RGB unimodal models.
    """
    data_stats = DatasetStatistics()

    if args.linear_eval:
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ])
    else:
        if args.model == "dino":
            train_transform = DINOTransform()
        elif args.model == "simclr":
            train_transform = SimCLRTransform()
        elif args.model == "byol":
            train_transform = BYOLTransform()
        else:
            raise ValueError(f"Unknown model type for RGB unimodal: {args.model}")

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ])

    target_transform = T.Compose([
        ChangeBandOrder(),
        Normalize(data_stats),
        ToTensor()
    ])

    samples_df = args.samples_file if isinstance(args.samples_file, pd.DataFrame) else pd.read_csv(args.samples_file, index_col="idx")
    samples_df = samples_df[~np.isnan(samples_df.no2)]

    train, val = split_samples_df(samples_df, test_size=0.2)

    train_dataset = UniModalRGBDataset(
        args=args,
        data_tabular=train,
        augmentation=train_transform,
        target_augmentation=target_transform,
    )
    val_dataset = UniModalRGBDataset(
        args=args,
        data_tabular=val,
        augmentation=val_transform,
        target_augmentation=target_transform,
    )

    return train_dataset, val_dataset, data_stats


def load_datasets(args):
    """
    Loads and prepares the training and validation datasets for multimodal models.
    """
    if args.datatype != "multimodal":
        raise ValueError(f"Unknown datatype {args.datatype}. Expected 'multimodal'.")

    data_stats = DatasetStatistics()
    
    train_default_transforms = T.Compose([
        ChangeBandOrder(),
        Normalize(data_stats),
        ToTensor(),
        RandomResizeCrop(size=224, scale=(0.08, 1.0)),
        Resize(224)
    ])
    
    if args.linear_eval:
        train_aug_transforms = train_default_transforms
    else:
        train_aug_transforms = T.Compose([
            ChangeBandOrder(),
            Normalize(data_stats),
            ToGray(out_channels=12),
            GaussianBlur(sigma=[0.1, 2]),
            RandomChannelDrop(),
            Randomize(),
            ToTensor(),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.0),
            CustomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            random_rotation_transform(rr_prob=0.0, rr_degrees=None),
            RandomResizeCrop(size=224, scale=(0.08, 1.0)),
            Resize(224),
        ])

    val_default_transforms = T.Compose([
        ChangeBandOrder(),
        Normalize(data_stats),
        ToTensor(),
        Resize(224)
    ])
    
    samples_df = args.samples_file if isinstance(args.samples_file, pd.DataFrame) else pd.read_csv(args.samples_file, index_col="idx")
    samples_df = samples_df[~np.isnan(samples_df.no2)]

    train, val = split_samples_df(samples_df, test_size=0.2)
    
    train_dataset = SatelliteContrastiveDataset(
        args=args,
        data_tabular=train,
        augmentation=train_aug_transforms,
        transforms=train_default_transforms,
    )
    val_dataset = SatelliteContrastiveDataset(
        args=args,
        data_tabular=val,
        augmentation=val_default_transforms,
        transforms=val_default_transforms,
    )

    return train_dataset, val_dataset, data_stats


def load_transfer_data(args, filename):
    """
    Loads and prepares data for transfer learning evaluation.
    """
    reg_data = pd.read_csv(os.path.join(args.tf_datapath, filename))
    print(f"Unique fuel types: {reg_data['fuel_type'].unique()}")

    if args.datatype == "rgb_unimodal":
        args.tf_channels = [3, 2, 1]
        args.num_channels = len(args.tf_channels)
    else:
        args.tf_channels = [int(c) for c in args.tf_channels.split(',')]
        args.num_channels = len(args.tf_channels)

    data_train_120x120 = create_dataset(
        args=args,
        data_path=args.tf_datapath, 
        mode='training/120x120/', 
        reg_data=reg_data, 
        mult=4, 
        train=True, 
        channels=args.tf_channels, 
        new_imgsize=args.img_size
    )

    data_train_300x300 = create_dataset(
        args=args,
        data_path=args.tf_datapath, 
        mode='training/300x300/', 
        reg_data=reg_data, 
        mult=4, 
        train=True, 
        channels=args.tf_channels, 
        size=300,
        new_imgsize=args.img_size
    )

    data_val = create_dataset(
        args=args,
        data_path=args.tf_datapath, 
        mode='validation/', 
        train=False, 
        reg_data=reg_data, 
        mult=1, 
        channels=args.tf_channels,
        new_imgsize=args.img_size
    )

    data_train = ConcatDataset([data_train_120x120, data_train_300x300])

    print(f"Number of training samples: {len(data_train)}")
    print(f"Number of validation samples: {len(data_val)}")

    train_sampler = RandomSampler(
        data_train, 
        replacement=True, 
        num_samples=int(2 * len(data_train) / 3)
    )

    train_dl = DataLoader(
        data_train, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True, 
        sampler=train_sampler
    )

    val_dl = DataLoader(data_val, batch_size=args.batch_size)
    
    return train_dl, val_dl


def load_segmentation_data(args):
    """
    Loads and prepares data for segmentation evaluation.
    """
    if args.datatype == "rgb_unimodal":
        args.tf_channels = [3, 2, 1]
        args.num_channels = len(args.tf_channels)
    else:
        args.tf_channels = [int(c) for c in args.tf_channels.split(',')]
        args.num_channels = len(args.tf_channels)

    return init_segmentation_datasets(args)
