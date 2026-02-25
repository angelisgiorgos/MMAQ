# MMAQ: A Multi-Modal Self-Supervised Approach For Estimating Air Quality From Remote Sensing Data

This repository contains the official model architecture, training, and evaluation scripts for multimodal and unimodal Self-Supervised Learning (SSL), tailored primarily for satellite, remote-sensing, and plant datasets.

## Paper

This repository contains the official implementation of:

> **MMAQ: A Multi-Modal Self-Supervised Approach For Estimating Air Quality From Remote Sensing Data**  
> *G. Angelis et al., 2024 IEEE International Conference on Image Processing (ICIP)*  
> [IEEE Xplore Abstract](https://ieeexplore.ieee.org/abstract/document/10647792)

```bibtex
@inproceedings{angelis2024mmaq,
  title={MMAQ: A Multi-Modal Self-Supervised Approach For Estimating Air Quality From Remote Sensing Data},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)},
  pages={319--325},
  year={2024},
  organization={IEEE}
}
```

## Pretrained Checkpoints

You can find the best performing pretrained checkpoint for the model here:  
[`checkpoints/mmaq_best.ckpt`](./checkpoints/mmaq_best.ckpt)

## Running the Code

### 1. Pretraining (`main.py`)
Entry point for self-supervised pretraining. It integrates with contrastive frameworks (e.g., SimCLR, BYOL, DINO) on multimodal datasets.

```bash
python main.py \
    --datatype multimodal \
    --samples_file path/to/metadata.csv \
    --max_epochs 100 \
    --wandb_project "AQ Multimodal SSL"
```

### 2. Linear Probing (`linear_prob.py`)
Evaluates the frozen pretrained embeddings via linear regression.

```bash
python linear_prob.py \
    --datatype multimodal \
    --samples_file path/to/metadata.csv \
    --ckpt_path checkpoints/mmaq_best.ckpt \
    --max_num_epochs 90
```

### 3. Supervised Fine-Tuning (`fine_tune.py`)
End-to-end supervised fine-tuning of the pretrained representations.

```bash
python fine_tune.py \
    --datatype multimodal \
    --samples_file path/to/metadata.csv \
    --ckpt_path checkpoints/mmaq_best.ckpt \
    --max_num_epochs 90
```

### 4. Transfer Learning - Classification (`transfer_learning_eval.py`)
Evaluates the pretrained features on auxiliary multi-class classification tasks using a transfer learning dataset.

```bash
python transfer_learning_eval.py \
    --datatype multimodal \
    --tf_datapath /data/angelisg/sociobee/CO2_MultiModal/data \
    --ckpt_path checkpoints/mmaq_best.ckpt \
    --max_num_epochs 90 \
    --freeze
```

### 5. Transfer Learning - Segmentation (`tf_segm_eval.py`)
Adapts the embeddings for semantic segmentation evaluation.

```bash
python tf_segm_eval.py \
    --datatype multimodal \
    --tf_datapath /data/angelisg/sociobee/CO2_MultiModal/data \
    --ckpt_path checkpoints/mmaq_best.ckpt \
    --max_num_epochs 90 \
    --freeze
```

## Dataset and Data Loaders Structure

All data ingestion logic is grouped under the `dataset/` directory to remain isolated from training business logic:

- **`dataset/data_loaders.py`**: A centralized data loader factory. It provides primary loading functions (`load_datasets`, `load_rgb_unimodal_datasets`, `load_transfer_data`, `load_segmentation_data`) which parse configurations and yield data.
- **`dataset/SatelliteConstrativeDataset.py`**: Custom PyTorch dataset built for multimodal (e.g., multip-spectral/tabular) self-supervised pairing.
- **`dataset/UniModalDataset.py`**: Custom dataset dedicated to processing default standard 3-channel RGB imagery.
- **`dataset/plant_dataset.py` / `dataset/segmentation_dataset.py`**: Transfer learning and downstream datasets logic. 
- **`dataset/transforms.py` / `dataset/tabular_transforms.py`**: Specialized transformations handling non-standard image manipulation like multi-band randomization, tabular shifts, etc.

## Logging and Configuration

Hyperparameters are handled by modules in the `opts/` directory (e.g., `TrainOptions`, `TransferLearningOptions`).
Logging is integrated heavily with PyTorch Lightning and Weights & Biases (WandB).
