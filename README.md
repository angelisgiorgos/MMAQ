# MMAQ: A Multi-Modal Self-Supervised Approach For Estimating Air Quality
[![Paper](https://img.shields.io/badge/IEEE-Paper-blue)](https://ieeexplore.ieee.org/abstract/document/10647792)
[![Conference](https://img.shields.io/badge/ICIP-2024-green)](https://2024.ieeeicip.org/)

This repository contains the official implementation of **MMAQ**, as presented at ICIP 2024. MMAQ is a multi-modal self-supervised learning (SSL) framework that integrates Sentinel-2, Sentinel-5P, and tabular land-use data to estimate ground-level air quality.

## 📊 Dataset & Architecture Structure

The pretraining dataset is structured as follows:

```
data/
├── data/
│   ├── editted/
│   │   └── pollutant_ssl.csv
├── sentinel-2
├── sentinel-5p/
```
Data Modalities:
- Sentinel-2: 12 multi-spectral bands resized to $224^2$.
- Sentinel-5P: Tropospheric column density reprojected to EPSG:32614.
- Tabular: Land use data including Altitude, Population Density, Area Type, and Station Type.

The Sentinel-2 and Sentinel-5p data can be downloaded from [here](https://zenodo.org/records/5764262)

## 📊 Benchmarks & Results

MMAQ achieves state-of-the-art performance by capturing both inter-modal and intra-modal correspondences.

### 1. $NO_2$ Concentration Estimation
Evaluated on a dataset of ~3K samples across Europe.

| Method | Type | $R^2$ (Linear) | $R^2$ (Fine-Tuning) | MSE (FT) |
| :--- | :--- | :---: | :---: | :---: |
| **MMAQ (Ours)** | **Multi-Modal** | **0.5863** | **0.7507** | **32.601** |
| AQNet [24] | Multi-Modal | - | 0.6923 | 39.524 |
| DeCUR [31] | Multi-Modal | 0.5038 | 0.5753 | 73.846 |
| Barlow Twins [11]| Unimodal | 0.4136 | 0.4281 | 82.486 |

> **Key Result:** MMAQ provides a **17% improvement** in MSE over the previous state-of-the-art.

### 2. Environmental Transfer Learning
Performance on fossil fuel power plant tasks (Classification & Segmentation).

| Task | Metric | Frozen Backbone | Fine-Tuned (FT) |
| :--- | :--- | :---: | :---: |
| **Fuel Classification** | Accuracy (%) |88.4% | **95.9%** |
| **Plume Segmentation** | mIoU | 61.8 | **75.5** |

---

## 🛠️ Architecture Details

The model leverages a multi-modal redundancy reduction loss (extended Barlow Twins).

* **Sentinel-2 Branch:** ResNet-50 backbone processing all 12 multi-spectral bands.
* **Sentinel-5P Branch:** ResNet-50 backbone processing tropospheric column density.
* **Tabular Branch:** 3 linear layers + Layer Normalization + Attention mechanism.
* **Loss Weighting:** Automatic weighting learns task-dependent weights ($\sigma_w$) during training.

---

## 🚀 Usage

### Installation
```bash
conda create -n mmaq python=3.11 -y
conda activate mmaq
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt
```

<!-- ## 📁 Pretrained Checkpoints

You can find the best-performing pretrained checkpoints for the MMAQ model here:
* **MMAQ Best**: [`checkpoints/mmaq_best.ckpt`](./checkpoints/mmaq_best.ckpt) — The full multi-modal architecture including Sentinel-2, Sentinel-5P, and Tabular encoders.
* **MMAQ- (S2 only)**: [`checkpoints/mmaq_s2_only.ckpt`](./checkpoints/mmaq_s2_only.ckpt) — Effective version using only Sentinel-2 encoders when other modalities are absent. -->

---

## 🚀 Running the Code

We provide a unified entry point, `run.py`, controlled via the `--task` argument. All experiments utilize the **LARS optimizer** with a batch size of **128**.

### 1. Self-Supervised Pretraining
This task trains the encoders from scratch using the multi-modal redundancy reduction loss. 
It captures both inter-modal (between different sources) and intra-modal (within the same source) correspondences.

```bash
python run.py \
    --task pretrain \
    --datatype multimodal \
    --samples_file ./data/data/editted/pollutant_ssl.csv \
    --max_epochs 500 \
    --lr 0.2 \
    --wandb_project "MMAQ_Pretraining"
    --model mmaq
    --channels 12
    --uncertainty
```
The current codebase supports self-supervised pretraining for both multimodal and unimodal models, and is capable of handling both multispectral and rgb-based data.
Models Supported:
* MMAQ (Multimodal, Sentinel-2(Multispectral), Sentinel-5P, Tabular )
* DeCUR (Multimodal, Sentinel-2(RGB) and Sentinel-5P)
* MMCL (Multimodal, Sentinel-2(RGB), Tabular)
* MM-Barlow Twins (Multimodal, Sentinel-2(Multispectral), Sentinel-5P, Tabular)
* MM-BYOL (Multimodal, Sentinel-2(Multispectral), Sentinel-5P, Tabular)
* SimCLR (Unimodal, S-2(RGB))
* SimSiam (Unimodal, S-2(RGB))
* DINO (Unimodal, S-2(RGB))
* Barlow Twins  (Unimodal, S-2(RGB))
* BYOL (Unimodal, S-2(RGB))
* ViCReg (Unimodal, S-2(RGB))
* MoCoV2 (Unimodal, S-2(RGB))

For pretraining DINO model, use the following command:
```bash
python run.py \
    --task pretrain \
    --datatype rgb_unimodal \
    --samples_file ./data/data/editted/pollutant_ssl.csv \
    --max_epochs 500 \
    --lr 0.2 \
    --wandb_project "MMAQ_Pretraining"
    --model dino
    --channels 3
```

### 2. Linear Probing
Evaluates the quality of learned representations by freezing the backbone and training a linear regressor for 90 epochs.

```bash
python run.py \
    --task linear_eval \
    --datatype multimodal \
    --samples_file ./data/data/editted/pollutant_ssl.csv \
    --max_epochs 90 \
    --lr 0.2 \
    --wandb_project "MMAQ_Linear_Evaluation"
    --model mmaq
    --channels 12
    --ckpt_path ./checkpoints/mmaq_best.ckpt
```

### 3. Linear Evaluation (Fine-Tuning)
This task evaluates the performance of the pre-trained encoders on a linear regression, by fine-tuning the encoders on the  NO2 Concentration Estimation task.
```bash
python run.py \
    --task fine_tune \
    --datatype multimodal \
    --samples_file ./data/data/editted/pollutant_ssl.csv \
    --max_epochs 90 \
    --lr 0.2 \
    --wandb_project "MMAQ_Linear_Evaluation"
    --model mmaq
    --channels 12
    --ckpt_path ./checkpoints/mmaq_best.ckpt
```

### 4. Transfer Learning: Classification & Segmentation 
Evaluates the backbone's versatility on auxiliary environmental tasks like fuel classification and industrial plume segmentation

```bash
# Classification
python run.py --task transfer_learning --tf_datapath /path/to/data --ckpt_path checkpoints/mmaq_best.ckpt

# Segmentation
python run.py --task transfer_segmentation --tf_datapath /path/to/data --ckpt_path checkpoints/mmaq_best.ckpt
```

### Citation
```
@inproceedings{angelis2024mmaq,
  title={MMAQ: A Multi-Modal Self-Supervised Approach For Estimating Air Quality From Remote Sensing Data},
  author={Angelis, G. F. and Emvoliadis, A. and Drosou, A. and Tzovaras, D.},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)},
  pages={319--325},
  year={2024},
  organization={IEEE}
}
```
