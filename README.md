# MMAQ: A Multi-Modal Self-Supervised Approach For Estimating Air Quality

This repository contains the official implementation of **MMAQ**, as presented at ICIP 2024. MMAQ is a multi-modal self-supervised learning (SSL) framework that integrates Sentinel-2, Sentinel-5P, and tabular land-use data to estimate ground-level air quality.

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
| **Fuel Classification** | Accuracy (%) | **95.9%** | 88.4% |
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

## 📁 Pretrained Checkpoints

You can find the best-performing pretrained checkpoints for the MMAQ model here:
* **MMAQ Best**: [`checkpoints/mmaq_best.ckpt`](./checkpoints/mmaq_best.ckpt) — The full multi-modal architecture including Sentinel-2, Sentinel-5P, and Tabular encoders.
* **MMAQ- (S2 only)**: [`checkpoints/mmaq_s2_only.ckpt`](./checkpoints/mmaq_s2_only.ckpt) — Effective version using only Sentinel-2 encoders when other modalities are absent.

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
    --learning_rate 0.2 \
    --wandb_project "MMAQ_Pretraining"
```
