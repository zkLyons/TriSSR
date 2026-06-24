<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-DBEDFA"></a>
  <a href="./README_zh.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-DFE0E5"></a>
</p>

![](https://komarev.com/ghpvc/?username=zkLyons&label=Views&color=yellow)


# TriSSR: Tri-expert fusion in state space models for sequential recommendation

## Project Structure

```
TriSSR/
├── dataset
├── main.py                  # Entry point for training and evaluation
├── trissr.py                # Core model implementation (TriSSR, TriSSRLayer, FrequencyLayer, TimeFourier, TriExpertFusion)
├── custom_utils.py          # Custom dataset & DataLoader
├── custom_trainer.py        # Custom trainer (mixed-precision training, evaluation logic, TensorBoard logging)
├── config.yaml              # Model and training hyperparameter configuration
├── environment.yaml         # Conda environment configuration
├── images/
│   ├── model_architecture.pdf   # Model architecture diagram
│   ├── frequency_picture.pdf    # Frequency filtering schematic
│   └── results.png              # Experimental results comparison
└── README.md
```

---

## Model Overview

### Architecture

This work is published in **Neurocomputing**: [TriSSR: Tri-expert fusion in state space models for sequential recommendation](https://www.sciencedirect.com/science/article/abs/pii/S0925231226009707)

![](./assets/model_architecture.png)

## Paper Introduction

**TriSSR** is a novel sequential recommendation model that innovatively integrates three complementary feature extraction paradigms:

1. **State Space Model (Mamba2)**: Models long- and short-term dependencies in user behavior sequences in a bidirectional manner
2. **Frequency Domain Filtering (Frequency Layer)**: Decomposes user interaction sequences into multi-band (low/mid/high) components in the Fourier domain and reconstructs them
3. **Time Encoding (TimeFourier)**: Explicitly models time intervals between user actions using Fourier features

The three feature streams are adaptively aggregated via a learnable **TriExpertFusion** module, achieving superior recommendation performance across multiple public benchmarks.

### Challenges Addressed

Traditional sequential recommendation models face the following challenges:

1. User behavior encompasses both short-term interests and long-term preferences, which manifest as different patterns in the frequency domain. A simple two-level division is insufficient to capture user characteristics — more fine-grained modeling is required.

2. User interests evolve over time, making it necessary to incorporate time interval information as a complement to user feature modeling.

3. RNN/CNN models suffer from gradient vanishing or limited receptive fields on long sequences. While self-attention offers strong long-sequence modeling capability, it faces a significant computational bottleneck. We introduce an SSM (State Space Model) that combines strong sequence modeling power with linear computational complexity.

## Experimental Results

![Experimental Results](./assets/results.png)

The model is evaluated on four public Amazon datasets against various baselines using **NDCG@10/20**, **MRR@10/20**, and **Hit@10/20** metrics. Experimental results show that TriSSR achieves competitive performance across all datasets.

---

## Environment Setup

Install via Conda:

```bash
git clone https://github.com/zkLyons/TriSSR.git
cd TriSSR
conda env create -f environment.yaml
conda activate your_conda_env

```

All experiments are conducted on an NVIDIA 24GB 3090 GPU. The main required packages are as follows:

- Python 3.10
- PyTorch 2.1.1 + CUDA 11.8
- mamba-ssm 2.2.2
- RecBole 1.2.0
- causal-conv1d 1.4.0

### Datasets

We use four public datasets: Amazon-Beauty, Amazon-Video-Games, Amazon-Sports, and Amazon-Toys. These datasets can be downloaded from the following link: [Google Drive](https://drive.google.com/drive/folders/1jsF4n1dge4KgdfD8HKxNjOcyHUKdEHka)

Setup steps:

1. Create a `dataset` folder
2. Download the datasets from: [Google Drive](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj)
3. Place the data files into the `dataset/` directory

The datasets follow RecBole's Atomic File format (`.inter` files), containing three main fields: `user_id`, `item_id`, and `timestamp`.

---

## Quick Start

Training and evaluation:

```bash
python main.py
```

### Configuration

All hyperparameters are configured in `config.yaml`. Key parameters include:

| Parameter              | Description                   | Default |
| ---------------------- | ----------------------------- | ------- |
| `hidden_size`          | Feature dimension             | 256     |
| `d_state`              | SSM state expansion dimension | 64      |
| `d_conv`               | Local convolution width       | 4       |
| `expand`               | Block expansion factor        | 2       |
| `num_layers`           | Number of TriSSR layers       | 1       |
| `dropout_prob`         | Dropout probability           | 0.4     |
| `beta`                 | Backward Mamba weight         | 0.1     |
| `maskratio`            | Sequence masking ratio        | 0.2     |
| `learning_rate`        | Learning rate                 | 0.0001  |
| `train_batch_size`     | Training batch size           | 1024    |
| `MAX_ITEM_LIST_LENGTH` | Maximum sequence length       | 50      |

To switch datasets, modify the corresponding dataset configuration block in `config.yaml` by uncommenting the desired dataset and commenting out the others.

## Acknowledgment

This project is built upon the following excellent open-source works, to which we extend our sincere gratitude:

- [SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation](https://dl.acm.org/doi/10.1145/3773038)

- **Mamba / Mamba2**: [state-spaces/mamba](https://github.com/state-spaces/mamba) — State space model providing an efficient backbone for sequence modeling
- **RecBole**: [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) — Unified recommendation framework providing data processing, evaluation, and other infrastructure

---

## Citation

If you use TriSSR in your research, please cite our paper:

```
@article{Zhang2026TriSSR,
  title={TriSSR: Tri-expert fusion in state space models for sequential recommendation},
  author={Kang Zhang and Quan Wen and Yujian Huang and Yanmei Hu and Na Dong and Xiaomeng Yang and Ruixing Huang},
  journal={Neurocomputing},
  year={2026},
  volume={685},
  pages={133573},
  url={https://www.sciencedirect.com/science/article/abs/pii/S0925231226009707}
}
```
