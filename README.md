# Transformers for Hyperspectral Image Classification

This repository contains an end-to-end PyTorch implementation of transformer-based models for hyperspectral image (HSI) classification, built and trained from scratch on standard benchmark datasets such as Indian Pines, Pavia University, and others commonly used in the literature. The goal is to explore how pure transformers and hybrid CNN-Transformer architectures capture spectral-spatial dependencies in HSIs and how they compare to conventional CNN-based models.

## Motivation

Hyperspectral images contain hundreds of contiguous spectral bands, which provide rich information about materials on the Earth's surface but also lead to high dimensionality and strong band correlations. Transformers are well-suited for modeling long-range dependencies and complex relationships across both spatial locations and spectral channels, making them a natural fit for HSI classification tasks.

## Features

- PyTorch implementation of transformer-based architectures tailored for hyperspectral images (spectral, spatial, and spectral-spatial attention)
- Support for widely used HSI datasets (e.g., Indian Pines, Pavia University, Kennedy Space Center, Houston2018), assuming data are available in `.mat` format
- Configurable training scripts with options for patch size, number of heads, depth, learning rate, and data augmentation
- Modular code structure to help experiment with:
  - Pure transformer backbones for HSI
  - CNN + Transformer hybrids (local feature extraction + global attention)

## Repository Structure

```
.
├── data/               # Scripts or notes for organizing hyperspectral datasets
├── models/             # Transformer, CNN-Transformer, and utility modules
├── scripts/            # Training and evaluation entry points for different datasets
├── utils/              # Helper functions for metrics, visualization, logging
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/dikshitrishii/Transformers-Implementation-On-Hyperspectral-Images.git
cd Transformers-Implementation-On-Hyperspectral-Images
```

2. Create and activate a Python environment (conda or venv):
```bash
# Using conda
conda create -n hsi-transformer python=3.8
conda activate hsi-transformer

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn matplotlib tqdm
pip install tensorboard opencv-python
```

## Data Preparation

This project assumes access to standard public HSI benchmarks that are distributed as MATLAB `.mat` files. Typical directory structure:

```
data/
├── IndianPines/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PaviaU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── KSC/
│   ├── KSC.mat
│   └── KSC_gt.mat
└── Houston2018/
    ├── Houston2018.mat
    └── Houston2018_gt.mat
```

Download the datasets from their official sources and place them in the appropriate folders.

## Usage

### Training

Example commands for training on different datasets:

```bash
# Train on Indian Pines
python train_indian_pines.py --model transformer --epochs 200 --lr 1e-4

# Train on Pavia University
python train_paviaU.py --model transformer --epochs 200 --lr 1e-4

# Train a CNN + Transformer hybrid
python train_indian_pines.py --model cnn_transformer --epochs 200 --lr 1e-4
```

Key arguments:
- `--model`: backbone choice (e.g., `transformer`, `cnn_transformer`)
- `--patch_size`: spatial patch size for embedding
- `--depth`: transformer depth
- `--num_heads`: number of attention heads
- `--mlp_dim`: MLP dimension in transformer blocks
- `--batch_size`: training batch size
- `--epochs`: number of training epochs
- `--lr`: learning rate

### Evaluation

After training, evaluate the model and generate classification maps:

```bash
python test_indian_pines.py --checkpoint path/to/checkpoint.pth
python test_paviaU.py --checkpoint path/to/checkpoint.pth
```

Evaluation outputs include:
- Overall accuracy (OA)
- Average accuracy (AA)
- Kappa coefficient
- Per-class accuracies
- Confusion matrix
- Classification maps

## Model Architecture

The models in this repository are inspired by spectral-spatial transformer networks and vision transformers adapted for hyperspectral data. Core components include:

### Pure Transformer
- Patch or voxel embedding of HSI cubes into token sequences
- Multi-head self-attention over spatial tokens, spectral tokens, or combined spectral-spatial tokens
- Residual connections and layer normalization
- MLP blocks for feature transformation
- Classification head for final predictions

### CNN + Transformer Hybrid
- CNN front-end for local feature extraction
- Transformer layers for capturing global context
- Combined spectral-spatial feature learning
- Enhanced classification through multi-scale feature fusion

Key design choices:
- Spectral positional encodings to preserve band ordering information
- Factorized attention mechanisms for computational efficiency
- Data augmentation strategies tailored for hyperspectral data

## Results

Experimental results on benchmark datasets:

| Dataset        | Model              | OA (%) | AA (%) | Kappa |
|---------------|-------------------|--------|--------|-------|
| Indian Pines  | Transformer        |   TBD  |   TBD  |  TBD  |
| Indian Pines  | CNN + Transformer  |   TBD  |   TBD  |  TBD  |
| PaviaU        | Transformer        |   TBD  |   TBD  |  TBD  |
| PaviaU        | CNN + Transformer  |   TBD  |   TBD  |  TBD  |

*Note: Update with actual experimental results after training.*

### Classification Maps

Visual results showing predicted land cover maps will be added here after experiments are completed.

## Project Structure Details

### Models (`models/`)
- `transformer.py`: Pure transformer architecture for HSI classification
- `cnn_transformer.py`: Hybrid CNN-Transformer model
- `attention.py`: Various attention mechanism implementations
- `positional_encoding.py`: Spectral and spatial positional encoding modules
- `patch_embedding.py`: HSI patch embedding layers

### Scripts (`scripts/`)
- `train_*.py`: Training scripts for different datasets
- `test_*.py`: Evaluation and testing scripts
- `visualize.py`: Visualization utilities for classification maps

### Utils (`utils/`)
- `data_loader.py`: Dataset loading and preprocessing
- `metrics.py`: Evaluation metrics (OA, AA, Kappa)
- `augmentation.py`: Data augmentation for HSI
- `logger.py`: Training logging and monitoring

## Roadmap

Future improvements and extensions:

- [ ] Add self-supervised pretraining via masked image modeling
- [ ] Support additional datasets (WHU-Hi series, Chikusei)
- [ ] Implement Swin-style windowed attention for efficiency
- [ ] Add hierarchical transformer architectures
- [ ] Integrate cross-attention mechanisms for multi-modal fusion
- [ ] Develop uncertainty quantification methods
- [ ] Add model compression and quantization techniques
- [ ] Create comprehensive documentation and tutorials

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If this repository contributes to your research, please consider citing:

```bibtex
@misc{transformers_hsi_2024,
  author = {Rishii Dikshit},
  title = {Transformers Implementation for Hyperspectral Image Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/dikshitrishii/Transformers-Implementation-On-Hyperspectral-Images}
}
```

## Acknowledgements

This project builds on ideas from prior work on hyperspectral vision transformers and spectral-spatial transformer networks for HSI classification. Public benchmark datasets used in this repository are released and maintained by their original authors and institutions.

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please open an issue on GitHub or contact the repository owner.

---

**Note**: This is an active research project. Results, documentation, and code structure may be updated.
