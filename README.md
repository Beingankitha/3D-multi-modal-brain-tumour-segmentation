# 3D Multi-Modal Brain Tumour Segmentation

[![CI](https://github.com/Beingankitha/3D-multi-modal-brain-tumour-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/Beingankitha/3D-multi-modal-brain-tumour-segmentation/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-quality research codebase for 3D multi-modal brain tumour segmentation with explainability using PyTorch and MONAI.

## Features

✨ **Production-Ready**
- Config-driven architecture with YAML configurations
- Reproducible experiments with seed management
- Cross-platform support (CUDA GPU, Apple Silicon MPS, CPU)
- Comprehensive test suite with CI/CD

🧠 **Medical Imaging**
- Medical Segmentation Decathlon Task01_BrainTumour dataset
- 4 MRI modalities: T1, T1Gd, T2, FLAIR
- Multi-class segmentation: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)

🏗️ **Architecture**
- 3D U-Net with MONAI
- Sliding window inference for large volumes
- Mixed precision training (AMP)
- Multi-device support (auto-detection)

📊 **Evaluation**
- Dice coefficient, IoU
- Region-based metrics (WT, TC, ET)
- Hausdorff distance

🔍 **Explainability**
- Occlusion sensitivity analysis
- Grad-CAM visualization
- Saliency maps

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- MONAI 1.3+

### From Source

```bash
# Clone repository
git clone https://github.com/Beingankitha/3D-multi-modal-brain-tumour-segmentation.git
cd 3D-multi-modal-brain-tumour-segmentation

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
```

## Dataset Setup

1. Download the Medical Segmentation Decathlon Task01_BrainTumour dataset:
   - Visit: http://medicaldecathlon.com/
   - Download: `Task01_BrainTumour.tar`

2. Extract to the data directory:
   ```bash
   mkdir -p data
   tar -xvf Task01_BrainTumour.tar -C data/
   ```

3. Verify structure:
   ```
   data/Task01_BrainTumour/
   ├── dataset.json
   ├── imagesTr/
   │   ├── BRATS_001.nii.gz
   │   └── ...
   └── labelsTr/
       ├── BRATS_001.nii.gz
       └── ...
   ```

**Important:** Do not commit dataset files to the repository (they are in `.gitignore`).

## Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default_config.yaml

# Train with custom settings
python scripts/train.py \
    --config configs/default_config.yaml \
    --data-root ./data/Task01_BrainTumour \
    training.batch_size=4 \
    training.num_epochs=100

# Quick training (smaller model, faster iteration)
python scripts/train.py --config configs/quick_train.yaml
```

### Inference

```bash
# Run inference on a single image
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output prediction.nii.gz

# With test-time augmentation
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output prediction.nii.gz \
    --use-tta

# With post-processing
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output prediction.nii.gz \
    --post-process
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/default_config.yaml \
    --split test

# Evaluate on validation set
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --split val
```

### Explainability

```bash
# Generate explainability visualizations
python scripts/explain.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output-dir explainability_results \
    --methods occlusion saliency

# Analyze specific class
python scripts/explain.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output-dir explainability_results \
    --methods occlusion gradcam saliency \
    --target-class 3
```

## Configuration

The project uses YAML configuration files. Key configurations:

- `configs/default_config.yaml`: Full training configuration
- `configs/quick_train.yaml`: Fast training for testing

### Key Configuration Sections

```yaml
data:
  root_dir: "./data/Task01_BrainTumour"
  num_workers: 4
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  name: "unet"
  in_channels: 4
  out_channels: 4
  channels: [32, 64, 128, 256, 512]

training:
  device: "auto"  # auto-detect: cuda > mps > cpu
  batch_size: 2
  num_epochs: 300
  amp: true  # mixed precision (CUDA only)
  
  loss:
    type: "dice_ce"
    dice_weight: 0.5
    ce_weight: 0.5
  
  optimizer:
    type: "adam"
    lr: 1e-4
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=brain_tumor_segmentation --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## Project Structure

```
.
├── src/brain_tumor_segmentation/
│   ├── data/              # Data loading and transforms
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities (loss, metrics, trainer)
│   ├── inference/         # Inference pipeline
│   ├── explainability/    # Explainability methods
│   └── utils/             # Utilities (device, config, etc.)
├── configs/               # Configuration files
├── scripts/               # Training, inference, evaluation scripts
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks (tutorials)
└── docs/                  # Documentation
```

## Platform Support

### CUDA (NVIDIA GPUs)
- Full support with mixed precision training (AMP)
- Recommended for training large models

### Apple Silicon (MPS)
- Native support for M1/M2/M3 chips
- Faster than CPU, no mixed precision yet

### CPU
- Works on any platform
- Slower but useful for development/testing

The framework automatically detects the best available device when `device: "auto"` is set.

## Colab/Kaggle Support

The codebase is designed to work seamlessly in Google Colab and Kaggle:

```python
# In Colab/Kaggle notebook
!git clone https://github.com/Beingankitha/3D-multi-modal-brain-tumour-segmentation.git
%cd 3D-multi-modal-brain-tumour-segmentation
!pip install -r requirements.txt

# Download and prepare dataset
# ... (download Task01_BrainTumour.tar)

# Train
!python scripts/train.py --config configs/default_config.yaml
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{brain_tumor_segmentation_2024,
  title={3D Multi-Modal Brain Tumour Segmentation},
  author={Your Name},
  year={2024},
  url={https://github.com/Beingankitha/3D-multi-modal-brain-tumour-segmentation}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical Segmentation Decathlon for the dataset
- MONAI project for medical imaging tools
- PyTorch team for the deep learning framework

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions and issues:
- Open an issue on GitHub
- Check existing issues and discussions

## Roadmap

- [ ] Add nnU-Net architecture
- [ ] Support for more datasets
- [ ] Docker containerization
- [ ] Web demo interface
- [ ] Pre-trained model zoo
- [ ] Multi-GPU training support

---

**Reproducible PyTorch/MONAI pipeline for 3D multi-modal brain tumour segmentation (MSD Task01) with training, inference, metrics, and explainability, plus tests and CI.**
