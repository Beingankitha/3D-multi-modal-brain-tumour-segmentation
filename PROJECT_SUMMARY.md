# Project Summary

## Production-Quality 3D Multi-Modal Brain Tumour Segmentation

This repository contains a complete, production-ready research codebase for 3D multi-modal brain tumour segmentation using PyTorch and MONAI, targeting the Medical Segmentation Decathlon Task01_BrainTumour dataset.

## Key Features Implemented

### 1. Architecture
- **Config-driven**: YAML configurations using OmegaConf
- **Cross-platform**: Auto-detection and support for CUDA GPUs, Apple Silicon (MPS), and CPU
- **Production-ready**: Proper package structure with setup.py, comprehensive tests, and CI/CD

### 2. Data Pipeline
- Loader for MSD Task01 format (4 MRI modalities: T1, T1Gd, T2, FLAIR)
- MONAI-based transforms with augmentation
- Reproducible train/val/test splits with seeding
- Multi-class segmentation: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)

### 3. Model
- 3D U-Net architecture using MONAI
- Configurable channels and depth
- Support for Swin UNETR (transformer-based)
- Device-agnostic implementation

### 4. Training
- Combined loss functions (Dice + Cross-Entropy)
- Multiple optimizers (Adam, AdamW, SGD)
- Learning rate schedulers (Cosine, Step, Plateau)
- Early stopping and checkpointing
- Mixed precision training (AMP) for CUDA
- TensorBoard integration (optional)
- Comprehensive metrics (Dice, IoU, Hausdorff)

### 5. Inference & Evaluation
- Sliding window inference for large volumes
- Test-time augmentation (TTA)
- Post-processing (small object removal)
- Region-based evaluation (WT, TC, ET)
- Batch and single-image inference

### 6. Explainability
- Occlusion sensitivity analysis
- Gradient-based saliency maps
- Grad-CAM visualization
- 3D volume visualization utilities

### 7. Testing & Quality
- 14 unit tests covering all major components
- pytest-based test suite
- GitHub Actions CI/CD workflow
- Cross-platform testing (Ubuntu, macOS)
- Code quality tools (black, flake8, isort)

### 8. Documentation
- Comprehensive README with usage examples
- Tutorial notebook for quick start
- API documentation in docstrings
- Configuration guides

## File Structure

```
.
├── src/brain_tumor_segmentation/    # Main package
│   ├── data/                        # Data loading and transforms
│   ├── models/                      # Model architectures
│   ├── training/                    # Training utilities
│   ├── inference/                   # Inference pipeline
│   ├── explainability/              # Explainability methods
│   └── utils/                       # Utilities
├── configs/                         # YAML configurations
├── scripts/                         # CLI scripts
│   ├── train.py                     # Training script
│   ├── inference.py                 # Inference script
│   ├── evaluate.py                  # Evaluation script
│   └── explain.py                   # Explainability script
├── tests/                           # Unit tests
├── notebooks/                       # Tutorial notebooks
├── .github/workflows/               # CI/CD configuration
├── requirements.txt                 # Dependencies
├── setup.py                         # Package installation
└── README.md                        # Documentation
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Download Dataset
Download Task01_BrainTumour.tar from http://medicaldecathlon.com/ and extract to `data/Task01_BrainTumour/`

### Training
```bash
python scripts/train.py --config configs/default_config.yaml
```

### Inference
```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output prediction.nii.gz
```

### Evaluation
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --split test
```

### Explainability
```bash
python scripts/explain.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output-dir explainability_results \
    --methods occlusion saliency
```

## Platform Support

The codebase automatically detects and uses the best available device:

- **CUDA (NVIDIA GPUs)**: Full support with mixed precision training (AMP)
- **Apple Silicon (MPS)**: Native support for M1/M2/M3 chips
- **CPU**: Works on any platform for development/testing

## Reproducibility

- Deterministic seed setting for reproducibility
- Configuration saving for experiment tracking
- Comprehensive logging (TensorBoard, console)
- No dataset files committed to repository

## Testing

All 14 unit tests pass successfully:
```bash
pytest tests/ -v
```

## CI/CD

GitHub Actions workflow runs automatically on push:
- Cross-platform testing (Ubuntu, macOS)
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Code quality checks (black, flake8, isort)
- Test coverage reporting

## Requirements Met

✅ **Production-quality research codebase**  
✅ **Config-driven with YAML configurations**  
✅ **Reproducible with seed management**  
✅ **Testable with comprehensive test suite**  
✅ **Runnable on Apple Silicon (MPS/CPU) and CUDA GPUs**  
✅ **PyTorch + MONAI implementation**  
✅ **Medical Segmentation Decathlon Task01_BrainTumour target**  
✅ **No dataset files committed**  
✅ **Works on Colab/Kaggle**  

## License

MIT License - see LICENSE file for details.
