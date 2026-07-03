# HSI Band Selection Stability Study on Hyperspectral City Dataset for Semantic Segmentation

This repository provides a PyTorch framework for evaluating hyperspectral band selection strategies for semantic segmentation. It supports multiple segmentation architectures using SMP, configurable band subsets, and automated experimentation across different random seeds, and band ablation settings.

---

## Features

* Train semantic segmentation models using HSI
* Compare multiple encoder-decoder architectures, Such as:
  * U-Net
  * DeepLabV3+
  * SegFormer
* Evaluate different hyperspectral band selection methods: CMIM, JMIM, mRMR, Sim-LP, ...
* Automated band ablation experiments
* Multi-seed evaluation for reproducibility
* Mixed-precision training (AMP)
* Early stopping with checkpointing
* Evaluation metrics: mIoU, F1, Recall, Precision, Kappa


---

## Repository Structure

```text
.
├── train.py                  # Main training and evaluation pipeline
├── misc.py                   # Utility functions, metrics, losses, cleanup
├── dataset_config.json       # Dataset configuration
├── bandSelect_Config.json    # Selected spectral bands
├── experiments/              # Output directory
│   ├── <seed>/
│   └── ...
└── README.md
```

---

## Supported Models

The framework currently supports models from `segmentation_models_pytorch`: U-Net, DeepLabV3+, SegFormer, or any other supported by SMP.

Example encoder backbones include (We used in our study): EfficientNet-B0 and MiT-B0

Note: Additional architectures/backbones supported by `segmentation_models_pytorch` can easily be added by modifying `MODEL_CONFIGS` inside `train.py`.

---

## Dataset Format

The framework expects the following directory structure:

```text
dataset/
├── train/
│   ├── images/
│   │   ├── image_001.npy
│   │   ├── image_002.npy
│   │   └── ...
│   └── masks/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── validation/
    ├── images/
    └── masks/
```

### Image Requirements

Hyperspectral cubes should be stored as NumPy arrays (`.npy`).

Both layouts are supported:

* `(H, W, C)`
* `(C, H, W)`

The loader automatically detects the orientation.

### Mask Requirements

Masks may be provided as:

* PNG
* TIFF
* NumPy (`.npy`)

Ignored pixels should be labelled using:

```python
255
```

---

## Configuration

### Dataset Configuration

Datasets are defined inside:

```text
dataset_config.json
```

Each dataset configuration includes:

* Number of classes
* Input channels
* Batch size
* Dataset paths
* Class weights
* Gradient accumulation settings

Example:

```json
{
  "HSI_City": {
    "classes": 2,
    "in_channels": 224,
    "batch_size": 16,
    "class_weights": [...],
    "OriginalDataset": [
        "train/images",
        "train/masks",
        "val/images",
        "val/masks"
    ]
  }
}
```

---

### Band Selection Configuration

Selected spectral bands are loaded from

```text
bandSelect_Config.json
```

Each entry corresponds to a band selection strategy.

Example:

```json
{
    "Original": None,
    "JMIM": [...],
    "CMIM": [...],
    "Any Custom BS": [...]
}
```

---

## Training

Run all experiments using

```bash
python train.py
```

The training script automatically iterates over:

* Random seeds
* Datasets
* Band selection methods
* Band ablation levels
* Model architectures

---

## Training Pipeline

Each experiment performs:

1. Dataset loading
2. Band subset selection
3. Model initialization
4. Mixed-precision training
5. Early stopping
6. Best model checkpointing
7. Prediction generation
8. Metric computation
9. Experiment summary export

---

## Loss Function

Training uses a weighted combination of:

* Dice Loss
* Cross Entropy Loss

```text
Loss = 0.5 × Dice + 0.5 × Cross Entropy
```

Class weighting is supported through the dataset configuration.

---

## Evaluation Metrics

The framework reports:

### Overall Metrics

* Mean and Weighted IoU
* Mean and Weighted F1
* Mean and Weighted Precision
* Mean and Weighted Recall
* Mean and Weighted Accuracy
* Mean and Weighted Specificity
* Cohen's Kappa

### Per-Class Metrics

For every class:

* IoU
* Precision
* Recall
* F1 Score
* Accuracy
* Specificity

### Per-Image Metrics

Each validation image receives an individual IoU score, exported as

```text
per_image_iou.csv
```

---

## Output Structure

Each experiment generates

```text
experiments/
└── <seed>/
    └── Dataset/
        └── BandSelection/
            └── Ablation/
                └── Architecture_Backbone/
                    ├── best_model.pth
                    ├── experiment_summary.csv
                    ├── per_image_iou.csv
                    └── predictions/
                        ├── image001_pred.png
                        └── ...
```

---

## Reproducibility

The framework fixes:

* Python random seed
* NumPy random seed
* PyTorch random seed
* CUDA random seed

Deterministic CuDNN execution is enabled where possible.

---

## Requirements

* Python 3.10+
* PyTorch
* torchvision
* segmentation-models-pytorch
* NumPy
* pandas
* Pillow
* scikit-learn
* tqdm

Install dependencies with:

```bash
pip install torch torchvision segmentation-models-pytorch pandas numpy pillow scikit-learn tqdm
```

---

## Citation

If you use this repository in your research, please cite the associated publication (to be added).

---

## License



---

## Acknowledgements

This project builds upon:

* PyTorch
* segmentation_models_pytorch
* NumPy
* scikit-learn
