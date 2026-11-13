# Band-Selection Stability and Semantic Segmentation Performance: A Study on Hyperspectral City

[![arXiv](https://img.shields.io/badge/arXiv-2609.31074-b31b1b.svg)](https://arxiv.org/abs/2609.31074)

Official code for the paper **"Band-Selection Stability and Semantic Segmentation Performance: A Study on Hyperspectral City"**.

This repository provides a PyTorch framework for evaluating hyperspectral band selection strategies for semantic segmentation. It supports multiple segmentation architectures using SMP, configurable band subsets, and automated experimentation across different random seeds and band ablation settings.

---

## Paper Summary

The study examines how sensitive hyperspectral band-selection methods are to the data sample they are run on, and how the selected bands affect downstream semantic segmentation. Six band-selection methods are evaluated on ten class-balanced regions of interest from the [Hyperspectral City V2 dataset](https://pbdl-ws.github.io/pbdl2021/challenge/download.html) (128 bands, 450–950 nm), using top-K band subsets and three segmentation models against a full-band baseline. Please follow the dataset's license and terms of use.

Key findings:

* Sim-LP is the most stable method, and together with JMIM+CSNR gives the best segmentation results.
* Top-K models remain competitive with the full 128-band baseline (up to +2.01 mIoU and +1.72 mF1) while running 18-22x faster on CPU at K=9.
* Performance does not rise steadily with K, and stability does not consistently predict segmentation performance, so stability alone is an unreliable guide to downstream results.

---

## Features

* Train semantic segmentation models using HSI
* Compare multiple encoder-decoder architectures, such as U-Net, DeepLabV3+ and SegFormer
* Evaluate different hyperspectral band selection methods: CMIM, JMIM, mRMR, Sim-LP, JMIM+CSNR, ...
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

The framework currently supports models from [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) (SMP): U-Net, DeepLabV3+ and SegFormer. Please follow SMP's license and terms of use.

Example encoder backbones include: EfficientNet-B0 and MiT-B0. Any additional architectures/backbones supported by SMP can easily be added by modifying `MODEL_CONFIGS` inside `train.py`.

---

## Dataset

### Dataset Format

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

The original hypercubes were converted to NumPy arrays (`.npy`).

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

Each dataset entry includes:

* Number of classes
* Input channels
* Batch size
* Dataset paths
* Class weights
* Gradient accumulation settings

Example (values in angle brackets and `...` are placeholders to be replaced, so the snippet is not valid JSON as written):

```json
{
  "<dataset_name>": {
    "classes": "<number of classes>",
    "in_channels": "<number of channels>",
    "batch_size": 8,
    "class_weights": ["<w_0>", "<w_1>", "..."],
    "OriginalDataset": [
        "../train/images",
        "../train/masks",
        "../validation/images",
        "../validation/masks"
    ]
  }
}
```

Adjust the dataset paths to match the location of your dataset.

---

### Band Selection Configuration

Selected spectral bands are loaded from:

```text
bandSelect_Config.json
```

Each entry corresponds to a band selection strategy. Use `null` for the full-band baseline and a list of band indices (0-based, in `[0, 127]` for Hyperspectral City V2) for each method.

Example (`<...>` and `...` are placeholders):

```json
{
    "Original": null,
    "JMIM": [0, 12, 35],
    "CMIM": [3, 20, 41],
    "<Any Custom bandselection>": ["<b_1>", "<b_2>", "..."]
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

If you use this repository in your research, please cite our paper:

```bibtex
@misc{li2026bandselection,
  title         = {Band-Selection Stability and Semantic Segmentation Performance: A Study on Hyperspectral City},
  author        = {Li, Jiarong and Shah, Imad Ali and Ward, Enda and Glavin, Martin and Jones, Edward and Deegan, Brian},
  year          = {2026},
  eprint        = {2609.31074},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.31074}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This project builds upon, please follow their license and terms of use:

* PyTorch
* segmentation_models_pytorch
* NumPy
* scikit-learn
* Hyperspectral City V2 dataset
