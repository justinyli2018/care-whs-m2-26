3D UNet for Whole-Heart Segmentation

A PyTorch implementation of 3D UNet for multi-class cardiac segmentation on the WHS++ dataset, with support for Dice/CE/topology-constrained losses and Sharpness-Aware Minimization (SAM).

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (strongly recommended; 128³ volumes are memory-intensive)

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Configure your data path

**The data directory is hardcoded in `config.py`.** Before running, open [config.py](config.py) and update line 7 to point to your local copy of the WHS++ training dataset:

```python
data_dir: str = "/path/to/your/Wholeheart_Train_Dataset/"
```

The expected folder structure is:

```
Wholeheart_Train_Dataset/
├── CT/
│   ├── Case0001_image.nii.gz
│   ├── Case0001_label.nii.gz
│   └── ...
└── MR/
    ├── Case0001_image.nii.gz
    ├── Case0001_label.nii.gz
    └── ...
```

### 2. Train with defaults

```bash
python train.py
```

This runs for 100 epochs with Adam optimizer, cosine LR schedule, Dice loss, and early stopping (patience = 20).

### 3. Common overrides

```bash
# Specify a different loss
python train.py --loss_type dice_ce
python train.py --loss_type topology --topology_loss_weight 1e-6

# Filter to a single modality
python train.py --modality ct
python train.py --modality mri

# Use Sharpness-Aware Minimization
python train.py --use_sam --sam_rho 0.05

# Use an external validation set
python train.py --val_data_dir /path/to/external/val/

# Adjust training parameters
python train.py --num_epochs 200 --batch_size 1 --learning_rate 5e-4
```

---

## Hardcoded Dataset Assumptions

This codebase was written specifically for the **WHS++ (Whole Heart Segmentation++) dataset** and contains several hardcoded values that reflect this:

| Location | What is hardcoded |
|---|---|
| [config.py:7](config.py) | Absolute path to the training data directory |
| [dataset.py:16](dataset.py) | `WHOLEHEART_LABEL_MAP` — raw NIfTI label integers mapped to class indices |
| [train.py:30–39](train.py) | `STRUCTURE_NAMES` — 8 cardiac structure names for per-class Dice logging |
| [config.py:30](config.py) | `num_classes = 8` — matches the 8 WHS++ label classes |

**If you use a different dataset**, you must update at minimum `WHOLEHEART_LABEL_MAP`, `STRUCTURE_NAMES`, and `num_classes`.

The 8 label classes are:

| Index | Structure | Raw label ID |
|---|---|---|
| 0 | Background | 0 |
| 1 | LV Myocardium | 205 |
| 2 | Left Atrium | 420 |
| 3 | Left Ventricle | 500 |
| 4 | Right Atrium | 550 |
| 5 | Right Ventricle | 600 |
| 6 | Ascending Aorta | 820 |
| 7 | Pulmonary Artery | 850 |

---

## File Overview

| File | Description |
|---|---|
| [config.py](config.py) | All hyperparameters via Python dataclasses |
| [dataset.py](dataset.py) | NIfTI loading, preprocessing, train/val/test split |
| [unet_3d.py](unet_3d.py) | 3D UNet architecture |
| [utils.py](utils.py) | Dice, DiceCE, topology-constrained loss functions; metrics |
| [sam.py](sam.py) | Sharpness-Aware Minimization optimizer |
| [train.py](train.py) | Full training loop with CLI |

---

## Topology-Constrained Loss

The topology loss (`--loss_type topology`) is an implementation of the approach from:

> Zhang et al., "Preserving Cardiac Integrity: A Topology-Infused Approach to Whole Heart Segmentation," arXiv:2410.10551.

It penalizes voxels that violate anatomical adjacency and containment constraints between cardiac structures during training.

---

## Notes

- Data is split 70/15/15 (train/val/test) by default, stratified per modality folder.
- Volumes are center-cropped or zero-padded to 128×128×128.
- Intensity normalization uses z-score over non-zero voxels.
- The dataset `__main__` block in [dataset.py](dataset.py) also contains a hardcoded test path and is intended only for standalone debugging.
