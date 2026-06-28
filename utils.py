import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, weight: List[float] = None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.softmax(predictions, dim=1)

        num_classes = predictions.shape[1]
        batch_size = predictions.shape[0]

        predictions_flat = predictions.view(batch_size, num_classes, -1)
        targets_flat = targets.view(batch_size, -1)

        targets_one_hot = torch.zeros_like(predictions_flat)
        targets_one_hot.scatter_(1, targets_flat.unsqueeze(1), 1)

        dice_scores = []
        for c in range(num_classes):
            pred_c = predictions_flat[:, c, :]
            target_c = targets_one_hot[:, c, :]
            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            if self.weight is not None:
                dice = dice * self.weight[c]
            dice_scores.append(dice.mean())

        return 1.0 - torch.stack(dice_scores).mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight: List[float] = None, lambda_dice: float = 1.0, lambda_ce: float = 1.0):
        super(DiceCrossEntropyLoss, self).__init__()
        self.dice_loss = DiceLoss(weight=weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(weight) if weight else None)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.lambda_dice * dice + self.lambda_ce * ce


class TopologyConstrainedLoss(nn.Module):
    """
    Zhang et al., "Preserving Cardiac Integrity: A Topology-Infused Approach
    to Whole Heart Segmentation" (arXiv:2410.10551).
    Total loss: L = L_CE + L_Dice + λ · L_tp
    L_tp = L_CE(p ⊙ N, g ⊙ N) where N masks topology-violating voxels.
    """

    # Pairs that must never be spatially adjacent.
    # Genuinely adjacent pairs (valve connections, shared septum) are intentionally omitted.
    EXCLUSION_PAIRS: List[Tuple[int, int]] = [
        (4, 6),  # Right Atrium   ↔ Ascending Aorta
        (2, 6),  # Left Atrium    ↔ Ascending Aorta
        (3, 4),  # Left Ventricle ↔ Right Atrium
        (2, 5),  # Left Atrium    ↔ Right Ventricle
        (4, 7),  # Right Atrium   ↔ Pulmonary Artery
    ]

    # (container_class, contained_class)
    CONTAINMENT_PAIRS: List[Tuple[int, int]] = [
        (1, 3),  # LV Myocardium must surround Left Ventricle
    ]

    def __init__(self, num_classes: int = 8, lambda_tp: float = 1e-6):
        super(TopologyConstrainedLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_tp = lambda_tp

        self.dice_loss = DiceLoss()
        self.ce_loss_mean = nn.CrossEntropyLoss()
        self.ce_loss_per_voxel = nn.CrossEntropyLoss(reduction='none')

        # 3×3×3 all-ones kernel for morphological dilation
        self.register_buffer('_dilation_kernel', torch.ones(1, 1, 3, 3, 3))

    def _dilate(self, binary_mask: torch.Tensor) -> torch.Tensor:
        x = binary_mask.unsqueeze(1).float()
        dilated = F.conv3d(x, self._dilation_kernel, padding=1) > 0
        return dilated.squeeze(1)

    @torch.no_grad()
    def _violation_mask(self, pred_classes: torch.Tensor) -> torch.Tensor:
        violation = torch.zeros_like(pred_classes, dtype=torch.bool)

        for cls_a, cls_b in self.EXCLUSION_PAIRS:
            mask_a = pred_classes == cls_a
            mask_b = pred_classes == cls_b
            if not (mask_a.any() and mask_b.any()):
                continue
            violation |= mask_b & self._dilate(mask_a)
            violation |= mask_a & self._dilate(mask_b)

        for cls_container, cls_contained in self.CONTAINMENT_PAIRS:
            mask_contained = pred_classes == cls_contained
            if not mask_contained.any():
                continue
            mask_container = pred_classes == cls_container
            outside = ~(mask_container | mask_contained)
            if not outside.any():
                continue
            violation |= mask_contained & self._dilate(outside)

        return violation

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce_loss_mean(predictions, targets)
        loss_dice = self.dice_loss(predictions, targets)

        pred_classes = torch.argmax(predictions.detach(), dim=1)
        N = self._violation_mask(pred_classes)

        if N.any():
            per_voxel = self.ce_loss_per_voxel(predictions, targets)
            loss_tp = (per_voxel * N.float()).sum() / N.float().sum()
        else:
            loss_tp = predictions.new_zeros(1).squeeze()

        return loss_ce + loss_dice + self.lambda_tp * loss_tp


def per_class_dice_coefficient(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1.0
) -> List[float]:
    scores = []
    for c in range(num_classes):
        pred_c = (predictions == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        scores.append(((2 * intersection + smooth) / (union + smooth)).item())
    return scores


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> float:
    if predictions.dtype != targets.dtype:
        predictions = predictions.float()
        targets = targets.float()
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module):
    total, trainable = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total * 4 / 1024 / 1024:.2f} MB (assuming float32)")
