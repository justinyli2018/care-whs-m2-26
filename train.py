"""
Training script for 3D Medical Image Segmentation.

Usage:
    python train.py --experiment_name baseline --num_epochs 100
    python train.py --loss_type topology --topology_loss_weight 1e-6
    python train.py --use_sam --sam_rho 0.05
"""

import argparse
import warnings
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from config import ExperimentConfig, get_default_config
from sam import SAM
from unet_3d import create_unet_3d
from dataset import MedicalImageDataset, SegmentationDataLoader
from utils import (
    DiceLoss, DiceCrossEntropyLoss, TopologyConstrainedLoss,
    count_parameters, print_model_summary,
    per_class_dice_coefficient, iou_score
)

STRUCTURE_NAMES = [
    "Background",
    "LV Myocardium",
    "Left Atrium",
    "Left Ventricle",
    "Right Atrium",
    "Right Ventricle",
    "Ascending Aorta",
    "Pulmonary Artery",
]

warnings.filterwarnings('ignore')


class SegmentationTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.training.device)

        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.patience_counter = 0

        print(f"Training device: {self.device}")

    def setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        print("\n" + "=" * 60)
        print("SETTING UP DATA")
        print("=" * 60)

        print(f"\nLoading dataset from: {self.config.data.data_dir}")
        dataset = MedicalImageDataset(
            data_dir=self.config.data.data_dir,
            target_size=self.config.data.target_size,
            normalize=self.config.data.normalize_intensity,
            cache=self.config.data.cache_in_memory,
        )
        print(f"Total samples: {len(dataset)}")

        modality = self.config.data.modality.lower()
        if modality != "both":
            keyword = "ct" if modality == "ct" else "mr"
            dataset.cases = [c for c in dataset.cases if keyword in c['folder'].lower()]
            print(f"Modality filter '{modality}': {len(dataset.cases)} cases remaining")

        train_cases_split, val_cases_split, test_cases = SegmentationDataLoader.split_cases(
            dataset.cases,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
            seed=self.config.data.random_seed,
        )

        # Shared kwargs so all subset datasets use identical preprocessing
        dataset_kwargs = dict(
            target_size=self.config.data.target_size,
            normalize=self.config.data.normalize_intensity,
            cache=self.config.data.cache_in_memory,
        )

        if self.config.data.val_data_dir:
            train_cases = train_cases_split + val_cases_split

            print(f"\nLoading external validation dataset from: {self.config.data.val_data_dir}")
            val_dataset = MedicalImageDataset(
                data_dir=self.config.data.val_data_dir,
                **dataset_kwargs,
            )
            if modality != "both":
                keyword = "ct" if modality == "ct" else "mr"
                val_dataset.cases = [c for c in val_dataset.cases
                                     if keyword in c['folder'].lower()]
                print(f"Modality filter '{modality}': "
                      f"{len(val_dataset.cases)} external val cases remaining")

            print(f"Data split: {len(train_cases)} train (incl. internal val), "
                  f"{len(val_dataset.cases)} val (external), {len(test_cases)} test")
        else:
            train_cases = train_cases_split
            val_dataset = MedicalImageDataset(self.config.data.data_dir, **dataset_kwargs)
            val_dataset.cases = val_cases_split

            print(f"Data split: {len(train_cases)} train, "
                  f"{len(val_cases_split)} val, {len(test_cases)} test")

        train_dataset = MedicalImageDataset(self.config.data.data_dir, **dataset_kwargs)
        train_dataset.cases = train_cases

        test_dataset = MedicalImageDataset(self.config.data.data_dir, **dataset_kwargs)
        test_dataset.cases = test_cases

        loader_kwargs = dict(
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=self.config.data.shuffle_train,
            drop_last=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader

    def setup_model(self) -> nn.Module:
        print("\n" + "=" * 60)
        print("SETTING UP MODEL")
        print("=" * 60 + "\n")

        model = create_unet_3d(
            in_channels=self.config.model.in_channels,
            num_classes=self.config.model.num_classes,
            base_channels=self.config.model.base_channels,
            dropout_p=self.config.model.dropout_p,
            device=str(self.device),
        )

        print_model_summary(model)
        return model

    def setup_optimizer_and_scheduler(self, model: nn.Module):
        print("\n" + "=" * 60)
        print("SETTING UP OPTIMIZER AND SCHEDULER")
        print("=" * 60)

        opt_kwargs = dict(
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        opt_map = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
        if self.config.training.optimizer not in opt_map:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        base_optimizer_cls = opt_map[self.config.training.optimizer]
        if self.config.training.optimizer == "sgd":
            opt_kwargs["momentum"] = self.config.training.momentum

        if self.config.training.use_sam:
            optimizer = SAM(
                model.parameters(),
                base_optimizer_cls,
                rho=self.config.training.sam_rho,
                adaptive=self.config.training.sam_adaptive,
                **opt_kwargs,
            )
            print(f"Optimizer: SAM({self.config.training.optimizer}) "
                  f"rho={self.config.training.sam_rho} "
                  f"adaptive={self.config.training.sam_adaptive}")
        else:
            optimizer = base_optimizer_cls(model.parameters(), **opt_kwargs)
            print(f"Optimizer: {self.config.training.optimizer}")

        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"Weight decay: {self.config.training.weight_decay}")

        scheduler = None
        if self.config.training.use_scheduler:
            if self.config.training.scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.training.t_max,
                    eta_min=self.config.training.eta_min,
                )
            elif self.config.training.scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.config.training.lr_step_size,
                    gamma=self.config.training.lr_gamma,
                )
            elif self.config.training.scheduler_type == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=self.config.training.lr_gamma,
                )
            print(f"Scheduler: {self.config.training.scheduler_type}")

        return optimizer, scheduler

    def setup_loss(self):
        print("\n" + "=" * 60)
        print("SETTING UP LOSS FUNCTION")
        print("=" * 60)

        if self.config.training.loss_type == "dice":
            criterion = DiceLoss()
            print("Loss: Dice Loss")
        elif self.config.training.loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
            print("Loss: Cross-Entropy Loss")
        elif self.config.training.loss_type == "dice_ce":
            criterion = DiceCrossEntropyLoss()
            print("Loss: Dice + Cross-Entropy Loss")
        elif self.config.training.loss_type == "topology":
            criterion = TopologyConstrainedLoss(
                num_classes=self.config.model.num_classes,
                lambda_tp=self.config.training.topology_loss_weight,
            )
            print(f"Loss: Topology-Constrained Loss "
                  f"[lambda_tp={self.config.training.topology_loss_weight}]")
        else:
            raise ValueError(f"Unknown loss type: {self.config.training.loss_type}")

        return criterion.to(self.device)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        model.train()
        total_loss = 0.0
        num_batches = 0
        use_sam = isinstance(optimizer, SAM)

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            if use_sam:
                # First forward-backward: move to sharpest point w + e(w)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if self.config.training.gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             self.config.training.gradient_clipping)
                optimizer.first_step(zero_grad=True)

                # Second forward-backward: update from the perturbed point
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if self.config.training.gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             self.config.training.gradient_clipping)
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if self.config.training.gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             self.config.training.gradient_clipping)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {total_loss / num_batches:.4f}")

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, model: nn.Module, val_loader: DataLoader,
                 criterion: nn.Module) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        num_classes = self.config.model.num_classes
        all_per_class_dice = [[] for _ in range(num_classes)]
        all_iou_scores = []
        num_batches = 0

        for batch in val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            for i in range(predictions.shape[0]):
                class_dices = per_class_dice_coefficient(
                    predictions[i], labels[i], num_classes
                )
                for c, d in enumerate(class_dices):
                    all_per_class_dice[c].append(d)

                iou = iou_score(
                    (predictions[i] > 0).float(),
                    (labels[i] > 0).float(),
                )
                all_iou_scores.append(iou)

            num_batches += 1

        per_class_means = [float(np.mean(all_per_class_dice[c])) for c in range(num_classes)]
        mean_fg_dice = float(np.mean(per_class_means[1:]))  # exclude background

        return {
            'val_loss': total_loss / num_batches,
            'val_dice': mean_fg_dice,
            'val_dice_per_class': per_class_means,
            'val_iou': float(np.mean(all_iou_scores)),
        }

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, optimizer: optim.Optimizer,
              scheduler, criterion: nn.Module):
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch + 1

            print(f"\nEpoch [{self.current_epoch}/{self.config.training.num_epochs}]")
            print("-" * 60)

            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            print(f"Train Loss: {train_loss:.4f}")

            val_metrics = self.validate(model, val_loader, criterion)
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Dice (mean fg): {val_metrics['val_dice']:.4f}")
            for c, (name, score) in enumerate(
                zip(STRUCTURE_NAMES, val_metrics['val_dice_per_class'])
            ):
                print(f"  [{c}] {name:<20s} {score:.4f}")
            print(f"Val IoU: {val_metrics['val_iou']:.4f}")

            if scheduler is not None:
                scheduler.step()

            if val_metrics['val_dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['val_dice']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.config.training.use_early_stopping:
                if self.patience_counter >= self.config.training.patience:
                    print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                    break

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best validation Dice: {self.best_val_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train 3D UNet for medical image segmentation")
    parser.add_argument('--experiment_name', type=str, default='baseline')
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--topology_loss_weight', type=float, default=None)
    parser.add_argument('--modality', type=str, default=None, choices=['ct', 'mri', 'both'])
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['dice', 'cross_entropy', 'dice_ce', 'topology'])
    parser.add_argument('--val_data_dir', type=str, default=None)
    parser.add_argument('--use_sam', action='store_true', default=None)
    parser.add_argument('--sam_rho', type=float, default=None)
    args = parser.parse_args()

    config = get_default_config()
    config.experiment_name = args.experiment_name

    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.topology_loss_weight is not None:
        config.training.topology_loss_weight = args.topology_loss_weight
    if args.modality:
        config.data.modality = args.modality
    if args.loss_type:
        config.training.loss_type = args.loss_type
    if args.val_data_dir:
        config.data.val_data_dir = args.val_data_dir
    if args.use_sam:
        config.training.use_sam = True
    if args.sam_rho is not None:
        config.training.sam_rho = args.sam_rho

    print(f"Experiment: {config.experiment_name}")

    trainer = SegmentationTrainer(config)
    train_loader, val_loader, test_loader = trainer.setup_data()
    model = trainer.setup_model()
    criterion = trainer.setup_loss()
    optimizer, scheduler = trainer.setup_optimizer_and_scheduler(model)

    trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
