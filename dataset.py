import os
import glob
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


WHOLEHEART_LABEL_MAP = {0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        cache: bool = False,
        exclude_folders: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.cache = cache
        self.exclude_folders = exclude_folders or []

        self._cache: Dict[str, np.ndarray] = {}

        self.cases = self._find_cases()

        if len(self.cases) == 0:
            raise RuntimeError(f"No case pairs found in {data_dir}")

        print(f"Found {len(self.cases)} training cases")

    def _find_cases(self) -> List[Dict[str, str]]:
        cases = []

        for folder_path in self.data_dir.glob("*"):
            if not folder_path.is_dir():
                continue
            if folder_path.name in self.exclude_folders:
                continue

            image_files = sorted(folder_path.glob("*_image.nii.gz"))

            for image_file in image_files:
                case_name = image_file.stem.replace("_image.nii", "")
                label_file = image_file.parent / f"{case_name}_label.nii.gz"

                if label_file.exists():
                    cases.append({
                        'image': str(image_file),
                        'label': str(label_file),
                        'case_id': case_name,
                        'folder': folder_path.name
                    })

            print(f"  Found {len([c for c in cases if c['folder'] == folder_path.name])} cases in {folder_path.name}")

        return cases

    def _load_nii(self, filepath: str) -> np.ndarray:
        nib_img = nib.load(filepath)
        return np.asarray(nib_img.dataobj, dtype=np.float32)

    def _load_label_nii(self, filepath: str) -> np.ndarray:
        nib_img = nib.load(filepath)
        return np.asarray(nib_img.dataobj, dtype=np.int32)

    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        mask = image > 0
        if np.sum(mask) == 0:
            return image
        mean = np.mean(image[mask])
        std = np.std(image[mask])
        if std > 0:
            image = (image - mean) / std
        return image

    def _resize_volume(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
        mode: str = 'nearest'
    ) -> np.ndarray:
        current_size = volume.shape
        if current_size == target_size:
            return volume

        output = np.zeros(target_size, dtype=volume.dtype)
        slices_in = []
        slices_out = []

        for i, (curr, target) in enumerate(zip(current_size, target_size)):
            if curr >= target:
                start = (curr - target) // 2
                slices_in.append(slice(start, start + target))
                slices_out.append(slice(None))
            else:
                pad_total = target - curr
                pad_start = pad_total // 2
                slices_in.append(slice(None))
                slices_out.append(slice(pad_start, pad_start + curr))

        output[tuple(slices_out)] = volume[tuple(slices_in)]
        return output

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx]
        case_id = case['case_id']

        if self.cache and case_id in self._cache:
            image = self._cache[case_id]['image'].copy()
            label = self._cache[case_id]['label'].copy()
        else:
            image = self._load_nii(case['image'])
            label = self._load_label_nii(case['label'])

            if self.normalize:
                image = self._normalize_intensity(image)

            image = self._resize_volume(image, self.target_size, mode='nearest')
            label = self._resize_volume(label, self.target_size, mode='nearest')

            remapped = np.zeros_like(label, dtype=np.int64)
            for raw_id, class_idx in WHOLEHEART_LABEL_MAP.items():
                remapped[label == raw_id] = class_idx
            label = remapped

            if self.cache:
                self._cache[case_id] = {
                    'image': image.copy(),
                    'label': label.copy()
                }

        image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.from_numpy(label).unsqueeze(0).long()

        return {
            'image': image,
            'label': label.squeeze(0),
            'case_id': case_id,
            'folder': case['folder']
        }


class SegmentationDataLoader:
    @staticmethod
    def split_cases(
        cases: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        np.random.seed(seed)

        by_folder = {}
        for case in cases:
            folder = case['folder']
            if folder not in by_folder:
                by_folder[folder] = []
            by_folder[folder].append(case)

        train_cases = []
        val_cases = []
        test_cases = []

        for folder, folder_cases in by_folder.items():
            indices = np.arange(len(folder_cases))
            np.random.shuffle(indices)

            train_idx = int(train_ratio * len(folder_cases))
            val_idx = int((train_ratio + val_ratio) * len(folder_cases))

            train_cases.extend([folder_cases[i] for i in indices[:train_idx]])
            val_cases.extend([folder_cases[i] for i in indices[train_idx:val_idx]])
            test_cases.extend([folder_cases[i] for i in indices[val_idx:]])

            print(f"  {folder}: {len(indices[:train_idx])} train, "
                  f"{len(indices[train_idx:val_idx])} val, {len(indices[val_idx:])} test")

        return train_cases, val_cases, test_cases


if __name__ == "__main__":
    data_dir = "/path/to/dataset"

    print("Creating dataset...")
    dataset = MedicalImageDataset(
        data_dir=data_dir,
        target_size=(128, 128, 128),
        normalize=True,
        cache=False
    )

    print(f"\nDataset size: {len(dataset)}")

    print("\nLoading sample...")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Case ID: {sample['case_id']}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"Label unique values: {torch.unique(sample['label'])}")
