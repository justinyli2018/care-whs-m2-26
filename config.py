from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DataConfig:
    # Change this line to the correct data directory for your project
    data_dir: str = "/data2/archive/justin/care-heart/Wholeheart_Train_Dataset/"
    val_data_dir: Optional[str] = None

    target_size: Tuple[int, int, int] = (128, 128, 128)
    normalize_intensity: bool = True
    cache_in_memory: bool = False

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    modality: str = "both"

    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True


@dataclass
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 8
    base_channels: int = 64
    dropout_p: float = 0.0


@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    momentum: float = 0.9

    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = False

    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "exponential"

    lr_step_size: int = 30
    lr_gamma: float = 0.1

    t_max: int = 100
    eta_min: float = 1e-6

    loss_type: str = "dice"  # "dice", "cross_entropy", "dice_ce", "topology"
    topology_loss_weight: float = 1e-6

    gradient_clipping: Optional[float] = 1.0

    device: str = "cuda"

    use_early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4


@dataclass
class ExperimentConfig:
    experiment_name: str = "wholeheart_unet_baseline"
    description: str = "3D UNet for cardiac segmentation"

    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
