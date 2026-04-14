from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_root: Path
    sample_rate: int = 22050
    clip_duration_s: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 0
    fmax: int | None = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    num_workers: int = 0
    val_fold: int = 1
    seed: int = 42
    output_path: Path = Path("artifacts/best_model.pt")

    # Training quality knobs
    amp: bool = True
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 8

    # Augmentations (train only)
    time_shift_pct: float = 0.15
    noise_snr_db_min: float = 15.0
    noise_snr_db_max: float = 30.0

    # SpecAugment (train only, on mel)
    specaug_freq_mask: int = 16
    specaug_time_mask: int = 32
    specaug_num_freq_masks: int = 2
    specaug_num_time_masks: int = 2

    # Normalization
    normalize: str = "per_sample"  # "none" | "per_sample"
