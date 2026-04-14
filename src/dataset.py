from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ESC50Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        sample_rate: int,
        clip_duration_s: float,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: int = 0,
        fmax: int | None = None,
        val_fold: int = 1,
        time_shift_pct: float = 0.0,
        noise_snr_db_min: float = 0.0,
        noise_snr_db_max: float = 0.0,
        specaug_freq_mask: int = 0,
        specaug_time_mask: int = 0,
        specaug_num_freq_masks: int = 0,
        specaug_num_time_masks: int = 0,
        normalize: str = "per_sample",
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'")

        self.data_root = Path(data_root)
        meta_path = self.data_root / "meta" / "esc50.csv"
        audio_dir = self.data_root / "audio"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        meta = pd.read_csv(meta_path)
        self.class_to_idx = self._build_class_map(meta)

        if split == "train":
            self.meta = meta[meta["fold"] != val_fold].reset_index(drop=True)
        else:
            self.meta = meta[meta["fold"] == val_fold].reset_index(drop=True)

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_duration_s)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        self.is_train = split == "train"
        self.time_shift_pct = float(time_shift_pct)
        self.noise_snr_db_min = float(noise_snr_db_min)
        self.noise_snr_db_max = float(noise_snr_db_max)

        self.specaug_freq_mask = int(specaug_freq_mask)
        self.specaug_time_mask = int(specaug_time_mask)
        self.specaug_num_freq_masks = int(specaug_num_freq_masks)
        self.specaug_num_time_masks = int(specaug_num_time_masks)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.meta.iloc[index]
        audio_path = self.audio_dir / row["filename"]

        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        waveform = self._fix_length(waveform)
        if self.is_train:
            waveform = self._augment_waveform(waveform)

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        if self.normalize == "per_sample":
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        if self.is_train:
            mel_spec = self._spec_augment(mel_spec)

        mel_spec = mel_spec.unsqueeze(0)
        label = torch.tensor(self.class_to_idx[row["category"]], dtype=torch.long)
        return mel_spec, label

    def _fix_length(self, waveform: np.ndarray) -> np.ndarray:
        if len(waveform) == self.target_samples:
            return waveform
        if len(waveform) > self.target_samples:
            return waveform[: self.target_samples]

        padded = np.zeros(self.target_samples, dtype=np.float32)
        padded[: len(waveform)] = waveform
        return padded

    def _augment_waveform(self, waveform: np.ndarray) -> np.ndarray:
        # Time shift (circular roll) up to time_shift_pct of length
        if self.time_shift_pct > 0:
            max_shift = int(self.target_samples * self.time_shift_pct)
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                waveform = np.roll(waveform, shift)

        # Additive white noise with random SNR
        if self.noise_snr_db_max > 0 and self.noise_snr_db_max >= self.noise_snr_db_min:
            snr_db = np.random.uniform(self.noise_snr_db_min, self.noise_snr_db_max)
            signal_power = np.mean(waveform**2) + 1e-12
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(
                np.float32
            )
            waveform = waveform + noise
            waveform = np.clip(waveform, -1.0, 1.0)

        return waveform

    def _spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        if self.specaug_freq_mask <= 0 and self.specaug_time_mask <= 0:
            return mel

        out = mel.clone()
        n_mels, n_frames = out.shape

        for _ in range(max(0, self.specaug_num_freq_masks)):
            if self.specaug_freq_mask <= 0:
                continue
            width = int(np.random.randint(0, min(self.specaug_freq_mask, n_mels) + 1))
            if width == 0:
                continue
            f0 = int(np.random.randint(0, max(1, n_mels - width + 1)))
            out[f0 : f0 + width, :] = 0.0

        for _ in range(max(0, self.specaug_num_time_masks)):
            if self.specaug_time_mask <= 0:
                continue
            width = int(np.random.randint(0, min(self.specaug_time_mask, n_frames) + 1))
            if width == 0:
                continue
            t0 = int(np.random.randint(0, max(1, n_frames - width + 1)))
            out[:, t0 : t0 + width] = 0.0

        return out

    @staticmethod
    def _build_class_map(meta: pd.DataFrame) -> Dict[str, int]:
        labels = sorted(meta["category"].unique().tolist())
        return {label: idx for idx, label in enumerate(labels)}
