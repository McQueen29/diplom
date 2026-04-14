import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from src.model import AudioResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict class probabilities for one audio file")
    parser.add_argument("--audio-path", type=Path, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to ESC-50 root with meta/esc50.csv")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--top-k", type=int, default=10, help="How many top classes to print first")
    return parser.parse_args()


def fix_length(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    if len(waveform) == target_samples:
        return waveform
    if len(waveform) > target_samples:
        return waveform[:target_samples]

    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: len(waveform)] = waveform
    return padded


def load_class_names(data_root: Path) -> list[str]:
    meta_path = data_root / "meta" / "esc50.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {meta_path}")
    meta = pd.read_csv(meta_path)
    return sorted(meta["category"].unique().tolist())


def build_input_tensor(audio_path: Path, cfg: dict) -> torch.Tensor:
    sample_rate = int(cfg.get("sample_rate", 22050))
    clip_duration_s = float(cfg.get("clip_duration_s", 5.0))
    n_fft = int(cfg.get("n_fft", 2048))
    hop_length = int(cfg.get("hop_length", 512))
    n_mels = int(cfg.get("n_mels", 128))
    fmin = int(cfg.get("fmin", 0))
    fmax = cfg.get("fmax", None)
    normalize = str(cfg.get("normalize", "per_sample"))

    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = fix_length(waveform, int(sample_rate * clip_duration_s))

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)

    if normalize == "per_sample":
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

    return mel_spec.unsqueeze(0).unsqueeze(0)


def main() -> None:
    args = parse_args()
    if not args.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    num_classes = int(checkpoint["num_classes"])

    class_names = load_class_names(args.data_root)
    if len(class_names) != num_classes:
        raise ValueError(
            f"Class count mismatch: checkpoint has {num_classes}, metadata has {len(class_names)}"
        )

    x = build_input_tensor(args.audio_path, cfg).to(device)

    model = AudioResNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pairs = list(zip(class_names, probs, strict=False))
    pairs_sorted = sorted(pairs, key=lambda p: p[1], reverse=True)

    top_k = max(1, min(args.top_k, len(pairs_sorted)))
    print(f"Device: {device}")
    print(f"Audio: {args.audio_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print("\nTop predictions:")
    for cls_name, prob in pairs_sorted[:top_k]:
        print(f"{cls_name:20s}  {prob * 100:6.2f}%")

    print("\nAll classes with probabilities:")
    for cls_name, prob in pairs_sorted:
        print(f"{cls_name:20s}  {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
