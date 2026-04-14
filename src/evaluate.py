import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.dataset import ESC50Dataset
from src.model import AudioResNet


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained ESC-50 model")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get("config", {})
    num_classes = int(checkpoint["num_classes"])

    sample_rate = int(ckpt_cfg.get("sample_rate", 22050))
    clip_duration_s = float(ckpt_cfg.get("clip_duration_s", 5.0))
    n_fft = int(ckpt_cfg.get("n_fft", 2048))
    hop_length = int(ckpt_cfg.get("hop_length", 512))
    n_mels = int(ckpt_cfg.get("n_mels", 128))
    fmin = int(ckpt_cfg.get("fmin", 0))
    fmax = ckpt_cfg.get("fmax", None)
    normalize = str(ckpt_cfg.get("normalize", "per_sample"))

    val_ds = ESC50Dataset(
        data_root=args.data_root,
        split="val",
        sample_rate=sample_rate,
        clip_duration_s=clip_duration_s,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        val_fold=args.val_fold,
        normalize=normalize,
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AudioResNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict(model=model, loader=loader, device=device)
    acc = accuracy_score(y_true, y_pred)
    idx_to_class = {v: k for k, v in val_ds.class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    main()
