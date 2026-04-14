import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainConfig
from src.dataset import ESC50Dataset
from src.model import AudioResNet
from src.report import save_classification_report, save_confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(preds)
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_pred)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            use_amp = bool(scaler is not None)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, running_correct / total


def make_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, int]:
    train_ds = ESC50Dataset(
        data_root=cfg.data_root,
        split="train",
        sample_rate=cfg.sample_rate,
        clip_duration_s=cfg.clip_duration_s,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        val_fold=cfg.val_fold,
        time_shift_pct=cfg.time_shift_pct,
        noise_snr_db_min=cfg.noise_snr_db_min,
        noise_snr_db_max=cfg.noise_snr_db_max,
        specaug_freq_mask=cfg.specaug_freq_mask,
        specaug_time_mask=cfg.specaug_time_mask,
        specaug_num_freq_masks=cfg.specaug_num_freq_masks,
        specaug_num_time_masks=cfg.specaug_num_time_masks,
        normalize=cfg.normalize,
    )
    val_ds = ESC50Dataset(
        data_root=cfg.data_root,
        split="val",
        sample_rate=cfg.sample_rate,
        clip_duration_s=cfg.clip_duration_s,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        val_fold=cfg.val_fold,
        normalize=cfg.normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, len(train_ds.class_to_idx)


def train(cfg: TrainConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = make_dataloaders(cfg)
    model = AudioResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_metrics: Dict[str, float] = {}
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = cfg.output_path.parent / "history.json"
    history: list[dict] = []
    epochs_since_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"| lr={lr:.6f} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improve = 0
            best_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "config": vars(cfg),
                    "best_metrics": best_metrics,
                },
                cfg.output_path,
            )
            print(f"Saved best checkpoint to: {cfg.output_path}")
        else:
            epochs_since_improve += 1
            if cfg.early_stopping_patience > 0 and epochs_since_improve >= cfg.early_stopping_patience:
                print(
                    f"Early stopping: no val_acc improvement for {cfg.early_stopping_patience} epochs."
                )
                break

    # Final validation reports for the last model state (typically close to best).
    y_true, y_pred = predict_epoch(model=model, loader=val_loader, device=device)
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        out_path=cfg.output_path.parent / "confusion_matrix.png",
        labels=None,
        normalize="true",
    )
    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        out_path=cfg.output_path.parent / "classification_report.txt",
        labels=None,
    )

    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESC-50 baseline model")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to ESC-50 root (contains audio/ and meta/)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_fold=args.val_fold,
        output_path=args.output,
        amp=not args.no_amp,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience,
    )
    best_metrics = train(cfg)
    print("Best metrics:", best_metrics)


if __name__ == "__main__":
    main()
