from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    labels: list[str] | None = None,
    normalize: str | None = "true",
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True, values_format=".2f")
    ax.set_title("Confusion matrix" + (f" (normalize={normalize})" if normalize else ""))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    labels: list[str] | None = None,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
