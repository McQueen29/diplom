from pathlib import Path
import shutil

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "artifacts" / "best_model.onnx"
    meta_path = root / "data" / "ESC-50-master" / "meta" / "esc50.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}. Run export first with src.export_onnx."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {meta_path}")

    out_dir = root / "mobile" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy model
    model_out = out_dir / "best_model.onnx"
    shutil.copy2(model_path, model_out)

    # 2) Build labels in the same class order as training (sorted unique labels)
    meta = pd.read_csv(meta_path)
    labels = sorted(meta["category"].unique().tolist())
    labels_out = out_dir / "labels.txt"
    labels_out.write_text("\n".join(labels) + "\n", encoding="utf-8")

    print("Mobile assets are ready:")
    print(f"- Model:  {model_out}")
    print(f"- Labels: {labels_out}")
    print(f"- Classes count: {len(labels)}")


if __name__ == "__main__":
    main()
