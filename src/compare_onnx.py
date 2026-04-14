import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.model import AudioResNet
from src.predict import build_input_tensor, load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX outputs on one audio file")
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--onnx-model", type=Path, default=Path("artifacts/best_model.onnx"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    num_classes = int(checkpoint["num_classes"])
    class_names = load_class_names(args.data_root)

    x = build_input_tensor(args.audio_path, cfg).to(device)

    # PyTorch output
    model = AudioResNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pt_logits = model(x).cpu().numpy()
        pt_probs = torch.softmax(torch.tensor(pt_logits), dim=1).numpy()

    # ONNX output
    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": x.cpu().numpy()})[0]
    onnx_probs = softmax_np(onnx_logits)

    max_abs_diff = float(np.max(np.abs(pt_probs - onnx_probs)))
    mean_abs_diff = float(np.mean(np.abs(pt_probs - onnx_probs)))

    pt_top = int(np.argmax(pt_probs[0]))
    onnx_top = int(np.argmax(onnx_probs[0]))

    print(f"Audio: {args.audio_path}")
    print(f"Max |prob diff|: {max_abs_diff:.8f}")
    print(f"Mean |prob diff|: {mean_abs_diff:.8f}")
    print(
        f"Top-1 class (PyTorch): {class_names[pt_top]} ({pt_probs[0, pt_top] * 100:.2f}%)"
    )
    print(
        f"Top-1 class (ONNX):    {class_names[onnx_top]} ({onnx_probs[0, onnx_top] * 100:.2f}%)"
    )

    top_k = max(1, min(args.top_k, num_classes))
    print("\nTop classes (PyTorch vs ONNX):")
    order = np.argsort(-pt_probs[0])[:top_k]
    for idx in order:
        print(
            f"{class_names[idx]:20s}  PT={pt_probs[0, idx] * 100:6.2f}%  ONNX={onnx_probs[0, idx] * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()
