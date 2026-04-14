import argparse
from pathlib import Path

import torch

from src.model import AudioResNet
from src.predict import build_input_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained ESC-50 model to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/best_model.onnx"))
    parser.add_argument("--sample-audio", type=Path, required=True, help="Audio used to build example tensor")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    num_classes = int(checkpoint["num_classes"])

    model = AudioResNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    example = build_input_tensor(args.sample_audio, cfg).to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        example,
        args.output,
        export_params=True,
        dynamo=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"Exported ONNX model: {args.output}")
    print(f"Input shape used for export: {tuple(example.shape)}")


if __name__ == "__main__":
    main()
