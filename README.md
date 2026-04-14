# ESC-50 Baseline Training (Diploma Project)

This repository contains a baseline pipeline for environmental sound classification using the ESC-50 dataset and PyTorch.

## What is included

- ESC-50 metadata loading from `esc50.csv`
- Audio preprocessing into log-mel spectrograms
- A compact CNN classifier
- Train/validation loop with metrics
- Saving best model checkpoint

## Dataset setup

1. Download ESC-50 from the official source.
2. Extract it so you have this structure:

```text
data/ESC-50-master/
  audio/
  meta/esc50.csv
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run training

```bash
python -m src.train --data-root data/ESC-50-master --epochs 20 --batch-size 32
```

The best model will be saved to `artifacts/best_model.pt`.

## Export and mobile assets

Export ONNX:

```bash
python -m src.export_onnx --checkpoint artifacts/best_model.pt --output artifacts/best_model.onnx --sample-audio data/ESC-50-master/audio/1-100032-A-0.wav
```

Compare ONNX vs PyTorch:

```bash
python -m src.compare_onnx --audio-path data/ESC-50-master/audio/1-100032-A-0.wav --data-root data/ESC-50-master --checkpoint artifacts/best_model.pt --onnx-model artifacts/best_model.onnx --top-k 5
```

Prepare mobile assets (`mobile/assets/best_model.onnx` + `mobile/assets/labels.txt`):

```bash
python scripts/prepare_mobile_assets.py
```
