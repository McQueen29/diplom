# Flutter Sound Classifier Prototype

Prototype mobile app that:
- records audio from microphone,
- runs local ONNX inference on device,
- shows predicted sound classes with probabilities.

Model format used: **ONNX** (`assets/best_model.onnx`).

## Features

- Start / Stop recording button
- 5-second mono WAV recording at 22050 Hz
- Local preprocessing (log-mel style features)
- Local ONNX inference (no server)
- Top class probabilities on screen

## Project structure

- `lib/main.dart` - app entrypoint
- `lib/sound_classifier_page.dart` - UI and flow
- `lib/audio_recorder_service.dart` - microphone recording
- `lib/audio_preprocessor.dart` - wav decoding + feature extraction
- `lib/onnx_sound_classifier.dart` - ONNX runtime inference
- `assets/best_model.onnx` - trained model
- `assets/labels.txt` - class labels

## Run

1. Install Flutter SDK and add `flutter` to PATH.
2. In this directory run:

```bash
flutter pub get
flutter run
```

## Notes

- This is a prototype focused on end-to-end local inference.
- For production, move FFT/mel preprocessing to native optimized code for speed.
