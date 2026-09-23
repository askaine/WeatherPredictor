# Weather Vision 🌤️

An offline, on-device weather prediction app for React Native/Expo. Point your phone at the sky (or upload a photo), and a fine-tuned computer vision model classifies the cloud formation and returns a short-term weather forecast — no server, no internet connection required.

## How it works

1. **Model training (Python/PyTorch)** — A ResNet18 CNN, pre-trained on ImageNet, is fine-tuned via transfer learning on a labeled dataset of cloud images (11 classes: Ac, As, Cb, Cc, Ci, Cs, Ct, Cu, Ns, Sc, St). The final classification layer and `layer4` are unfrozen and retrained; everything else stays frozen to keep training lightweight (runs on a 4GB VRAM GPU or CPU).
2. **Export to ONNX** — The trained `.pth` weights are traced and exported to `.onnx`, a portable, language-agnostic model format.
3. **On-device inference (React Native)** — The `.onnx` model ships bundled inside the app. `onnxruntime-react-native` runs inference directly on the phone's CPU/NPU — no API calls, no cloud costs, works in airplane mode.
4. **Forecast mapping** — The predicted cloud class is mapped to a plain-language short-term forecast (e.g. *Cumulonimbus → thunderstorms likely, heavy rain and gusty winds expected soon*).

## Features

- 📷 Live camera capture with digital zoom
- 🖼️ Upload an existing photo from your gallery
- 🧠 Fully offline inference — the model runs locally, nothing is sent to a server
- ☁️ 11-class cloud classification mapped to human-readable forecasts
- 📊 Confidence score shown alongside each prediction

## Tech stack

| Layer | Tools |
|---|---|
| Model training | PyTorch, torchvision (ResNet18, transfer learning) |
| Model export | ONNX |
| Mobile app | React Native, Expo, `onnxruntime-react-native` |
| Camera / media | `expo-camera`, `expo-image-picker`, `expo-image-manipulator` |

## Project structure

```
weather-vision/
├── App.js                     # Main app: camera, upload, inference, UI
├── assets/
│   └── weather_classifier.onnx  # Exported, fine-tuned model
├── training/
│   ├── train.py                # PyTorch fine-tuning script
│   └── export_onnx.py          # .pth → .onnx conversion script
├── app.json
├── eas.json
└── package.json
```

## Running it locally

This app uses a native ML runtime, so it **cannot run in the standard Expo Go app** — it needs a custom development build.

```bash
# 1. Install dependencies
npm install

# 2. Build and install a development client on a connected device/emulator
npx expo run:android
# or
npx expo run:ios

# 3. Start the dev server for fast-refresh development
npx expo start --dev-client
```

## Training your own model

```bash
cd training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or the CPU build
pip install -r requirements.txt

# Organize your dataset as: dataset/<class_name>/*.jpg
python train.py

# Export the trained weights to ONNX
python export_onnx.py
```

Drop the resulting `weather_classifier.onnx` into `assets/` and rebuild the app.

## Model performance

Trained for 15 epochs on a labeled cloud-image dataset, the fine-tuned model reached ~94% training accuracy across 11 cloud classes (vs. a ~9% random-guess baseline).

## Notes & limitations

- The forecast text is a simplified heuristic mapping from cloud type to likely near-term weather — it is not a substitute for a real meteorological forecast.
- Model accuracy depends heavily on dataset quality and balance; results will vary on cloud types or lighting conditions underrepresented in training data.

## License

MIT