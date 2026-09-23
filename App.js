import { Buffer } from 'buffer';
global.Buffer = global.Buffer || Buffer;
import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { Asset } from 'expo-asset';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import jpeg from 'jpeg-js';

const WEATHER_MAP = {
  Ac: { name: "Altocumulus", forecast: "Fair weather likely, though isolated afternoon showers are possible.", icon: "⛅" },
  As: { name: "Altostratus", forecast: "Overcast skies, with light rain or drizzle possible within a few hours.", icon: "🌥️" },
  Cb: { name: "Cumulonimbus", forecast: "Thunderstorms likely. Expect heavy rain, gusty winds, and possible lightning soon.", icon: "⛈️" },
  Cc: { name: "Cirrocumulus", forecast: "Generally fair, but a weather change may be approaching in the next day.", icon: "🌤️" },
  Ci: { name: "Cirrus", forecast: "Fair weather for now, but these often precede a system arriving in 24–48 hours.", icon: "🌤️" },
  Cs: { name: "Cirrostratus", forecast: "Rain or snow possible within 12–24 hours as a system approaches.", icon: "🌥️" },
  Ct: { name: "Contrail", forecast: "No significant weather indicated — this is aircraft exhaust, not a natural cloud.", icon: "✈️" },
  Cu: { name: "Cumulus", forecast: "Fair, pleasant weather expected to continue.", icon: "🌤️" },
  Ns: { name: "Nimbostratus", forecast: "Steady, prolonged rain or snow likely.", icon: "🌧️" },
  Sc: { name: "Stratocumulus", forecast: "Mostly cloudy, low chance of light rain.", icon: "☁️" },
  St: { name: "Stratus", forecast: "Overcast and possibly drizzly, low visibility conditions likely.", icon: "🌫️" },
};

const CLASS_NAMES = Object.keys(WEATHER_MAP);

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [session, setSession] = useState(null);
  const [prediction, setPrediction] = useState(null); 
  const [isProcessing, setIsProcessing] = useState(false);
  const [zoom, setZoom] = useState(0);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      if (!permission?.granted) {
        await requestPermission();
      }

      try {
        const modelAsset = Asset.fromModule(require('./assets/weather_classifier.onnx'));
        await modelAsset.downloadAsync();
        const mySession = await InferenceSession.create(modelAsset.localUri);
        setSession(mySession);
        console.log("ONNX Model loaded successfully");
      } catch (e) {
        console.error("Failed to load ONNX model", e);
      }
    })();
  }, []);

  const processImageToTensor = async (uri) => {
    const manipulated = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: 224, height: 224 } }],
      { base64: true, format: ImageManipulator.SaveFormat.JPEG }
    );

    const rawImageData = jpeg.decode(Buffer.from(manipulated.base64, 'base64'), { useTArray: true });

    const float32Data = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
      float32Data[i] = ((rawImageData.data[i * 4] / 255.0) - mean[0]) / std[0];
      float32Data[i + 224 * 224] = ((rawImageData.data[i * 4 + 1] / 255.0) - mean[1]) / std[1];
      float32Data[i + 2 * 224 * 224] = ((rawImageData.data[i * 4 + 2] / 255.0) - mean[2]) / std[2];
    }

    return new Tensor('float32', float32Data, [1, 3, 224, 224]);
  };

  const runPrediction = async (uri) => {
    if (!session) return;

    setIsProcessing(true);
    setPrediction(null);

    try {
      const tensor = await processImageToTensor(uri);
      const results = await session.run({ input: tensor });
      const outputArray = Array.from(results.output.data);
      const probabilities = softmax(outputArray);

      let maxIndex = 0;
      let maxProb = probabilities[0];
      for (let i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      const code = CLASS_NAMES[maxIndex];
      const info = WEATHER_MAP[code];

      setPrediction({
        code,
        name: info.name,
        forecast: info.forecast,
        icon: info.icon,
        confidence: (maxProb * 100).toFixed(1),
      });
    } catch (error) {
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  };

  const takePictureAndPredict = async () => {
    if (!cameraRef.current || !session) return;
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.5 });
    await runPrediction(photo.uri);
  };

  const pickFromGallery = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8,
    });

    if (!result.canceled && result.assets?.length > 0) {
      await runPrediction(result.assets[0].uri);
    }
  };

  const adjustZoom = (delta) => {
    setZoom(prev => Math.min(1, Math.max(0, prev + delta)));
  };

  if (!permission) return <View />;
  if (!permission.granted) return <Text>No access to camera</Text>;

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing="back" zoom={zoom} ref={cameraRef}>
        <View style={styles.overlay}>
          {prediction && (
            <View style={styles.resultBox}>
              <Text style={styles.resultIcon}>{prediction.icon}</Text>
              <Text style={styles.resultTitle}>{prediction.name}</Text>
              <Text style={styles.resultConfidence}>{prediction.confidence}% confidence</Text>
              <Text style={styles.resultForecast}>{prediction.forecast}</Text>
            </View>
          )}

          <View style={styles.zoomRow}>
            <TouchableOpacity style={styles.zoomButton} onPress={() => adjustZoom(-0.1)}>
              <Text style={styles.zoomButtonText}>−</Text>
            </TouchableOpacity>
            <Text style={styles.zoomLabel}>{Math.round(zoom * 100)}%</Text>
            <TouchableOpacity style={styles.zoomButton} onPress={() => adjustZoom(0.1)}>
              <Text style={styles.zoomButtonText}>+</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={pickFromGallery}
              disabled={isProcessing || !session}
            >
              <Text style={styles.secondaryButtonText}>Upload</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.captureButton}
              onPress={takePictureAndPredict}
              disabled={isProcessing || !session}
            >
              {isProcessing ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={styles.buttonText}>Predict</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 30,
  },
  resultBox: {
    backgroundColor: 'rgba(0,0,0,0.75)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    width: '85%',
    alignItems: 'center',
  },
  resultIcon: { fontSize: 36, marginBottom: 4 },
  resultTitle: { color: 'white', fontSize: 22, fontWeight: 'bold' },
  resultConfidence: { color: '#9fd3ff', fontSize: 14, marginBottom: 8 },
  resultForecast: { color: 'white', fontSize: 15, textAlign: 'center' },
  zoomRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 4,
    marginBottom: 16,
  },
  zoomButton: { paddingHorizontal: 12, paddingVertical: 4 },
  zoomButtonText: { color: 'white', fontSize: 20, fontWeight: 'bold' },
  zoomLabel: { color: 'white', fontSize: 14, width: 44, textAlign: 'center' },
  buttonRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  secondaryButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 50,
  },
  secondaryButtonText: { color: 'white', fontSize: 16, fontWeight: '600' },
  captureButton: {
    backgroundColor: '#2196F3',
    padding: 20,
    borderRadius: 50,
    width: 140,
    alignItems: 'center',
  },
  buttonText: { color: 'white', fontSize: 18, fontWeight: 'bold' },
});