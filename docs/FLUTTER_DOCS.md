# Use the TFLite model in Flutter

This guide assumes the Baybayin model is already trained. The model recognizes **one cropped text line per inference**; it does not locate text in a full camera image.

## 1. Get the model and its vocabulary

Place these matching files in your Flutter project:

```text
assets/models/baybayin_mobile.tflite
assets/models/baybayin_mobile.vocab.json
```

The vocabulary file maps the model's output class numbers to Unicode characters. It must come from the same checkpoint used to export the model. The project exporter writes both files:

```bash
python export_litert.py \
  --checkpoint checkpoints/best_model.pt \
  --output checkpoints/baybayin_mobile.tflite
```

Copy the resulting `.tflite` and `.vocab.json` files into the Flutter assets. Do not build the mapping from a separately sorted character list.

## 2. Add Flutter dependencies and assets

From the Flutter project root:

```bash
flutter pub add tflite_flutter
```

Register both assets in `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/baybayin_mobile.tflite
    - assets/models/baybayin_mobile.vocab.json
```

Then run `flutter pub get`. See the [`tflite_flutter` package guide](https://pub.dev/packages/tflite_flutter) for platform setup and runtime options.

## 3. Load the interpreter and vocabulary

```dart
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

late final Interpreter interpreter;
late final List<String> characters;

Future<void> loadOcr() async {
  interpreter = await Interpreter.fromAsset(
    'assets/models/baybayin_mobile.tflite',
  );

  final jsonText = await rootBundle.loadString(
    'assets/models/baybayin_mobile.vocab.json',
  );
  final vocabulary = jsonDecode(jsonText) as Map<String, dynamic>;
  characters = List<String>.from(vocabulary['characters'] as List);

  // Confirm the artifact matches this guide/model export.
  assert(interpreter.getInputTensor(0).shape.toString() == '[1, 1, 48, 320]');
  assert(interpreter.getOutputTensor(0).shape[0] == 1);
  assert(interpreter.getOutputTensor(0).shape[2] == characters.length + 1);
}
```

The expected input is float32 `[1, 1, 48, 320]` (NCHW). Output is `[1, T, C]`, where `C` is `characters.length + 1`; output class `0` is the CTC blank. Read tensor shapes from the loaded model as above in case you use a different export.

## 4. Crop and preprocess a text line

Before inference, detect or let the user select a single line from the photo/camera frame. Convert it to grayscale, preserve its aspect ratio, resize to fit within **48 px high × 320 px wide**, and place it on a white `48 × 320` canvas. Center it vertically and align it to the left. Do not stretch the line to fill the canvas.

Normalize each grayscale pixel `p` (0–255) to a float with:

```text
x = (p / 255.0 - 0.5) / 0.5
```

This produces values from `-1` (black) to `+1` (white). Store them in NCHW order: batch, channel, row, column. Match the training preprocessing: area interpolation when shrinking, cubic interpolation when enlarging. Camera images commonly use RGB; use the corresponding grayscale conversion rather than treating RGB bytes as BGR.

Calculate the valid CTC time steps from the resized line width **before padding**:

```text
validSteps = (resizedWidth + 1) ~/ 2
```

The model output has 160 time steps for a 320 px input, but only the first `validSteps` correspond to the unpadded line.

## 5. Run inference and decode the CTC output

Build a four-dimensional input with shape `[1][1][48][320]`, then run the interpreter. Allocate the output with the model's reported shape `[1][T][C]`:

```dart
final input = /* nested float lists [1][1][48][320] from preprocessing */;
final shape = interpreter.getOutputTensor(0).shape;
final output = List.generate(
  shape[0],
  (_) => List.generate(
    shape[1],
    (_) => List<double>.filled(shape[2], 0),
  ),
);

interpreter.run(input, output);
final text = decodeGreedy(output[0], validSteps, characters);
```

For each valid timestep, select the index with the largest output score. Collapse consecutive repeated indices, discard index `0`, and map each remaining index `i` to `characters[i - 1]`:

```dart
String decodeGreedy(
  List<List<double>> steps,
  int validSteps,
  List<String> characters,
) {
  final result = StringBuffer();
  var previous = -1;

  for (final scores in steps.take(validSteps)) {
    var best = 0;
    for (var i = 1; i < scores.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }
    if (best != 0 && best != previous) {
      result.write(characters[best - 1]);
    }
    previous = best;
  }
  return result.toString();
}
```

This mapping includes the Baybayin characters and the Philippine punctuation marks `᜵` and `᜶` when they are present in the exported vocabulary. For camera use, run preprocessing and inference off the UI isolate (the package provides `IsolateInterpreter`) so the preview and controls remain responsive.

## 6. Release resources and verify on device

Keep the interpreter loaded while the OCR screen is active, and close it when that screen is disposed:

```dart
interpreter.close();
```

Try several cropped lines from the target phone camera. Compare the results with desktop predictions and check both punctuation marks, line cropping, and light/dark images. Test inference speed on the actual Android and iOS devices you plan to support.
