import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import 'image_utils.dart';

class YoloService {
  late final tfl.Interpreter _interpreter;
  tfl.GpuDelegateV2? _gpu;
  final List<String> labels;

  bool _loaded = false;

  int inputSize = 640;
  late final tfl.TensorType _inType;
  late final double _inScale;
  late final int _inZeroPoint;

  late final tfl.TensorType _outType;
  late final double _outScale;
  late final int _outZeroPoint;

  final double scoreThreshold;
  final double nmsThreshold;

  YoloService({
    required this.labels,
    this.scoreThreshold = 0.35,
    this.nmsThreshold = 0.45,
  });

  Future<void> load({
    String assetPath = 'assets/model/yolov8n_int8.tflite',
  }) async {
    final options = tfl.InterpreterOptions()..threads = 2;

    if (Platform.isAndroid) {
      try {
        _gpu = tfl.GpuDelegateV2();
        options.addDelegate(_gpu!);
      } catch (_) {}
    } else if (Platform.isIOS) {
      try {
        final gpu = tfl.GpuDelegate();
        options.addDelegate(gpu);
      } catch (_) {}
    }

    _interpreter = await tfl.Interpreter.fromAsset(assetPath, options: options);
    _loaded = true;

    final inTensor = _interpreter.getInputTensors().first;
    final outTensor = _interpreter.getOutputTensors().first;

    _inType = inTensor.type;
    _outType = outTensor.type;

    if (inTensor.shape.length == 4) {
      inputSize = inTensor.shape[1];
    }

    final inQuant = inTensor.params;
    _inScale = inQuant.scale;
    _inZeroPoint = inQuant.zeroPoint;

    final outQuant = outTensor.params;
    _outScale = outQuant.scale;
    _outZeroPoint = outQuant.zeroPoint;
  }

  bool get isLoaded => _loaded;

  String get inputTypeStr {
    switch (_inType) {
      case tfl.TensorType.float32:
        return 'float32';
      case tfl.TensorType.int8:
        return 'int8';
      case tfl.TensorType.uint8:
        return 'uint8';
      default:
        return 'other';
    }
  }

  double get inputScale => _inScale;
  int get inputZeroPoint => _inZeroPoint;

  Future<List<_RawDet>> inferNhwc(Object nhwcInput) async {
    if (!_loaded) throw StateError("Interpreter belum dimuat");

    Object input;
    if (_inType == tfl.TensorType.float32) {
      input = (nhwcInput as Float32List).reshape([1, inputSize, inputSize, 3]);
    } else if (_inType == tfl.TensorType.int8) {
      input = (nhwcInput as Int8List).reshape([1, inputSize, inputSize, 3]);
    } else {
      input = (nhwcInput as Uint8List).reshape([1, inputSize, inputSize, 3]);
    }

    final outTensor = _interpreter.getOutputTensors().first;
    final shape = outTensor.shape;
    late Object output;

    if (shape.length == 3 && shape[1] == 8400) {
      if (_outType == tfl.TensorType.float32) {
        output = List.generate(
          1,
          (_) => List.generate(8400, (_) => List<double>.filled(84, 0)),
        );
      } else {
        output = List.generate(
          1,
          (_) => List.generate(8400, (_) => List<int>.filled(84, 0)),
        );
      }
    } else {
      if (_outType == tfl.TensorType.float32) {
        output = List.generate(
          1,
          (_) => List.generate(84, (_) => List<double>.filled(8400, 0)),
        );
      } else {
        output = List.generate(
          1,
          (_) => List.generate(84, (_) => List<int>.filled(8400, 0)),
        );
      }
    }

    _interpreter.run(input, output);
    return _decodeOutput(output);
  }

  double _deq(num v) {
    if (_outType == tfl.TensorType.float32) return v.toDouble();
    return _outScale * (v.toDouble() - _outZeroPoint);
  }

  List<_RawDet> _decodeOutput(Object outputs) {
    late List<List<double>> dets;

    final out = (outputs as List).first;
    if (out is List && out.isNotEmpty && out[0] is List) {
      final dynamic tensor = out;
      final int d0 = (tensor as List).length;
      final int d1 = (tensor[0] as List).length;

      if (d0 == 84) {
        final int N = d1;
        dets = List.generate(N, (i) => List<double>.filled(84, 0));
        for (int k = 0; k < 84; k++) {
          final List row = tensor[k] as List;
          for (int i = 0; i < N; i++) {
            dets[i][k] = _deq(row[i] as num);
          }
        }
      } else {
        dets = List<List<double>>.generate(
          d0,
          (i) => List<double>.generate(
            d1,
            (j) => _deq((tensor[i] as List)[j] as num),
            growable: false,
          ),
          growable: false,
        );
      }
    } else {
      dets = const [];
    }

    final List<_RawDet> results = [];
    for (final d in dets) {
      final cx = d[0];
      final cy = d[1];
      final w = d[2];
      final h = d[3];

      int bestClass = -1;
      double bestScore = 0.0;
      for (int c = 4; c < d.length; c++) {
        final s = d[c];
        if (s > bestScore) {
          bestScore = s;
          bestClass = c - 4;
        }
      }

      if (bestClass >= 0 && bestScore >= scoreThreshold) {
        results.add(
          _RawDet(
            box: Rect.fromCenter(center: Offset(cx, cy), width: w, height: h),
            classIndex: bestClass,
            score: bestScore,
          ),
        );
      }
    }
    return results;
  }

  Future<List<RectLabelScore>> postprocessToPreview(
    List<_RawDet> raw, {
    required double previewW,
    required double previewH,
  }) async {
    final all = <RectLabelScore>[];
    final byClass = <int, List<_RawDet>>{};

    for (final r in raw) {
      byClass.putIfAbsent(r.classIndex, () => []).add(r);
    }

    byClass.forEach((cls, list) {
      final boxes = <Rect>[];
      final scores = <double>[];
      for (final r in list) {
        boxes.add(r.box);
        scores.add(r.score);
      }
      final keep = nonMaxSuppression(boxes, scores, nmsThreshold);

      for (final idx in keep) {
        final proj = projectBoxToPreview(
          boxes[idx],
          previewW,
          previewH,
          modelSize: inputSize.toDouble(),
        );
        final label = (cls >= 0 && cls < labels.length)
            ? labels[cls]
            : "unknown";
        all.add(RectLabelScore(rect: proj, label: label, score: scores[idx]));
      }
    });

    return all;
  }
}

class _RawDet {
  final Rect box;
  final int classIndex;
  final double score;
  _RawDet({required this.box, required this.classIndex, required this.score});
}

class RectLabelScore {
  final Rect rect;
  final String label;
  final double score;
  RectLabelScore({
    required this.rect,
    required this.label,
    required this.score,
  });
}
