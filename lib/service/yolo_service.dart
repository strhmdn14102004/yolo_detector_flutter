// ignore_for_file: library_private_types_in_public_api

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
  final double scoreThreshold;
  final double nmsThreshold;

  YoloService({
    required this.labels,
    this.scoreThreshold = 0.35,
    this.nmsThreshold = 0.45,
  });

  Future<void> load({
    String assetPath = 'assets/model/yolo11n_int8.tflite',
  }) async {
    final options = tfl.InterpreterOptions()..threads = 2;

    if (Platform.isAndroid) {
      try {
        _gpu = tfl.GpuDelegateV2();
        options.addDelegate(_gpu!);
      } catch (_) {}
    }

    _interpreter = await tfl.Interpreter.fromAsset(assetPath, options: options);
    _loaded = true;
  }

  bool get isLoaded => _loaded;

  List<_RawDet> _decodeOutput(List outputs) {
    late List<List<double>> dets;

    final out = outputs.first;
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
            dets[i][k] = (row[i] as num).toDouble();
          }
        }
      } else {
        dets = List<List<double>>.from(
          tensor.map<List<double>>(
            (e) => List<double>.from(
              (e as List).map<double>((x) => (x as num).toDouble()),
            ),
          ),
        );
      }
    } else {
      dets = [];
    }

    final List<_RawDet> results = [];
    for (final List<double> d in dets) {
      final double cx = d[0];
      final double cy = d[1];
      final double w = d[2];
      final double h = d[3];

      int bestClass = -1;
      double bestScore = 0.0;

      for (int c = 4; c < d.length; c++) {
        final double s = d[c];
        if (s > bestScore) {
          bestScore = s;
          bestClass = c - 4;
        }
      }

      if (bestClass >= 0 && bestScore >= scoreThreshold) {
        final Rect box = Rect.fromCenter(
          center: Offset(cx, cy),
          width: w,
          height: h,
        );
        results.add(_RawDet(box: box, classIndex: bestClass, score: bestScore));
      }
    }
    return results;
  }

  Future<List<_RawDet>> infer(Float32List nhwcInput) async {
    if (!_loaded) throw StateError("Interpreter belum dimuat");

    final input = nhwcInput.reshape([1, inputSize, inputSize, 3]);

    final outputShapes = _interpreter
        .getOutputTensors()
        .map((e) => e.shape)
        .toList();

    List output;
    if (outputShapes.first.length == 3 && outputShapes.first[1] == 8400) {
      output = List.generate(
        1,
        (_) => List.generate(8400, (_) => List<double>.filled(84, 0)),
      );
    } else {
      output = List.generate(
        1,
        (_) => List.generate(84, (_) => List<double>.filled(8400, 0)),
      );
    }

    _interpreter.run(input, output);
    return _decodeOutput(output);
  }

  Future<List<RectLabelScore>> postprocessToPreview(
    List<_RawDet> raw, {
    required double previewW,
    required double previewH,
  }) async {
    final List<RectLabelScore> all = [];
    final Map<int, List<_RawDet>> byClass = {};
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
        final Rect proj = projectBoxToPreview(
          boxes[idx],
          previewW,
          previewH,
          modelSize: inputSize.toDouble(),
        );
        final String label = (cls >= 0 && cls < labels.length)
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
