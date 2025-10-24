import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';

enum ScanMode { fast, accurate }

class DetectionResult {
  final Rect box;
  final String label;
  final double score;

  const DetectionResult({
    required this.box,
    required this.label,
    required this.score,
  });
}

class ScanState extends Equatable {
  final bool isReady;
  final bool isStreaming;
  final List<CameraDescription> cameras;
  final CameraDescription? activeCamera;
  final List<DetectionResult> detections;
  final String message;

  const ScanState({
    required this.isReady,
    required this.isStreaming,
    required this.cameras,
    required this.activeCamera,
    required this.detections,
    required this.message,
  });

  factory ScanState.initial() => const ScanState(
    isReady: false,
    isStreaming: false,
    cameras: <CameraDescription>[],
    activeCamera: null,
    detections: <DetectionResult>[],
    message: "Memuat…",
  );

  @override
  List<Object?> get props => [
    isReady,
    isStreaming,
    cameras,
    activeCamera,
    detections,
    message,
  ];
}
