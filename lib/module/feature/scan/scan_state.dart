import 'dart:ui';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';

enum ScanMode { fast, accurate }

class DetectionResult {
  final Rect box; // koordinat pada ruang preview (px)
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

  // // Tambahan
  // final ScanMode mode;
  // final bool smartFallback; // FAST akan auto-fallback/return
  // final double fps; // moving average FPS

  const ScanState({
    required this.isReady,
    required this.isStreaming,
    required this.cameras,
    required this.activeCamera,
    required this.detections,
    required this.message,
    // required this.mode,
    // required this.smartFallback,
    // required this.fps,
  });

  factory ScanState.initial() => const ScanState(
        isReady: false,
        isStreaming: false,
        cameras: <CameraDescription>[],
        activeCamera: null,
        detections: <DetectionResult>[],
        message: "Memuat…",
        // mode: ScanMode.fast,
        // smartFallback: true,
        // fps: 0.0,
      );

  @override
  List<Object?> get props => [
        isReady,
        isStreaming,
        cameras,
        activeCamera,
        detections,
        message,
        // mode,
        // smartFallback,
        // fps,
      ];
}
