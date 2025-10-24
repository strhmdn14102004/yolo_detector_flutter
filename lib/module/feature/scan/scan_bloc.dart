// ignore_for_file: depend_on_referenced_packages, avoid_types_as_parameter_names
import 'dart:async';
import 'dart:typed_data';
import 'dart:ui';

import 'package:bloc/bloc.dart';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

import 'package:face_recognition/service/image_utils.dart';
import 'package:face_recognition/service/yolo_service.dart';

import 'scan_event.dart';
import 'scan_state.dart';

class ScanBloc extends Bloc<ScanEvent, ScanState> {
  final YoloService yolo;
  CameraController? _controller;

  bool _busy = false;
  DateTime _lastRun = DateTime.fromMillisecondsSinceEpoch(0);

  // Dinamis bergantung mode
  int _intervalMs = 150; // FAST default ~6-7 FPS target

  // FPS tracking (EMA)
  DateTime? _lastFrameDoneAt;
  double _emaFps = 0.0;

  // Auto fallback / return timers
  DateTime? _lowFpsSince;
  DateTime? _highFpsSince;
  int _consecutiveErrors = 0;

  ScanBloc({required this.yolo}) : super(ScanState.initial()) {
    on<ScanInitRequested>(_onInit);
    on<ScanSwitchCameraRequested>(_onSwitch);
    on<ScanStartStreamRequested>(_onStartStream);
    on<ScanStopStreamRequested>(_onStopStream);
    on<ScanOnCameraImage>(_onImage, transformer: _droppable());

    on<ScanModeChanged>(_onModeChanged);
    on<ScanSmartFallbackToggled>(_onSmartToggled);
  }

  EventTransformer<ScanEvent> _droppable<ScanEvent>() {
    return (events, mapper) => events.asyncExpand(mapper);
  }

  // ---------------- INIT / CAMERA ----------------

  Future<void> _onInit(ScanInitRequested event, Emitter<ScanState> emit) async {
    try {
      final List<CameraDescription> cams = await availableCameras();

      CameraDescription? active;
      if (cams.any((c) => c.lensDirection == CameraLensDirection.back)) {
        active = cams.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
        );
      } else {
        active = cams.isNotEmpty ? cams.first : null;
      }

      if (!yolo.isLoaded) {
        await yolo.load();
      }

      if (active != null) {
        _controller = CameraController(
          active,
          ResolutionPreset.low, // jaga ringan
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.yuv420,
        );
        await _controller!.initialize();
      }

      // Set interval sesuai mode awal (FAST)
      _applyModeInternal(ScanMode.fast);

      emit(
        ScanState(
          isReady: _controller?.value.isInitialized ?? false,
          isStreaming: false,
          cameras: cams,
          activeCamera: active,
          detections: const [],
          message: cams.isEmpty ? "Kamera tidak ditemukan" : "Siap",
          mode: ScanMode.fast,
          smartFallback: true,
          fps: 0.0,
        ),
      );
    } catch (e) {
      emit(
        ScanState(
          isReady: false,
          isStreaming: false,
          cameras: const [],
          activeCamera: null,
          detections: const [],
          message: "Gagal init: $e",
          mode: ScanMode.fast,
          smartFallback: true,
          fps: 0.0,
        ),
      );
    }
  }

  Future<void> _onSwitch(
    ScanSwitchCameraRequested event,
    Emitter<ScanState> emit,
  ) async {
    if (state.cameras.isEmpty) return;

    final bool toFront =
        state.activeCamera?.lensDirection != CameraLensDirection.front;
    final CameraDescription next = state.cameras.firstWhere(
      (c) =>
          c.lensDirection ==
          (toFront ? CameraLensDirection.front : CameraLensDirection.back),
      orElse: () => state.cameras.first,
    );

    await _stopStreamInternal();
    await _controller?.dispose();

    _controller = CameraController(
      next,
      ResolutionPreset.low, // tetap ringan
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await _controller!.initialize();

    emit(
      ScanState(
        isReady: _controller!.value.isInitialized,
        isStreaming: false,
        cameras: state.cameras,
        activeCamera: next,
        detections: const [],
        message: "Kamera: ${next.name}",
        mode: state.mode,
        smartFallback: state.smartFallback,
        fps: 0.0,
      ),
    );
  }

  // ---------------- STREAM ----------------

  Future<void> _onStartStream(
    ScanStartStreamRequested event,
    Emitter<ScanState> emit,
  ) async {
    if (!state.isReady || _controller == null) return;

    // reset trackers
    _lastFrameDoneAt = null;
    _emaFps = 0.0;
    _lowFpsSince = null;
    _highFpsSince = null;
    _consecutiveErrors = 0;

    await _controller!.startImageStream((CameraImage image) {
      final int rotationDegrees = _controller!.description.sensorOrientation;
      add(ScanOnCameraImage(image, rotationDegrees));
    });

    emit(
      ScanState(
        isReady: state.isReady,
        isStreaming: true,
        cameras: state.cameras,
        activeCamera: state.activeCamera,
        detections: const [],
        message: "Streaming…",
        mode: state.mode,
        smartFallback: state.smartFallback,
        fps: 0.0,
      ),
    );
  }

  Future<void> _onStopStream(
    ScanStopStreamRequested event,
    Emitter<ScanState> emit,
  ) async {
    await _stopStreamInternal();
    emit(
      ScanState(
        isReady: state.isReady,
        isStreaming: false,
        cameras: state.cameras,
        activeCamera: state.activeCamera,
        detections: const [],
        message: "Berhenti",
        mode: state.mode,
        smartFallback: state.smartFallback,
        fps: 0.0,
      ),
    );
  }

  // ---------------- MODE ----------------

  Future<void> _onModeChanged(
    ScanModeChanged event,
    Emitter<ScanState> emit,
  ) async {
    _applyModeInternal(event.mode);

    emit(
      ScanState(
        isReady: state.isReady,
        isStreaming: state.isStreaming,
        cameras: state.cameras,
        activeCamera: state.activeCamera,
        detections: state.detections,
        message: event.mode == ScanMode.fast ? "Mode: FAST" : "Mode: AKURAT",
        mode: event.mode,
        smartFallback: state.smartFallback,
        fps: state.fps,
      ),
    );
  }

  Future<void> _onSmartToggled(
    ScanSmartFallbackToggled event,
    Emitter<ScanState> emit,
  ) async {
    emit(
      ScanState(
        isReady: state.isReady,
        isStreaming: state.isStreaming,
        cameras: state.cameras,
        activeCamera: state.activeCamera,
        detections: state.detections,
        message: event.enabled ? "Smart fallback ON" : "Smart fallback OFF",
        mode: state.mode,
        smartFallback: event.enabled,
        fps: state.fps,
      ),
    );
  }

  void _applyModeInternal(ScanMode mode) {
    // Hanya ubah interval inferensi. (Kamera tetap di resolusi rendah untuk stabilitas.)
    if (mode == ScanMode.fast) {
      _intervalMs = 150; // target ~6-7 FPS
    } else {
      _intervalMs = 300; // target ~3-4 FPS (longgar, akurat/stabil)
    }
  }

  // ---------------- PIPELINE ----------------

  Future<void> _onImage(
    ScanOnCameraImage event,
    Emitter<ScanState> emit,
  ) async {
    final now = DateTime.now();
    if (now.difference(_lastRun).inMilliseconds < _intervalMs) return;
    if (_busy) return;
    _busy = true;
    _lastRun = now;

    if (_controller == null || !_controller!.value.isInitialized) {
      _busy = false;
      return;
    }
    if (!state.isStreaming) {
      _busy = false;
      return;
    }

    try {
      final Map<String, dynamic> payload = {
        'width': event.image.width,
        'height': event.image.height,
        'dst': yolo.inputSize,
        'y': event.image.planes[0].bytes,
        'u': event.image.planes[1].bytes,
        'v': event.image.planes[2].bytes,
        'yRowStride': event.image.planes[0].bytesPerRow,
        'uvRowStride': event.image.planes[1].bytesPerRow,
        'uvPixelStride': event.image.planes[1].bytesPerPixel ?? 1,
        'targetType': yolo.inputTypeStr,    // 'float32' | 'int8' | 'uint8'
        'inScale': yolo.inputScale,
        'inZeroPoint': yolo.inputZeroPoint,
      };

      final Object nhwc = await compute(
        yuv420ToNhwcQuantAwareCompute,
        payload,
      );

      final raw = await yolo.inferNhwc(nhwc);

      final previewSize = _controller!.value.previewSize!;
      final double pW = previewSize.height;
      final double pH = previewSize.width;

      var dets = await yolo.postprocessToPreview(
        raw,
        previewW: pW,
        previewH: pH,
      );

      // Mirror front camera
      if (state.activeCamera?.lensDirection == CameraLensDirection.front) {
        dets = dets
            .map((e) {
              final Rect r = e.rect;
              final Rect flipped =
                  Rect.fromLTWH(pW - r.right, r.top, r.width, r.height);
              return RectLabelScore(
                  rect: flipped, label: e.label, score: e.score);
            })
            .toList();
      }

      final newDets = dets
          .map(
            (e) => DetectionResult(
              box: e.rect,
              label: e.label,
              score: e.score,
            ),
          )
          .toList();

      // ------- FPS update -------
      final end = DateTime.now();
      if (_lastFrameDoneAt != null) {
        final dt = end.difference(_lastFrameDoneAt!).inMilliseconds / 1000.0;
        if (dt > 0) {
          final instFps = 1.0 / dt;
          // EMA lembut
          _emaFps = _emaFps == 0.0 ? instFps : (_emaFps * 0.8 + instFps * 0.2);
        }
      }
      _lastFrameDoneAt = end;

      // Emit hanya jika berubah banyak
      final prev = state.detections;
      final bool changed =
          newDets.length != prev.length ||
          (newDets.isNotEmpty &&
              (newDets.first.label != (prev.isNotEmpty ? prev.first.label : null)));

      if (changed || (end.millisecondsSinceEpoch % 8 == 0)) {
        emit(
          ScanState(
            isReady: state.isReady,
            isStreaming: state.isStreaming,
            cameras: state.cameras,
            activeCamera: state.activeCamera,
            detections: newDets,
            message: changed ? "Deteksi: ${newDets.length}" : state.message,
            mode: state.mode,
            smartFallback: state.smartFallback,
            fps: double.parse(_emaFps.toStringAsFixed(2)),
          ),
        );
      }

      // ------- Smart fallback / return -------
      if (state.smartFallback) {
        if (state.mode == ScanMode.fast) {
          // jika FPS < 3 selama ≥1s atau error beruntun (di catch) => fallback
          if (_emaFps > 0 && _emaFps < 3.0) {
            _lowFpsSince ??= end;
            if (end.difference(_lowFpsSince!).inMilliseconds >= 1000) {
              _switchModeAuto(ScanMode.accurate, emit, reason: "FPS rendah");
            }
          } else {
            _lowFpsSince = null;
          }
        } else {
          // accurate -> kembali ke fast jika FPS > 6 selama ≥2s
          if (_emaFps > 6.0) {
            _highFpsSince ??= end;
            if (end.difference(_highFpsSince!).inMilliseconds >= 2000) {
              _switchModeAuto(ScanMode.fast, emit, reason: "FPS pulih");
            }
          } else {
            _highFpsSince = null;
          }
        }
      }

      // Berhasil → reset error counter
      _consecutiveErrors = 0;
    } catch (_) {
      // error satu frame: hitung untuk trigger fallback
      _consecutiveErrors++;
      if (state.smartFallback &&
          state.mode == ScanMode.fast &&
          _consecutiveErrors >= 2) {
        _switchModeAuto(ScanMode.accurate, emit, reason: "Error beruntun");
      }
    } finally {
      _busy = false;
    }
  }

  void _switchModeAuto(ScanMode to, Emitter<ScanState> emit, {required String reason}) {
    if (state.mode == to) return;
    _applyModeInternal(to);
    _lowFpsSince = null;
    _highFpsSince = null;
    _consecutiveErrors = 0;

    emit(
      ScanState(
        isReady: state.isReady,
        isStreaming: state.isStreaming,
        cameras: state.cameras,
        activeCamera: state.activeCamera,
        detections: state.detections,
        message:
            "Auto switch → ${to == ScanMode.fast ? "FAST" : "AKURAT"} ($reason)",
        mode: to,
        smartFallback: state.smartFallback,
        fps: state.fps,
      ),
    );
  }

  Future<void> _stopStreamInternal() async {
    try {
      if (_controller?.value.isStreamingImages == true) {
        await _controller?.stopImageStream();
      }
    } catch (_) {}
  }

  @override
  Future<void> close() async {
    await _stopStreamInternal();
    await _controller?.dispose();
    return super.close();
  }

  CameraController? get controller => _controller;
}
