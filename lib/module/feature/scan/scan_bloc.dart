// ignore_for_file: depend_on_referenced_packages, avoid_types_as_parameter_names

import 'dart:async';

import 'package:bloc/bloc.dart';
import 'package:camera/camera.dart';
import 'package:face_recognition/service/image_utils.dart';
import 'package:face_recognition/service/yolo_service.dart';
import 'package:flutter/foundation.dart';

import 'scan_event.dart';
import 'scan_state.dart';

class ScanBloc extends Bloc<ScanEvent, ScanState> {
  final YoloService yolo;
  CameraController? _controller;

  bool _busy = false;

  DateTime _lastRun = DateTime.fromMillisecondsSinceEpoch(0);
  final int _intervalMs = 700;

  ScanBloc({required this.yolo}) : super(ScanState.initial()) {
    on<ScanInitRequested>(_onInit);
    on<ScanSwitchCameraRequested>(_onSwitch);
    on<ScanStartStreamRequested>(_onStartStream);
    on<ScanStopStreamRequested>(_onStopStream);
    on<ScanOnCameraImage>(_onImage, transformer: _droppable());
  }

  EventTransformer<ScanEvent> _droppable<ScanEvent>() {
    return (events, mapper) => events.asyncExpand(mapper);
  }

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

          ResolutionPreset.low,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.yuv420,
        );
        await _controller!.initialize();
      }

      emit(
        ScanState(
          isReady: _controller?.value.isInitialized ?? false,
          isStreaming: false,
          cameras: cams,
          activeCamera: active,
          detections: const [],
          message: cams.isEmpty ? "Kamera tidak ditemukan" : "Siap",
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
      ResolutionPreset.high,
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
      ),
    );
  }

  Future<void> _onStartStream(
    ScanStartStreamRequested event,
    Emitter<ScanState> emit,
  ) async {
    if (!state.isReady || _controller == null) return;

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
      ),
    );
  }

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
      };

      final Float32List nhwc = await compute(
        yuv420ToRgbNormalizedResizeCompute,
        payload,
      );

      final raw = await yolo.infer(nhwc);

      final previewSize = _controller!.value.previewSize!;
      final dets = await yolo.postprocessToPreview(
        raw,
        previewW: previewSize.height,
        previewH: previewSize.width,
      );

      final prev = state.detections;
      final bool changed =
          dets.length != prev.length ||
          (dets.isNotEmpty &&
              (dets.first.label !=
                  (prev.isNotEmpty ? prev.first.label : null)));

      if (changed) {
        emit(
          ScanState(
            isReady: state.isReady,
            isStreaming: state.isStreaming,
            cameras: state.cameras,
            activeCamera: state.activeCamera,
            detections: dets
                .map(
                  (e) => DetectionResult(
                    box: e.rect,
                    label: e.label,
                    score: e.score,
                  ),
                )
                .toList(),
            message: "Deteksi: ${dets.length}",
          ),
        );
      }
    } catch (_) {
    } finally {
      _busy = false;
    }
  }

  Future<void> _stopStreamInternal() async {
    try {
      await _controller?.stopImageStream();
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
