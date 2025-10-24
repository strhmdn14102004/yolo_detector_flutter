// ignore_for_file: avoid_types_as_parameter_names, depend_on_referenced_packages

import 'dart:async';
import 'dart:io';

import 'package:bloc/bloc.dart';
import 'package:camera/camera.dart';
import 'package:face_recognition/service/image_utils.dart';
import 'package:face_recognition/service/yolo_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'scan_event.dart';
import 'scan_state.dart';

class ScanBloc extends Bloc<ScanEvent, ScanState> {
  final YoloService yolo;
  CameraController? _controller;

  bool _busy = false;
  DateTime _lastRun = DateTime.fromMillisecondsSinceEpoch(0);

  final int _intervalMs = Platform.isIOS ? 0 : 120;

  DateTime _winStart = DateTime.now();

  ScanBloc({required this.yolo}) : super(ScanState.initial()) {
    on<ScanInitRequested>(_onInit);
    on<ScanSwitchCameraRequested>(_onSwitch);
    on<ScanStartStreamRequested>(_onStartStream);
    on<ScanStopStreamRequested>(_onStopStream);
    on<ScanOnCameraImage>(_onImage, transformer: _droppable());
  }

  EventTransformer<ScanEvent> _droppable<ScanEvent>() =>
      (events, mapper) => events.asyncExpand(mapper);

  Future<void> _onInit(ScanInitRequested event, Emitter<ScanState> emit) async {
    try {
      final cams = await availableCameras();

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
          imageFormatGroup: Platform.isIOS
              ? ImageFormatGroup.bgra8888
              : ImageFormatGroup.yuv420,
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
    final next = state.cameras.firstWhere(
      (c) =>
          c.lensDirection ==
          (toFront ? CameraLensDirection.front : CameraLensDirection.back),
      orElse: () => state.cameras.first,
    );

    await _stopStreamInternal();
    await _controller?.dispose();

    _controller = CameraController(
      next,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: Platform.isIOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420,
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

    await _controller!.startImageStream((image) {
      final rotationDegrees = _controller!.description.sensorOrientation;
      add(ScanOnCameraImage(image, rotationDegrees));
    });

    _winStart = DateTime.now();

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
      final payloadCommon = <String, dynamic>{
        'dst': yolo.inputSize,
        'targetType': yolo.inputTypeStr,
        'inScale': yolo.inputScale,
        'inZeroPoint': yolo.inputZeroPoint,
      };

      Object nhwc;

      if (event.image.planes.length == 1) {
        nhwc = await compute(bgra8888ToNhwcQuantAwareCompute, {
          ...payloadCommon,
          'width': event.image.width,
          'height': event.image.height,
          'bytes': event.image.planes.first.bytes,
        });
      } else {
        nhwc = await compute(yuv420ToNhwcQuantAwareCompute, {
          ...payloadCommon,
          'width': event.image.width,
          'height': event.image.height,
          'y': event.image.planes[0].bytes,
          'u': event.image.planes[1].bytes,
          'v': event.image.planes[2].bytes,
          'yRowStride': event.image.planes[0].bytesPerRow,
          'uvRowStride': event.image.planes[1].bytesPerRow,
          'uvPixelStride': event.image.planes[1].bytesPerPixel ?? 1,
        });
      }

      final raw = await yolo.inferNhwc(nhwc);

      final pv = _controller!.value.previewSize!;
      final pW = pv.height;
      final pH = pv.width;

      var dets = await yolo.postprocessToPreview(
        raw,
        previewW: pW,
        previewH: pH,
      );

      if (state.activeCamera?.lensDirection == CameraLensDirection.front) {
        dets = dets.map((e) {
          final r = e.rect;
          final flipped = Rect.fromLTWH(pW - r.right, r.top, r.width, r.height);
          return RectLabelScore(rect: flipped, label: e.label, score: e.score);
        }).toList();
      }

      final newDets = dets
          .map(
            (e) => DetectionResult(box: e.rect, label: e.label, score: e.score),
          )
          .toList();

      emit(
        ScanState(
          isReady: state.isReady,
          isStreaming: state.isStreaming,
          cameras: state.cameras,
          activeCamera: state.activeCamera,
          detections: newDets,
          message: "Deteksi: ${newDets.length}",
        ),
      );

      final dur = now.difference(_winStart).inMilliseconds;
      if (dur >= 1000) {
        _winStart = now;
      }
    } catch (_) {
    } finally {
      _busy = false;
    }
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
