import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';

import 'scan_state.dart';

abstract class ScanEvent extends Equatable {
  @override
  List<Object?> get props => [];
}

class ScanInitRequested extends ScanEvent {}

class ScanSwitchCameraRequested extends ScanEvent {}

class ScanStartStreamRequested extends ScanEvent {}

class ScanStopStreamRequested extends ScanEvent {}

class ScanOnCameraImage extends ScanEvent {
  final CameraImage image;
  final int rotationDegrees;
  ScanOnCameraImage(this.image, this.rotationDegrees);

  @override
  List<Object?> get props => [image, rotationDegrees];
}

class ScanModeChanged extends ScanEvent {
  final ScanMode mode;
  ScanModeChanged(this.mode);
  @override
  List<Object?> get props => [mode];
}

class ScanSmartFallbackToggled extends ScanEvent {
  final bool enabled;
  ScanSmartFallbackToggled(this.enabled);
  @override
  List<Object?> get props => [enabled];
}
