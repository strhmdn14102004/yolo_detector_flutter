import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';

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
