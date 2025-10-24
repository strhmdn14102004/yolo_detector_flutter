import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'scan_bloc.dart';
import 'scan_event.dart';
import 'scan_state.dart';

class ScanPage extends StatefulWidget {
  const ScanPage({super.key});
  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    context.read<ScanBloc>().add(ScanInitRequested());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final bloc = context.read<ScanBloc>();
    if (state == AppLifecycleState.paused) {
      bloc.add(ScanStopStreamRequested());
    }
    super.didChangeAppLifecycleState(state);
  }

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<ScanBloc, ScanState>(
      listenWhen: (p, c) => p.message != c.message,
      listener: (context, state) {
        if (state.message.isNotEmpty) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(state.message),
              duration: const Duration(milliseconds: 600),
            ),
          );
        }
      },
      buildWhen: (p, c) =>
          p.isReady != c.isReady ||
          p.isStreaming != c.isStreaming ||
          p.activeCamera != c.activeCamera ||
          !listEquals(p.detections, c.detections),
      builder: (context, state) {
        final bloc = context.read<ScanBloc>();
        final CameraController? ctrl = bloc.controller;

        return Scaffold(
          appBar: AppBar(
            title: const Text("YOLO11 TFLite Scanner"),
            actions: [
              IconButton(
                icon: const Icon(Icons.cameraswitch),
                onPressed: state.isReady
                    ? () => bloc.add(ScanSwitchCameraRequested())
                    : null,
              ),
            ],
          ),
          body: !state.isReady || ctrl == null
              ? const Center(child: CircularProgressIndicator())
              : _PreviewWithOverlay(
                  controller: ctrl,
                  detections: state.detections,
                ),
          bottomNavigationBar: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: state.isReady && !state.isStreaming
                        ? () => context.read<ScanBloc>().add(
                            ScanStartStreamRequested(),
                          )
                        : null,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text("Start"),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: state.isReady && state.isStreaming
                        ? () => context.read<ScanBloc>().add(
                            ScanStopStreamRequested(),
                          )
                        : null,
                    icon: const Icon(Icons.stop),
                    label: const Text("Stop"),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _PreviewWithOverlay extends StatelessWidget {
  final CameraController controller;
  final List<DetectionResult> detections;
  const _PreviewWithOverlay({
    required this.controller,
    required this.detections,
  });

  @override
  Widget build(BuildContext context) {
    final pv = controller.value.previewSize!;
    final srcW = pv.height;
    final srcH = pv.width;

    return LayoutBuilder(
      builder: (context, constraints) {
        final dstW = constraints.maxWidth;
        final dstH = constraints.maxHeight;

        final scale = math.max(dstW / srcW, dstH / srcH);
        final paintW = srcW * scale;
        final paintH = srcH * scale;

        return Center(
          child: SizedBox(
            width: paintW,
            height: paintH,
            child: CameraPreview(
              controller,
              child: CustomPaint(
                painter: _DetectionsPainter(
                  detections,
                  logicalW: srcW,
                  logicalH: srcH,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _DetectionsPainter extends CustomPainter {
  final List<DetectionResult> dets;
  final double logicalW;
  final double logicalH;
  _DetectionsPainter(
    this.dets, {
    required this.logicalW,
    required this.logicalH,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / logicalW;
    final sy = size.height / logicalH;

    final paintOval = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final paintBg = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xCC000000);

    const textStyle = TextStyle(
      color: Colors.white,
      fontSize: 13,
      fontWeight: FontWeight.w600,
    );

    for (final d in dets) {
      paintOval.color = HSVColor.fromAHSV(
        1,
        (d.label.hashCode % 360).toDouble(),
        0.85,
        0.95,
      ).toColor();

      final r = Rect.fromLTWH(
        d.box.left * sx,
        d.box.top * sy,
        d.box.width * sx,
        d.box.height * sy,
      );

      canvas.drawOval(r, paintOval);

      final caption = "${d.label} ${(d.score * 100).toStringAsFixed(1)}%";
      final tp = TextPainter(
        text: TextSpan(text: caption, style: textStyle),
        textDirection: TextDirection.ltr,
        maxLines: 1,
        ellipsis: '…',
      )..layout(maxWidth: size.width * 0.9);

      final tagLeft = (r.center.dx - tp.width / 2).clamp(
        0.0,
        size.width - tp.width - 8,
      );
      double tagTop = r.top - tp.height - 8;
      if (tagTop < 0) tagTop = r.top + 2;

      final tag = Rect.fromLTWH(tagLeft, tagTop, tp.width + 8, tp.height + 4);
      canvas.drawRRect(
        RRect.fromRectAndRadius(tag, const Radius.circular(6)),
        paintBg,
      );
      tp.paint(canvas, Offset(tag.left + 4, tag.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionsPainter old) =>
      old.dets != dets || old.logicalW != logicalW || old.logicalH != logicalH;
}
