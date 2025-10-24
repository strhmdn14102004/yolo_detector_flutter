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
              duration: const Duration(milliseconds: 700),
            ),
          );
        }
      },
      buildWhen: (p, c) =>
          p.isReady != c.isReady ||
          p.isStreaming != c.isStreaming ||
          p.activeCamera != c.activeCamera ||
          p.mode != c.mode ||
          p.smartFallback != c.smartFallback ||
          p.fps != c.fps ||
          !listEquals(p.detections, c.detections),
      builder: (context, state) {
        final bloc = context.read<ScanBloc>();
        final CameraController? ctrl = bloc.controller;

        return Scaffold(
          appBar: AppBar(
            title: const Text("YOLO11 TFLite Scanner"),
            actions: [
              Center(
                child: Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: Text(
                    state.fps > 0 ? "FPS ${state.fps.toStringAsFixed(1)}" : "FPS -",
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                ),
              ),
              IconButton(
                icon: const Icon(Icons.cameraswitch),
                onPressed: state.isReady
                    ? () => bloc.add(ScanSwitchCameraRequested())
                    : null,
                tooltip: "Switch Camera",
              ),
            ],
          ),
          body: !state.isReady || ctrl == null
              ? const Center(child: CircularProgressIndicator())
              : Column(
                  children: [
                    // Toolbar mode
                    Padding(
                      padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
                      child: Row(
                        children: [
                          SegmentedButton<ScanMode>(
                            segments: const [
                              ButtonSegment(
                                  value: ScanMode.fast, label: Text("FAST")),
                              ButtonSegment(
                                  value: ScanMode.accurate,
                                  label: Text("AKURAT")),
                            ],
                            selected: {state.mode},
                            onSelectionChanged: (set) {
                              final mode = set.first;
                              context.read<ScanBloc>().add(ScanModeChanged(mode));
                            },
                          ),
                          const SizedBox(width: 12),
                          FilterChip(
                            label: const Text("Smart"),
                            tooltip: "Auto fallback/return",
                            selected: state.smartFallback,
                            onSelected: (v) => context
                                .read<ScanBloc>()
                                .add(ScanSmartFallbackToggled(v)),
                          ),
                        ],
                      ),
                    ),
                    Expanded(
                      child: _PreviewWithOverlay(
                        controller: ctrl,
                        detections: state.detections,
                      ),
                    ),
                  ],
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
    final Size pv = controller.value.previewSize!;
    // Rotasi portrait => (width,height) tertukar
    final double srcW = pv.height;
    final double srcH = pv.width;

    return LayoutBuilder(
      builder: (context, constraints) {
        final double dstW = constraints.maxWidth;
        final double dstH = constraints.maxHeight;

        // Fit kamera di tengah (cover), sama seperti CameraPreview
        final double scale = math.max(dstW / srcW, dstH / srcH);
        final double paintW = srcW * scale;
        final double paintH = srcH * scale;

        return Center(
          child: SizedBox(
            width: paintW,
            height: paintH,
            child: RepaintBoundary(
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
    // preview logical -> actual size
    final double sx = size.width / logicalW;
    final double sy = size.height / logicalH;

    final Paint paintBox = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final Paint paintBg = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xAA000000);

    const TextStyle textStyle = TextStyle(
      color: Colors.white,
      fontSize: 12,
      fontWeight: FontWeight.w600,
    );

    for (final d in dets) {
      paintBox.color = HSVColor.fromAHSV(
        1,
        (d.label.hashCode % 360).toDouble(),
        0.8,
        0.95,
      ).toColor();

      final Rect r = Rect.fromLTWH(
        d.box.left * sx,
        d.box.top * sy,
        d.box.width * sx,
        d.box.height * sy,
      );

      // kotak
      canvas.drawRect(r, paintBox);

      // label + score
      final String caption = "${d.label} ${(d.score * 100).toStringAsFixed(1)}%";
      final TextPainter tp = TextPainter(
        text: TextSpan(text: caption, style: textStyle),
        textDirection: TextDirection.ltr,
        maxLines: 1,
        ellipsis: '…',
      )..layout(maxWidth: size.width * 0.9);

      final double tagLeft = r.left.clamp(0.0, size.width - tp.width - 8);
      // coba di atas kotak, kalau mentok naikkan ke dalam
      double tagTop = r.top - tp.height - 6;
      if (tagTop < 0) tagTop = r.top + 2;

      final Rect tag = Rect.fromLTWH(
        tagLeft,
        tagTop,
        tp.width + 8,
        tp.height + 4,
      );
      final RRect rr = RRect.fromRectAndRadius(tag, const Radius.circular(6));
      canvas.drawRRect(rr, paintBg);
      tp.paint(canvas, Offset(tag.left + 4, tag.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionsPainter oldDelegate) {
    return oldDelegate.dets != dets ||
        oldDelegate.logicalW != logicalW ||
        oldDelegate.logicalH != logicalH;
  }
}
