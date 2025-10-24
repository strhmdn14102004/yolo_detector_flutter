import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart' as img;

Object _rgbImageToNhwcQuantAware({
  required img.Image rgb,
  required int dst,
  required String targetType,
  required double inScale,
  required int inZeroPoint,
}) {
  final resized = img.copyResize(
    rgb,
    width: dst,
    height: dst,
    interpolation: img.Interpolation.linear,
  );

  if (targetType == 'int8') {
    final out = Int8List(dst * dst * 3);
    int i = 0;
    for (int y = 0; y < dst; y++) {
      for (int x = 0; x < dst; x++) {
        final p = resized.getPixel(x, y);
        final rn = p.r / 255.0;
        final gn = p.g / 255.0;
        final bn = p.b / 255.0;

        int rq = (inZeroPoint + rn / inScale).round();
        int gq = (inZeroPoint + gn / inScale).round();
        int bq = (inZeroPoint + bn / inScale).round();

        if (rq < -128) rq = -128;
        if (rq > 127) rq = 127;
        if (gq < -128) gq = -128;
        if (gq > 127) gq = 127;
        if (bq < -128) bq = -128;
        if (bq > 127) bq = 127;

        out[i++] = rq;
        out[i++] = gq;
        out[i++] = bq;
      }
    }
    return out;
  } else {
    final out = Float32List(dst * dst * 3);
    int i = 0;
    for (int y = 0; y < dst; y++) {
      for (int x = 0; x < dst; x++) {
        final p = resized.getPixel(x, y);
        out[i++] = p.r / 255.0;
        out[i++] = p.g / 255.0;
        out[i++] = p.b / 255.0;
      }
    }
    return out;
  }
}

Object yuv420ToNhwcQuantAwareCompute(Map<String, dynamic> args) {
  final int width = args['width'];
  final int height = args['height'];
  final int dst = args['dst'];

  final yBytes = args['y'] as Uint8List;
  final uBytes = args['u'] as Uint8List;
  final vBytes = args['v'] as Uint8List;

  final int yRowStride = args['yRowStride'];
  final int uvRowStride = args['uvRowStride'];
  final int uvPixelStride = args['uvPixelStride'];

  final String targetType = args['targetType'];
  final double inScale = (args['inScale'] ?? 1.0) as double;
  final int inZeroPoint = (args['inZeroPoint'] ?? 0) as int;

  final rgb = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    final pY = yRowStride * y;
    final uvY = uvRowStride * (y >> 1);
    for (int x = 0; x < width; x++) {
      final yi = pY + x;
      final uvX = (x >> 1) * uvPixelStride;

      final yp = yBytes[yi];
      final up = uBytes[uvY + uvX];
      final vp = vBytes[uvY + uvX];

      final yf = yp.toDouble();
      final uf = up.toDouble() - 128.0;
      final vf = vp.toDouble() - 128.0;

      int r = (yf + 1.402 * vf).round();
      int g = (yf - 0.344136 * uf - 0.714136 * vf).round();
      int b = (yf + 1.772 * uf).round();

      if (r < 0) r = 0;
      if (r > 255) r = 255;
      if (g < 0) g = 0;
      if (g > 255) g = 255;
      if (b < 0) b = 0;
      if (b > 255) b = 255;

      rgb.setPixelRgb(x, y, r, g, b);
    }
  }

  return _rgbImageToNhwcQuantAware(
    rgb: rgb,
    dst: dst,
    targetType: targetType,
    inScale: inScale,
    inZeroPoint: inZeroPoint,
  );
}

Object bgra8888ToNhwcQuantAwareCompute(Map<String, dynamic> args) {
  final int width = args['width'];
  final int height = args['height'];
  final int dst = args['dst'];

  final bytes = args['bytes'] as Uint8List;
  final String targetType = args['targetType'];
  final double inScale = (args['inScale'] ?? 1.0) as double;
  final int inZeroPoint = (args['inZeroPoint'] ?? 0) as int;

  final rgb = img.Image(width: width, height: height);

  int idx = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final b = bytes[idx++];
      final g = bytes[idx++];
      final r = bytes[idx++];
      idx++;
      rgb.setPixelRgb(x, y, r, g, b);
    }
  }

  return _rgbImageToNhwcQuantAware(
    rgb: rgb,
    dst: dst,
    targetType: targetType,
    inScale: inScale,
    inZeroPoint: inZeroPoint,
  );
}

Rect projectBoxToPreview(
  Rect boxModel,
  double pw,
  double ph, {
  double modelSize = 640,
}) {
  final sx = pw / modelSize;
  final sy = ph / modelSize;
  return Rect.fromLTWH(
    boxModel.left * sx,
    boxModel.top * sy,
    boxModel.width * sx,
    boxModel.height * sy,
  );
}

List<int> nonMaxSuppression(
  List<Rect> boxes,
  List<double> scores,
  double iouThreshold,
) {
  final idxs = List<int>.generate(boxes.length, (i) => i)
    ..sort((a, b) => scores[b].compareTo(scores[a]));

  final keep = <int>[];
  while (idxs.isNotEmpty) {
    final current = idxs.removeAt(0);
    keep.add(current);

    idxs.removeWhere((other) {
      final a = boxes[current];
      final b = boxes[other];
      final interW = math.max(
        0.0,
        math.min(a.right, b.right) - math.max(a.left, b.left),
      );
      final interH = math.max(
        0.0,
        math.min(a.bottom, b.bottom) - math.max(a.top, b.top),
      );
      final inter = interW * interH;
      final union = a.width * a.height + b.width * b.height - inter;
      final iou = union == 0 ? 0 : inter / union;
      return iou > iouThreshold;
    });
  }
  return keep;
}
