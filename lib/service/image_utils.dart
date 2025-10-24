import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

Object _allocOut(String targetType, int dst) =>
    targetType == 'int8' ? Int8List(dst * dst * 3) : Float32List(dst * dst * 3);

@pragma('vm:prefer-inline')
int _clamp8(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

Object bgra8888ToNhwcQuantAwareCompute(Map<String, dynamic> args) {
  final int width = args['width'];
  final int height = args['height'];
  final int dst = args['dst'];

  final Uint8List bytes = args['bytes'];
  final String targetType = args['targetType'];
  final double inScale = (args['inScale'] ?? 1.0) as double;
  final int inZeroPoint = (args['inZeroPoint'] ?? 0) as int;

  final double sx = width / dst;
  final double sy = height / dst;

  final out = _allocOut(targetType, dst);
  int k = 0;

  for (int y = 0; y < dst; y++) {
    final int syi = (y * sy).floor();
    final int rowBase = syi * width * 4;
    for (int x = 0; x < dst; x++) {
      final int sxi = (x * sx).floor();
      final int idx = rowBase + (sxi << 2);

      final int b = bytes[idx + 0];
      final int g = bytes[idx + 1];
      final int r = bytes[idx + 2];

      if (out is Float32List) {
        out[k++] = r / 255.0;
        out[k++] = g / 255.0;
        out[k++] = b / 255.0;
      } else {
        int rq = (inZeroPoint + (r / 255.0) / inScale).round();
        int gq = (inZeroPoint + (g / 255.0) / inScale).round();
        int bq = (inZeroPoint + (b / 255.0) / inScale).round();

        if (rq < -128) rq = -128;
        if (rq > 127) rq = 127;
        if (gq < -128) gq = -128;
        if (gq > 127) gq = 127;
        if (bq < -128) bq = -128;
        if (bq > 127) bq = 127;
        (out as Int8List)[k++] = rq;
        out[k++] = gq;
        out[k++] = bq;
      }
    }
  }
  return out;
}

Object yuv420ToNhwcQuantAwareCompute(Map<String, dynamic> args) {
  final int width = args['width'];
  final int height = args['height'];
  final int dst = args['dst'];

  final Uint8List yBytes = args['y'];
  final Uint8List uBytes = args['u'];
  final Uint8List vBytes = args['v'];
  final int yRowStride = args['yRowStride'];
  final int uvRowStride = args['uvRowStride'];
  final int uvPixelStride = args['uvPixelStride'];

  final String targetType = args['targetType'];
  final double inScale = (args['inScale'] ?? 1.0) as double;
  final int inZeroPoint = (args['inZeroPoint'] ?? 0) as int;

  final double sx = width / dst;
  final double sy = height / dst;

  final out = _allocOut(targetType, dst);
  int k = 0;

  for (int y = 0; y < dst; y++) {
    final int syi = (y * sy).floor();
    final int yRow = syi * yRowStride;
    final int uvRow = (syi >> 1) * uvRowStride;

    for (int x = 0; x < dst; x++) {
      final int sxi = (x * sx).floor();

      final int yi = yRow + sxi;
      final int uvi = uvRow + ((sxi >> 1) * uvPixelStride);

      final int Y = yBytes[yi];
      final int U = uBytes[uvi];
      final int V = vBytes[uvi];

      final double yf = Y.toDouble();
      final double uf = U.toDouble() - 128.0;
      final double vf = V.toDouble() - 128.0;

      int r = (yf + 1.402 * vf).round();
      int g = (yf - 0.344136 * uf - 0.714136 * vf).round();
      int b = (yf + 1.772 * uf).round();
      r = _clamp8(r);
      g = _clamp8(g);
      b = _clamp8(b);

      if (out is Float32List) {
        out[k++] = r / 255.0;
        out[k++] = g / 255.0;
        out[k++] = b / 255.0;
      } else {
        int rq = (inZeroPoint + (r / 255.0) / inScale).round();
        int gq = (inZeroPoint + (g / 255.0) / inScale).round();
        int bq = (inZeroPoint + (b / 255.0) / inScale).round();
        if (rq < -128) rq = -128;
        if (rq > 127) rq = 127;
        if (gq < -128) gq = -128;
        if (gq > 127) gq = 127;
        if (bq < -128) bq = -128;
        if (bq > 127) bq = 127;
        (out as Int8List)[k++] = rq;
        out[k++] = gq;
        out[k++] = bq;
      }
    }
  }
  return out;
}

Rect projectBoxToPreview(
  Rect boxModel,
  double pw,
  double ph, {
  double modelSize = 640,
}) {
  final double sx = pw / modelSize;
  final double sy = ph / modelSize;
  return Rect.fromLTWH(
    boxModel.left * sx,
    boxModel.top * sy,
    boxModel.width * sx,
    boxModel.height * sy,
  );
}

List<int> nonMaxSuppression(List<Rect> boxes, List<double> scores, double th) {
  final idxs = List<int>.generate(boxes.length, (i) => i)
    ..sort((a, b) => scores[b].compareTo(scores[a]));
  final keep = <int>[];
  while (idxs.isNotEmpty) {
    final cur = idxs.removeAt(0);
    keep.add(cur);
    idxs.removeWhere((o) {
      final a = boxes[cur], b = boxes[o];
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
      return iou > th;
    });
  }
  return keep;
}
