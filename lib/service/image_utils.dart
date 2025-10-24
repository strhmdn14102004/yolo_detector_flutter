import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart' as img;

/// Konversi YUV420 -> RGB, resize ke [dst] dan keluarkan NHWC:
/// - jika targetType == 'float32': Float32List [0..1]
/// - jika targetType == 'int8'   : Int8List terkuantisasi dg (scale, zeroPoint)
///
/// Argumen yang wajib:
/// width, height, dst,
/// y, u, v (Uint8List),
/// yRowStride, uvRowStride, uvPixelStride,
/// targetType ('float32' | 'int8'),
/// inScale (double), inZeroPoint (int) -> hanya dipakai untuk 'int8'
Object yuv420ToNhwcQuantAwareCompute(Map<String, dynamic> args) {
  final int width = args['width'] as int;
  final int height = args['height'] as int;
  final int dst = args['dst'] as int;

  final Uint8List yBytes = args['y'] as Uint8List;
  final Uint8List uBytes = args['u'] as Uint8List;
  final Uint8List vBytes = args['v'] as Uint8List;

  final int yRowStride = args['yRowStride'] as int;
  final int uvRowStride = args['uvRowStride'] as int;
  final int uvPixelStride = args['uvPixelStride'] as int;

  final String targetType = args['targetType'] as String;
  final double inScale = (args['inScale'] ?? 1.0) as double;
  final int inZeroPoint = (args['inZeroPoint'] ?? 0) as int;

  // YUV420 -> RGB (image lib)
  final img.Image rgbImage = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    final int pY = yRowStride * y;
    final int uvY = uvRowStride * (y >> 1);

    for (int x = 0; x < width; x++) {
      final int yi = pY + x;
      final int uvX = (x >> 1) * uvPixelStride;

      final int yp = yBytes[yi];
      final int up = uBytes[uvY + uvX];
      final int vp = vBytes[uvY + uvX];

      double yf = yp.toDouble();
      double uf = up.toDouble() - 128.0;
      double vf = vp.toDouble() - 128.0;

      int r = (yf + 1.402 * vf).round();
      int g = (yf - 0.344136 * uf - 0.714136 * vf).round();
      int b = (yf + 1.772 * uf).round();

      rgbImage.setPixelRgb(
        x,
        y,
        r.clamp(0, 255),
        g.clamp(0, 255),
        b.clamp(0, 255),
      );
    }
  }

  // Resize ke ukuran input model
  final img.Image resized = img.copyResize(
    rgbImage,
    width: dst,
    height: dst,
    interpolation: img.Interpolation.linear,
  );

  if (targetType == 'int8') {
    // Kuantisasi: q = round(zp + (float/255)/scale)
    final Int8List out = Int8List(dst * dst * 3);
    int i = 0;
    for (int y = 0; y < dst; y++) {
      for (int x = 0; x < dst; x++) {
        final p = resized.getPixel(x, y);
        final rn = (p.r / 255.0);
        final gn = (p.g / 255.0);
        final bn = (p.b / 255.0);

        int rq = (inZeroPoint + rn / inScale).round();
        int gq = (inZeroPoint + gn / inScale).round();
        int bq = (inZeroPoint + bn / inScale).round();

        // int8 clamp -128..127
        rq = rq.clamp(-128, 127);
        gq = gq.clamp(-128, 127);
        bq = bq.clamp(-128, 127);

        out[i++] = rq;
        out[i++] = gq;
        out[i++] = bq;
      }
    }
    return out;
  } else {
    // FLOAT32 [0..1]
    final Float32List out = Float32List(dst * dst * 3);
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

List<int> nonMaxSuppression(
  List<Rect> boxes,
  List<double> scores,
  double iouThreshold,
) {
  final List<int> idxs = List<int>.generate(boxes.length, (i) => i)
    ..sort((a, b) => scores[b].compareTo(scores[a]));

  final List<int> keep = [];
  while (idxs.isNotEmpty) {
    final int current = idxs.removeAt(0);
    keep.add(current);

    idxs.removeWhere((other) {
      final Rect a = boxes[current];
      final Rect b = boxes[other];
      final double inter =
          (math.max(0.0, math.min(a.right, b.right) - math.max(a.left, b.left))) *
          (math.max(0.0, math.min(a.bottom, b.bottom) - math.max(a.top, b.top)));
    final double union = a.width * a.height + b.width * b.height - inter;
    final double iou = union == 0 ? 0 : inter / union;
    return iou > iouThreshold;
    });
  }
  return keep;
}
