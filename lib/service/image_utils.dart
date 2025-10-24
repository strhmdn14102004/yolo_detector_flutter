import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart' as img;

Float32List yuv420ToRgbNormalizedResizeCompute(Map<String, dynamic> args) {
  final int width = args['width'] as int;
  final int height = args['height'] as int;
  final int dst = args['dst'] as int;

  final Uint8List yBytes = args['y'] as Uint8List;
  final Uint8List uBytes = args['u'] as Uint8List;
  final Uint8List vBytes = args['v'] as Uint8List;

  final int yRowStride = args['yRowStride'] as int;
  final int uvRowStride = args['uvRowStride'] as int;
  final int uvPixelStride = args['uvPixelStride'] as int;

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

  final img.Image resized = img.copyResize(
    rgbImage,
    width: dst,
    height: dst,
    interpolation: img.Interpolation.cubic,
  );

  final Float32List out = Float32List(dst * dst * 3);
  int i = 0;
  for (int y = 0; y < dst; y++) {
    for (int x = 0; x < dst; x++) {
      final img.Pixel p = resized.getPixel(x, y);
      out[i++] = p.r / 255.0;
      out[i++] = p.g / 255.0;
      out[i++] = p.b / 255.0;
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
          (math.max(
            0.0,
            math.min(a.right, b.right) - math.max(a.left, b.left),
          )) *
          (math.max(
            0.0,
            math.min(a.bottom, b.bottom) - math.max(a.top, b.top),
          ));
      final double union = a.width * a.height + b.width * b.height - inter;
      final double iou = union == 0 ? 0 : inter / union;
      return iou > iouThreshold;
    });
  }
  return keep;
}
