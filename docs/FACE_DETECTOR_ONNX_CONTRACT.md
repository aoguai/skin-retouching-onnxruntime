# face_detector.onnx contract

`face_detector.onnx` replaces the original ModelScope face-detection pipeline.
The current exporter creates a raw RetinaFace ONNX model, and the skin
retouching runtime performs prior decode, landmark decode and NMS in numpy.

## Input

- One input tensor named `image`.
- BGR image in NCHW layout.
- `float32`.
- Mean-subtracted by `(104, 117, 123)`.
- Height and width should be multiples of 32. The runtime resizes the long side
  to at most `1024`, pads bottom/right to a multiple of 32, and feeds this tensor.

## Outputs

The raw RetinaFace model must expose:

1. `loc`: shape `1 x N x 4`
2. `conf`: shape `1 x N x 2`, softmax scores
3. `landms`: shape `1 x N x 10`

The runtime also keeps fallback support for an already post-processed detector
with `boxes`, `scores`, and `keypoints` / `landmarks` outputs, but the bundled
export script writes the raw RetinaFace form above.

## Why this is separate

The downloaded ModelScope model directory contains `pytorch_model.pt` and
configuration metadata, but no Python implementation files. The local exporter
therefore includes a minimal RetinaFace ResNet50/FPN/SSH definition that matches
the downloaded state dict. Keeping decode and NMS in numpy keeps runtime
dependencies limited to `onnxruntime`, `opencv-python`, and `numpy`.
