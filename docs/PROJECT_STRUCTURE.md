# Project Structure

This repository is organized as a small ONNXRuntime inference package plus an
offline exporter.

```text
.
├── README.md
├── requirements.txt
├── requirements-export.txt
├── retouch_onnx.py
├── ms_wrapper.py
├── export_skin_retouching_onnx.py
├── configuration.json
├── .gitattributes
├── .gitignore
├── docs/
│   ├── ONNX_DEPLOYMENT_REPORT.md
│   ├── FACE_DETECTOR_ONNX_CONTRACT.md
│   ├── PROJECT_STRUCTURE.md
│   └── UPSTREAMS.md
├── images/
│   ├── skin_retouching_examples_1.jpg
│   ├── skin_retouching_examples_2.jpg
│   ├── examples.jpg
│   └── ABPN_framework.jpg
└── onnx_skin_retouching/
    ├── __init__.py
    ├── runtime.py
    ├── face.py
    └── utils.py
```

## Runtime Files

- `retouch_onnx.py`: command line inference entrypoint.
- `ms_wrapper.py`: compatibility wrapper for simple script-style usage.
- `onnx_skin_retouching/runtime.py`: orchestrates all ONNX sessions and the
  retouching pipeline.
- `onnx_skin_retouching/face.py`: RetinaFace preprocessing, prior decode,
  landmark decode and NMS in numpy.
- `onnx_skin_retouching/utils.py`: image utilities for ROI crops, blend-layer
  composition, whitening and local patch stitching.

Runtime dependencies are intentionally limited to:

```text
numpy
opencv-python
onnxruntime
```

## Export Files

- `export_skin_retouching_onnx.py`: offline exporter from PyTorch checkpoints
  to ONNX.
- `requirements-export.txt`: dependencies needed only for ONNX export.

The exporter can read:

```text
pytorch_model.pt
joint_20210926.pth
cv_resnet50_face-detection_retinaface/pytorch_model.pt
```

and produce:

```text
retouch_generator.onnx
local_detection.onnx
local_inpainting.onnx
face_detector.onnx
```

## Model Artifacts

Large model files are ignored by `.gitignore`:

```text
*.onnx
*.pt
*.pth
cv_resnet50_face-detection_retinaface/
```

For public GitHub hosting, publish model artifacts through Git LFS, GitHub
Releases, or external model hosting. If you intentionally want to commit model
files through Git LFS, remove or override the corresponding `.gitignore` rules
and keep `.gitattributes`.
