# ModelScope-Free ONNX Skin Retouching Export and Deployment Report

This document explains how to prepare the project from scratch, export ONNX models, run inference, and understand the runtime boundaries and known troubleshooting cases.

## 1. Goals and Summary

The original `ms_wrapper.py` depended on the ModelScope pipeline:

- `model.onnx` was only used for the skin-mask / whitening stage.
- The main skin-smoothing network came from `pytorch_model.pt`.
- The optional local blemish removal networks came from `joint_20210926.pth`.
- Face boxes and five-point landmarks came from `cv_resnet50_face-detection_retinaface` / RetinaFace.

The adapted project is designed with these goals:

- no `modelscope` dependency at runtime
- no `torch` dependency at runtime
- `onnxruntime + opencv-python + numpy` as the runtime stack
- `torch` and `onnx` allowed only during offline export

Implemented outputs:

- `retouch_generator.onnx`: main skin-smoothing ONNX model
- `local_detection.onnx`: local blemish detection ONNX model
- `local_inpainting.onnx`: local blemish inpainting ONNX model
- `face_detector.onnx`: RetinaFace ONNX model with raw `loc/conf/landms` outputs
- numpy runtime decode for RetinaFace priors, boxes, five landmarks, and NMS
- default `CPUExecutionProvider` to avoid noisy CUDA provider load failures on Windows systems without matching CUDA/cuDNN DLLs

## 2. Important Files

| File | Purpose |
| --- | --- |
| `retouch_onnx.py` | ONNXRuntime CLI inference entrypoint. |
| `ms_wrapper.py` | Lightweight compatibility wrapper backed by `SkinRetoucher`; no ModelScope dependency. |
| `export_skin_retouching_onnx.py` | Offline exporter from `.pt` / `.pth` checkpoints to ONNX. |
| `onnx_skin_retouching/runtime.py` | Runtime orchestration: ONNX sessions, face ROI, smoothing, whitening, and optional local retouching. |
| `onnx_skin_retouching/face.py` | RetinaFace preprocessing, prior decode, landmark decode, and NMS in numpy. |
| `onnx_skin_retouching/utils.py` | ROI, blend-layer composition, whitening, and patch partition/aggregation utilities. |
| `docs/FACE_DETECTOR_ONNX_CONTRACT.md` | Input/output contract for `face_detector.onnx`. |
| `requirements.txt` | Runtime dependencies. |
| `requirements-export.txt` | Offline export dependencies. |
| `README.md` | Concise project overview and quickstart. |

## 3. Model File Inventory

### 3.1 Files Required Before Export

The project root should contain:

| File | Source | Purpose |
| --- | --- | --- |
| `model.onnx` | Original upstream artifact | Skin mask / whitening. This is not the full retouching model. |
| `pytorch_model.pt` | Original upstream artifact | Main smoothing generator weights. |
| `joint_20210926.pth` | Original upstream artifact | Local blemish detection / inpainting weights. |

The face detection snapshot should contain:

```text
cv_resnet50_face-detection_retinaface/
  configuration.json
  pytorch_model.pt
```

By default, the exporter reads RetinaFace weights from `cv_resnet50_face-detection_retinaface/pytorch_model.pt`. Use `--face-model-dir` if the snapshot is stored elsewhere.

### 3.2 ONNX Files Required After Export

| File | Required | Purpose |
| --- | --- | --- |
| `model.onnx` | yes | Skin mask / whitening. |
| `retouch_generator.onnx` | yes | Main skin smoothing. |
| `face_detector.onnx` | yes | Face boxes and five-point landmarks. |
| `local_detection.onnx` | only with `--enable-local` | Local blemish detection. |
| `local_inpainting.onnx` | only with `--enable-local` | Local blemish inpainting. |

Example ONNX artifacts produced in the local workspace:

| File | Size |
| --- | ---: |
| `model.onnx` | 102544002 bytes |
| `retouch_generator.onnx` | 53567920 bytes |
| `local_detection.onnx` | 72134598 bytes |
| `local_inpainting.onnx` | 144272586 bytes |
| `face_detector.onnx` | 109110170 bytes |

## 4. Environment Setup

The examples below use Windows PowerShell.

### 4.1 Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4.2 Install Runtime Dependencies

For inference only:

```powershell
python -m pip install -r requirements.txt
```

### 4.3 Install Export-Time Dependencies

For exporting `.pt` / `.pth` checkpoints to ONNX:

```powershell
python -m pip install -r requirements-export.txt
```

Notes:

- `torch`, `onnx`, and `onnxscript` are only needed for offline export.
- Runtime inference does not import `torch`, `onnx`, `onnxscript`, or `modelscope`.

### 4.4 Optional GPU Runtime

The default inference provider is:

```text
CPUExecutionProvider
```

This avoids repeated CUDA provider errors on machines with `onnxruntime-gpu` installed but without matching CUDA/cuDNN DLLs, for example:

```text
Failed to create CUDAExecutionProvider
cudnn64_9.dll is missing
```

To use GPU inference, install matching CUDA 12, cuDNN 9, and the MSVC runtime, then opt in explicitly:

```powershell
python retouch_onnx.py --input images/skin_retouching_examples_1.jpg --output result.png --model-dir . --providers CUDAExecutionProvider,CPUExecutionProvider
```

## 5. ONNX Export

### 5.1 Export All Models

Run from the project root:

```powershell
python export_skin_retouching_onnx.py --model-dir .
```

Expected output:

```text
exported retouch_generator.onnx
exported local_detection.onnx
exported local_inpainting.onnx
exported face_detector.onnx
```

### 5.2 Export Main Smoothing and Face Detection Only

```powershell
python export_skin_retouching_onnx.py --model-dir . --skip-local
```

Use this mode when you do not plan to run inference with `--enable-local`.

### 5.3 Export Face Detection Only

If the skin-retouching ONNX files already exist and only `face_detector.onnx` is missing:

```powershell
python export_skin_retouching_onnx.py --model-dir . --only-face
```

### 5.4 Use a Custom Face Model Directory

```powershell
python export_skin_retouching_onnx.py --model-dir . --only-face --face-model-dir path\to\cv_resnet50_face-detection_retinaface
```

### 5.5 Export Warnings

You may see this PyTorch warning:

```text
Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op.
```

This is a constant-folding warning, not an export failure. If the command prints `exported xxx.onnx`, the corresponding model file was written successfully.

## 6. ONNX Contracts

### 6.1 Retouching Models

| File | Input | Output |
| --- | --- | --- |
| `model.onnx` | RGB HWC float32, long side resized to 800 | skin mask |
| `retouch_generator.onnx` | `1 x 3 x 512 x 512`, RGB normalized to `[-1, 1]` | `pred_mg` |
| `local_detection.onnx` | `1 x 3 x 768 x 768`, RGB normalized to `[-1, 1]` | `mask_logits` |
| `local_inpainting.onnx` | image patch + mask patch | `inpainted` |

### 6.2 Face Detection Model

`face_detector.onnx` input:

```text
image: [1, 3, height, width], tensor(float)
```

The runtime performs:

- RGB to BGR conversion
- scaling to `[0, 1]` by dividing by `255`
- long-side limit to `1024`
- bottom/right padding to a multiple of 32

`face_detector.onnx` outputs:

```text
loc:    [1, num_priors, 4]
conf:   [1, num_priors, 2]
landms: [1, num_priors, 10]
```

`onnx_skin_retouching/face.py` then performs:

- prior generation
- bbox decode
- landmark decode
- score thresholding
- NMS
- coordinate scaling back to the original image

## 7. Inference

### 7.1 Basic Inference

```powershell
python retouch_onnx.py --input images/skin_retouching_examples_1.jpg --output result.png --model-dir .
```

### 7.2 Enable Local Blemish Removal

```powershell
python retouch_onnx.py --input images/skin_retouching_examples_1.jpg --output result_local.png --model-dir . --enable-local
```

When `--enable-local` is used, these files must exist:

```text
local_detection.onnx
local_inpainting.onnx
```

### 7.3 Tune Strengths

```powershell
python retouch_onnx.py --input input.jpg --output output.png --model-dir . --retouch-degree 0.7 --whitening-degree 0.8
```

Defaults:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--retouch-degree` | `0.7` | Main smoothing blend strength. |
| `--whitening-degree` | `0.8` | Whitening strength. |

### 7.4 Use GPU

Use this only when CUDA/cuDNN is configured correctly:

```powershell
python retouch_onnx.py --input input.jpg --output output.png --model-dir . --providers CUDAExecutionProvider,CPUExecutionProvider
```

Otherwise, keep the default CPU provider.

## 8. Python API

```python
import cv2
from onnx_skin_retouching import SkinRetoucher

image = cv2.imread("images/skin_retouching_examples_1.jpg", cv2.IMREAD_UNCHANGED)

retoucher = SkinRetoucher(
    model_dir=".",
    retouch_degree=0.7,
    whitening_degree=0.8,
    enable_local=False,
)

result = retoucher.retouch(image)
cv2.imwrite("result.png", result)
```

Compatibility wrapper:

```python
import cv2
from ms_wrapper import OUTPUT_IMG, SkinRetouchingTorchPipeline

pipeline = SkinRetouchingTorchPipeline(model=".", enable_local=False)
result = pipeline("images/skin_retouching_examples_1.jpg")[OUTPUT_IMG]
cv2.imwrite("result.png", result)
```

## 9. Production Deployment

### 9.1 Export Machine

The export machine needs:

```text
torch
onnx
numpy
opencv-python
```

It also needs the source checkpoints:

```text
pytorch_model.pt
joint_20210926.pth
cv_resnet50_face-detection_retinaface/pytorch_model.pt
```

### 9.2 Inference Machine

The inference machine only needs:

```text
numpy
opencv-python
onnxruntime
```

Deploy these files:

```text
retouch_onnx.py
ms_wrapper.py
onnx_skin_retouching/
model.onnx
retouch_generator.onnx
face_detector.onnx
local_detection.onnx          # only if --enable-local is used
local_inpainting.onnx         # only if --enable-local is used
```

The inference machine does not need:

```text
modelscope
torch
onnx
onnxscript
pytorch_model.pt
joint_20210926.pth
cv_resnet50_face-detection_retinaface/
```

If local blemish removal is not used, do not deploy:

```text
local_detection.onnx
local_inpainting.onnx
```

## 10. Execution and Validation Notes

### 10.1 Completed Code Changes

- Added the ModelScope-free runtime package: `onnx_skin_retouching`.
- Added CLI inference: `retouch_onnx.py`.
- Replaced `ms_wrapper.py` with a lightweight compatibility wrapper.
- Added offline export script: `export_skin_retouching_onnx.py`.
- Added RetinaFace ONNX export structure and numpy postprocessing.
- Changed the default provider to `CPUExecutionProvider`.
- Updated README and `docs/FACE_DETECTOR_ONNX_CONTRACT.md`.

### 10.2 Static Checks

Runtime code does not import:

```text
from modelscope
import modelscope
from torch
import torch
```

Notes:

- `torch` is still present in `export_skin_retouching_onnx.py`, which is export-only.
- `onnxruntime` is present in the runtime, as expected.

### 10.3 ONNXRuntime Contract Check

`face_detector.onnx` was loaded with ONNXRuntime and reported:

```text
inputs:
  image: [1, 3, 'height', 'width'] tensor(float)

outputs:
  loc:    [1, 'num_priors', 4] tensor(float)
  conf:   [1, 'num_priors', 2] tensor(float)
  landms: [1, 'num_priors', 10] tensor(float)
```

### 10.4 User-Side Runtime Result

The user reported:

- images are generated successfully
- the output has visible retouching effects
- CUDA provider dependency errors did not prevent CPU fallback inference

The default provider was changed to CPU afterward to avoid CUDA/cuDNN warning spam.

## 11. Troubleshooting

### 11.1 `ModuleNotFoundError: No module named 'onnxscript'`

Cause: newer PyTorch ONNX exporter paths may import `onnxscript`.

The exporter now uses the legacy exporter path, but if this error still appears, install:

```powershell
python -m pip install onnxscript
```

### 11.2 `torch.onnx.OnnxExporterError: Module onnx is not installed!`

Cause: the ONNX Python package is required when exporting ONNX files.

Fix:

```powershell
python -m pip install onnx
```

### 11.3 CUDA Provider Warning Spam

Typical log:

```text
Failed to create CUDAExecutionProvider
cudnn64_9.dll is missing
```

Cause: `onnxruntime-gpu` is installed, but CUDA 12 / cuDNN 9 / MSVC runtime dependencies are incomplete.

Fix:

- use the latest code and do not pass `--providers`
- or force CPU explicitly:

```powershell
python retouch_onnx.py --input input.jpg --output output.png --model-dir . --providers CPUExecutionProvider
```

To use GPU, install the full CUDA/cuDNN runtime stack and then run:

```powershell
python retouch_onnx.py --input input.jpg --output output.png --model-dir . --providers CUDAExecutionProvider,CPUExecutionProvider
```

### 11.4 `missing ONNX model`

The runtime is missing a required model file. Check:

```powershell
Get-Item model.onnx, retouch_generator.onnx, face_detector.onnx
```

If local retouching is enabled, also check:

```powershell
Get-Item local_detection.onnx, local_inpainting.onnx
```

### 11.5 No Face Detected, Output Looks Unchanged

If no valid face box is detected, the runtime returns the original BGR image. Try:

- using a clearer portrait image
- confirming `face_detector.onnx` exists and loads in ONNXRuntime
- lowering `min_face_score` through the Python API: `SkinRetoucher(min_face_score=...)`

### 11.6 `--enable-local` Fails Because Local Models Are Missing

Local blemish removal requires:

```text
local_detection.onnx
local_inpainting.onnx
```

If you only need main smoothing and whitening, do not pass `--enable-local`.

## 12. Shortest From-Zero Workflow

```powershell
# 1. Create environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Install export + runtime dependencies
python -m pip install -r requirements-export.txt

# 3. Confirm source model files
Get-Item model.onnx, pytorch_model.pt, joint_20210926.pth
Get-Item cv_resnet50_face-detection_retinaface\pytorch_model.pt

# 4. Export all ONNX models
python export_skin_retouching_onnx.py --model-dir .

# 5. Run default CPU inference
python retouch_onnx.py --input images/skin_retouching_examples_1.jpg --output result.png --model-dir .

# 6. Optional: enable local blemish removal
python retouch_onnx.py --input images/skin_retouching_examples_1.jpg --output result_local.png --model-dir . --enable-local
```

If skin-retouching ONNX files already exist and only the face detector is missing:

```powershell
python export_skin_retouching_onnx.py --model-dir . --only-face
```

If you only deploy inference and do not need export:

```powershell
python -m pip install -r requirements.txt
python retouch_onnx.py --input input.jpg --output output.png --model-dir .
```

## 13. Known Boundaries

- `model.onnx` is not the full retouching model; it only handles skin mask / whitening.
- `face_detector.onnx` outputs raw `loc/conf/landms`; decode and NMS run in numpy.
- CPU is the stable default. GPU requires a correctly matched CUDA/cuDNN/MSVC runtime stack.
- `onnx_skin_retouching/utils.py` reproduces the key image-processing behavior of the original pipeline. Minor differences from unreleased internal ModelScope utility implementations may remain.
- The exporter embeds the required network definitions to avoid a runtime ModelScope dependency. If upstream checkpoints or architectures change, verify state-dict key compatibility again.
