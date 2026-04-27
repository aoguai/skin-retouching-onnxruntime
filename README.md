# ModelScope-Free ONNX Skin Retouching

This repository adapts Alibaba ModelScope `cv_unet_skin_retouching_torch` into a ModelScope-free ONNXRuntime inference project for high-resolution portrait skin retouching.

The goals are:

- no `modelscope` dependency at runtime
- no `torch` dependency at runtime
- `onnxruntime + opencv-python + numpy` inference for face detection, ROI cropping, skin smoothing, whitening, and optional local blemish removal
- offline export scripts that convert upstream `.pt` / `.pth` checkpoints to ONNX

For a full deployment guide, see [ONNX_DEPLOYMENT_REPORT.md](docs/ONNX_DEPLOYMENT_REPORT.md).

## Features

- Uses the upstream `model.onnx` as the skin-mask / whitening model.
- Exports `pytorch_model.pt` to `retouch_generator.onnx`.
- Exports `joint_20210926.pth` to `local_detection.onnx` and `local_inpainting.onnx`.
- Exports `cv_resnet50_face-detection_retinaface/pytorch_model.pt` to `face_detector.onnx`.
- Runs RetinaFace prior decode, landmark decode, and NMS in numpy.
- Defaults to `CPUExecutionProvider` to avoid noisy CUDA/cuDNN DLL errors on Windows systems with incomplete GPU runtimes.

## Project Layout

```text
.
├── retouch_onnx.py                    # CLI inference entrypoint
├── ms_wrapper.py                      # lightweight compatibility wrapper
├── export_skin_retouching_onnx.py      # offline PyTorch -> ONNX exporter
├── onnx_skin_retouching/               # runtime package
├── docs/                               # deployment, structure, upstream notes
├── images/                             # sample images and upstream figures
├── requirements.txt                    # runtime dependencies
└── requirements-export.txt             # export-time dependencies
```

For the full layout, see [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Model Files

This repository is intended to be published without large model artifacts in normal Git commits. `.gitignore` excludes `*.onnx`, `*.pt`, `*.pth`, and downloaded model snapshots by default.

To run the full pipeline, prepare these files in the project root:

```text
model.onnx
retouch_generator.onnx
face_detector.onnx
local_detection.onnx        # only needed with --enable-local
local_inpainting.onnx       # only needed with --enable-local
```

To export the ONNX files yourself, prepare the upstream checkpoints:

```text
model.onnx
pytorch_model.pt
joint_20210926.pth
cv_resnet50_face-detection_retinaface/pytorch_model.pt
```

## Install

Runtime only:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Export + runtime:

```powershell
python -m pip install -r requirements-export.txt
```

`torch`, `onnx`, and `onnxscript` are export-time dependencies only. The ONNXRuntime inference path does not import them.

## Export ONNX

Export all ONNX models:

```powershell
python export_skin_retouching_onnx.py --model-dir .
```

Export only the face detector:

```powershell
python export_skin_retouching_onnx.py --model-dir . --only-face
```

Skip local blemish detection / inpainting export:

```powershell
python export_skin_retouching_onnx.py --model-dir . --skip-local
```

If the RetinaFace snapshot is not under the default directory:

```powershell
python export_skin_retouching_onnx.py --model-dir . --only-face --face-model-dir path\to\cv_resnet50_face-detection_retinaface
```

## Run

Basic CPU inference:

```powershell
python retouch_onnx.py --input images/skin_retouching_examples_1.jpg --output result.png --model-dir .
```

Enable local blemish detection / inpainting:

```powershell
python retouch_onnx.py --input images/skin_retouching_examples_1.jpg --output result_local.png --model-dir . --enable-local
```

Tune retouching strengths:

```powershell
python retouch_onnx.py --input input.jpg --output output.png --model-dir . --retouch-degree 0.7 --whitening-degree 0.8
```

Use CUDA only when CUDA 12, cuDNN 9, and the MSVC runtime are installed correctly:

```powershell
python retouch_onnx.py --input input.jpg --output output.png --model-dir . --providers CUDAExecutionProvider,CPUExecutionProvider
```

## Python API

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

## Documentation

- [Deployment report](docs/ONNX_DEPLOYMENT_REPORT.md)
- [Face detector ONNX contract](docs/FACE_DETECTOR_ONNX_CONTRACT.md)
- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Upstream sources and citations](docs/UPSTREAMS.md)

## Upstream Sources

This repository adapts upstream ModelScope assets and papers:

- Skin retouching ModelScope model: <https://modelscope.cn/models/damo/cv_unet_skin_retouching_torch/summary>
- ABPN / CRHD-3K project: <https://github.com/youngLBW/CRHD-3K>
- RetinaFace ModelScope model: <https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary>
- Pytorch_Retinaface implementation referenced by ModelScope: <https://github.com/biubug6/Pytorch_Retinaface>

See [UPSTREAMS.md](docs/UPSTREAMS.md) for license notes and BibTeX entries.

## Citation

If you use the skin-retouching model, cite ABPN:

```bibtex
@inproceedings{lei2022abpn,
  title={ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo},
  author={Lei, Biwen and Guo, Xiefan and Yang, Hongyu and Cui, Miaomiao and Xie, Xuansong and Huang, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2108--2117},
  year={2022}
}
```

If you use the face detector, cite RetinaFace:

```bibtex
@inproceedings{deng2020retinaface,
  title={RetinaFace: Single-shot Multi-level Face Localisation in the Wild},
  author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5203--5212},
  year={2020}
}
```

## License Notes

This repository contains adaptation code. Upstream model weights, model cards, papers, and third-party implementations remain governed by their own licenses and redistribution terms. Verify redistribution rights before publishing pretrained weights.
