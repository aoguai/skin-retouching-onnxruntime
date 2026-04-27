# Upstream Sources, Licenses and Citations

This repository is a ModelScope-free ONNXRuntime adaptation of Alibaba
ModelScope skin-retouching and face-detection assets. It does not claim
authorship of the upstream models, papers, or pretrained weights.

## Skin Retouching Upstream

- ModelScope model card:
  <https://modelscope.cn/models/damo/cv_unet_skin_retouching_torch/summary>
- The original README in this snapshot describes the model as a
  "high-resolution portrait skin retouching model based on blend layers".
- The model card lists the license as Apache License 2.0.
- Paper:
  "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of
  Ultra High-Resolution Photo", CVPR 2022.
- Related GitHub repository:
  <https://github.com/youngLBW/CRHD-3K>

The upstream skin-retouching pipeline contains multiple stages:

- skin mask / whitening: existing `model.onnx`
- main skin smoothing: `pytorch_model.pt`
- optional local blemish detection and inpainting: `joint_20210926.pth`

This repository exports the PyTorch stages to ONNX and reimplements the runtime
orchestration without ModelScope.

## Face Detection Upstream

- ModelScope model card:
  <https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary>
- The downloaded model card lists the license as MIT License.
- The model is based on RetinaFace:
  <https://arxiv.org/abs/1905.00641>
- The ModelScope model card references the open-source implementation:
  <https://github.com/biubug6/Pytorch_Retinaface>

The downloaded ModelScope snapshot contains `pytorch_model.pt` and metadata, but
not the Python network source. This repository therefore includes a minimal
RetinaFace ResNet50/FPN/SSH definition in `export_skin_retouching_onnx.py` for
offline export only. Runtime inference uses ONNXRuntime and numpy postprocess.

## What This Repository Changes

- Removes runtime dependency on `modelscope`.
- Removes runtime dependency on `torch`.
- Adds ONNX export scripts for all required PyTorch checkpoints.
- Adds ONNXRuntime runtime orchestration.
- Adds numpy/opencv RetinaFace postprocessing.
- Defaults ONNXRuntime to CPU to avoid CUDA/cuDNN DLL errors on Windows.

## Citation

If you use the skin-retouching model, cite the ABPN paper:

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

- Code added in this repository should be distributed under the license you
  choose for this repository.
- Upstream model weights, model cards, and third-party implementations remain
  governed by their own upstream licenses and terms.
- Before publishing pretrained weights, verify redistribution rights for each
  upstream artifact.
