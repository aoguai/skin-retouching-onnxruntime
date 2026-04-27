# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility wrapper that no longer depends on ModelScope.

This file keeps a small pipeline-like surface for local scripts, but the
implementation is now backed by ``onnx_skin_retouching.SkinRetoucher`` and
requires exported ONNX models in ``model_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from onnx_skin_retouching import SkinRetoucher
from onnx_skin_retouching.utils import ensure_bgr_uint8


OUTPUT_IMG = "output_img"


class SkinRetouchingTorchPipeline:
    """Small local pipeline replacement for the original ModelScope wrapper."""

    def __init__(
        self,
        model: Union[str, Path] = ".",
        device: Optional[str] = None,
        enable_local: bool = False,
        retouch_degree: float = 0.7,
        whitening_degree: float = 0.8,
    ):
        providers = None
        if device is not None and "cuda" in device.lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device is not None and "cpu" in device.lower():
            providers = ["CPUExecutionProvider"]

        self.retoucher = SkinRetoucher(
            model_dir=model,
            providers=providers,
            retouch_degree=retouch_degree,
            whitening_degree=whitening_degree,
            enable_local=enable_local,
        )

    def __call__(self, input_image: Union[str, Path, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.forward(self.preprocess(input_image))

    def preprocess(self, input_image: Union[str, Path, np.ndarray]) -> Dict[str, np.ndarray]:
        if isinstance(input_image, (str, Path)):
            image = cv2.imread(str(input_image), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"failed to read input image: {input_image}")
        else:
            image = input_image
        return {"img": ensure_bgr_uint8(image)}

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {OUTPUT_IMG: self.retoucher.retouch(inputs["img"])}

    def postprocess(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return inputs


def retouch_image(
    input_image: Union[str, Path, np.ndarray],
    model_dir: Union[str, Path] = ".",
    output_path: Optional[Union[str, Path]] = None,
    enable_local: bool = False,
) -> np.ndarray:
    pipeline = SkinRetouchingTorchPipeline(model=model_dir, enable_local=enable_local)
    result = pipeline(input_image)[OUTPUT_IMG]
    if output_path is not None:
        ok = cv2.imwrite(str(output_path), result)
        if not ok:
            raise RuntimeError(f"failed to write output image: {output_path}")
    return result


if __name__ == "__main__":
    image_path = "images/skin_retouching_examples_1.jpg"
    result_path = "result.png"
    retouch_image(image_path, model_dir=".", output_path=result_path)
    print({"output": result_path})
