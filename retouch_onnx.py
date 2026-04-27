# Copyright (c) Alibaba, Inc. and its affiliates.
"""Command line entrypoint for the ModelScope-free ONNXRuntime retoucher."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2

from onnx_skin_retouching import SkinRetoucher


def _parse_providers(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run skin retouching with ONNXRuntime only.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--model-dir", default=".", help="Directory containing ONNX model files.")
    parser.add_argument("--retouch-degree", type=float, default=0.7, help="Main retouch blend strength.")
    parser.add_argument("--whitening-degree", type=float, default=0.8, help="Whitening blend strength.")
    parser.add_argument("--enable-local", action="store_true", help="Enable local blemish detection/inpainting.")
    parser.add_argument(
        "--providers",
        default=None,
        help=(
            "Comma-separated ONNXRuntime providers. Defaults to CPUExecutionProvider. "
            "Use CUDAExecutionProvider,CPUExecutionProvider only when CUDA/cuDNN are installed."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read input image: {args.input}")

    retoucher = SkinRetoucher(
        model_dir=args.model_dir,
        providers=_parse_providers(args.providers),
        retouch_degree=args.retouch_degree,
        whitening_degree=args.whitening_degree,
        enable_local=args.enable_local,
    )
    result = retoucher.retouch(image)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), result)
    if not ok:
        raise RuntimeError(f"failed to write output image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
