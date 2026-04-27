# Copyright (c) Alibaba, Inc. and its affiliates.
"""Export skin-retouching PyTorch weights to ONNX.

This script is intentionally isolated from the runtime path. It may depend on
PyTorch, but it does not import ModelScope. The existing ``model.onnx`` is
already the skin-mask model and is not regenerated here.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import utils as onnx_utils


GENERATOR_PT = "pytorch_model.pt"
LOCAL_PT = "joint_20210926.pth"
GENERATOR_ONNX = "retouch_generator.onnx"
LOCAL_DETECTION_ONNX = "local_detection.onnx"
LOCAL_INPAINTING_ONNX = "local_inpainting.onnx"
FACE_MODEL_DIR = "cv_resnet50_face-detection_retinaface"
FACE_PT = "pytorch_model.pt"
FACE_ONNX = "face_detector.onnx"


def check_export_dependencies() -> None:
    """Fail early with a clear message for export-only dependencies."""
    missing = []
    for package_name in ("onnx",):
        if importlib.util.find_spec(package_name) is None:
            missing.append(package_name)
    if missing:
        joined = " ".join(missing)
        raise RuntimeError(
            "Missing export dependency: "
            f"{', '.join(missing)}. Install it in the active environment with "
            f"`python -m pip install {joined}` and rerun this script. "
            "This dependency is only needed while exporting ONNX files; the "
            "ONNXRuntime inference path does not import it."
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.dsoutc4 = OutConv(256, n_classes)
        self.dsoutc3 = OutConv(128, n_classes)
        self.dsoutc2 = OutConv(64, n_classes)
        self.dsoutc1 = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        x0 = self.sigmoid(self.outc(x11))
        if self.deep_supervision:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode="bilinear")
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode="bilinear")
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode="bilinear")
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode="bilinear")
            return x0, x11, x22, x33, x44
        return x0


class ConvBNActiv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn: bool = True,
        sample: str = "none-3",
        activ: Optional[str] = "relu",
        bias: bool = False,
    ):
        super().__init__()
        if sample == "down-7":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias)
        elif sample == "down-5":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=bias)
        elif sample == "down-3":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, images):
        outputs = self.conv(images)
        if hasattr(self, "bn"):
            outputs = self.bn(outputs)
        if hasattr(self, "activation"):
            outputs = self.activation(outputs)
        return outputs


class DetectionUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, up_sampling_node: str = "nearest"):
        super().__init__()
        self.n_classes = n_classes
        self.up_sampling_node = up_sampling_node
        self.ec_images_1 = ConvBNActiv(n_channels, 64, bn=False, sample="down-3")
        self.ec_images_2 = ConvBNActiv(64, 128, sample="down-3")
        self.ec_images_3 = ConvBNActiv(128, 256, sample="down-3")
        self.ec_images_4 = ConvBNActiv(256, 512, sample="down-3")
        self.ec_images_5 = ConvBNActiv(512, 512, sample="down-3")
        self.ec_images_6 = ConvBNActiv(512, 512, sample="down-3")
        self.dc_images_6 = ConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_5 = ConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_4 = ConvBNActiv(512 + 256, 256, activ="leaky")
        self.dc_images_3 = ConvBNActiv(256 + 128, 128, activ="leaky")
        self.dc_images_2 = ConvBNActiv(128 + 64, 64, activ="leaky")
        self.dc_images_1 = nn.Conv2d(64 + n_channels, n_classes, kernel_size=1)

    def forward(self, input_images):
        ec_images = {"ec_images_0": input_images}
        ec_images["ec_images_1"] = self.ec_images_1(input_images)
        ec_images["ec_images_2"] = self.ec_images_2(ec_images["ec_images_1"])
        ec_images["ec_images_3"] = self.ec_images_3(ec_images["ec_images_2"])
        ec_images["ec_images_4"] = self.ec_images_4(ec_images["ec_images_3"])
        ec_images["ec_images_5"] = self.ec_images_5(ec_images["ec_images_4"])
        ec_images["ec_images_6"] = self.ec_images_6(ec_images["ec_images_5"])
        logits = ec_images["ec_images_6"]
        for idx in range(6, 0, -1):
            logits = F.interpolate(logits, scale_factor=2, mode=self.up_sampling_node)
            logits = torch.cat((logits, ec_images[f"ec_images_{idx - 1}"]), dim=1)
            logits = getattr(self, f"dc_images_{idx}")(logits)
        return logits


class GatedConvBNActiv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn: bool = True,
        sample: str = "none-3",
        activ: Optional[str] = "relu",
        bias: bool = False,
    ):
        super().__init__()
        if sample == "down-7":
            kernel_size, stride, padding = 7, 2, 3
        elif sample == "down-5":
            kernel_size, stride, padding = 5, 2, 2
        elif sample == "down-3":
            kernel_size, stride, padding = 3, 2, 1
        else:
            kernel_size, stride, padding = 3, 1, 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.gate = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        images = self.conv(x)
        gates = self.sigmoid(self.gate(x))
        if hasattr(self, "bn"):
            images = self.bn(images)
        if hasattr(self, "activation"):
            images = self.activation(images)
        return images * gates


class RetouchingNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, up_sampling_node: str = "nearest"):
        super().__init__()
        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node
        self.ec_images_1 = GatedConvBNActiv(in_channels, 64, bn=False, sample="down-3")
        self.ec_images_2 = GatedConvBNActiv(64, 128, sample="down-3")
        self.ec_images_3 = GatedConvBNActiv(128, 256, sample="down-3")
        self.ec_images_4 = GatedConvBNActiv(256, 512, sample="down-3")
        self.ec_images_5 = GatedConvBNActiv(512, 512, sample="down-3")
        self.ec_images_6 = GatedConvBNActiv(512, 512, sample="down-3")
        self.dc_images_6 = GatedConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_5 = GatedConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_4 = GatedConvBNActiv(512 + 256, 256, activ="leaky")
        self.dc_images_3 = GatedConvBNActiv(256 + 128, 128, activ="leaky")
        self.dc_images_2 = GatedConvBNActiv(128 + 64, 64, activ="leaky")
        self.dc_images_1 = GatedConvBNActiv(
            64 + in_channels,
            out_channels,
            bn=False,
            sample="none-3",
            activ=None,
            bias=True,
        )
        self.tanh = nn.Tanh()

    def forward(self, input_images, input_masks):
        ec_images = {"ec_images_0": torch.cat((input_images, input_masks), dim=1)}
        ec_images["ec_images_1"] = self.ec_images_1(ec_images["ec_images_0"])
        ec_images["ec_images_2"] = self.ec_images_2(ec_images["ec_images_1"])
        ec_images["ec_images_3"] = self.ec_images_3(ec_images["ec_images_2"])
        ec_images["ec_images_4"] = self.ec_images_4(ec_images["ec_images_3"])
        ec_images["ec_images_5"] = self.ec_images_5(ec_images["ec_images_4"])
        ec_images["ec_images_6"] = self.ec_images_6(ec_images["ec_images_5"])
        dc_images = ec_images["ec_images_6"]
        for idx in range(6, 0, -1):
            dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images = torch.cat((dc_images, ec_images[f"ec_images_{idx - 1}"]), dim=1)
            dc_images = getattr(self, f"dc_images_{idx}")(dc_images)
        return self.tanh(dc_images)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_ec_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


def conv_bn(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_bn_no_relu(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1x1(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return OrderedDict([("1", layer2), ("2", layer3), ("3", layer4)])


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels: int):
        super().__init__()
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels)
        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)

    def forward(self, inputs):
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        up3 = F.interpolate(output3, size=output2.shape[2:], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=output1.shape[2:], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return [output1, output2, output3]


class SSH(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv3x3 = self.conv3X3(input)
        conv5x5_1 = self.conv5X5_1(input)
        conv5x5 = self.conv5X5_2(conv5x5_1)
        conv7x7_2 = self.conv7X7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        return self.relu(out)


class ClassHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, phase: str = "test", in_channel: int = 256, out_channel: int = 256):
        super().__init__()
        self.phase = phase
        self.body = ResNetBackbone()
        self.fpn = FPN([512, 1024, 2048], out_channel)
        self.ssh1 = SSH(out_channel, out_channel)
        self.ssh2 = SSH(out_channel, out_channel)
        self.ssh3 = SSH(out_channel, out_channel)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=in_channel)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=in_channel)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=in_channel)

    @staticmethod
    def _make_class_head(fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2):
        return nn.ModuleList([ClassHead(inchannels, anchor_num) for _ in range(fpn_num)])

    @staticmethod
    def _make_bbox_head(fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2):
        return nn.ModuleList([BboxHead(inchannels, anchor_num) for _ in range(fpn_num)])

    @staticmethod
    def _make_landmark_head(fpn_num: int = 3, inchannels: int = 64, anchor_num: int = 2):
        return nn.ModuleList([LandmarkHead(inchannels, anchor_num) for _ in range(fpn_num)])

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(list(out.values()))
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        if self.phase == "train":
            return bbox_regressions, classifications, ldm_regressions
        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions


def _strip_module_prefix(state_dict):
    stripped = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        stripped[key] = value
    return stripped


def _load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return torch.load(str(path), map_location="cpu")


def export_generator(model_dir: Path, opset: int) -> Path:
    checkpoint = _load_checkpoint(model_dir / GENERATOR_PT)
    model = UNet(3, 3)
    state = checkpoint["generator"] if isinstance(checkpoint, dict) and "generator" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    output_path = model_dir / GENERATOR_ONNX
    dummy = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
    onnx_utils.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["pred_mg"],
        opset_version=opset,
        do_constant_folding=True,
    )
    return output_path


def export_local_models(model_dir: Path, opset: int) -> Tuple[Path, Path]:
    checkpoint = _load_checkpoint(model_dir / LOCAL_PT)

    detection_net = DetectionUNet(n_channels=3, n_classes=1)
    detection_net.load_state_dict(checkpoint["detection_net"])
    detection_net.eval()
    detection_path = model_dir / LOCAL_DETECTION_ONNX
    onnx_utils.export(
        detection_net,
        torch.zeros(1, 3, 768, 768, dtype=torch.float32),
        str(detection_path),
        input_names=["image"],
        output_names=["mask_logits"],
        opset_version=opset,
        do_constant_folding=True,
    )

    inpainting_net = RetouchingNet(in_channels=4, out_channels=3)
    inpainting_net.load_state_dict(checkpoint["inpainting_net"])
    inpainting_net.eval()
    inpainting_path = model_dir / LOCAL_INPAINTING_ONNX
    image_patch = torch.zeros(1, 3, 576, 576, dtype=torch.float32)
    mask_patch = torch.ones(1, 1, 576, 576, dtype=torch.float32)
    onnx_utils.export(
        inpainting_net,
        (image_patch, mask_patch),
        str(inpainting_path),
        input_names=["image", "mask"],
        output_names=["inpainted"],
        opset_version=opset,
        do_constant_folding=True,
    )
    return detection_path, inpainting_path


def export_face_detector(model_dir: Path, face_model_dir: Path, opset: int) -> Path:
    checkpoint = _load_checkpoint(face_model_dir / FACE_PT)
    model = RetinaFace(phase="test", in_channel=256, out_channel=256)
    model.load_state_dict(_strip_module_prefix(checkpoint))
    model.eval()

    output_path = model_dir / FACE_ONNX
    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    onnx_utils.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["loc", "conf", "landms"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "loc": {1: "num_priors"},
            "conf": {1: "num_priors"},
            "landms": {1: "num_priors"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export skin retouching PyTorch weights to ONNX.")
    parser.add_argument("--model-dir", default=".", help="Directory containing pytorch_model.pt and joint_20210926.pth.")
    parser.add_argument(
        "--face-model-dir",
        default=FACE_MODEL_DIR,
        help="Directory containing the RetinaFace pytorch_model.pt.",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument("--skip-local", action="store_true", help="Only export retouch_generator.onnx.")
    parser.add_argument("--skip-face", action="store_true", help="Do not export face_detector.onnx.")
    parser.add_argument("--only-face", action="store_true", help="Only export face_detector.onnx.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    check_export_dependencies()
    model_dir = Path(args.model_dir)
    face_model_dir = Path(args.face_model_dir)
    if not args.only_face:
        generator_path = export_generator(model_dir, args.opset)
        print(f"exported {generator_path}")
        if not args.skip_local:
            detection_path, inpainting_path = export_local_models(model_dir, args.opset)
            print(f"exported {detection_path}")
            print(f"exported {inpainting_path}")
    if not args.skip_face:
        face_path = export_face_detector(model_dir, face_model_dir, args.opset)
        print(f"exported {face_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
