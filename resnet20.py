
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    """Normalize CIFAR-10 images that arrive in raw [0, 1] pixel space."""

    def __init__(
        self,
        mean: tuple[float, float, float] = CIFAR10_MEAN,
        std: tuple[float, float, float] = CIFAR10_STD,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class CIFARResNet(nn.Module):
    """
    CIFAR-style ResNet.

    For ResNet-20:
        depth = 6n + 2  ->  20 = 6*3 + 2
        therefore n = 3 BasicBlocks per stage.
    """

    def __init__(self, depth: int = 20, num_classes: int = 10) -> None:
        super().__init__()

        if (depth - 2) % 6 != 0:
            raise ValueError("For CIFAR ResNet, depth must satisfy depth = 6n + 2")

        blocks_per_stage = (depth - 2) // 6
        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_stage(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_stage(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_stage(64, blocks_per_stage, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks: list[nn.Module] = []

        for block_stride in strides:
            blocks.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=block_stride,
                )
            )
            self.in_channels = out_channels * BasicBlock.expansion

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_resnet20(
    num_classes: int = 10,
    normalize_inside_model: bool = True,
) -> nn.Module:
    """
    Official staff model.

    Input:
        Float tensor [N, 3, 32, 32] in [0, 1]

    Output:
        Logits [N, 10]
    """
    classifier = CIFARResNet(depth=20, num_classes=num_classes)

    if normalize_inside_model:
        return nn.Sequential(
            Normalize(),
            classifier,
        )

    return classifier


def extract_state_dict(checkpoint: object) -> dict[str, Tensor]:
    """
    Accept either:
      1) a raw state_dict, or
      2) a checkpoint containing 'state_dict' or 'model_state_dict'.

    The official student guide asks students to submit a raw state_dict.
    """
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif (
            "model_state_dict" in checkpoint
            and isinstance(checkpoint["model_state_dict"], dict)
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError("The uploaded file does not contain a valid state_dict")

    cleaned: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue

        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        cleaned[new_key] = value

    return cleaned


def load_official_model(
    weights_path: str,
    device: str | torch.device = "cpu",
) -> nn.Module:
    model = build_resnet20(normalize_inside_model=True)

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
