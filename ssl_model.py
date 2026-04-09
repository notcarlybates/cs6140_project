"""
ResNet-V2 1D backbone for accelerometer-based activity recognition.

Architecture: 18-layer pre-activation ResNet with 1D convolutions.
Input:  (N, 3, 300)  — 3 axes × 300 samples (30 Hz × 10 s)
Output: (N, 1024)    — shared feature vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock1D(nn.Module):
    """Pre-activation residual block (ResNet V2) with 1D convolutions.

    Order: BN → ReLU → Conv1d → BN → ReLU → Conv1d
    The projection shortcut operates on the pre-activated input to preserve
    the identity-mapping property (He et al., 2016b).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet1D(nn.Module):
    """
    1D ResNet-V2 (pre-activation) backbone.

    Follows the ResNet-18 block structure (2-2-2-2) with 1D convolutions.
    Channel widths: 64 → 128 → 256 → 512, with a linear projection to
    `feature_dim` (default 1024) after global average pooling.
    Total parameters approximately 10 million including the projection.
    """

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        # Stem: downsample by 4× (stride-2 conv + maxpool)
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)
        # Final BN+ReLU required by V2 (pre-activation leaves the last block
        # without a trailing activation)
        self.final_bn = nn.BatchNorm1d(512)
        # 1024-dim feature projection
        self.fc = nn.Linear(512, feature_dim)

    def _make_layer(self, in_ch: int, out_ch: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        layers = [PreActBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(PreActBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.final_bn(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x)
