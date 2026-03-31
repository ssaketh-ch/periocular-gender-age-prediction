"""
PeriGender: Custom CNN for periocular gender classification.

Architecture inspired by:
"Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features"
https://file.techscience.com/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class SkipConnection1(nn.Module):
    """Skip connection from layer 1: Conv(3x3) -> MaxPool(8,stride=8)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8, padding=1)

    def forward(self, x):
        return self.pool(self.conv(x))


class SkipConnection2(nn.Module):
    """Skip connection from layer 2: Conv(1x1) -> MaxPool(5, stride=4)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=3, kernel_size=1, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=4, padding=0)

    def forward(self, x):
        return self.pool(self.conv(x))


class SkipConnection3(nn.Module):
    """Skip connection from layer 3: Conv(1x1) -> MaxPool(4, stride=2)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=3, kernel_size=1, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)

    def forward(self, x):
        return self.pool(self.conv(x))


class SkipConnection4(nn.Module):
    """Skip connection from layer 4: Conv(1x1) -> MaxPool((1,2))"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

    def forward(self, x):
        return self.pool(self.conv(x))


class PeriGender(nn.Module):
    """
    PeriGender model for binary gender classification from periocular images.

    Input:  (B, 3, 224, 112)
    Output: (B, 2)  — [female, male] logits

    Architecture:
        Conv -> MaxPool
        -> [Skip1 || ResBlock1]
        -> [Skip2 || ResBlock2]
        -> [Skip3 || ResBlock3]
        -> [Skip4 || MaxPool2]
        -> Concat all skip outputs
        -> AdaptiveAvgPool -> Dropout -> FC(2)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.skip1 = SkipConnection1(64, 64)
        self.resblock1 = nn.Sequential(ResBlock(64, 64, stride=2), ResBlock(64, 128))

        self.skip2 = SkipConnection2(128, 128)
        self.resblock2 = nn.Sequential(ResBlock(128, 128, stride=2), ResBlock(128, 256))

        self.skip3 = SkipConnection3(256, 256)
        self.resblock3 = nn.Sequential(ResBlock(256, 256, stride=2), ResBlock(256, 512))

        self.skip4 = SkipConnection4(512, 4)
        self.resblock4 = nn.Sequential(ResBlock(512, 512, stride=2), ResBlock(512, 512))

        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(524, num_classes)

    def forward(self, x):
        out = self.maxpool1(self.conv(x))

        skip1 = self.skip1(out)
        out = self.resblock1(out)

        skip2 = self.skip2(out)
        out = self.resblock2(out)

        skip3 = self.skip3(out)
        out = self.resblock3(out)

        skip4 = self.skip4(out)
        out = self.maxpool2(out)

        out = torch.cat([skip1, skip2, skip3, skip4, out], dim=1)
        out = self.avg_pool(out).squeeze()
        out = self.fc(self.dropout(out))
        return out


class PeriGenderV2(nn.Module):
    """
    Improved gender model with learnable skip fusion and a stronger classifier head.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.skip1 = SkipConnection1(64, 64)
        self.resblock1 = nn.Sequential(ResBlock(64, 64, stride=2), ResBlock(64, 128))

        self.skip2 = SkipConnection2(128, 128)
        self.resblock2 = nn.Sequential(ResBlock(128, 128, stride=2), ResBlock(128, 256))

        self.skip3 = SkipConnection3(256, 256)
        self.resblock3 = nn.Sequential(ResBlock(256, 256, stride=2), ResBlock(256, 512))

        self.skip4 = SkipConnection4(512, 4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

        self.fusion = nn.Sequential(
            nn.Conv2d(524, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.maxpool1(self.conv(x))

        skip1 = self.skip1(out)
        out = self.resblock1(out)

        skip2 = self.skip2(out)
        out = self.resblock2(out)

        skip3 = self.skip3(out)
        out = self.resblock3(out)

        skip4 = self.skip4(out)
        out = self.maxpool2(out)

        out = torch.cat([skip1, skip2, skip3, skip4, out], dim=1)
        out = self.fusion(out)
        out = self.avg_pool(out)
        return self.head(out)
