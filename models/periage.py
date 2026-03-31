"""
PeriAge: Modified PeriGender architecture for age-range classification.

Predicts one of 10 age-decade buckets (1-10, 11-20, ..., 91-100)
from a periocular region image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights

from models.perigender import ResBlock, SkipConnection1, SkipConnection2, SkipConnection3


class SkipConnection4(nn.Module):
    """
    Skip connection 4 for PeriAge — includes an upsample step
    to match spatial dimensions before concatenation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.resize = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.resize(self.pool(self.conv(x)))
        return x


class Upsample(nn.Module):
    """Transposed conv upsample used in the main path of PeriAge."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.pad(x, (0, 1, 0, 0))
        return x


class PeriAge(nn.Module):
    """
    PeriAge model for age-range classification from periocular images.

    Input:  (B, 3, 224, 224)
    Output: (B, 10) — logits over 10 age-decade buckets

    Age bucket mapping:
        0 → 1-10    1 → 11-20   2 → 21-30   3 → 31-40   4 → 41-50
        5 → 51-60   6 → 61-70   7 → 71-80   8 → 81-90   9 → 91-100
    """
    def __init__(self, num_classes=10):
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
        self.upsample = Upsample(512, 512)

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
        out = self.upsample(self.maxpool2(out))

        out = torch.cat([skip1, skip2, skip3, skip4, out], dim=1)
        out = self.avg_pool(out).squeeze()
        out = self.fc(self.dropout(out))
        return out


class PeriAgeV2(nn.Module):
    """
    A stronger age model that keeps the multiscale skip backbone but replaces
    the final classifier with a learnable fusion block and a 2-layer head.
    """

    def __init__(self, num_classes=10):
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
        self.upsample = Upsample(512, 512)

        self.fusion = nn.Sequential(
            nn.Conv2d(524, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
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
        out = self.upsample(self.maxpool2(out))

        out = torch.cat([skip1, skip2, skip3, skip4, out], dim=1)
        out = self.fusion(out)
        out = self.avg_pool(out)
        return self.head(out)


class PeriAgeResNet34(nn.Module):
    """
    Hybrid age model:
    - pretrained ResNet34 backbone
    - multiscale fusion from layer2/layer3/layer4
    - lightweight MLP head for age-bucket prediction
    """

    def __init__(self, num_classes=10, weights=ResNet34_Weights.DEFAULT):
        super().__init__()
        backbone = models.resnet34(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.proj2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(384),
            nn.Dropout(p=0.25),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        f2 = self.pool(self.proj2(x2))
        f3 = self.pool(self.proj3(x3))
        f4 = self.pool(self.proj4(x4))
        fused = torch.cat([f2, f3, f4], dim=1)
        return self.head(fused)

    def backbone_parameters(self):
        modules = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in modules:
            yield from module.parameters()

    def head_parameters(self):
        modules = [self.proj2, self.proj3, self.proj4, self.head]
        for module in modules:
            yield from module.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone_parameters():
            param.requires_grad = trainable
