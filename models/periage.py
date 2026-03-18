"""
PeriAge: Modified PeriGender architecture for age-range classification.

Predicts one of 10 age-decade buckets (1-10, 11-20, ..., 91-100)
from a periocular region image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
