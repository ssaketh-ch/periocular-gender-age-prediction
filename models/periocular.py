"""
PeriOcular: Multi-task model combining gender and age prediction.

Shares a single backbone (same as PeriGender) but has two output heads:
  - fc  → 2  classes (gender: female / male)
  - fc2 → 10 classes (age range, decade buckets)
"""

import torch
import torch.nn as nn
from models.perigender import (
    ResBlock, SkipConnection1, SkipConnection2, SkipConnection3, SkipConnection4
)


class PeriOcular(nn.Module):
    """
    PeriOcular multi-task model.

    Input:  (B, 3, 224, 112)
    Output: tuple (gender_logits, age_logits)
              gender_logits: (B, 2)
              age_logits:    (B, 10)

    Loss = CrossEntropy(gender) + CrossEntropy(age)
    Optimizer: Adam, lr=0.01
    """
    def __init__(self, num_gender_classes=2, num_age_classes=10):
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

        # Two output heads
        self.fc = nn.Linear(524, num_gender_classes)   # gender head
        self.fc2 = nn.Linear(524, num_age_classes)     # age head

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
        out = self.dropout(out)

        gender_out = self.fc(out)
        age_out = self.fc2(out)
        return gender_out, age_out
