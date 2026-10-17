# models/resnet1d.py
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))


class ECGResNet(nn.Module):
    """
    Lightweight 1D ResNet for 12-lead ECG multilabel classification.
    Input:  (batch, 12, 1000)
    Output: (batch, num_classes) — raw logits
    """
    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            ResBlock1D(64,  64,  dropout=dropout),
            ResBlock1D(64,  128, stride=2, dropout=dropout),
            ResBlock1D(128, 256, stride=2, dropout=dropout),
            ResBlock1D(256, 512, stride=2, dropout=dropout),
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)   # (batch, 512)
        return self.classifier(x)      # raw logits — no sigmoid here

    def get_logits(self, x):
        """Alias for forward — explicit name used in calibration."""
        return self.forward(x)
