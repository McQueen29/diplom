import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class AudioResNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.layer1 = nn.Sequential(
            ResidualBlock(c1, c1, stride=1, dropout=dropout),
            ResidualBlock(c1, c1, stride=1, dropout=dropout),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2, dropout=dropout),
            ResidualBlock(c2, c2, stride=1, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, dropout=dropout),
            ResidualBlock(c3, c3, stride=1, dropout=dropout),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(c3, c4, stride=2, dropout=dropout),
            ResidualBlock(c4, c4, stride=1, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)
