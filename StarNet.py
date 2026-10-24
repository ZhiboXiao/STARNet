"""StarNet: grouped sensor encoding and U-shaped signal reconstruction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StarNet(nn.Module):
    """Map [B, C, S, T] to [B, 1, S * R, T].

    n_channels is C; the historical name n_classes denotes the expansion
    factor R, not semantic classes. sensors is S and must be supplied.
    """

    def __init__(self, n_channels, n_classes, bilinear=False, *, sensors):
        super().__init__()
        if min(n_channels, n_classes) < 1 or sensors < 16:
            raise ValueError("Channels and expansion must be positive; sensors must be >= 16.")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sensors = sensors

        self.inc = InConv(n_channels, 64, sensors)
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.apply(xavier_init)


    def forward(self, x):
        if x.ndim != 4 or x.shape[1:3] != (self.n_channels, self.sensors):
            raise ValueError(f"Expected [B, {self.n_channels}, {self.sensors}, T], got {tuple(x.shape)}")
        if x.shape[-1] < 16:
            raise ValueError("T must be >= 16 for four pooling stages.")
        if self.training and x.shape[0] * (self.sensors // 16) * (x.shape[-1] // 16) <= 1:
            raise ValueError("BatchNorm needs B * floor(S/16) * floor(T/16) > 1 during training.")
        batch, _, sensors, length = x.shape
        # Adjacent channels belong to one sensor in the grouped Conv1d input.
        x = x.permute(0, 2, 1, 3).reshape(batch, sensors * self.n_channels, length)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # output[b, 0, s * R + r, t] = logits[b, r, s, t]
        return logits.permute(0, 2, 1, 3).reshape(batch, 1, sensors * self.n_classes, length)

class DoubleConv(nn.Module):
    """Two convolution / BatchNorm / Tanh stages."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Tanh(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels - out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Align spatial dimensions with the encoder skip connection.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = DoubleConv(in_channels,in_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x+x1)
        return x2


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, sensors):
        super(InConv, self).__init__()
        self.sensors = sensors
        self.dim = 16
        self.conv1 = nn.Sequential(
            # Each group processes the channels of a single sensor.
            nn.Conv1d(in_channels=in_channels*sensors,
                     out_channels=4*sensors,
                     kernel_size=5,
                     padding=2,
                     groups=sensors),
            nn.BatchNorm1d(4*sensors),
            nn.Tanh(),
            nn.Conv1d(4*sensors,
                     self.dim*sensors,
                     kernel_size=3,
                     padding=1,
                     groups=sensors),
            nn.BatchNorm1d(self.dim*sensors),
            nn.Tanh(),
            nn.Conv1d(self.dim*sensors,
                     out_channels*sensors,
                     kernel_size=3,
                     padding=1,
                     groups=sensors)
        )
        self.conv2 = DoubleConv(out_channels, out_channels)
        self.conv3 = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        B, _, W = x.shape
        x1 = self.conv1(x)
        x1 = x1.view(B, self.sensors, -1, W)
        x1 = x1.permute(0,2,1,3)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1+x2)
        return x3
def xavier_init(module):
    if isinstance(module, nn.Conv2d):
        # 使用 Xavier 初始化（假设后续接 Tanh）
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('tanh'))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
