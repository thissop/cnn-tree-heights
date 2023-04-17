import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_label_channel, layer_count=64, gaussian_noise=0.1, weight_file=None):
        super().__init__()
        self.input_label_channel = input_label_channel
        self.layer_count = layer_count

        self.inc = DoubleConv(1, layer_count)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(layer_count, 2 * layer_count))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(2 * layer_count, 4 * layer_count))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(4 * layer_count, 8 * layer_count))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(8 * layer_count, 16 * layer_count))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(16 * layer_count, 8 * layer_count, kernel_size=2, stride=2), DoubleConv(16 * layer_count, 8 * layer_count))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(8 * layer_count, 4 * layer_count, kernel_size=2, stride=2), DoubleConv(8 * layer_count, 4 * layer_count))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(4 * layer_count, 2 * layer_count, kernel_size=2, stride=2), DoubleConv(4 * layer_count, 2 * layer_count))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(2 * layer_count, layer_count, kernel_size=2, stride=2), DoubleConv(2 * layer_count, layer_count))
        self.outc = nn.Conv2d(layer_count, len(input_label_channel), kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x + x4)
        x = self.up3(x + x3)
        x = self.up4(x + x2)
        x = self.outc(x + x1)
        return torch.sigmoid(x)

'''

'''