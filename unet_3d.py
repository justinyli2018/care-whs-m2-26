import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super(EncoderBlock3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels, dropout_p)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv_block(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, dropout_p: float = 0.0):
        super(DecoderBlock3D, self).__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=2, stride=2, bias=True
        )
        self.conv_block = ConvBlock3D(in_channels + skip_channels, skip_channels, dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        dropout_p: float = 0.0,
    ):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        channels = [base_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 4]

        self.initial_conv = ConvBlock3D(in_channels, channels[0], dropout_p)

        self.encoder1 = EncoderBlock3D(channels[0], channels[1], dropout_p)
        self.encoder2 = EncoderBlock3D(channels[1], channels[2], dropout_p)
        self.encoder3 = EncoderBlock3D(channels[2], channels[3], dropout_p)
        self.encoder4 = EncoderBlock3D(channels[3], channels[4], dropout_p)

        self.bottleneck = ConvBlock3D(channels[4], channels[4], dropout_p)

        self.decoder4 = DecoderBlock3D(channels[4], channels[4], dropout_p)
        self.decoder3 = DecoderBlock3D(channels[4], channels[3], dropout_p)
        self.decoder2 = DecoderBlock3D(channels[3], channels[2], dropout_p)
        self.decoder1 = DecoderBlock3D(channels[2], channels[1], dropout_p)

        self.final_conv = nn.Conv3d(channels[0], num_classes, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.initial_conv(x)

        skip1, x = self.encoder1(x0)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.final_conv(x)


def create_unet_3d(
    in_channels: int = 1,
    num_classes: int = 2,
    base_channels: int = 64,
    dropout_p: float = 0.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> UNet3D:
    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout_p=dropout_p
    )
    return model.to(device)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing UNet3D on device: {device}")

    model = create_unet_3d(in_channels=1, num_classes=2, device=device)

    dummy_input = torch.randn(2, 1, 64, 64, 64, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
