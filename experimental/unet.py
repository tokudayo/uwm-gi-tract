import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.LayerNorm):
    """Permute the input tensor so that the channel dimension is the last one."""
    def __init__(self, num_features: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(num_features, eps=eps, **kwargs)
    
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)


        if use_batchnorm:
            # bn = LayerNorm(out_channels)
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Identity()
        super().__init__(conv2d, upsampling, activation)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = nn.Identity()
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(self, channels=[16, 32, 64, 128, 256, 512]):
        super().__init__()
        channels = channels[::-1]
        blocks = []
        for i in range(3):
            blocks.append(
                DecoderBlock(
                    channels[i],
                    channels[i + 1],
                    channels[i + 1],
                    use_batchnorm=True,
                )
            )
        for i in range(3, len(channels) - 1):
            blocks.append(
                DecoderBlock(
                    channels[i],
                    0,
                    channels[i + 1],
                    use_batchnorm=True,
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, return_features=False):
        z = x[0]
        x = x[1:]
        features = []
        for i, block in enumerate(self.blocks):
            if i >= len(x):
                z = block(z)
            else:
                z = block(z, x[i])
                features.append(z.clone())
        if return_features:
            return features
        else:
            return z

class Unet(nn.Module):
    def __init__(self, encoder, channels=[16, 32, 64, 128, 256, 512]):
        super().__init__()
        self.encoder = encoder
        self.channels = channels
        self.decoder = UnetDecoder(channels)
        self.segmentation_head = SegmentationHead(
            in_channels=channels[0],
            out_channels=3,
            activation=None,
            kernel_size=3,
        )
    
    def forward(self, x):
        features = self.encoder(x)[::-1]
        x = self.decoder(features)
        x = self.segmentation_head(x)
        return x