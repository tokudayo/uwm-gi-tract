from functools import partial

import torch
from torch import nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from experimental.convnext import LayerNorm


class Permute(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.permute(*self.dim)


class Encoder(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(out_indices)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # TODO: Fix
        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Decoder(nn.Module):
    def __init__(self, depths=[1, 2, 1, 1, 1], dims=[256, 128, 64, 32, 32, 16], 
                 encoder_dims=[512, 256, 128, 64], layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3, 4]
                 ):
        super().__init__()

        self.n_stages = len(depths)
        tdims = dims.copy()

        for i in range(self.n_stages):
            if i + 1 < len(encoder_dims):
                tdims[i] += encoder_dims[i + 1]

        self.upsample_layers = nn.ModuleList()

        self.upsample_layers.append(
            nn.Sequential(
                Permute(0, 2, 3, 1),
                nn.Linear(encoder_dims[0], dims[0]),
                LayerNorm(dims[0], eps=1e-6),
                # nn.ConvTranspose2d(encoder_dims[0], dims[0], kernel_size=2, stride=2),
                
                Permute(0, 3, 1, 2),
                nn.UpsamplingNearest2d(scale_factor=2),
            )
        )

        for i in range(self.n_stages - 1):
            upsample_layer = nn.Sequential(
                Permute(0, 2, 3, 1),
                nn.Linear(tdims[i], dims[i+1]),
                LayerNorm(dims[i+1], eps=1e-6),
                
                Permute(0, 3, 1, 2),
                nn.UpsamplingNearest2d(scale_factor=2),
                # LayerNorm(tdims[i], eps=1e-6, data_format="channels_first"),
                # nn.ConvTranspose2d(tdims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)
        
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.n_stages):
            stage = nn.Sequential(
                *[Block(dim=tdims[i],
                layer_scale_init_value=layer_scale_init_value) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(out_indices)):
            layer = norm_layer(tdims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_enc):
        x_enc = x_enc[::-1]
        outs = []
        x = x_enc[0]
        for i in range(self.n_stages):
            x = self.upsample_layers[i](x)
            if i + 1 < len(x_enc):
                x = torch.concat([x, x_enc[i+1]], 1)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
            

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class UNext(nn.Module):
    def __init__(self, n_class=3):
        super().__init__()
        self.encoder = Encoder(in_chans=3, depths=[1, 2, 4, 1], dims=[64, 128, 256, 512], drop_path_rate=0.2)
        self.decoder = Decoder(depths=[1, 4, 2, 1, 1], dims=[256, 128, 64, 32, 32, 16], 
        encoder_dims=[512, 256, 128, 64], layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3, 4])
        self.head = nn.Conv2d(in_channels=32, out_channels=n_class, kernel_size=1, stride=1, padding=0)
        # TODO: Seg head
        # TODO: Check
        # TODO: Self distillation: takes final output and output from branches and perform los
    def forward(self, x):
        x_enc = self.encoder(x)
        x = self.decoder(x_enc)
        x = self.head(x[-1])
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x