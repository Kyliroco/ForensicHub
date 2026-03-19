import torch
import jpegio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable
from .fph import FPH
from .dwt import DWTFPN
from .backbone_convnext import ConvNeXt
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from functools import partial
from typing import Optional, Union, List, Dict, Any

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_norm_layer(
        use_norm: Union[bool, str, Dict[str, Any]], out_channels: int
) -> nn.Module:
    supported_norms = ("inplace", "batchnorm", "identity", "layernorm", "instancenorm")

    if use_norm is True:
        norm_params = {"type": "batchnorm"}
    elif use_norm is False:
        norm_params = {"type": "identity"}
    elif isinstance(use_norm, str):
        norm_str = use_norm.lower()
        if norm_str == "inplace":
            norm_params = {
                "type": "inplace",
                "activation": "leaky_relu",
                "activation_param": 0.0,
            }
        elif norm_str in supported_norms:
            norm_params = {"type": norm_str}
        else:
            raise ValueError(
                f"Unrecognized normalization type string provided: {use_norm}. Should be in "
                f"{supported_norms}"
            )
    elif isinstance(use_norm, dict):
        norm_params = use_norm
    else:
        raise ValueError(
            f"Invalid type for use_norm should either be a bool (batchnorm/identity), "
            f"a string in {supported_norms}, or a dict like {{'type': 'batchnorm', **kwargs}}"
        )

    if "type" not in norm_params:
        raise ValueError(
            f"Malformed dictionary given in use_norm: {use_norm}. Should contain key 'type'."
        )
    if norm_params["type"] not in supported_norms:
        raise ValueError(
            f"Unrecognized normalization type string provided: {use_norm}. Should be in {supported_norms}"
        )

    norm_type = norm_params["type"]
    norm_kwargs = {k: v for k, v in norm_params.items() if k != "type"}

    if norm_type == "inplace":
        norm = InPlaceABN(out_channels, **norm_kwargs)
    elif norm_type == "batchnorm":
        norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
    elif norm_type == "identity":
        norm = nn.Identity()
    elif norm_type == "layernorm":
        norm = nn.LayerNorm(out_channels, **norm_kwargs)
    elif norm_type == "instancenorm":
        norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        raise ValueError(f"Unrecognized normalization type: {norm_type}")

    return norm


def weighted_statistics_pooling(x, log_w=None):
    """Compute weighted statistics pooling over spatial dimensions.

    Computes min, max, mean and mean-square statistics using soft weights
    derived from log_w. When log_w is None, uniform weights are used.
    """
    b = x.shape[0]
    c = x.shape[1]
    x = x.view(b, c, -1)

    if log_w is None:
        log_w = torch.zeros((b, 1, x.shape[-1]), device=x.device)
    else:
        assert log_w.shape[0] == b
        assert log_w.shape[1] == 1
        log_w = log_w.view(b, 1, -1)
        assert log_w.shape[-1] == x.shape[-1]

    log_w = F.log_softmax(log_w, dim=-1)
    x_min = -torch.logsumexp(log_w - x, dim=-1)
    x_max = torch.logsumexp(log_w + x, dim=-1)

    w = torch.exp(log_w)
    x_avg = torch.sum(w * x, dim=-1)
    x_msq = torch.sum(w * x * x, dim=-1)

    return torch.cat((x_min, x_max, x_avg, x_msq), dim=1)


class TruForDetectionHead(nn.Module):
    """Image-level detection head inspiré de TruFor, avec confidence-weighted
    statistical pooling stabilisé.

    Les features sont normalisées par L2 avant le pooling pour éviter
    l'explosion des logsumexp. La confidence map est clampée pour éviter
    les underflows dans le log_softmax.

    Le output est un logit non sigmoïdé (compatible autocast + BCEWithLogits).
    """

    def __init__(self, in_channels: int, mid_channels: int = 256):
        super().__init__()
        self.feat_norm = nn.LayerNorm(in_channels)
        self.conf_map = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mid_channels, 1),
        )

    def _stable_weighted_pooling(
        self, x: torch.Tensor, log_w: torch.Tensor
    ) -> torch.Tensor:
        """Weighted statistics pooling avec stabilisation numérique.

        Args:
            x: Features normalisées de shape (B, C, N).
            log_w: Log-poids de shape (B, 1, N), après log_softmax.

        Returns:
            Tenseur concatené (x_min, x_max, x_mean, x_msq) de shape (B, 4*C).
        """
        log_w = torch.clamp(log_w, min=-20.0)

        x_min = -torch.logsumexp(log_w - x, dim=-1)
        x_max =  torch.logsumexp(log_w + x, dim=-1)

        w = torch.exp(log_w)
        x_avg = torch.sum(w * x,     dim=-1)
        x_msq = torch.sum(w * x * x, dim=-1)

        return torch.cat((x_min, x_max, x_avg, x_msq), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute image-level logit from spatial feature map.

        Args:
            features: Bottleneck feature map de shape (B, C, H, W).

        Returns:
            Logit tensor de shape (B, 1), non sigmoïdé.
        """
        b, c, h, w = features.shape

        # Normalisation L2 des features pour borner les valeurs dans logsumexp
        x = features.permute(0, 2, 3, 1)       # (B, H, W, C)
        x = self.feat_norm(x)                   # LayerNorm sur C
        x = x.permute(0, 3, 1, 2)              # (B, C, H, W)
        x = x.view(b, c, -1)                   # (B, C, N)

        log_w = self.conf_map(features)         # (B, 1, H, W)
        log_w = F.log_softmax(
            log_w.view(b, 1, -1), dim=-1        # (B, 1, N)
        )

        pooled = self._stable_weighted_pooling(x, log_w)   # (B, 4*C)
        return self.classifier(pooled)                      # (B, 1)

class PSCCNetDetectionHead(nn.Module):
    """Image-level detection head following PSCCNet's explicit detection branch.

    Applies Global Average Pooling over the bottleneck features followed by a
    two-layer MLP classifier, matching the detection head design in PSCCNet
    (Liu et al., 2022). The output score ∈ [0, 1], where 1 means manipulated.
    """

    def __init__(self, in_channels: int, mid_channels: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mid_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute image-level score from spatial feature map.

        Args:
            features: Bottleneck feature map of shape (B, C, H, W).

        Returns:
            Score tensor of shape (B, 1).
        """
        x = self.gap(features).flatten(1)
        return self.classifier(x)


class Conv2dReLU(nn.Sequential):
    """Conv2d followed by optional BatchNorm and ReLU activation."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int = 0,
            stride: int = 1,
            use_batchnorm: bool = True,
    ):
        if use_batchnorm:
            norm = get_norm_layer(use_batchnorm, out_channels)

        is_identity = isinstance(norm, nn.Identity)
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=is_identity,
        )
        activation = nn.ReLU(inplace=True)
        super(Conv2dReLU, self).__init__(conv, norm, activation)


class SCSEModule(nn.Module):
    """Concurrent spatial and channel squeeze-and-excitation module."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        """Apply channel and spatial excitation and sum the results."""
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """ConvNeXt-style depthwise convolution block with layer scale and stochastic depth."""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Apply depthwise conv, layer norm, pointwise MLPs and residual connection."""
        ipt = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return ipt + self.drop_path(x)


class AddCoords(nn.Module):
    """Appends normalized x, y (and optionally radial r) coordinate channels to the input."""

    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """Concatenate coordinate channels to input_tensor."""
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim, dtype=input_tensor.dtype),
                                    torch.arange(y_dim, dtype=input_tensor.dtype))
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim)
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class DecoderBlock(nn.Module):
    """Upsampling decoder block with optional skip connection concatenation."""

    def __init__(self, cin, cadd, cout):
        super().__init__()
        self.cin = (cin + cadd)
        self.cout = cout
        self.conv1 = Conv2dReLU(self.cin, self.cout, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv2 = Conv2dReLU(self.cout, self.cout, kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, x1, x2=None):
        """Upsample x1, optionally concatenate x2, and apply two conv layers."""
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:, :self.cin])
        return self.conv2(x1)


class ConvBNReLU(nn.Module):
    """Convolution with optional BatchNorm+ReLU and optional residual connection."""

    def __init__(self, in_c, out_c, ks, stride=1, norm=True, res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False)
        self.res = res

    def forward(self, x):
        """Apply convolution with optional residual addition."""
        if self.res:
            return x + self.conv(x)
        return self.conv(x)


class FUSE1(nn.Module):
    """Top-down feature fusion module across 4 scales."""

    def __init__(self, in_channels_list=(96, 192, 384, 768)):
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 1)
        self.c32 = ConvBNReLU(in_channels_list[3], in_channels_list[2], 1)
        self.c33 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 3)
        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)
        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        """Fuse features top-down from scale 3 to scale 0."""
        x, x1, x2, x3 = x
        h, w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3), size=(h, w)) + self.c31(x2))
        h, w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2), size=(h, w)) + self.c21(x1))
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w)) + self.c11(x))
        return x, x1, x2, x3


class FUSE2(nn.Module):
    """Top-down feature fusion module across 3 scales."""

    def __init__(self, in_channels_list=(96, 192, 384)):
        super(FUSE2, self).__init__()
        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)
        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        """Fuse features top-down from scale 2 to scale 0."""
        x, x1, x2 = x
        h, w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2), size=(h, w), mode='bilinear', align_corners=True) + self.c21(x1))
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w), mode='bilinear', align_corners=True) + self.c11(x))
        return x, x1, x2


class FUSE3(nn.Module):
    """Top-down feature fusion module across 2 scales."""

    def __init__(self, in_channels_list=(96, 192)):
        super(FUSE3, self).__init__()
        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        """Fuse features top-down from scale 1 to scale 0."""
        x, x1 = x
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w), mode='bilinear', align_corners=True) + self.c11(x))
        return x, x1


class MID(nn.Module):
    """Multi-scale iterative decoder with nested dense skip connections."""

    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [96]
        self.out_channels = decoder_channels
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1]
        )
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, *features):
        """Run nested dense decoding with multi-scale fusion."""
        decoder_features = {}
        features = self.fuse1(features)[::-1]
        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0], features[1])
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1], features[2])
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2], features[3])
        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2(
            (decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]))
        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](
            decoder_features["x_0_0"],
            torch.cat((decoder_features["x_1_1"], features[2]), 1)
        )
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](
            decoder_features["x_1_1"],
            torch.cat((decoder_features["x_2_2"], features[3]), 1)
        )
        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3(
            (decoder_features["x_1_2"], decoder_features["x_0_1"]))
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](
            decoder_features["x_0_1"],
            torch.cat((decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]), 1)
        )
        return self.decoder_convs["x_0_3"](
            torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"]), 1)
        )


@register_model("FFDN")
class FFDN(BaseModel):
    """Frequency Feature Decomposition Network for document tampering detection.

    Combines a ConvNeXt visual backbone with a frequency processing head (FPH)
    and a DWT-based feature pyramid decoder (DWTFPN) for pixel-level mask
    prediction. Optionally adds an image-level detection head via second_head.

    Args:
        decoder_channels: Output channel sizes for each decoder stage.
        classes: Number of output segmentation classes.
        weight_path: Path to pretrained ConvNeXt backbone weights.
        second_head: Detection head variant. One of {"trufor", "psccnet", ""}.
            - "trufor": Confidence-weighted statistical pooling head.
            - "psccnet": Global average pooling MLP head.
            - "": No image-level detection head (localization only).
    """

    # Channel dimension of the DWTFPN bottleneck output fed to the detection head
    _BOTTLENECK_CHANNELS = 256

    def __init__(
        self,
        decoder_channels=(384, 192, 96, 64),
        classes=2,
        weight_path='/mnt/data1/public_datasets/Doc/Hub/ffdn/convnext_small.pth',
        second_head="",
        first_head=True,
        freeze_for_det=False,
    ):
        super().__init__()
        self.vph = ConvNeXt()
        self.fph = FPH()
        self.second_head = second_head.lower()
        self.addcoords = AddCoords()
        self.FU = nn.Sequential(
            SCSEModule(448),
            nn.Conv2d(448, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.Conv2d(192, 192, 1, 1, 0),
        )
        self.FU[-1].weight.data.zero_()
        self.decoder = DWTFPN([96, 192, 384, 768], self._BOTTLENECK_CHANNELS)
        if first_head:
            self.head = nn.Sequential(
                nn.Conv2d(self._BOTTLENECK_CHANNELS, self._BOTTLENECK_CHANNELS, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(self._BOTTLENECK_CHANNELS, 2, 1, 1, 0),
            )
        else:
            self.head = None
        
        self.head_2 = self._build_detection_head(self.second_head)
        self.init_vph(weight_path)
        self.freeze_for_det = freeze_for_det
        if freeze_for_det:
            self._freeze_all_except_detection_head()
        
    def _freeze_all_except_detection_head(self) -> None:
        """Freeze backbone, FPH, decoder and seg head. Only head_2 trains.

        Called during phase-2 training so the detection head learns from
        stable, forensically-rich frozen features.
        """
        modules_to_freeze = [self.vph, self.fph, self.FU,
                            self.decoder, self.addcoords]
        if self.head is not None:
            modules_to_freeze.append(self.head)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
                
    def _build_detection_head(self, second_head: str) -> Optional[nn.Module]:
        """Instantiate the image-level detection head based on the selected variant.

        Args:
            second_head: One of {"trufor", "psccnet", ""}.

        Returns:
            The detection head module, or None if second_head is empty.
        """
        match second_head:
            case "trufor":
                return TruForDetectionHead(
                    in_channels=self._BOTTLENECK_CHANNELS,
                    mid_channels=self._BOTTLENECK_CHANNELS,
                )
            case "psccnet":
                return PSCCNetDetectionHead(
                    in_channels=self._BOTTLENECK_CHANNELS,
                    mid_channels=self._BOTTLENECK_CHANNELS,
                )
            case _:
                return None

    def init_vph(self, weight_path: str) -> None:
        """Load pretrained ConvNeXt backbone weights from checkpoint.

        Args:
            weight_path: Path to the checkpoint file containing 'state_dict'.
        """
        try:
            weights = torch.load(weight_path,weights_only=False)['state_dict']
        except:
            weights = torch.load(weight_path,weights_only=False)['model']
        dels = [k for k in weights.keys() if not k.startswith('backbone.')]
        for k in dels:
            del weights[k]
        new_weights = {'.'.join(k.split('.')[1:]): v for k, v in weights.items()}
        self.vph.load_state_dict(new_weights)

    def cal_seg_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        """Compute cross-entropy segmentation loss with bilinear upsampling.

        Args:
            pred: Raw logits of shape (B, C, H', W').
            gt: Ground-truth mask of shape (B, H, W) with long dtype.

        Returns:
            Tuple of (scalar loss, upsampled prediction of shape (B, C, H, W)).
        """
        h, w = gt.shape[-2:]
        pred = F.interpolate(pred, size=(h, w), mode='bilinear')
        ce_loss = F.cross_entropy(pred, gt)
        return ce_loss, pred

    def _compute_detection_loss(
        self, score: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross-entropy detection loss from image-level score.

        The image-level label is derived from the pixel mask: an image is
        considered manipulated if at least one pixel is forged.

        Args:
            score: Predicted score of shape (B, 1), values in [0, 1].
            mask: Ground-truth pixel mask of shape (B, H, W) with long dtype.

        Returns:
            Scalar BCE loss.
        """
        image_label = (mask.sum(dim=(-1, -2)) > 0).float().unsqueeze(1)
        return F.binary_cross_entropy_with_logits(score, image_label)

    def forward(self, image, dct, qt, mask, **kwargs):
        """Run the full forward pass: localization and optional detection.

        Args:
            image: RGB input tensor of shape (B, 3, H, W).
            dct: DCT coefficient map of shape (B, 1, H, W), long dtype.
            qt: Quantization table of shape (B, 1, 8, 8) or (B, 8, 8), long dtype.
            mask: Ground-truth mask of shape (B, 1, H, W), long dtype.

        Returns:
            Dict with keys: backward_loss, pred_mask, pred_score (optional),
            visual_loss, visual_image.
        """
        DCT_coef = dct
        qtables = qt
        x = image
        mask = mask.squeeze(1).long()
        DCT_coef = DCT_coef.squeeze(1).long()
        if len(qtables.shape) == 3:
            qtables = qtables.unsqueeze(1)

        features = self.vph.forward_features(x, end_index=2)
        features[1] = self.FU(torch.cat((features[1], self.fph(DCT_coef, qtables)), 1)) + features[1]
        features.extend(self.vph.forward_features(features[1], start_index=2, end_index=4))

        decoder_output = self.decoder(features)
        bottleneck = decoder_output[0]
        output_dict = {"visual_loss": {}, "visual_image": {}}
        if not self.training or (self.head is not None and not self.freeze_for_det):
            output = self.head(bottleneck)
            seg_loss, output = self.cal_seg_loss(output, mask)
            output_dict["pred_mask"] = F.softmax(output, dim=1)[:, 1:]
            output_dict["visual_loss"]["seg_loss"] = seg_loss
            output_dict["visual_image"]["pred_mask"] = F.softmax(output, dim=1)[:, 1:]
            total_loss = seg_loss
        else:
            total_loss = torch.tensor(0.0, device=image.device)

        # Phase 2 ou entraînement joint : tête de détection active
        if self.head_2 is not None:
            score = self.head_2(bottleneck)
            det_loss = self._compute_detection_loss(score, mask)
            total_loss = total_loss + 0.5 * det_loss
            output_dict["pred_label"] = torch.sigmoid(score)
            output_dict["visual_loss"]["det_loss"] = det_loss

        output_dict["backward_loss"] = total_loss
        return output_dict


if __name__ == "__main__":
    img = torch.ones((1, 3, 512, 512))
    mask = torch.ones((1, 1, 512, 512), dtype=torch.int64)
    dct = torch.ones((1, 1, 512, 512), dtype=torch.int64)
    qt = torch.ones((1, 1, 8, 8), dtype=torch.int64)

    for variant in ["", "psccnet", "trufor"]:
        model = FFDN(second_head=variant)
        pred = model(img, dct, qt, mask)
        score_info = f"  score={pred['pred_score'].item():.4f}" if "pred_score" in pred else ""
        print(f"[{variant or 'none':8s}] loss={pred['backward_loss'].item():.4f}{score_info}")