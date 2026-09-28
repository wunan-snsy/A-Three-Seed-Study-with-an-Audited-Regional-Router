"""Compact RGB/RGB-D saliency predictors and UCRR regional utility router."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    """ResNet-18 feature pyramid with optional ImageNet initialization."""
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        if in_channels != 3:
            old = model.conv1
            conv = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride,
                             old.padding, bias=False)
            with torch.no_grad():
                conv.weight[:, :1].copy_(old.weight.mean(dim=1, keepdim=True))
                if in_channels > 1:
                    conv.weight[:, 1:].zero_()
            model.conv1 = conv
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = model.layer1, model.layer2, model.layer3, model.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        a = self.layer1(x); b = self.layer2(a); c = self.layer3(b); d = self.layer4(c)
        return [a, b, c, d]


class PyramidDecoder(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int = 32):
        super().__init__()
        self.proj = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.fuse = nn.ModuleList(block(out_channels * 2, out_channels) for _ in range(3))
        self.pred = nn.Conv2d(out_channels, 1, 1)

    def forward(self, feats: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        xs = [p(f) for p, f in zip(self.proj, feats)]
        x = xs[-1]
        for i, lateral in zip((0, 1, 2), reversed(xs[:-1])):
            x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            x = self.fuse[i](torch.cat((x, lateral), dim=1))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.pred(x))


class RGBReference(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = Encoder(3, pretrained)
        self.decoder = PyramidDecoder([64, 128, 256, 512])

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(rgb), rgb.shape[-2:])


class RGBDCandidate(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.rgb_encoder = Encoder(3, pretrained)
        self.depth_encoder = Encoder(2, pretrained)
        self.rgb_proj = nn.ModuleList(nn.Conv2d(c, 32, 1) for c in (64, 128, 256, 512))
        self.depth_proj = nn.ModuleList(nn.Conv2d(c, 32, 1) for c in (64, 128, 256, 512))
        self.fuse = nn.ModuleList(block(65, 32) for _ in range(4))
        self.decoder = PyramidDecoder([32, 32, 32, 32])

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        rf, df = self.rgb_encoder(rgb), self.depth_encoder(torch.cat((depth, valid), dim=1))
        fused = []
        for a, b, rp, dp, layer in zip(rf, df, self.rgb_proj, self.depth_proj, self.fuse):
            v = F.interpolate(valid, size=a.shape[-2:], mode="nearest")
            fused.append(layer(torch.cat((rp(a), dp(b), v), dim=1)))
        return self.decoder(fused, rgb.shape[-2:])


def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


class UtilityRouter(nn.Module):
    def __init__(self, region_size: int = 16):
        super().__init__()
        self.region_size = region_size
        self.net = nn.Sequential(block(8, 32), block(32, 32), nn.Conv2d(32, 1, 3, padding=1))

    def features(self, ref: torch.Tensor, cand: torch.Tensor, depth: torch.Tensor,
                 valid: torch.Tensor) -> torch.Tensor:
        dx = F.pad(depth[..., :, 1:] - depth[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(depth[..., 1:, :] - depth[..., :-1, :], (0, 0, 0, 1))
        vx = valid * F.pad(valid[..., :, 1:], (0, 1, 0, 0))
        vy = valid * F.pad(valid[..., 1:, :], (0, 0, 0, 1))
        grad = dx.abs() * vx + dy.abs() * vy
        pix = torch.cat((ref, cand, depth, valid, (ref - cand).abs(), entropy(ref), entropy(cand), grad), dim=1)
        return F.avg_pool2d(pix, self.region_size, self.region_size)

    def forward(self, ref: torch.Tensor, cand: torch.Tensor, depth: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        return self.predict_features(self.features(ref, cand, depth, valid))

    def predict_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


def utility_target(ref: torch.Tensor, cand: torch.Tensor, gt: torch.Tensor,
                   region_size: int = 16) -> torch.Tensor:
    utility = F.avg_pool2d((ref - gt).abs() - (cand - gt).abs(), region_size, region_size)
    return ((utility + 1.0) / 2.0).clamp(0, 1)


def route_predictions(ref: torch.Tensor, cand: torch.Tensor, score: torch.Tensor,
                      valid: torch.Tensor, threshold: float, region_size: int = 16):
    usable = F.avg_pool2d(valid, region_size, region_size) >= 0.5
    choose = (2 * score - 1 > threshold) & usable
    expanded = F.interpolate(choose.float(), size=ref.shape[-2:], mode="nearest")
    return expanded * cand + (1 - expanded) * ref, choose


def segmentation_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, gt)
    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = (pred + gt - pred * gt).sum(dim=(1, 2, 3))
    iou = ((inter + 1) / (union + 1)).mean()
    return bce + 1 - iou
