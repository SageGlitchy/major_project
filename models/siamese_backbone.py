"""Module 4: Stage C2 — Weight-Shared Siamese 3D Backbone (`models/siamese_backbone.py`).

Processes gated scan pair X_{gated, 1} and X_{gated, 2} of shape (B, 1, 128, 128, 128)
through a single weight-shared instance of `ResAxial3DEncoder` f_theta(.):

Layer progression inside `ResAxial3DEncoder`:
- Stem: Conv3d(1, 16, kernel=3, stride=2, padding=1) + BatchNorm3d(16) + LeakyReLU(0.01)
  -> (B, 16, 64, 64, 64)
- Stage 1: ResBlock3D(16, 32) + MaxPool3d(2)
  -> (B, 32, 32, 32, 32)
- Stage 2: ResBlock3D(32, 64) + AxialAttention3D(64, spatial_dim=32) + MaxPool3d(2)
  -> (B, 64, 16, 16, 16)
- Stage 3: ResBlock3D(64, 128) + AxialAttention3D(128, spatial_dim=16) + MaxPool3d(2)
  -> (B, 128, 8, 8, 8)
- Global pool: AdaptiveAvgPool3d((1, 1, 1)) + Flatten()
  -> z_t in R^{B x 128}
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .axial_attention_3d import AxialAttention3D


class ResBlock3D(nn.Module):
    """3D Residual Block with two 3x3x3 convolutions, BatchNorm3d, LeakyReLU(0.01),
    and a 1x1x1 projection shortcut when channel dimensions or strides change."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self._init_weights(negative_slope)

    def _init_weights(self, negative_slope: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, a=negative_slope, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act2(out)
        return out


class ResAxial3DEncoder(nn.Module):
    """Single-branch 3D Res-Axial Encoder f_theta(.) mapping (B, 1, 128, 128, 128) -> (B, 128)."""

    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 16,
        stage1_channels: int = 32,
        stage2_channels: int = 64,
        stage3_channels: int = 128,
        stage2_spatial_dim: int = 32,
        stage3_spatial_dim: int = 16,
        num_heads: int = 4,
        axial_dropout: float = 0.0,
        leaky_relu_slope: float = 0.01,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = stage3_channels

        # Stem: Conv3d(1, 16, kernel=3, stride=2, padding=1) + BatchNorm3d(16) + LeakyReLU(0.01)
        # Output shape: (B, 16, 64, 64, 64)
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
        )

        # Stage 1: ResBlock3D(16, 32) + MaxPool3d(2) -> (B, 32, 32, 32, 32)
        self.stage1_block = ResBlock3D(
            stem_channels, stage1_channels, negative_slope=leaky_relu_slope
        )
        self.stage1_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Stage 2: ResBlock3D(32, 64) + AxialAttention3D(64, spatial_dim=32) + MaxPool3d(2)
        # -> (B, 64, 16, 16, 16)
        self.stage2_block = ResBlock3D(
            stage1_channels, stage2_channels, negative_slope=leaky_relu_slope
        )
        self.stage2_attn = AxialAttention3D(
            channels=stage2_channels,
            spatial_dim=stage2_spatial_dim,
            num_heads=num_heads,
            dropout=axial_dropout,
            use_checkpoint=use_checkpoint,
        )
        self.stage2_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Stage 3: ResBlock3D(64, 128) + AxialAttention3D(128, spatial_dim=16) + MaxPool3d(2)
        # -> (B, 128, 8, 8, 8)
        self.stage3_block = ResBlock3D(
            stage2_channels, stage3_channels, negative_slope=leaky_relu_slope
        )
        self.stage3_attn = AxialAttention3D(
            channels=stage3_channels,
            spatial_dim=stage3_spatial_dim,
            num_heads=num_heads,
            dropout=axial_dropout,
            use_checkpoint=use_checkpoint,
        )
        self.stage3_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Global pool: AdaptiveAvgPool3d((1, 1, 1)) + Flatten() -> (B, 128)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run encoder and return both final embedding z_t and intermediate stage tensors."""
        stem_out = self.stem(x)  # (B, 16, 64, 64, 64)

        s1_res = self.stage1_block(stem_out)  # (B, 32, 64, 64, 64)
        s1_out = self.stage1_pool(s1_res)  # (B, 32, 32, 32, 32)

        s2_res = self.stage2_block(s1_out)  # (B, 64, 32, 32, 32)
        s2_attn = self.stage2_attn(s2_res)  # (B, 64, 32, 32, 32)
        s2_out = self.stage2_pool(s2_attn)  # (B, 64, 16, 16, 16)

        s3_res = self.stage3_block(s2_out)  # (B, 128, 16, 16, 16)
        s3_attn = self.stage3_attn(s3_res)  # (B, 128, 16, 16, 16)
        s3_out = self.stage3_pool(s3_attn)  # (B, 128, 8, 8, 8)

        pooled = self.global_pool(s3_out)  # (B, 128, 1, 1, 1)
        z = self.flatten(pooled)  # (B, 128)

        intermediates = {
            "stem": stem_out,
            "stage1": s1_out,
            "stage2_pre_pool": s2_attn,
            "stage2": s2_out,
            "stage3_pre_pool": s3_attn,
            "stage3": s3_out,
            "z": z,
        }
        return z, intermediates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single gated 3D MRI scan into latent representation z_t in R^{B x 128}."""
        z, _ = self.forward_with_intermediates(x)
        return z


class Siamese3DBackbone(nn.Module):
    """Weight-shared Siamese 3D Res-Axial Backbone processing both baseline and follow-up scans."""

    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 16,
        stage1_channels: int = 32,
        stage2_channels: int = 64,
        stage3_channels: int = 128,
        stage2_spatial_dim: int = 32,
        stage3_spatial_dim: int = 16,
        num_heads: int = 4,
        axial_dropout: float = 0.0,
        leaky_relu_slope: float = 0.01,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        # Strict weight sharing: a single instance of ResAxial3DEncoder processes both timepoints
        self.encoder = ResAxial3DEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage1_channels=stage1_channels,
            stage2_channels=stage2_channels,
            stage3_channels=stage3_channels,
            stage2_spatial_dim=stage2_spatial_dim,
            stage3_spatial_dim=stage3_spatial_dim,
            num_heads=num_heads,
            axial_dropout=axial_dropout,
            leaky_relu_slope=leaky_relu_slope,
            use_checkpoint=use_checkpoint,
        )

    def encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single timepoint volume through the shared encoder."""
        return self.encoder(x)

    def forward(
        self,
        X_gated_1: torch.Tensor,
        X_gated_2: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, torch.Tensor]]],
    ]:
        """Process both gated scans through the shared 3D Res-Axial Encoder f_theta(.)."""
        if return_intermediates:
            z_1, inter_1 = self.encoder.forward_with_intermediates(X_gated_1)
            z_2, inter_2 = self.encoder.forward_with_intermediates(X_gated_2)
            return z_1, z_2, {"t1": inter_1, "t2": inter_2}

        z_1 = self.encoder(X_gated_1)
        z_2 = self.encoder(X_gated_2)
        return z_1, z_2
