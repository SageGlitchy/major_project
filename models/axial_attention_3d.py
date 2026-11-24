"""Module 3: Stage C1 — 3D Factorized Axial Self-Attention (`models/axial_attention_3d.py`).

Applies sequential 1D Multi-Head Self-Attention (MHSA) along the Depth (D), Height (H),
and Width (W) axes with learnable 1D positional encodings, LayerNorm, residual connections,
and optional gradient checkpointing (`torch.utils.checkpoint.checkpoint`) for memory
efficiency on 3D MRI feature volumes.

Dimensional progression for F in R^{B x C x D x H x W}:
- Depth axis (D):
    Permute/reshape F -> (B * H * W, D, C), add 1D pos encoding, apply LayerNorm +
    MultiheadAttention(embed_dim=C, num_heads=4, batch_first=True), reshape back to
    (B, C, D, H, W), and add residual connection.
- Height axis (H):
    Permute/reshape F -> (B * D * W, H, C), add 1D pos encoding, apply LayerNorm +
    MultiheadAttention(embed_dim=C, num_heads=4, batch_first=True), reshape back to
    (B, C, D, H, W), and add residual connection.
- Width axis (W):
    Permute/reshape F -> (B * D * H, W, C), add 1D pos encoding, apply LayerNorm +
    MultiheadAttention(embed_dim=C, num_heads=4, batch_first=True), reshape back to
    (B, C, D, H, W), and add residual connection.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AxialAttention3D(nn.Module):
    """Factorized 3D Axial Self-Attention sequentially across Depth (D), Height (H), and Width (W)."""

    def __init__(
        self,
        channels: int,
        spatial_dim: Union[int, Tuple[int, int, int]] = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )

        if isinstance(spatial_dim, int):
            d_dim, h_dim, w_dim = spatial_dim, spatial_dim, spatial_dim
        elif len(spatial_dim) == 3:
            d_dim, h_dim, w_dim = int(spatial_dim[0]), int(spatial_dim[1]), int(spatial_dim[2])
        else:
            raise ValueError(f"Invalid spatial_dim: {spatial_dim}")

        self.channels = channels
        self.spatial_dim = (d_dim, h_dim, w_dim)
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        # Learnable 1D positional encodings along D, H, and W
        self.pos_embed_d = nn.Parameter(torch.randn(1, d_dim, channels) * 0.02)
        self.pos_embed_h = nn.Parameter(torch.randn(1, h_dim, channels) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, w_dim, channels) * 0.02)

        # Depth-axis LayerNorm and MultiheadAttention
        self.norm_d = nn.LayerNorm(channels)
        self.attn_d = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Height-axis LayerNorm and MultiheadAttention
        self.norm_h = nn.LayerNorm(channels)
        self.attn_h = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Width-axis LayerNorm and MultiheadAttention
        self.norm_w = nn.LayerNorm(channels)
        self.attn_w = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @staticmethod
    def _get_1d_pos_embed(pos_embed: torch.Tensor, target_len: int) -> torch.Tensor:
        """Return (1, target_len, C) positional embedding, interpolating only if target_len differs."""
        if pos_embed.shape[1] == target_len:
            return pos_embed
        # pos_embed: (1, L, C) -> permute to (1, C, L) for 1D linear interpolation -> (1, target_len, C)
        pos_t = pos_embed.permute(0, 2, 1)
        pos_interp = F.interpolate(pos_t, size=target_len, mode="linear", align_corners=False)
        return pos_interp.permute(0, 2, 1)

    def _forward_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D self-attention along Depth (D) axis with residual connection."""
        B, C, D, H, W = x.shape
        # Permute (B, C, D, H, W) -> (B, H, W, D, C) -> reshape to (B * H * W, D, C)
        seq_d = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, D, C)
        pos_d = self._get_1d_pos_embed(self.pos_embed_d, D)
        qkv_d = self.norm_d(seq_d + pos_d)
        attn_out_d, _ = self.attn_d(qkv_d, qkv_d, qkv_d, need_weights=False)
        # Reshape back to (B, H, W, D, C) -> permute to (B, C, D, H, W)
        attn_out_d = attn_out_d.view(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()
        return x + attn_out_d

    def _forward_height(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D self-attention along Height (H) axis with residual connection."""
        B, C, D, H, W = x.shape
        # Permute (B, C, D, H, W) -> (B, D, W, H, C) -> reshape to (B * D * W, H, C)
        seq_h = x.permute(0, 2, 4, 3, 1).contiguous().view(B * D * W, H, C)
        pos_h = self._get_1d_pos_embed(self.pos_embed_h, H)
        qkv_h = self.norm_h(seq_h + pos_h)
        attn_out_h, _ = self.attn_h(qkv_h, qkv_h, qkv_h, need_weights=False)
        # Reshape back to (B, D, W, H, C) -> permute to (B, C, D, H, W)
        attn_out_h = attn_out_h.view(B, D, W, H, C).permute(0, 4, 1, 3, 2).contiguous()
        return x + attn_out_h

    def _forward_width(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D self-attention along Width (W) axis with residual connection."""
        B, C, D, H, W = x.shape
        # Permute (B, C, D, H, W) -> (B, D, H, W, C) -> reshape to (B * D * H, W, C)
        seq_w = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H, W, C)
        pos_w = self._get_1d_pos_embed(self.pos_embed_w, W)
        qkv_w = self.norm_w(seq_w + pos_w)
        attn_out_w, _ = self.attn_w(qkv_w, qkv_w, qkv_w, need_weights=False)
        # Reshape back to (B, D, H, W, C) -> permute to (B, C, D, H, W)
        attn_out_w = attn_out_w.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return x + attn_out_w

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential axial self-attention across D -> H -> W."""
        out = self._forward_depth(x)
        out = self._forward_height(out)
        out = self._forward_width(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 3D Factorized Axial Attention.

        Args:
            x: 3D feature volume of shape (B, C, D, H, W).

        Returns:
            Output 3D feature volume of shape (B, C, D, H, W).
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D feature tensor (B, C, D, H, W), got {x.shape}.")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.shape[1]} in tensor of shape {x.shape}."
            )

        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)
