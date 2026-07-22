from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# VDT: https://github.com/RERV/VDT/blob/main/models.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
"""Sequential diffusion model with a VMamba restoration backbone."""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers import DropPath
from einops import rearrange

from lib.models.vmamba_blocks import (
    BidirectionalTemporalMamba,
    HighFrequencyRefinementHead,
    MultiScaleVMambaSpatial,
)


def modulate(x, shift, scale, T):
    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DateEmbedder(nn.Module):
    """
    Modified from https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/positional_encoding.py
    """
    def __init__(self, d: int, T: int = 10000, repeat: int | None = None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * torch.div(torch.arange(offset, offset + d).float(), 2, rounding_mode='floor') / d
        )

    def forward(self, dates):

        self.denom = self.denom.to(dates.device)

        # B x T x C, where B is equal to batch_size * H * W
        sinusoid_table = (
            dates[:, :, None] / self.denom[None, None, :]
        )
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


#################################################################################
#                                 Core Denoising Transformer                    #
#################################################################################

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, SAR=None):
        B, N, C = x.shape
        # BNC -> BNH(C/H) -> BHN(C/H)
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(SAR).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(SAR).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Lets PyTorch select FlashAttention, memory-efficient attention, or
        # its math fallback for the active GPU. This avoids materializing the
        # B x heads x tokens x tokens attention matrix when a fused kernel is
        # available (notably on Ampere/Ada GPUs).
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=self.scale
        ).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CloudAwareAttention(nn.Module):
    """
    Self-attention that softly prefers clean key/value tokens.

    cloud_mask is expected to be a patch-level cloud fraction in [0, 1], where 1
    means fully cloudy/missing. It is used as a log-prior on attention keys.
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            cloud_bias_scale=2.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.cloud_bias_scale = cloud_bias_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cloud_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if cloud_mask is not None:
            clean_conf = (1.0 - cloud_mask.float()).clamp(1e-4, 1.0)
            key_bias = self.cloud_bias_scale * torch.log(clean_conf)
            attn = attn + key_bias[:, None, None, :]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            num_frames=16
    ):
        super().__init__()
        self.num_frames = num_frames
        self.norm_x = norm_layer(hidden_size)
        self.norm_SAR = norm_layer(hidden_size)
        self.spatial_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.temporal_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, c, SAR, sar_gate=None):
        # c   [B, M]
        # x   [(B T), N, M]
        # SAR [(B T), N, M]
        shift_ca, scale_ca, gate_ca = self.adaLN_modulation(c).chunk(3, dim=1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T

        x_ln = modulate(self.norm_x(x), shift_ca, scale_ca, self.num_frames)
        SAR = self.norm_SAR(SAR)

        # cross attention on spatial dimension
        res_cross = self.drop_path(self.spatial_attn(x_ln, SAR))

        # cross attention on temporal dimension
        res_cross = rearrange(res_cross, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)
        SAR = rearrange(SAR, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)
        res_cross = self.drop_path(self.temporal_attn(res_cross, SAR))

        # gate
        res_cross = rearrange(res_cross, '(b n) t m -> b (t n) m', b=B, t=T, n=N, m=M)
        res_cross = gate_ca.unsqueeze(1) * res_cross
        res_cross = rearrange(res_cross, 'b (t n) m -> (b t) n m', b=B, t=T, n=N, m=M)
        if sar_gate is not None:
            res_cross = res_cross * sar_gate.unsqueeze(-1)

        x = x + res_cross # (b t) n m

        return x

# x = torch.rand([16,128,256])
# SAR = torch.rand([16,128,256])
# c = torch.rand([1,256])
# module = CrossAttentionBlock(256, 8)
# out = module(x, c, SAR)
# print(out.shape)

class ConditionVMambaBlock(nn.Module):
    """Restoration block with cross-modal attention and VMamba state scans.

    Cross attention is retained only for aligned S1/S2 correspondence. Temporal
    and spatial self-attention are replaced by date-aware bidirectional Mamba
    and four-direction multi-scale visual selective scans.
    """

    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            num_frames=16,
            if_cross_attention=True,
            vmamba_expansion=1.5,
            vmamba_date_scale_days=45.0,
            vmamba_multiscale=True,
            **block_kwargs,
    ):
        super().__init__()
        self.CrossAttentionBlock = CrossAttentionBlock(
            hidden_size, num_heads, num_frames=num_frames
        )
        self.TemporalMamba = BidirectionalTemporalMamba(
            hidden_size,
            num_frames=num_frames,
            expansion=vmamba_expansion,
            date_scale_days=vmamba_date_scale_days,
        )
        self.SpatialVMamba = MultiScaleVMambaSpatial(
            hidden_size,
            num_frames=num_frames,
            expansion=vmamba_expansion,
            use_multiscale=vmamba_multiscale,
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size)
        )
        self.num_frames = num_frames
        self.ifCrossAttention = if_cross_attention

    def forward(
            self,
            x,
            c,
            SAR,
            cloud_mask=None,
            landcover_gates=None,
            dense_sar=None,
            date_values=None,
    ):
        total_frames, num_tokens, hidden_size = x.shape
        batch_size = total_frames // self.num_frames
        sar_gate = temporal_gate = spatial_gate = None
        if landcover_gates is not None:
            sar_gate = landcover_gates.get("sar")
            temporal_gate = landcover_gates.get("temporal")
            spatial_gate = landcover_gates.get("spatial")

        if self.ifCrossAttention and SAR is not None:
            x = self.CrossAttentionBlock(x, c, SAR, sar_gate=sar_gate)
        x = self.TemporalMamba(
            x,
            c,
            cloud_mask=cloud_mask,
            temporal_gate=temporal_gate,
            dense_sar=dense_sar,
            dates=date_values,
        )
        x = self.SpatialVMamba(
            x,
            c,
            cloud_mask=cloud_mask,
            spatial_gate=spatial_gate,
        )

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        mlp = self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp, self.num_frames))
        mlp = rearrange(
            mlp, '(b t) n m -> b (t n) m',
            b=batch_size, t=self.num_frames, n=num_tokens, m=hidden_size
        )
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(
            mlp, 'b (t n) m -> (b t) n m',
            b=batch_size, t=self.num_frames, n=num_tokens, m=hidden_size
        )
        return x + mlp

class FinalLayer(nn.Module):
    """
    The final layer of SDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class TargetQualityRefiner(nn.Module):
    """
    Estimates target reliability and builds an online temporal consensus target.

    This module operates inside the model during training. It does not remove
    samples or rewrite the dataset.
    """

    def __init__(
            self,
            hidden_channels=16,
            residual_temperature=0.12,
            temporal_scale_days=45.0,
            sar_temperature=0.5,
            correction_limit=2.0,
    ):
        super().__init__()
        self.residual_temperature = residual_temperature
        self.temporal_scale_days = temporal_scale_days
        self.sar_temperature = sar_temperature
        self.correction_limit = correction_limit

        self.reliability_net = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.reliability_net[-1].weight)
        nn.init.zeros_(self.reliability_net[-1].bias)

    def _date_weights(self, target, dates):
        B, T, _, H, W = target.shape
        dtype = target.dtype
        device = target.device

        if dates is None:
            date_distance = torch.arange(T, device=device, dtype=dtype)
            date_distance = (date_distance[:, None] - date_distance[None, :]).abs()
            date_distance = date_distance.unsqueeze(0).expand(B, -1, -1)
        else:
            date_distance = (dates[:, :, None] - dates[:, None, :]).abs().to(dtype)

        time_weight = torch.exp(-date_distance / max(self.temporal_scale_days, 1e-6))
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0)
        return time_weight * (1.0 - eye)

    def forward(self, target, cond=None, dates=None, known_cloud_mask=None):
        B, T, C, H, W = target.shape
        if known_cloud_mask is None:
            known_cloud_mask = torch.zeros(
                (B, T, 1, H, W), device=target.device, dtype=target.dtype
            )

        date_weight = self._date_weights(target, dates)
        known_clear = 1.0 - known_cloud_mask.float().clamp(0, 1)

        sar = None
        if cond is not None and cond.shape[1] == T:
            sar = cond.float()
            sar = torch.nn.functional.avg_pool2d(
                sar.reshape(B * T, sar.shape[2], H, W),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(B, T, sar.shape[2], H, W)

        consensus_sum = torch.zeros_like(target)
        denominator = torch.zeros((B, T, H, W), device=target.device, dtype=target.dtype)
        available_weight = torch.zeros_like(denominator)

        # Accumulate one reference frame at a time to avoid allocating a
        # B x T x T x H x W tensor.
        for reference_idx in range(T):
            pair_weight = date_weight[:, :, reference_idx, None, None]
            if sar is not None:
                sar_distance = (
                    sar - sar[:, reference_idx:reference_idx + 1]
                ).abs().mean(dim=2)
                pair_weight = pair_weight * torch.exp(
                    -sar_distance.to(target.dtype) / max(self.sar_temperature, 1e-6)
                )

            clear_weight = (
                pair_weight * known_clear[:, reference_idx, 0][:, None]
            )
            denominator = denominator + clear_weight
            available_weight = available_weight + pair_weight
            consensus_sum = consensus_sum + (
                clear_weight[:, :, None] * target[:, reference_idx:reference_idx + 1]
            )

        consensus = consensus_sum / denominator[:, :, None].clamp_min(1e-6)

        fallback = target.median(dim=1, keepdim=True).values.expand(-1, T, -1, -1, -1)
        has_reference = denominator[:, :, None] > 1e-6
        consensus = torch.where(has_reference, consensus, fallback)

        support = (denominator / available_weight.clamp_min(1e-6)).clamp(0, 1)
        support = support.unsqueeze(2)

        residual = (target - consensus).abs()
        residual_mean = residual.mean(dim=2, keepdim=True)
        residual_max = residual.amax(dim=2, keepdim=True)
        local_residual = torch.nn.functional.avg_pool2d(
            residual_mean.reshape(B * T, 1, H, W),
            kernel_size=5,
            stride=1,
            padding=2,
        ).reshape(B, T, 1, H, W)

        known_cloud = known_cloud_mask.float().clamp(0, 1)
        base_reliability = torch.exp(
            -residual_mean / max(self.residual_temperature, 1e-6)
        )
        base_reliability = (
            base_reliability * support + (1.0 - support)
        ).clamp(1e-4, 1.0 - 1e-4)
        calibration_base = (
            base_reliability * (1.0 - known_cloud) + 1e-4 * known_cloud
        ).clamp(1e-4, 1.0 - 1e-4)
        pseudo_reliability = calibration_base

        quality_features = torch.cat(
            [residual_mean, residual_max, local_residual, support, known_cloud], dim=2
        ).reshape(B * T, 5, H, W)
        correction = self.reliability_net(quality_features).reshape(B, T, 1, H, W)
        correction = self.correction_limit * torch.tanh(correction)

        base_logit = torch.logit(calibration_base)
        raw_reliability = torch.sigmoid(base_logit + correction)
        reliability = raw_reliability * (1.0 - known_cloud)

        return {
            'reliability': reliability,
            'raw_reliability': raw_reliability,
            'pseudo_reliability': pseudo_reliability.detach(),
            'consensus': consensus,
            'support': support,
        }


class LandCoverRouter(nn.Module):
    """
    Rule-guided soft land-cover router.

    The router does not require external land-cover labels. It estimates soft
    patch categories from S2 vegetation dynamics and SAR/optical structure,
    then maps them to separate gates for temporal, SAR cross, and spatial
    attention.
    """

    def __init__(
            self,
            red_index=2,
            nir_index=6,
            min_gate=0.25,
            built_temporal_gate=0.35,
            built_sar_gate=1.00,
            built_spatial_gate=1.00,
            crop_temporal_gate=1.00,
            crop_sar_gate=0.60,
            crop_spatial_gate=0.70,
            forest_temporal_gate=0.65,
            forest_sar_gate=0.50,
            forest_spatial_gate=0.80,
            other_temporal_gate=0.55,
            other_sar_gate=0.70,
            other_spatial_gate=0.65,
    ):
        super().__init__()
        self.red_index = red_index
        self.nir_index = nir_index
        self.min_gate = min_gate
        self.temporal_values = (
            built_temporal_gate,
            crop_temporal_gate,
            forest_temporal_gate,
            other_temporal_gate,
        )
        self.sar_values = (
            built_sar_gate,
            crop_sar_gate,
            forest_sar_gate,
            other_sar_gate,
        )
        self.spatial_values = (
            built_spatial_gate,
            crop_spatial_gate,
            forest_spatial_gate,
            other_spatial_gate,
        )

    @staticmethod
    def _to_unit_range(value):
        value = value.float()
        if value.detach().amin() < -0.05:
            value = (value + 1.0) / 2.0
        return value.clamp(0.0, 1.0)

    @staticmethod
    def _weighted_mean(value, valid):
        denom = valid.sum(dim=1).clamp_min(1e-6)
        return (value * valid).sum(dim=1) / denom

    @staticmethod
    def _weighted_change(value, valid):
        if value.shape[1] <= 1:
            return torch.zeros_like(value[:, 0])
        pair_valid = valid[:, 1:] * valid[:, :-1]
        diff = (value[:, 1:] - value[:, :-1]).abs()
        denom = pair_valid.sum(dim=1).clamp_min(1e-6)
        return (diff * pair_valid).sum(dim=1) / denom

    @staticmethod
    def _edge_strength(value):
        kernel_x = value.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 4.0
        kernel_y = kernel_x.transpose(-1, -2)
        grad_x = F.conv2d(value, kernel_x, padding=1)
        grad_y = F.conv2d(value, kernel_y, padding=1)
        return (grad_x.abs() + grad_y.abs()).clamp(0.0, 1.0)

    @staticmethod
    def _sigmoid_score(value, center, temperature):
        return torch.sigmoid((value - center) / max(temperature, 1e-6))

    def _class_probabilities(self, optical, cond=None, cloud_mask=None):
        B, T, C, H, W = optical.shape
        optical = self._to_unit_range(optical)
        if cloud_mask is None:
            clear = torch.ones((B, T, 1, H, W), device=optical.device, dtype=optical.dtype)
        else:
            clear = 1.0 - cloud_mask.float().clamp(0.0, 1.0)

        red_idx = min(max(self.red_index, 0), C - 1)
        nir_idx = min(max(self.nir_index, 0), C - 1)
        red = optical[:, :, red_idx:red_idx + 1]
        nir = optical[:, :, nir_idx:nir_idx + 1]
        ndvi = ((nir - red) / (nir + red + 1e-6)).clamp(-1.0, 1.0)
        mean_ndvi = self._weighted_mean(ndvi, clear)
        ndvi_change = self._weighted_change(ndvi, clear)
        mean_nir = self._weighted_mean(nir, clear)

        rgb_like = optical[:, :, :min(3, C)].mean(dim=2, keepdim=True)
        mean_rgb = self._weighted_mean(rgb_like, clear)
        optical_edge = self._edge_strength(mean_rgb)

        structure = optical_edge
        if cond is not None and cond.shape[1] == T:
            sar = self._to_unit_range(cond)
            sar_scalar = sar.mean(dim=2, keepdim=True)
            sar_valid = torch.ones_like(sar_scalar[:, :, :1])
            mean_sar = self._weighted_mean(sar_scalar, sar_valid)
            sar_edge = self._edge_strength(mean_sar)
            structure = torch.maximum(structure, sar_edge)

        low_veg = self._sigmoid_score(0.30 - mean_ndvi, 0.0, 0.08)
        low_change = self._sigmoid_score(0.08 - ndvi_change, 0.0, 0.035)
        structured = self._sigmoid_score(structure, 0.08, 0.035)
        built_score = low_veg * (0.4 + 0.6 * structured) * (0.5 + 0.5 * low_change)

        vegetated = self._sigmoid_score(mean_ndvi, 0.25, 0.08)
        seasonal = self._sigmoid_score(ndvi_change, 0.045, 0.025)
        crop_score = vegetated * seasonal

        high_veg = self._sigmoid_score(mean_ndvi, 0.38, 0.08)
        stable_veg = self._sigmoid_score(0.065 - ndvi_change, 0.0, 0.03)
        forest_score = high_veg * stable_veg

        low_nir = self._sigmoid_score(0.18 - mean_nir, 0.0, 0.08)
        water_or_shadow = low_nir * self._sigmoid_score(0.10 - mean_ndvi, 0.0, 0.08)
        other_score = 0.20 + water_or_shadow

        scores = torch.cat(
            [built_score, crop_score, forest_score, other_score], dim=1
        ).clamp_min(1e-4)
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _mix_gate(self, probabilities, values):
        gate_values = probabilities.new_tensor(values).view(1, 4, 1, 1)
        gate = (probabilities * gate_values).sum(dim=1, keepdim=True)
        return gate.clamp(self.min_gate, 1.0)

    def forward(self, optical, cond=None, cloud_mask=None):
        B, T, _, H, W = optical.shape
        probabilities = self._class_probabilities(optical, cond=cond, cloud_mask=cloud_mask)

        temporal_gate = self._mix_gate(probabilities, self.temporal_values)
        sar_gate = self._mix_gate(probabilities, self.sar_values)
        spatial_gate = self._mix_gate(probabilities, self.spatial_values)

        return {
            "landcover_probs": probabilities,
            "building_prob": probabilities[:, 0:1].unsqueeze(1).expand(-1, T, -1, -1, -1),
            "crop_prob": probabilities[:, 1:2].unsqueeze(1).expand(-1, T, -1, -1, -1),
            "forest_prob": probabilities[:, 2:3].unsqueeze(1).expand(-1, T, -1, -1, -1),
            "temporal_gate": temporal_gate.unsqueeze(1).expand(-1, T, -1, -1, -1),
            "sar_gate": sar_gate.unsqueeze(1).expand(-1, T, -1, -1, -1),
            "spatial_gate": spatial_gate.unsqueeze(1).expand(-1, T, -1, -1, -1),
        }


class SDT(nn.Module):
    """
    Sequential diffusion restoration model with a VMamba backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=3,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=False,
            num_frames=16,
            cond_in_channels=3,
            cross_attention=True,
            cloud_aware_attention=True,
            use_cloud_mask_embedding=True,
            target_quality_refinement=True,
            target_quality_hidden=16,
            target_quality_residual_temperature=0.12,
            target_quality_temporal_scale_days=45.0,
            target_quality_sar_temperature=0.5,
            landcover_router_enabled=False,
            landcover_min_gate=0.25,
            landcover_built_temporal_gate=0.35,
            landcover_built_sar_gate=1.00,
            landcover_built_spatial_gate=1.00,
            landcover_crop_temporal_gate=1.00,
            landcover_crop_sar_gate=0.60,
            landcover_crop_spatial_gate=0.70,
            landcover_forest_temporal_gate=0.65,
            landcover_forest_sar_gate=0.50,
            landcover_forest_spatial_gate=0.80,
            landcover_other_temporal_gate=0.55,
            landcover_other_sar_gate=0.70,
            landcover_other_spatial_gate=0.65,
            dense_sar_temporal_scale_days=18.0,
            dense_sar_topk=9,
            vmamba_expansion=1.5,
            vmamba_date_scale_days=45.0,
            detail_hidden_channels=64,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.cond_in_channels = cond_in_channels
        self.cloud_aware_attention = cloud_aware_attention
        self.use_cloud_mask_embedding = use_cloud_mask_embedding
        self.target_quality_refinement = target_quality_refinement
        self.landcover_router_enabled = landcover_router_enabled
        self.dense_sar_temporal_scale_days = dense_sar_temporal_scale_days
        self.dense_sar_topk = int(dense_sar_topk) if dense_sar_topk is not None else 0

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.cond_embedder = PatchEmbed(input_size, patch_size, cond_in_channels, hidden_size, bias=True)
        self.cloud_mask_embedder = PatchEmbed(input_size, patch_size, 1, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.date_embedder = DateEmbedder(hidden_size, T=10000)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # Class labels are not part of the satellite reconstruction objective.
        # Keep this legacy embedding out of DDP gradient synchronization.
        self.y_embedder.requires_grad_(False)
        self.target_quality_refiner = TargetQualityRefiner(
            hidden_channels=target_quality_hidden,
            residual_temperature=target_quality_residual_temperature,
            temporal_scale_days=target_quality_temporal_scale_days,
            sar_temperature=target_quality_sar_temperature,
        ) if target_quality_refinement else None
        self.landcover_router = LandCoverRouter(
            min_gate=landcover_min_gate,
            built_temporal_gate=landcover_built_temporal_gate,
            built_sar_gate=landcover_built_sar_gate,
            built_spatial_gate=landcover_built_spatial_gate,
            crop_temporal_gate=landcover_crop_temporal_gate,
            crop_sar_gate=landcover_crop_sar_gate,
            crop_spatial_gate=landcover_crop_spatial_gate,
            forest_temporal_gate=landcover_forest_temporal_gate,
            forest_sar_gate=landcover_forest_sar_gate,
            forest_spatial_gate=landcover_forest_spatial_gate,
            other_temporal_gate=landcover_other_temporal_gate,
            other_sar_gate=landcover_other_sar_gate,
            other_spatial_gate=landcover_other_spatial_gate,
        ) if landcover_router_enabled else None
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.num_frames = num_frames
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.time_drop = nn.Dropout(p=0)
        self.cross_attention = cross_attention


        self.blocks = nn.ModuleList(
            ConditionVMambaBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                num_frames=self.num_frames,
                if_cross_attention=self.cross_attention,
                vmamba_expansion=vmamba_expansion,
                vmamba_date_scale_days=vmamba_date_scale_days,
                vmamba_multiscale=True,
            )
            for _ in range(depth)
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.detail_refiner = HighFrequencyRefinementHead(
            out_channels=self.out_channels,
            input_channels=self.in_channels,
            cond_channels=self.cond_in_channels,
            hidden_channels=detail_hidden_channels,
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

        w = self.cloud_mask_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cloud_mask_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SDT blocks:
        for TSblock in self.blocks:
            if self.cross_attention:
                nn.init.constant_(TSblock.CrossAttentionBlock.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(TSblock.CrossAttentionBlock.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.TemporalMamba.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.TemporalMamba.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.SpatialVMamba.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.SpatialVMamba.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        self.detail_refiner.zero_init()

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _prepare_dense_dates(self, target_dates, source_dates, batch_size, target_length, source_length, device):
        if target_dates is None:
            target_dates = torch.arange(target_length, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            target_dates = target_dates.to(device)
            if target_dates.dim() == 1:
                target_dates = target_dates.unsqueeze(0).expand(batch_size, -1)

        if source_dates is None:
            source_dates = torch.arange(source_length, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            source_dates = source_dates.to(device)
            if source_dates.dim() == 1:
                source_dates = source_dates.unsqueeze(0).expand(batch_size, -1)

        return target_dates, source_dates

    def _aggregate_dense_sar(self, cond_dense, target_dates=None, source_dates=None):
        B, source_length, C, H, W = cond_dense.shape
        target_dates, source_dates = self._prepare_dense_dates(
            target_dates,
            source_dates,
            B,
            self.num_frames,
            source_length,
            cond_dense.device,
        )
        date_distance = (
            target_dates[:, :, None].float() - source_dates[:, None, :].float()
        ).abs()
        logits = -date_distance / max(float(self.dense_sar_temporal_scale_days), 1e-6)

        if 0 < self.dense_sar_topk < source_length:
            topk_indices = logits.topk(self.dense_sar_topk, dim=-1).indices
            topk_mask = torch.zeros_like(logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, True)
            logits = logits.masked_fill(~topk_mask, torch.finfo(logits.dtype).min)

        weights = logits.softmax(dim=-1).to(cond_dense.dtype)
        return torch.einsum('btd,bdchw->btchw', weights, cond_dense)

    def forward(
            self,
            x,
            t,
            date=None,
            cond=None,
            cond_dense=None,
            date_cond_dense=None,
            date_dense_target=None,
            cloud_mask=None,
            location=None,
            semantics=None,
            quality_target=None,
            quality_input_mask=None,
            quality_strength=1.0,
            quality_only=False,
            landcover_source=None,
            return_aux=False,
    ):
        """
        Forward pass of SDT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        date: (B, T) tensor of dates
        cond: same with x, conditional inputs such as SAR images
        cond_dense: denser SAR sequence, e.g. (B, 46, C_sar, H, W)
        quality_input_mask: simulated cloud mask used to build the diffusion input
        """

        quality_output = None
        if quality_target is not None and self.target_quality_refiner is not None:
            quality_output = self.target_quality_refiner(
                quality_target, cond=cond, dates=date, known_cloud_mask=cloud_mask
            )
            if quality_only:
                return quality_output
            if quality_input_mask is not None:
                if cloud_mask is None:
                    cloud_mask = torch.zeros_like(quality_input_mask)
                known_clear = 1.0 - cloud_mask.float().clamp(0.0, 1.0)
                strength = float(quality_strength)
                reliability = known_clear * (
                    (1.0 - strength) + strength * quality_output['reliability']
                )
                estimated_dirty = (1.0 - reliability).detach()
                quality_mask = torch.maximum(quality_input_mask, estimated_dirty)
                x = x * quality_mask + (1.0 - quality_mask) * quality_target
                cloud_mask = quality_mask
        elif quality_only:
            raise ValueError("quality_only=True requires target quality refinement and quality_target.")

        B, T, C, H, W = x.shape
        model_input_sequence = x
        condition_sequence = cond
        cloud_mask_sequence = cloud_mask
        landcover_output = None
        landcover_gates = None
        if self.landcover_router is not None:
            router_source = landcover_source if landcover_source is not None else x
            landcover_output = self.landcover_router(
                router_source, cond=cond, cloud_mask=cloud_mask_sequence
            )
            patch_size = self.x_embedder.patch_size[0]
            landcover_gates = {}
            for name in ("sar", "temporal", "spatial"):
                gate = landcover_output[f"{name}_gate"].reshape(B * T, 1, H, W)
                gate = F.avg_pool2d(
                    gate, kernel_size=patch_size, stride=patch_size
                ).flatten(1).clamp(0.0, 1.0)
                landcover_gates[name] = gate

        # N = int(H * W / self.patch_size ** 2)
        x = x.contiguous().view(-1, C, H, W)
        x = self.x_embedder(x) + self.pos_embed  # ((B T), N, M), where N = H * W / patch_size ** 2
        cloud_mask_tokens = None

        if cloud_mask is not None and (self.cloud_aware_attention or self.use_cloud_mask_embedding):
            cloud_mask = cloud_mask.contiguous().view(-1, 1, H, W).float().clamp(0, 1)
            patch_size = self.x_embedder.patch_size[0]
            cloud_mask_tokens = torch.nn.functional.avg_pool2d(
                cloud_mask, kernel_size=patch_size, stride=patch_size
            ).flatten(1).clamp(0, 1)

            if self.use_cloud_mask_embedding:
                x = x + self.cloud_mask_embedder(cloud_mask)

        # timestep embedding
        t = t.to(x.device)  # ?
        t = self.t_embedder(t)  # (B, M)
        c = t

        # if y is not None:
        #     y = self.y_embedder(y, self.training)  # (B, M)
        #     c = t + y  # (B, M)

        date_values = date
        date_tokens = None

        # date embedding:
        if date is not None:
            date_tokens = self.date_embedder(date)
            date_tokens = rearrange(date_tokens, 'b t m -> (b t) 1 m', b=B, t=T)
            x = x + date_tokens
        else:
            # if dates are not provided
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            x = x + self.time_embed

            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T)


        if location is not None:
            pass

        # Embed the date-aligned SAR condition at the same patch resolution.
        if cond is not None:
            cond = cond.contiguous().view(-1, self.cond_in_channels, H, W)
            cond = self.cond_embedder(cond) + self.pos_embed  # ((B T), N, M), where N = H * W / patch_size ** 2

            if date_tokens is not None:
                cond = cond + date_tokens
            else:
                cond = rearrange(cond, '(b t) n m -> (b n) t m', b=B, t=T)
                cond = cond + self.time_embed
                cond = self.time_drop(cond)
                cond = rearrange(cond, '(b n) t m -> (b t) n m', b=B, t=T)

        dense_sar_tokens = None
        dense_sar_dates = date_dense_target if date_dense_target is not None else date_values
        if cond_dense is not None:
            if dense_sar_dates is not None:
                dense_sar_dates = dense_sar_dates.to(cond_dense.device)
                if dense_sar_dates.dim() == 1:
                    dense_sar_dates = dense_sar_dates.unsqueeze(0).expand(B, -1)
            dense_sar_memory = self._aggregate_dense_sar(
                cond_dense,
                target_dates=dense_sar_dates,
                source_dates=date_cond_dense,
            )
            dense_sar_tokens = dense_sar_memory.contiguous().view(-1, self.cond_in_channels, H, W)
            dense_sar_tokens = self.cond_embedder(dense_sar_tokens) + self.pos_embed

            if dense_sar_dates is not None:
                dense_date_tokens = self.date_embedder(dense_sar_dates)
                dense_date_tokens = rearrange(
                    dense_date_tokens, 'b t m -> (b t) 1 m', b=B, t=T
                )
                dense_sar_tokens = dense_sar_tokens + dense_date_tokens
            else:
                dense_sar_tokens = rearrange(
                    dense_sar_tokens, '(b t) n m -> (b n) t m', b=B, t=T
                )
                dense_sar_tokens = dense_sar_tokens + self.time_embed
                dense_sar_tokens = self.time_drop(dense_sar_tokens)
                dense_sar_tokens = rearrange(
                    dense_sar_tokens, '(b n) t m -> (b t) n m', b=B, t=T
                )

        attn_cloud_mask = cloud_mask_tokens if self.cloud_aware_attention else None
        for block in self.blocks:
            x = block(
                x,
                c,
                cond,
                attn_cloud_mask,
                landcover_gates,
                dense_sar=dense_sar_tokens,
                date_values=dense_sar_dates,
            )


        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.detail_refiner(
            x,
            model_input_sequence,
            condition_sequence,
            cloud_mask_sequence,
        )
        if return_aux:
            output = {"prediction": x}
            if quality_output is not None:
                output["quality_output"] = quality_output
            if landcover_output is not None:
                output.update(landcover_output)
            return output
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# net = SDT(depth=3, input_size=128, hidden_size=256, patch_size=4, num_heads=4, num_frames=10, cond_in_channels=3)
# input = torch.randn(2, 10, 4, 128, 128)  # (batch, frames, channels, height, width)
# cond = torch.randn(2, 10, 3, 128, 128)
# date = torch.rand(2, 10)
# t = torch.randint(0, 100, (2,)).long()
# print(t.shape)
# output = net(input, t, date=date, cond=cond)
# print(output.shape)
# print(net)
