from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

try:
    # This import is intentionally optional: development machines and older
    # environments can still run the original PyTorch scan implementation.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


def selective_scan_backend_name() -> str:
    """Return the scan backend selected by this Python environment."""
    if selective_scan_fn is not None:
        return "mamba_ssm selective_scan_cuda (fused CUDA)"
    return "segmented PyTorch fallback"


def _repeat_condition(value: Tensor, repeats: int) -> Tensor:
    return value[:, None, :].expand(-1, repeats, -1).reshape(-1, value.shape[-1])


def _modulate(x: Tensor, shift: Tensor, scale: Tensor, num_frames: int) -> Tensor:
    condition = _repeat_condition(torch.cat([shift, scale], dim=-1), num_frames)
    shift_bt, scale_bt = condition.chunk(2, dim=-1)
    return x * (1.0 + scale_bt[:, None]) + shift_bt[:, None]


class SelectiveStateScan1D(nn.Module):
    """Bidirectional selective state scan with a CUDA fused fast path.

    On CUDA, the optional ``mamba-ssm`` selective-scan kernel executes the full
    recurrence in one fused operation. Environments without that extension keep
    the previous segmented PyTorch recurrence so checkpoints and CPU inference
    remain usable.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.5,
        state_dim: int = 16,
    ) -> None:
        super().__init__()
        inner_dim = max(dim, int(round(dim * expansion)))
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, inner_dim * 3)
        self.state_proj = nn.Linear(dim, state_dim * 2)
        decay = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.decay_log = nn.Parameter(decay.log().repeat(inner_dim, 1))
        self.skip = nn.Parameter(torch.ones(inner_dim))
        self.out_norm = nn.LayerNorm(inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim)
        # Recompute short scan segments during backward instead of retaining the
        # state transition tensors for the full spatial row or column.
        self.checkpoint_segment_length = 4

    @property
    def fused_cuda_available(self) -> bool:
        """Whether this environment can use Mamba's fused selective scan."""
        return selective_scan_fn is not None

    def _fused_scan_direction(
        self,
        value: Tensor,
        delta: Tensor,
        gate: Tensor,
        state_b: Tensor,
        state_c: Tensor,
        write_confidence: Optional[Tensor],
        step_scale: Optional[Tensor],
        reverse: bool,
    ) -> Tensor:
        """Run one scan direction through Mamba's CUDA selective-scan kernel."""
        if reverse:
            value = value.flip(1)
            delta = delta.flip(1)
            gate = gate.flip(1)
            state_b = state_b.flip(1)
            state_c = state_c.flip(1)
            if write_confidence is not None:
                write_confidence = write_confidence.flip(1)
            if step_scale is not None:
                step_scale = step_scale.flip(1)

        rate = F.softplus(delta) + 1e-4
        if step_scale is not None:
            rate = rate * step_scale.to(rate.dtype)
        if write_confidence is not None:
            rate = rate * write_confidence.to(rate.dtype).clamp(0.0, 1.0)

        # selective_scan_fn expects B/D/L. B and C are dynamically generated
        # from each input token; D is the learned skip connection. The original
        # scan used a manually expanded Euler update. This standard selective
        # SSM parameterization is mathematically stable and maps directly to
        # the fused CUDA implementation.
        # The CUDA kernel requires u, delta, and variable B/C to share one
        # scalar type. Date values and cloud confidences originate as fp32,
        # so explicitly restore the autocast input precision here.
        scan_dtype = value.dtype
        output = selective_scan_fn(
            value.transpose(1, 2).contiguous(),
            rate.to(dtype=scan_dtype).transpose(1, 2).contiguous(),
            -torch.exp(self.decay_log.float()),
            torch.tanh(state_b).to(dtype=scan_dtype).transpose(1, 2).contiguous(),
            torch.tanh(state_c).to(dtype=scan_dtype).transpose(1, 2).contiguous(),
            D=self.skip.float(),
            delta_softplus=False,
        ).transpose(1, 2)
        output = output / math.sqrt(self.state_dim)
        output = output * torch.sigmoid(gate)
        if reverse:
            output = output.flip(1)
        return output

    def _scan_direction(
        self,
        value: Tensor,
        delta: Tensor,
        gate: Tensor,
        state_b: Tensor,
        state_c: Tensor,
        write_confidence: Optional[Tensor],
        step_scale: Optional[Tensor],
        reverse: bool,
    ) -> Tensor:
        if reverse:
            value = value.flip(1)
            delta = delta.flip(1)
            gate = gate.flip(1)
            state_b = state_b.flip(1)
            state_c = state_c.flip(1)
            if write_confidence is not None:
                write_confidence = write_confidence.flip(1)
            if step_scale is not None:
                step_scale = step_scale.flip(1)

        rate = F.softplus(delta) + 1e-4
        if step_scale is not None:
            rate = rate * step_scale.to(rate.dtype)
        if write_confidence is not None:
            rate = rate * write_confidence.to(rate.dtype).clamp(0.0, 1.0)

        state = torch.zeros(
            value.shape[0],
            value.shape[2],
            self.state_dim,
            device=value.device,
            dtype=value.dtype,
        )
        continuous_a = -torch.exp(self.decay_log).to(value.dtype)
        def scan_segment(
            initial_state: Tensor,
            value_segment: Tensor,
            rate_segment: Tensor,
            gate_segment: Tensor,
            state_b_segment: Tensor,
            state_c_segment: Tensor,
        ) -> tuple[Tensor, Tensor]:
            segment_state = initial_state
            segment_outputs = []
            for index in range(value_segment.shape[1]):
                transition = torch.exp(
                    rate_segment[:, index, :, None] * continuous_a[None]
                ).clamp(1e-5, 1.0)
                input_state = (
                    (1.0 - transition)
                    * value_segment[:, index, :, None]
                    * torch.tanh(state_b_segment[:, index, None, :])
                )
                segment_state = transition * segment_state + input_state
                state_output = (
                    segment_state * torch.tanh(state_c_segment[:, index, None, :])
                ).sum(dim=-1) / math.sqrt(self.state_dim)
                output = (
                    state_output + self.skip * value_segment[:, index]
                ) * torch.sigmoid(gate_segment[:, index])
                segment_outputs.append(output)
            return segment_state, torch.stack(segment_outputs, dim=1)

        use_checkpoint = self.training and torch.is_grad_enabled()
        outputs = []
        for start in range(0, value.shape[1], self.checkpoint_segment_length):
            end = min(start + self.checkpoint_segment_length, value.shape[1])
            segment_args = (
                state,
                value[:, start:end],
                rate[:, start:end],
                gate[:, start:end],
                state_b[:, start:end],
                state_c[:, start:end],
            )
            if use_checkpoint:
                state, segment_output = checkpoint(
                    scan_segment,
                    *segment_args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                state, segment_output = scan_segment(*segment_args)
            outputs.append(segment_output)

        scanned = torch.cat(outputs, dim=1)
        if reverse:
            scanned = scanned.flip(1)
        return scanned

    def forward(
        self,
        x: Tensor,
        write_confidence: Optional[Tensor] = None,
        step_scale: Optional[Tensor] = None,
    ) -> Tensor:
        value, delta, gate = self.in_proj(x).chunk(3, dim=-1)
        state_b, state_c = self.state_proj(x).chunk(2, dim=-1)
        scan_direction = (
            self._fused_scan_direction
            if self.fused_cuda_available and value.is_cuda
            else self._scan_direction
        )
        forward = scan_direction(
            value, delta, gate, state_b, state_c,
            write_confidence, step_scale, reverse=False
        )
        backward = scan_direction(
            value, delta, gate, state_b, state_c,
            write_confidence, step_scale, reverse=True
        )
        output = self.out_norm(0.5 * (forward + backward))
        return self.out_proj(output)


class BidirectionalTemporalMamba(nn.Module):
    """Date-aware bidirectional state-space model over the S2 time axis."""

    def __init__(
        self,
        hidden_size: int,
        num_frames: int,
        expansion: float = 1.5,
        date_scale_days: float = 45.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.date_scale_days = max(float(date_scale_days), 1e-6)
        self.norm = nn.LayerNorm(hidden_size)
        self.scan = SelectiveStateScan1D(hidden_size, expansion=expansion)
        self.sar_change_gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size)
        )

    def zero_init(self) -> None:
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        cloud_mask: Optional[Tensor] = None,
        temporal_gate: Optional[Tensor] = None,
        dense_sar: Optional[Tensor] = None,
        dates: Optional[Tensor] = None,
    ) -> Tensor:
        shift, scale, residual_gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        total_frames, num_tokens, hidden_size = x.shape
        batch_size = total_frames // self.num_frames

        normalized = _modulate(self.norm(x), shift, scale, self.num_frames)
        sequence = normalized.view(batch_size, self.num_frames, num_tokens, hidden_size)
        sequence = sequence.permute(0, 2, 1, 3).reshape(
            batch_size * num_tokens, self.num_frames, hidden_size
        )

        write_confidence = None
        if cloud_mask is not None:
            clear = 1.0 - cloud_mask.float().clamp(0.0, 1.0)
            write_confidence = clear.view(batch_size, self.num_frames, num_tokens)
            write_confidence = write_confidence.permute(0, 2, 1).reshape(
                batch_size * num_tokens, self.num_frames, 1
            )

        step_scale = None
        if dates is not None:
            dates = dates.float()
            intervals = torch.ones_like(dates)
            if dates.shape[1] > 1:
                intervals[:, 1:] = (dates[:, 1:] - dates[:, :-1]).abs().clamp_min(1.0)
                intervals[:, 0] = intervals[:, 1]
            intervals = (intervals / self.date_scale_days).clamp(0.1, 4.0)
            step_scale = intervals[:, None, :, None].expand(
                batch_size, num_tokens, self.num_frames, 1
            ).reshape(batch_size * num_tokens, self.num_frames, 1)

        residual = self.scan(
            sequence,
            write_confidence=write_confidence,
            step_scale=step_scale,
        )

        if dense_sar is not None:
            sar = dense_sar.view(batch_size, self.num_frames, num_tokens, hidden_size)
            sar = sar.permute(0, 2, 1, 3).reshape_as(residual)
            residual = residual * (0.5 + 0.5 * self.sar_change_gate(sar))

        residual = residual.view(batch_size, num_tokens, self.num_frames, hidden_size)
        residual = residual.permute(0, 2, 1, 3).reshape_as(x)
        gate = _repeat_condition(residual_gate, self.num_frames)[:, None]
        residual = residual * gate
        if temporal_gate is not None:
            residual = residual * temporal_gate.unsqueeze(-1)
        return x + residual


class SelectiveScan2D(nn.Module):
    """Four-direction visual selective scan with a local convolution branch."""

    def __init__(self, hidden_size: int, expansion: float = 1.5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.horizontal = SelectiveStateScan1D(hidden_size, expansion=expansion)
        self.vertical = SelectiveStateScan1D(hidden_size, expansion=expansion)
        self.local = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 1),
        )
        self.fuse = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, x: Tensor, cloud_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, height, width, hidden_size = x.shape
        normalized = self.norm(x)

        horizontal = normalized.reshape(batch_size * height, width, hidden_size)
        horizontal_conf = None
        if cloud_mask is not None:
            horizontal_conf = (1.0 - cloud_mask).reshape(batch_size * height, width, 1)
        horizontal = self.horizontal(horizontal, write_confidence=horizontal_conf)
        horizontal = horizontal.reshape(batch_size, height, width, hidden_size)

        vertical = normalized.permute(0, 2, 1, 3).reshape(
            batch_size * width, height, hidden_size
        )
        vertical_conf = None
        if cloud_mask is not None:
            vertical_conf = (1.0 - cloud_mask).permute(0, 2, 1).reshape(
                batch_size * width, height, 1
            )
        vertical = self.vertical(vertical, write_confidence=vertical_conf)
        vertical = vertical.reshape(batch_size, width, height, hidden_size).permute(0, 2, 1, 3)

        local = self.local(normalized.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.fuse(torch.cat([horizontal, vertical, local], dim=-1))


class MultiScaleVMambaSpatial(nn.Module):
    """Full- and half-resolution VSS branches for restoration detail and context."""

    def __init__(
        self,
        hidden_size: int,
        num_frames: int,
        expansion: float = 1.5,
        use_multiscale: bool = True,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.use_multiscale = use_multiscale
        self.full_scan = SelectiveScan2D(hidden_size, expansion=expansion)
        self.coarse_scan = (
            SelectiveScan2D(hidden_size, expansion=expansion)
            if use_multiscale else None
        )
        self.scale_fusion = nn.Linear(
            hidden_size * (2 if use_multiscale else 1), hidden_size
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size)
        )

    def zero_init(self) -> None:
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        cloud_mask: Optional[Tensor] = None,
        spatial_gate: Optional[Tensor] = None,
    ) -> Tensor:
        shift, scale, residual_gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        total_frames, num_tokens, hidden_size = x.shape
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError("VMamba spatial tokens must form a square grid.")

        normalized = _modulate(x, shift, scale, self.num_frames)
        feature_map = normalized.view(total_frames, side, side, hidden_size)
        mask_map = cloud_mask.view(total_frames, side, side) if cloud_mask is not None else None
        full = self.full_scan(feature_map, mask_map)
        branches = [full]

        if self.coarse_scan is not None and side >= 2:
            coarse = F.avg_pool2d(
                feature_map.permute(0, 3, 1, 2), kernel_size=2, stride=2
            ).permute(0, 2, 3, 1)
            coarse_mask = None
            if mask_map is not None:
                coarse_mask = F.avg_pool2d(
                    mask_map[:, None], kernel_size=2, stride=2
                )[:, 0]
            coarse = self.coarse_scan(coarse, coarse_mask)
            coarse = F.interpolate(
                coarse.permute(0, 3, 1, 2),
                size=(side, side),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            branches.append(coarse)

        residual = self.scale_fusion(torch.cat(branches, dim=-1)).reshape_as(x)
        gate = _repeat_condition(residual_gate, self.num_frames)[:, None]
        residual = residual * gate
        if spatial_gate is not None:
            residual = residual * spatial_gate.unsqueeze(-1)
        return x + residual


class HighFrequencyRefinementHead(nn.Module):
    """Pixel-space residual head for roads, boundaries, and small structures."""

    def __init__(
        self,
        out_channels: int,
        input_channels: int,
        cond_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.cond_channels = cond_channels
        total_channels = out_channels + input_channels + cond_channels + 1
        self.in_proj = nn.Conv2d(total_channels, hidden_channels, 3, padding=1)
        self.branch_3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.branch_5 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2),
            nn.GELU(),
        )
        self.fuse = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)

    def zero_init(self) -> None:
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        prediction: Tensor,
        model_input: Tensor,
        condition: Optional[Tensor],
        cloud_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, frames, _, height, width = prediction.shape
        if condition is None:
            condition = prediction.new_zeros(
                batch_size, frames, self.cond_channels, height, width
            )
        if cloud_mask is None:
            cloud_mask = prediction.new_ones(batch_size, frames, 1, height, width)

        inputs = torch.cat([prediction, model_input, condition, cloud_mask], dim=2)
        inputs = inputs.reshape(batch_size * frames, inputs.shape[2], height, width)
        feature = self.in_proj(inputs)
        feature = self.fuse(torch.cat([self.branch_3(feature), self.branch_5(feature)], dim=1))
        residual = self.out_proj(feature).reshape_as(prediction)
        return prediction + cloud_mask * residual
