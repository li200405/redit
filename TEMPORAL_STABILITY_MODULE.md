# Temporal Stability Suppression

This module dynamically identifies spatial regions that change little over
the full input sequence and reduces their influence during training and
reconstruction.

## Focus map

The planner computes:

```text
S2 adjacent-frame change over valid pixels
S1 adjacent-frame change
spatial smoothing
continuous thresholding
```

The result is a dynamic focus map:

```text
0 = temporally stable region
1 = strongly changing region
```

Cloudy or missing S2 pairs are excluded. When valid S2 support is weak, the
module relies more on S1 change.

## Model integration

The focus map is pooled to patch tokens and applied to:

1. S1/S2 cross-attention residuals.
2. Temporal-attention residuals.
3. Spatial-attention residuals.
4. Spatial key priority in cloud-aware attention.
5. Reconstruction loss weights.

Stable regions are not removed. The default minimum attention and loss
weights are both `0.25`.

## Chongqing configuration

```yaml
SDT:
    temporal_stability_suppression: true
    stability_optical_threshold: 0.04
    stability_sar_threshold: 0.10
    stability_optical_temperature: 0.02
    stability_sar_temperature: 0.04
    stability_spatial_kernel: 5
    stability_attention_floor: 0.25

temporal_stability:
    enabled: true
    loss_floor: 0.25
```

Higher thresholds classify more regions as stable. Lower floor values
suppress stable regions more aggressively.

The module has no trainable parameters, so it does not change checkpoint
tensor shapes. Retraining is still recommended because the loss weighting
and attention behavior differ from earlier experiments.
