# Chongqing Dataset Training

The custom Chongqing dataset is loaded by:

```text
lib/datasets/ChongqingDataset.py
```

It does not require `metadata.geojson`.

## Dataset layout

Place the data under the repository:

```text
DATA/chongqin/
  dates.json
  low_clear_T_S2_CLEAR.json
  DATA_S2/
    S2_000093.npy
  DATA_S1A/
    S1_000093.npy
  REAL_MASKS_S2_CLEAR/
    S2_REAL_MASK_000093.npy
```

`low_clear_T_S2_CLEAR.json` contains cloudy S2 frame indices. The
complement is used as the candidate set for target images. Samples with
fewer than 13 candidate target frames are excluded.

For every training sequence:

1. Select 13 target S2 frames from the JSON complement.
2. Match S1 frames using the nearest real acquisition dates.
3. Sample simulated masks from the current sample's mask pool.
4. Use mask coverage bucket probabilities `65% / 25% / 7% / 3%`.
5. Train the model with masked S2, S1, and dates as inputs.

## Training

The configuration is:

```text
configs/config_chongqin_train.yaml
```

Run on GPUs 6, 7, and 8:

```bash
CUDA_VISIBLE_DEVICES=6,7,8 python run_train_PASTIS.py configs/config_chongqin_train.yaml --save_dir ./results/
```

## Reconstruct one NPY sample

Open:

```text
run_reconstruct_single.py
```

Edit the paths in its `USER SETTINGS` section:

```python
CONFIG_PATH = r"results/experiment/config.yaml"
CHECKPOINT_PATH = r"results/experiment/checkpoints/Model_best.pth"
DATA_ROOT = r"DATA/chongqin"
S2_PATH = r"DATA/chongqin/DATA_S2/S2_000093.npy"
OUTPUT_PATH = r"results/reconstruction/S2_000093_reconstructed.npy"
```

`S1_PATH`, `MASK_PATH`, and `DATES_PATH` can remain `None`; matching files
are found automatically from the S2 sample ID.

Run the file directly from an IDE or use:

```bash
python run_reconstruct_single.py
```

The script reconstructs the complete 30-frame sequence with overlapping
13-frame windows. Overlapping predictions use the same minimum-difference
switching strategy as the original reconstruction script instead of being
averaged. By default, only masked pixels are replaced and clear pixels retain
their original S2 values.

## VMamba restoration backbone

The SDT model now uses one fixed restoration backbone; there is no legacy
Transformer/VMamba runtime switch. The backbone contains:

1. `4 x 4` S2 and S1 patch embedding.
2. S1/S2 cross-modal attention for aligned SAR conditioning.
3. Date-aware bidirectional temporal Mamba over every spatial token.
4. Four-direction, multi-scale visual selective scans over every S2 frame.
5. A pixel-space high-frequency residual head for roads and boundaries.

The full dense S1 sequence is date-weighted for every S2 target date. It only
guides temporal state updates; the optical value stream remains in S2 feature
space.

This architecture requires newly trained weights. Checkpoints created by the
older Transformer backbone are not shape-compatible with this version.

## Training loss log

Every `Train:` line in `training.log` reports the actual weighted loss
contributions:

```text
reconstruction_loss
reliability_loss
building_edge_loss
spatial_gradient_loss
spectral_angle_loss
temporal_difference_loss
total_loss
```

The component values add up to `total_loss`. Model-best selection also uses
`total_loss`; it is no longer stored under the misleading
`l1_loss_occluded_input_pixels` label.
