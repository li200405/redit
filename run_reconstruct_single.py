"""Reconstruct one Chongqing S2 time series with a trained SDT checkpoint.

Edit the settings in the USER SETTINGS section, then run this file directly
from an IDE or with:

    python run_reconstruct_single.py
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime

import numpy as np
import torch
from diffusers.schedulers import DPMSolverMultistepScheduler

from SeqDiffusionPipeline import SeqDiffusionPipeline
from lib import config_utils
from lib.models.SDT import SDT


# ============================== USER SETTINGS ==============================

# Use the config.yaml saved in the same experiment directory as the checkpoint.
CONFIG_PATH = r"results/2026-06-11_15-15/config.yaml"
CHECKPOINT_PATH = r"results/2026-06-11_15-15/checkpoints/Model_best.pth"

# Dataset root and the single S2 file to reconstruct.
DATA_ROOT = r"D:\dpmm\DATA\chongqin"
S2_PATH = r"D:\dpmm\DATA\chongqin\DATA_S2\S2_004857.npy"

# Leave these as None to find the matching files automatically from S2_PATH.
S1_PATH = None
MASK_PATH = None
DATES_PATH = None

# Output settings.
OUTPUT_PATH = r"results/reconstruction/S2_004857_reconstructed.npy"
SAVE_PREDICTION_ONLY_PATH = None

# Diffusion/sliding-window settings.
INFERENCE_STEPS = 1
RANDOM_SEED = 0

# True: retain original pixels outside mask and replace only masked pixels.
# False: save the generated result over the entire image.
PRESERVE_CLEAR_PIXELS = False

# Convert model output [0, 1] back to the S2 reflectance scale used in training.
OUTPUT_SCALE = 8000.0

# "cuda" uses the first GPU visible to this process. Use "cpu" if necessary.
DEVICE = "cuda"

# ===========================================================================


def resolve_path(path: str, base_dir: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def sample_id_from_path(path: str) -> str:
    match = re.search(r"S2_(\d+)\.npy$", os.path.basename(path))
    if match is None:
        raise ValueError(
            "S2 filename must follow S2_XXXXXX.npy, got: {}".format(path)
        )
    return match.group(1)


def date_offsets(date_values, reference_date):
    return torch.tensor(
        [
            (
                datetime.strptime(str(value), "%Y%m%d")
                - reference_date
            ).days
            for value in date_values
        ],
        dtype=torch.long,
    )


def nearest_date_indices(target_dates, source_dates):
    differences = torch.abs(target_dates[:, None] - source_dates[None, :])
    return differences.argmin(dim=1)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch")
    else:
        state_dict = checkpoint
        epoch = None

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return epoch


def move_window_to_end(seq_length, window_size):
    return seq_length - window_size, seq_length


def move_window_next(start, seq_length, window_size, cloud_coverage):
    """Move the window using the same low-cloud boundary rule as old inference."""
    stride = window_size // 2
    candidate_start = start + stride

    if candidate_start + window_size > seq_length:
        return move_window_to_end(seq_length, window_size)

    if cloud_coverage[candidate_start] <= 0.1:
        return candidate_start, candidate_start + window_size

    search_radius = math.ceil(stride / 2)
    left = max(0, candidate_start - search_radius)
    right = min(candidate_start + search_radius + 1, seq_length)
    local_coverage = cloud_coverage[left:right]
    candidates = (
        local_coverage == local_coverage.min()
    ).nonzero(as_tuple=True)[0] + left
    next_start = candidates[
        torch.abs(candidates - candidate_start).argmin()
    ].item()
    next_end = next_start + window_size
    if next_end > seq_length:
        return move_window_to_end(seq_length, window_size)
    return next_start, next_end


def reconstruct_windows(
    pipeline,
    image,
    mask,
    dates,
    cond,
    window_size,
    inference_steps,
    seed,
):
    seq_length = image.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if seq_length <= window_size:
        starts = [0]
    else:
        starts = None

    def predict_window(start, end):
        valid_length = end - start

        image_window = image[:, start:end]
        mask_window = mask[:, start:end]
        date_window = dates[:, start:end]
        cond_window = cond[:, start:end]

        if valid_length < window_size:
            pad_length = window_size - valid_length
            image_window = torch.cat(
                [
                    image_window,
                    torch.zeros(
                        image.shape[0],
                        pad_length,
                        image.shape[2],
                        image.shape[3],
                        image.shape[4],
                        device=image.device,
                    ),
                ],
                dim=1,
            )
            mask_window = torch.cat(
                [
                    mask_window,
                    torch.ones(
                        mask.shape[0],
                        pad_length,
                        1,
                        mask.shape[3],
                        mask.shape[4],
                        device=mask.device,
                    ),
                ],
                dim=1,
            )
            date_window = torch.cat(
                [
                    date_window,
                    torch.zeros(
                        dates.shape[0], pad_length, device=dates.device
                    ),
                ],
                dim=1,
            )
            cond_window = torch.cat(
                [
                    cond_window,
                    torch.zeros(
                        cond.shape[0],
                        pad_length,
                        cond.shape[2],
                        cond.shape[3],
                        cond.shape[4],
                        device=cond.device,
                    ),
                ],
                dim=1,
            )

        prediction = pipeline(
            image=image_window,
            mask=mask_window,
            batch_positions=date_window,
            cond=cond_window,
            generator=generator,
            num_inference_steps=inference_steps,
            output_type="tensor",
            return_dict=False,
            quality_mask=mask_window,
        )
        return prediction[:, :valid_length]

    if starts is not None:
        prediction = predict_window(0, seq_length)
        print("Reconstructed frames 00-{:02d}".format(seq_length - 1))
        return prediction

    cloud_coverage = mask.mean(dim=(0, 2, 3, 4))
    start = 0
    end = window_size
    prediction_full = torch.zeros_like(image)
    previous_start = None
    previous_end = None

    while True:
        prediction_window = predict_window(start, end)
        print(
            "Reconstructed frames {:02d}-{:02d}".format(start, end - 1)
        )

        if previous_start is None:
            prediction_full[:, start:end] = prediction_window
        else:
            overlap_start = max(previous_start, start)
            overlap_end = min(previous_end, end)
            overlap_indices = torch.arange(
                overlap_start, overlap_end, device=image.device
            )

            if overlap_indices.numel() > 0:
                old_prediction = prediction_full[:, overlap_indices]
                new_prediction = prediction_window[
                    :, overlap_indices - start
                ]
                difference = torch.mean(
                    torch.abs(old_prediction - new_prediction),
                    dim=(0, 2, 3, 4),
                )
                switch_frame = (
                    difference.argmin().item() + overlap_start
                )
            else:
                switch_frame = start

            prediction_full[:, switch_frame:end] = prediction_window[
                :, switch_frame - start:
            ]

        if end == seq_length:
            break

        previous_start = start
        previous_end = end
        next_start, next_end = move_window_next(
            start, seq_length, window_size, cloud_coverage
        )
        if next_start == start and next_end == end:
            raise RuntimeError("Sliding window did not advance.")
        start, end = next_start, next_end

    return prediction_full


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = resolve_path(CONFIG_PATH, script_dir)
    checkpoint_path = resolve_path(CHECKPOINT_PATH, script_dir)
    data_root = resolve_path(DATA_ROOT, script_dir)
    s2_path = resolve_path(S2_PATH, script_dir)
    sample_id = sample_id_from_path(s2_path)

    s1_path = resolve_path(
        S1_PATH
        or os.path.join(data_root, "DATA_S1A", "S1_{}.npy".format(sample_id)),
        script_dir,
    )
    mask_path = resolve_path(
        MASK_PATH
        or os.path.join(
            data_root,
            "REAL_MASKS_S2_CLEAR",
            "S2_REAL_MASK_{}.npy".format(sample_id),
        ),
        script_dir,
    )
    dates_path = resolve_path(
        DATES_PATH or os.path.join(data_root, "dates.json"), script_dir
    )
    output_path = resolve_path(OUTPUT_PATH, script_dir)

    required = [
        config_path,
        checkpoint_path,
        s2_path,
        s1_path,
        mask_path,
        dates_path,
    ]
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError("Required file not found: {}".format(path))

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU.")
    device = torch.device(DEVICE)

    config = config_utils.read_config(config_path)
    model_args = dict(config.SDT)
    model = SDT(**model_args).to(device)
    epoch = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    s2_original = np.load(s2_path).astype(np.float32)
    s1_original = np.load(s1_path).astype(np.float32)
    mask_original = np.load(mask_path).astype(np.float32)

    if s2_original.ndim != 4 or s1_original.ndim != 4:
        raise ValueError("S2 and S1 arrays must use T,C,H,W layout.")
    if mask_original.ndim != 4 or mask_original.shape[1] != 1:
        raise ValueError("Mask array must use T,1,H,W layout.")
    if s2_original.shape[0] != mask_original.shape[0]:
        raise ValueError("S2 and mask frame counts do not match.")
    if s2_original.shape[1] != model_args["in_channels"]:
        raise ValueError(
            "S2 has {} channels but the model expects {}.".format(
                s2_original.shape[1], model_args["in_channels"]
            )
        )
    if s1_original.shape[1] != model_args["cond_in_channels"]:
        raise ValueError(
            "S1 has {} channels but the model expects {}.".format(
                s1_original.shape[1], model_args["cond_in_channels"]
            )
        )

    with open(dates_path, "r", encoding="utf-8") as file:
        date_config = json.load(file)
    reference_date = datetime.strptime(
        date_config.get("reference_date", "2022-01-01"), "%Y-%m-%d"
    )
    s2_dates = date_offsets(date_config["dates-S2"], reference_date)
    s1_dates = date_offsets(date_config["dates-S1A"], reference_date)
    if s2_dates.numel() != s2_original.shape[0]:
        raise ValueError("S2 frame count does not match dates.json.")
    if s1_dates.numel() != s1_original.shape[0]:
        raise ValueError("S1 frame count does not match dates.json.")

    s1_indices = nearest_date_indices(s2_dates, s1_dates)
    cond = torch.from_numpy(s1_original)[s1_indices]
    cond = torch.clamp(cond, -50, 10)
    cond = ((cond + 50) / 60) * 2 - 1

    image = torch.from_numpy(s2_original)
    image = torch.clamp(image, 0, OUTPUT_SCALE) / OUTPUT_SCALE
    image = image * 2 - 1

    mask = torch.from_numpy(mask_original)
    mask = torch.where(mask > 0, 1.0, 0.0)
    image[mask.expand_as(image) == 1] = 1

    positions = s2_dates.clone()
    if config.data.get("date_rescale", False):
        positions = ((positions / 10).round() * 10).int()

    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    positions = positions.unsqueeze(0).to(device)
    cond = cond.unsqueeze(0).to(device)

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        prediction_type=config.training_settings.prediction_type,
    )
    pipeline = SeqDiffusionPipeline(model=model, scheduler=scheduler)
    pipeline.set_progress_bar_config(disable=False)

    with torch.inference_mode():
        prediction_01 = reconstruct_windows(
            pipeline=pipeline,
            image=image,
            mask=mask,
            dates=positions,
            cond=cond,
            window_size=int(config.SDT.num_frames),
            inference_steps=INFERENCE_STEPS,
            seed=RANDOM_SEED,
        )

    prediction = (
        prediction_01.squeeze(0).cpu().numpy() * OUTPUT_SCALE
    ).astype(np.float32)
    prediction = np.nan_to_num(
        prediction, nan=0.0, posinf=OUTPUT_SCALE, neginf=0.0
    )
    prediction = np.clip(prediction, 0.0, OUTPUT_SCALE)
    binary_mask = mask_original > 0
    expanded_mask = np.broadcast_to(binary_mask, s2_original.shape)

    if PRESERVE_CLEAR_PIXELS:
        reconstructed = np.where(
            expanded_mask, prediction, s2_original
        ).astype(np.float32)
    else:
        reconstructed = prediction

    clear_pixels = ~expanded_mask
    if PRESERVE_CLEAR_PIXELS and np.any(clear_pixels):
        clear_difference = np.abs(
            reconstructed[clear_pixels] - s2_original[clear_pixels]
        )
        max_clear_difference = float(clear_difference.max())
    else:
        max_clear_difference = float("nan")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, reconstructed.astype(np.float32))

    if SAVE_PREDICTION_ONLY_PATH:
        prediction_path = resolve_path(
            SAVE_PREDICTION_ONLY_PATH, script_dir
        )
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        np.save(prediction_path, prediction)

    print("\nReconstruction complete")
    print("Sample ID: {}".format(sample_id))
    print("Checkpoint epoch: {}".format(epoch))
    print("Input shape: {}".format(s2_original.shape))
    print("Output shape: {}".format(reconstructed.shape))
    print(
        "Maximum difference outside mask: {}".format(max_clear_difference)
    )
    print("Saved to: {}".format(output_path))


if __name__ == "__main__":
    main()
