"""Reconstruct and validate one Chongqing S2 sequence, saving only a PNG.

Edit the USER SETTINGS section, then run this file directly:

    python run_reconstruct_validate_single.py

The default mode is "cloudy_real_clear_synthetic": real cloudy frames use their
real masks, while clear frames receive synthetic cloud masks. The whole sequence
is reconstructed together with temporal context, then displayed as original
visible pixels plus reconstructed masked pixels. Metrics are written directly
under the synthetic clear-frame reconstructions; no CSV/JSON files are produced.
"""

from __future__ import annotations

import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.schedulers import DPMSolverMultistepScheduler

try:
    import torchgeometry as tgm
except ImportError:  # pragma: no cover - optional local dependency fallback.
    tgm = None

from SeqDiffusionPipeline import SeqDiffusionPipeline
from lib import config_utils
from lib.models.SDT import SDT
from run_reconstruct_single import (
    date_offsets,
    load_checkpoint,
    nearest_date_indices,
    reconstruct_windows,
    resolve_config_path,
    resolve_path,
    sample_id_from_path,
)


# ============================== USER SETTINGS ==============================

CONFIG_PATH = r"D:\dpmm\REDiT\results\2026-06-15_17-17"
CHECKPOINT_PATH = r"D:\dpmm\REDiT\results\2026-06-15_17-17\checkpoints\Model_after_3000_epochs.pth"

DATA_ROOT = r"D:\dpmm\DATA\chongqin"
S2_PATH = r"D:\dpmm\DATA\chongqin\DATA_S2\S2_004857.npy"

# Leave these as None to find the matching files automatically from S2_PATH.
S1_PATH = None
MASK_PATH = None
DATES_PATH = None

# Only a PNG is saved. Per-frame metrics are printed below each reconstruction.
OUTPUT_PNG = r"results/validation/S2_004857_validate_tsm.png"

# "cloudy_real_clear_synthetic": real-cloud reconstruction plus clear-frame synthetic validation.
# "full_sequence": show all S2 dates; reconstruct real masked/cloudy pixels.
# "synthetic_clear": quantitative validation on clear frames with artificial masks.
# "real_cloud_diagnostic": reconstruct real cloudy frames; metrics use only clear pixels.
VALIDATION_MODE = "cloudy_real_clear_synthetic"

# Clear-frame threshold for synthetic validation, using the real mask coverage.
CLEAR_FRAME_MAX_COVERAGE = 0.02

# Real masks are selected with the same ordered coverage buckets used by eval/test:
# 10%-60%, 60%-80%, 80%-95%, 95%-100%.
EVAL_MASK_COVERAGE_BUCKETS = (
    (0.10, 0.60),
    (0.60, 0.80),
    (0.80, 0.95),
    (0.95, 1.01),
)

INFERENCE_STEPS = 10
RANDOM_SEED = 0
OUTPUT_SCALE = 8000.0
DEVICE = "cuda"

# Visual settings.
RGB_INDICES = (2, 1, 0)
MAX_FRAMES_TO_SHOW = 30
FIG_COLUMNS = 6
DISPLAY_PERCENTILES = (2, 98)

# ===========================================================================


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError("Required file not found: {}".format(path))


def load_date_config(dates_path: str):
    import json

    with open(dates_path, "r", encoding="utf-8") as file:
        date_config = json.load(file)
    reference_date = datetime.strptime(
        date_config.get("reference_date", "2022-01-01"), "%Y-%m-%d"
    )
    s2_dates = date_offsets(date_config["dates-S2"], reference_date)
    s1_dates = date_offsets(date_config["dates-S1A"], reference_date)
    s2_labels = [str(value) for value in date_config["dates-S2"]]
    return s2_dates, s1_dates, s2_labels


def normalize_s1(s1: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(s1.astype(np.float32))
    tensor = torch.clamp(tensor, -50, 10)
    return ((tensor + 50) / 60) * 2 - 1


def normalize_s2(s2: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(s2.astype(np.float32))
    tensor = torch.clamp(tensor, 0, OUTPUT_SCALE) / OUTPUT_SCALE
    return tensor * 2 - 1


def select_evenly(indices: np.ndarray, max_count: int) -> np.ndarray:
    if len(indices) <= max_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_count).round().astype(int)
    return indices[positions]


def select_eval_ordered_masks_by_coverage(mask_pool: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    coverage = mask_pool.mean(axis=(1, 2, 3))
    ordered_indices = []
    for low, high in EVAL_MASK_COVERAGE_BUCKETS:
        bucket = np.where((coverage >= low) & (coverage < high))[0]
        if bucket.size > 0:
            ordered_indices.append(bucket)

    if not ordered_indices:
        fallback = np.where(coverage > 0)[0]
        if fallback.size > 0:
            ordered_indices.append(fallback)

    if not ordered_indices:
        raise ValueError("No non-empty masks are available for synthetic validation.")

    ordered = np.concatenate(ordered_indices)
    repeats = int(np.ceil(count / ordered.size))
    chosen = np.tile(ordered, repeats)[:count]
    return (mask_pool[chosen] > 0).astype(np.float32), chosen


def prepare_validation_arrays(
    s2_original: np.ndarray,
    mask_original: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    mask_binary = (mask_original > 0).astype(np.float32)
    coverage = mask_binary.mean(axis=(1, 2, 3))

    if VALIDATION_MODE == "full_sequence":
        selected = np.arange(s2_original.shape[0])
        validation_mask = mask_binary
        mode_note = (
            "full_sequence: all S2 frames; cloudy/masked frames are reconstructed, "
            "metrics use visible clear pixels only"
        )
        return selected, validation_mask, mask_binary, mode_note

    if VALIDATION_MODE == "synthetic_clear":
        selected = np.where(coverage <= CLEAR_FRAME_MAX_COVERAGE)[0]
        if selected.size == 0:
            selected = np.argsort(coverage)[: min(MAX_FRAMES_TO_SHOW, len(coverage))]
        selected = select_evenly(selected, MAX_FRAMES_TO_SHOW)
        validation_mask, chosen_masks = select_eval_ordered_masks_by_coverage(mask_binary, len(selected))
        mode_note = (
            "synthetic_clear: clear target frames with sampled real masks "
            "(mask ids: {})".format(",".join(map(str, chosen_masks[:8])))
        )
        return selected, validation_mask, mask_binary[selected], mode_note

    if VALIDATION_MODE == "real_cloud_diagnostic":
        selected = select_evenly(np.arange(s2_original.shape[0]), MAX_FRAMES_TO_SHOW)
        validation_mask = mask_binary[selected]
        mode_note = "real_cloud_diagnostic: metrics use original clear pixels only"
        return selected, validation_mask, mask_binary[selected], mode_note

    raise ValueError("Unsupported VALIDATION_MODE: {}".format(VALIDATION_MODE))


def to_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = np.stack([frame[index] for index in RGB_INDICES], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, DISPLAY_PERCENTILES)
    if hi <= lo:
        lo, hi = 0.0, OUTPUT_SCALE
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def compute_sam_np(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    valid_2d = valid[:, 0].astype(bool)
    if not valid_2d.any():
        return float("nan")
    pred = predicted.transpose(0, 2, 3, 1)[valid_2d]
    tgt = target.transpose(0, 2, 3, 1)[valid_2d]
    dot = np.sum(pred * tgt, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    tgt_norm = np.linalg.norm(tgt, axis=1)
    ok = (pred_norm > 0) & (tgt_norm > 0)
    if not ok.any():
        return float("nan")
    cosine = np.clip(dot[ok] / (pred_norm[ok] * tgt_norm[ok]), -1.0, 1.0)
    return float(np.mean(np.arccos(cosine)))


def compute_ssim_full(predicted: np.ndarray, target: np.ndarray) -> float:
    if tgm is None:
        return float("nan")
    pred = torch.from_numpy(predicted.astype(np.float32))
    tgt = torch.from_numpy(target.astype(np.float32))
    with torch.no_grad():
        dssim = tgm.losses.SSIM(5, reduction="mean")(pred, tgt)
    return float((1 - 2 * dssim).item())


def compute_metrics(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    valid = valid.astype(bool)
    if not valid.any():
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "psnr": float("nan"),
            "sam": float("nan"),
            "valid_pixels": 0,
        }

    diff = predicted - target
    diff_valid = diff[valid.repeat(predicted.shape[1], axis=1)]
    mae = float(np.mean(np.abs(diff_valid)))
    mse = float(np.mean(np.square(diff_valid)))
    rmse = math.sqrt(mse)
    psnr = float(20 * math.log10(1.0 / max(rmse, 1e-12)))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "sam": compute_sam_np(predicted, target, valid),
        "valid_pixels": int(valid.sum()),
    }


def format_metric_line(metrics: dict[str, float], include_ssim: bool = False) -> str:
    parts = [
        "MAE={:.4f}".format(metrics["mae"]),
        "RMSE={:.4f}".format(metrics["rmse"]),
        "PSNR={:.2f}".format(metrics["psnr"]),
        "SAM={:.3f}".format(metrics["sam"]),
    ]
    if include_ssim:
        parts.append("SSIM={:.3f}".format(metrics.get("ssim", float("nan"))))
    return " ".join(parts)


def format_frame_metric(metrics: dict[str, float]) -> str:
    if metrics.get("valid_pixels", 0) <= 0:
        return "no clear GT pixels"
    return "MAE {:.4f}\nRMSE {:.4f}  PSNR {:.2f}\nSAM {:.3f}".format(
        metrics["mae"],
        metrics["rmse"],
        metrics["psnr"],
        metrics["sam"],
    )


def make_validation_figure(
    output_png: str,
    display_sequence: np.ndarray,
    date_labels: list[str],
    frame_status: list[str],
    per_frame_metrics: list[dict[str, float]],
    overall_metrics: dict[str, float],
    mode_note: str,
) -> None:
    n_frames = len(date_labels)
    ncols = min(FIG_COLUMNS, n_frames)
    nrows = int(math.ceil(n_frames / ncols))

    fig_width = 2.0 * ncols
    fig_height = 2.35 * nrows + 0.9
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    for ax in axes.reshape(-1):
        ax.axis("off")

    for frame_idx in range(n_frames):
        row = frame_idx // ncols
        col = frame_idx % ncols
        ax = axes[row, col]
        ax.imshow(to_rgb(display_sequence[frame_idx] * OUTPUT_SCALE))
        ax.axis("off")
        ax.set_title(
            "{}\n{}".format(date_labels[frame_idx], frame_status[frame_idx]),
            fontsize=8,
            pad=2,
        )
        ax.text(
            0.5,
            -0.08,
            format_frame_metric(per_frame_metrics[frame_idx]),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6,
        )

    summary = (
        "{}\nOverall metrics: {} | valid pixels={}".format(
            mode_note,
            format_metric_line(overall_metrics, include_ssim=True),
            overall_metrics["valid_pixels"],
        )
    )
    fig.suptitle(summary, fontsize=10)
    fig.subplots_adjust(
        left=0.02,
        right=0.985,
        top=0.90,
        bottom=0.04,
        wspace=0.08,
        hspace=0.54,
    )
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def reconstruct_subset(
    pipeline,
    s2_subset: np.ndarray,
    s2_dates_subset: torch.Tensor,
    s1_tensor: torch.Tensor,
    s1_dates: torch.Tensor,
    mask_subset: np.ndarray,
    config,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    s1_indices = nearest_date_indices(s2_dates_subset, s1_dates)
    cond = s1_tensor[s1_indices]
    cond_dense = s1_tensor

    image = normalize_s2(s2_subset)
    mask = torch.from_numpy(mask_subset.astype(np.float32))
    image[mask.expand_as(image) == 1] = 1

    positions = s2_dates_subset.clone()
    if config.data.get("date_rescale", False):
        positions = ((positions / 10).round() * 10).int()

    with torch.inference_mode():
        prediction = reconstruct_windows(
            pipeline=pipeline,
            image=image.unsqueeze(0).to(device),
            mask=mask.unsqueeze(0).to(device),
            dates=positions.unsqueeze(0).to(device),
            cond=cond.unsqueeze(0).to(device),
            cond_dense=cond_dense.unsqueeze(0).to(device),
            date_cond_dense=s1_dates.unsqueeze(0).to(device),
            date_dense_target=s2_dates_subset.unsqueeze(0).to(device),
            window_size=int(config.SDT.num_frames),
            inference_steps=INFERENCE_STEPS,
            seed=seed,
        )

    prediction = prediction.squeeze(0).cpu().numpy().astype(np.float32)
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(prediction, 0.0, 1.0)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = resolve_config_path(CONFIG_PATH, script_dir)
    checkpoint_path = resolve_path(CHECKPOINT_PATH, script_dir)
    data_root = resolve_path(DATA_ROOT, script_dir)
    s2_path = resolve_path(S2_PATH, script_dir)
    sample_id = sample_id_from_path(s2_path)

    s1_path = resolve_path(
        S1_PATH or os.path.join(data_root, "DATA_S1A", "S1_{}.npy".format(sample_id)),
        script_dir,
    )
    mask_path = resolve_path(
        MASK_PATH
        or os.path.join(data_root, "REAL_MASKS_S2_CLEAR", "S2_REAL_MASK_{}.npy".format(sample_id)),
        script_dir,
    )
    dates_path = resolve_path(DATES_PATH or os.path.join(data_root, "dates.json"), script_dir)
    output_png = resolve_path(OUTPUT_PNG, script_dir)

    for path in (config_path, checkpoint_path, s2_path, s1_path, mask_path, dates_path):
        require_file(path)

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
    s2_dates, s1_dates, s2_labels = load_date_config(dates_path)
    date_labels = s2_labels

    if s2_original.ndim != 4 or s1_original.ndim != 4:
        raise ValueError("S2 and S1 arrays must use T,C,H,W layout.")
    if mask_original.ndim != 4 or mask_original.shape[1] != 1:
        raise ValueError("Mask array must use T,1,H,W layout.")
    if s2_dates.numel() != s2_original.shape[0]:
        raise ValueError("S2 frame count does not match dates.json.")
    if s1_dates.numel() != s1_original.shape[0]:
        raise ValueError("S1 frame count does not match dates.json.")

    s1_tensor = normalize_s1(s1_original)
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        prediction_type=config.training_settings.prediction_type,
    )
    pipeline = SeqDiffusionPipeline(model=model, scheduler=scheduler)
    pipeline.set_progress_bar_config(disable=False)

    mask_binary = (mask_original > 0).astype(np.float32)
    coverage = mask_binary.mean(axis=(1, 2, 3))
    target_01 = (np.clip(s2_original, 0, OUTPUT_SCALE) / OUTPUT_SCALE).astype(np.float32)
    predicted_01 = target_01.copy()
    display_01 = target_01.copy()
    validation_mask_np = np.zeros_like(mask_binary, dtype=np.float32)
    valid_metric_mask = np.zeros_like(mask_binary, dtype=np.float32)
    frame_status = ["clear/reference" for _ in range(s2_original.shape[0])]

    if VALIDATION_MODE == "cloudy_real_clear_synthetic":
        cloudy_indices = np.where(coverage > CLEAR_FRAME_MAX_COVERAGE)[0]
        clear_indices = np.where(coverage <= CLEAR_FRAME_MAX_COVERAGE)[0]
        mode_note = (
            "cloudy_real_clear_synthetic: real cloudy frames are reconstructed; "
            "clear frames use synthetic clouds for metrics"
        )

        if cloudy_indices.size > 0:
            validation_mask_np[cloudy_indices] = mask_binary[cloudy_indices]
            for index in cloudy_indices:
                frame_status[index] = "real cloud recon {:.1f}%".format(coverage[index] * 100)

        if clear_indices.size > 0:
            synthetic_masks, _ = select_eval_ordered_masks_by_coverage(
                mask_binary, clear_indices.size
            )
            for offset, index in enumerate(clear_indices):
                validation_mask_np[index] = synthetic_masks[offset]
                valid_metric_mask[index] = synthetic_masks[offset]
                frame_status[index] = "eval mask {:.1f}%".format(
                    synthetic_masks[offset].mean() * 100
                )

        predicted_01 = reconstruct_subset(
            pipeline=pipeline,
            s2_subset=s2_original,
            s2_dates_subset=s2_dates,
            s1_tensor=s1_tensor,
            s1_dates=s1_dates,
            mask_subset=validation_mask_np,
            config=config,
            device=device,
            seed=RANDOM_SEED,
        )
        display_01 = (
            target_01 * (1.0 - validation_mask_np)
            + predicted_01 * validation_mask_np
        ).astype(np.float32)

    else:
        selected, validation_mask_np, original_cloud_mask_np, mode_note = prepare_validation_arrays(
            s2_original, mask_original
        )
        target_01 = target_01[selected]
        frame_status = []
        selected_coverage = validation_mask_np.mean(axis=(1, 2, 3))
        predicted_01 = reconstruct_subset(
            pipeline=pipeline,
            s2_subset=s2_original[selected],
            s2_dates_subset=s2_dates[selected],
            s1_tensor=s1_tensor,
            s1_dates=s1_dates,
            mask_subset=validation_mask_np,
            config=config,
            device=device,
            seed=RANDOM_SEED,
        )
        display_01 = (
            target_01 * (1.0 - validation_mask_np)
            + predicted_01 * validation_mask_np
        ).astype(np.float32)
        clear_target_mask = (original_cloud_mask_np == 0).astype(np.float32)
        if VALIDATION_MODE == "full_sequence":
            valid_metric_mask = (clear_target_mask > 0).astype(np.float32)
        elif VALIDATION_MODE == "real_cloud_diagnostic":
            valid_metric_mask = ((validation_mask_np == 0) & (clear_target_mask > 0)).astype(np.float32)
        else:
            valid_metric_mask = ((validation_mask_np > 0) & (clear_target_mask > 0)).astype(np.float32)
        for frame_coverage in selected_coverage:
            if frame_coverage > 0:
                frame_status.append("cloud/reconstruct {:.1f}%".format(frame_coverage * 100))
            else:
                frame_status.append("clear/reference")
        date_labels = [s2_labels[index] for index in selected]

    overall_metrics = compute_metrics(predicted_01, target_01, valid_metric_mask)
    if VALIDATION_MODE == "cloudy_real_clear_synthetic":
        clear_metric_frames = valid_metric_mask.reshape(valid_metric_mask.shape[0], -1).any(axis=1)
        if clear_metric_frames.any():
            overall_metrics["ssim"] = compute_ssim_full(
                predicted_01[clear_metric_frames], target_01[clear_metric_frames]
            )
        else:
            overall_metrics["ssim"] = float("nan")
    else:
        overall_metrics["ssim"] = compute_ssim_full(predicted_01, target_01)

    per_frame_metrics = []
    for index in range(predicted_01.shape[0]):
        frame_metrics = compute_metrics(
            predicted_01[index:index + 1],
            target_01[index:index + 1],
            valid_metric_mask[index:index + 1],
        )
        per_frame_metrics.append(frame_metrics)

    make_validation_figure(
        output_png=output_png,
        display_sequence=display_01,
        date_labels=date_labels,
        frame_status=frame_status,
        per_frame_metrics=per_frame_metrics,
        overall_metrics=overall_metrics,
        mode_note=mode_note,
    )

    print("\nValidation reconstruction complete")
    print("Sample ID: {}".format(sample_id))
    print("Checkpoint epoch: {}".format(epoch))
    print("Mode: {}".format(VALIDATION_MODE))
    print("Frames shown: {}".format(len(date_labels)))
    print("Overall: {}".format(format_metric_line(overall_metrics, include_ssim=True)))
    print("Saved PNG: {}".format(output_png))


if __name__ == "__main__":
    main()
