from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.utils.data as tdata
from omegaconf import DictConfig


class ChongqingDataset(tdata.Dataset):
    """Dataset reader for the custom Chongqing S2/S1 time series."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        channels: str = "all",
        filter_settings: Optional[DictConfig] = None,
        crop_settings: Optional[DictConfig] = None,
        max_seq_length: Optional[int] = 13,
        norm: bool = True,
        rescale: bool = True,
        ifCTHW: bool = False,
        date_rescale: bool = True,
        train_ratio: float = 0.9,
        limit_samples: Optional[int] = None,
        seed: int = 0,
        dates_file: str = "dates.json",
        cloudy_indices_file: str = "low_clear_T_S2_CLEAR.json",
        mask_kwargs: Optional[DictConfig] = None,
        augment: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.root = self._resolve_data_root(root)
        self.split = split
        self.channels = channels
        self.filter_settings = filter_settings
        self.crop_settings = crop_settings
        self.max_seq_length = max_seq_length
        self.norm = norm
        self.rescale = rescale
        self.ifCTHW = ifCTHW
        self.date_rescale = date_rescale
        self.augment = augment

        self.s2_dir = os.path.join(self.root, "DATA_S2")
        self.s1_dir = os.path.join(self.root, "DATA_S1A")
        self.mask_dir = os.path.join(self.root, "REAL_MASKS_S2_CLEAR")
        self.cloudy_index_file = os.path.join(self.root, cloudy_indices_file)
        self.dates_path = os.path.join(self.root, dates_file)

        required_paths = (
            self.s2_dir,
            self.s1_dir,
            self.mask_dir,
            self.cloudy_index_file,
            self.dates_path,
        )
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "Chongqing dataset path not found: {}".format(path)
                )

        with open(self.cloudy_index_file, "r", encoding="utf-8") as file:
            self.cloudy_indices = json.load(file)
        with open(self.dates_path, "r", encoding="utf-8") as file:
            date_config = json.load(file)

        reference_date = datetime.strptime(
            date_config.get("reference_date", "2022-01-01"), "%Y-%m-%d"
        )
        self.s2_dates = self._date_offsets(
            date_config["dates-S2"], reference_date
        )
        self.s1_dates = self._date_offsets(
            date_config["dates-S1A"], reference_date
        )

        ids = []
        for name in os.listdir(self.s2_dir):
            if not name.startswith("S2_") or not name.endswith(".npy"):
                continue
            sample_id = name[len("S2_"):-len(".npy")]
            s1_path = os.path.join(self.s1_dir, "S1_{}.npy".format(sample_id))
            mask_path = os.path.join(
                self.mask_dir, "S2_REAL_MASK_{}.npy".format(sample_id)
            )
            if (
                not os.path.exists(s1_path)
                or not os.path.exists(mask_path)
                or sample_id not in self.cloudy_indices
            ):
                continue

            s2_path = os.path.join(self.s2_dir, name)
            s2_shape = np.load(s2_path, mmap_mode="r").shape
            clear_count = self._count_clear_frames(
                self.cloudy_indices[sample_id], s2_shape[0]
            )
            enough_clear_frames = (
                clear_count > 0
                if self.max_seq_length is None
                else clear_count >= self.max_seq_length
            )
            if enough_clear_frames:
                ids.append(sample_id)

        ids = sorted(ids)
        rng = np.random.default_rng(seed)
        shuffled_ids = np.array(ids)
        rng.shuffle(shuffled_ids)
        split_at = int(len(shuffled_ids) * train_ratio)
        if split in ("train", "train+val"):
            self.ids = sorted(shuffled_ids[:split_at].tolist())
        elif split in ("val", "test"):
            self.ids = sorted(shuffled_ids[split_at:].tolist())
        else:
            raise ValueError("Unsupported split: {}".format(split))

        if limit_samples is not None:
            self.ids = self.ids[:int(limit_samples)]
        if not self.ids:
            raise RuntimeError(
                "No samples found for split={} under {}".format(split, self.root)
            )

        probe = np.load(
            os.path.join(self.s2_dir, "S2_{}.npy".format(self.ids[0])),
            mmap_mode="r",
        )
        self.image_size = (int(probe.shape[-2]), int(probe.shape[-1]))
        if channels == "all":
            self.s2_channels = list(range(min(10, probe.shape[1])))
            self.c_index_rgb = torch.tensor([2, 1, 0]).long()
            self.c_index_nir = torch.tensor([6]).long()
        elif channels == "bgr-nir":
            self.s2_channels = [0, 1, 2, 6]
            self.c_index_rgb = torch.tensor([2, 1, 0]).long()
            self.c_index_nir = torch.tensor([3]).long()
        elif channels == "bgr":
            self.s2_channels = [0, 1, 2]
            self.c_index_rgb = torch.tensor([2, 1, 0]).long()
            self.c_index_nir = torch.tensor(float("nan"))
        else:
            raise ValueError(
                "Unsupported channels setting for ChongqingDataset: {}".format(
                    channels
                )
            )

        self.num_channels = len(self.s2_channels)
        self.variable_seq_length = False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        sample_id = self.ids[item]
        s2 = torch.from_numpy(
            np.load(
                os.path.join(self.s2_dir, "S2_{}.npy".format(sample_id))
            ).astype(np.float32)
        )
        s1 = torch.from_numpy(
            np.load(
                os.path.join(self.s1_dir, "S1_{}.npy".format(sample_id))
            ).astype(np.float32)
        )
        masks_pool = torch.from_numpy(
            np.load(
                os.path.join(
                    self.mask_dir,
                    "S2_REAL_MASK_{}.npy".format(sample_id),
                )
            ).astype(np.float32)
        )

        self._validate_sequence_lengths(sample_id, s2, s1, masks_pool)
        clear_indices = self._clear_indices_from_cloudy(
            self.cloudy_indices[sample_id], s2.shape[0]
        )
        t_sampled = self._select_target_indices(clear_indices)

        target_cloud_mask = torch.where(
            masks_pool[t_sampled] > 0, 1.0, 0.0
        )
        s2 = s2[t_sampled][:, self.s2_channels]
        mask_indices = self._select_mask_indices_from_pool(
            masks_pool, len(t_sampled), self.split
        )
        masks = torch.where(masks_pool[mask_indices] > 0, 1.0, 0.0)

        if self.norm:
            s2 = torch.clamp(s2, 0, 8000) / 8000
            s1 = torch.clamp(s1, -50, 10)
            s1 = (s1 + 50) / 60
        if self.rescale:
            s2 = s2 * 2 - 1
            s1 = s1 * 2 - 1

        selected_s2_dates = self.s2_dates[t_sampled]
        s1_indices = self._match_nearest_dates(
            selected_s2_dates, self.s1_dates
        )
        cond = s1[s1_indices]
        selected_s1_dates = self.s1_dates[s1_indices]

        frames_input = s2.clone()
        frames_input[masks.expand_as(frames_input) == 1] = 1

        position_days = selected_s2_dates.clone()
        if self.date_rescale:
            position_days = ((position_days / 10).round() * 10).int()
        days = selected_s2_dates - selected_s2_dates[0]

        return {
            "x": frames_input,
            "cond": cond,
            "y": s2,
            "masks": masks,
            "position_days": position_days,
            "position_days_cond": selected_s1_dates,
            "days": days,
            "sample_index": sample_id,
            "label": torch.tensor(0, dtype=torch.long),
            "c_index_rgb": self.c_index_rgb,
            "c_index_nir": self.c_index_nir,
            "cloud_mask": target_cloud_mask,
        }

    def _select_target_indices(self, clear_indices):
        if self.max_seq_length is None or clear_indices.numel() <= self.max_seq_length:
            return clear_indices
        if self.split == "train":
            max_start = clear_indices.numel() - self.max_seq_length
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start = 0
        return clear_indices[start:start + self.max_seq_length]

    def _validate_sequence_lengths(self, sample_id, s2, s1, masks_pool):
        if s2.shape[0] != self.s2_dates.numel():
            raise ValueError(
                "S2 frame/date mismatch for {}: {} frames, {} dates".format(
                    sample_id, s2.shape[0], self.s2_dates.numel()
                )
            )
        if s1.shape[0] != self.s1_dates.numel():
            raise ValueError(
                "S1 frame/date mismatch for {}: {} frames, {} dates".format(
                    sample_id, s1.shape[0], self.s1_dates.numel()
                )
            )
        if masks_pool.shape[0] != s2.shape[0]:
            raise ValueError(
                "S2/mask frame mismatch for {}: {} and {}".format(
                    sample_id, s2.shape[0], masks_pool.shape[0]
                )
            )

    @staticmethod
    def _resolve_data_root(root):
        configured = os.path.abspath(os.path.expanduser(root))
        repository_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        candidates = [
            configured,
            os.path.abspath(os.path.join(repository_root, root)),
            os.path.abspath(os.path.join(repository_root, "..", root)),
        ]
        for candidate in dict.fromkeys(candidates):
            if (
                os.path.isdir(os.path.join(candidate, "DATA_S2"))
                and os.path.isdir(os.path.join(candidate, "DATA_S1A"))
            ):
                return candidate
        raise FileNotFoundError(
            "Chongqing DATA_S2/DATA_S1A folders not found under: {}".format(
                ", ".join(dict.fromkeys(candidates))
            )
        )

    @staticmethod
    def _date_offsets(date_values, reference_date):
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

    @staticmethod
    def _match_nearest_dates(target_dates, source_dates):
        differences = torch.abs(
            target_dates[:, None] - source_dates[None, :]
        )
        return differences.argmin(dim=1)

    @staticmethod
    def _clear_indices_from_cloudy(cloudy_indices, seq_length):
        cloudy = torch.as_tensor(cloudy_indices, dtype=torch.long)
        cloudy = cloudy[(cloudy >= 0) & (cloudy < seq_length)].unique()
        keep = torch.ones(seq_length, dtype=torch.bool)
        keep[cloudy] = False
        return keep.nonzero().view(-1)

    @staticmethod
    def _count_clear_frames(cloudy_indices, seq_length):
        cloudy = np.asarray(cloudy_indices, dtype=np.int64)
        cloudy = np.unique(cloudy[(cloudy >= 0) & (cloudy < seq_length)])
        return int(seq_length - cloudy.size)

    @staticmethod
    def _select_mask_indices_from_pool(masks_pool, num_frames, split):
        all_indices = torch.arange(masks_pool.shape[0], dtype=torch.long)
        coverages = (masks_pool > 0).float().mean(dim=(1, 2, 3))
        buckets = [
            all_indices[(coverages >= 0.10) & (coverages < 0.60)],
            all_indices[(coverages >= 0.60) & (coverages < 0.80)],
            all_indices[(coverages >= 0.80) & (coverages < 0.95)],
            all_indices[(coverages >= 0.95) & (coverages <= 1.00)],
        ]
        available = [
            index for index, bucket in enumerate(buckets)
            if bucket.numel() > 0
        ]
        if not available:
            buckets = [all_indices]
            available = [0]

        if split == "train":
            probabilities = torch.tensor(
                [0.65, 0.25, 0.07, 0.03], dtype=torch.float32
            )[available]
            probabilities = probabilities / probabilities.sum()
            sampled_buckets = torch.multinomial(
                probabilities, num_frames, replacement=True
            )
            selected = []
            for position in sampled_buckets.tolist():
                bucket = buckets[available[position]]
                selected.append(
                    bucket[torch.randint(0, bucket.numel(), (1,)).item()]
                )
            return torch.stack(selected).long()

        ordered = torch.cat([buckets[index] for index in available])
        repeats = int(np.ceil(num_frames / ordered.numel()))
        return ordered.repeat(repeats)[:num_frames]
