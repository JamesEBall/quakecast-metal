"""Catalogue-to-grid preprocessing and a synthetic aftershock dataset.

Author: James Edward Ball
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Event:
    time: datetime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float


@dataclass(frozen=True)
class CatalogueIndex:
    """Chronological numeric arrays for fast trigger-centred window queries."""

    times_seconds: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    depth_km: np.ndarray
    magnitude: np.ndarray

    @classmethod
    def from_normalized_csv(cls, path: Path) -> "CatalogueIndex":
        columns: dict[str, list[float]] = {
            "times_seconds": [],
            "latitude": [],
            "longitude": [],
            "depth_km": [],
            "magnitude": [],
        }
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                columns["times_seconds"].append(utc_time(row["time_utc"]).timestamp())
                for field in ("latitude", "longitude", "depth_km", "magnitude"):
                    columns[field].append(float(row[field]))
        order = np.argsort(np.asarray(columns["times_seconds"]), kind="stable")
        return cls(
            times_seconds=np.asarray(columns["times_seconds"], dtype=np.float64)[order],
            latitude=np.asarray(columns["latitude"], dtype=np.float32)[order],
            longitude=np.asarray(columns["longitude"], dtype=np.float32)[order],
            depth_km=np.asarray(columns["depth_km"], dtype=np.float32)[order],
            magnitude=np.asarray(columns["magnitude"], dtype=np.float32)[order],
        )

    def window(self, trigger_time: datetime) -> tuple[np.ndarray, ...]:
        trigger_seconds = trigger_time.timestamp()
        start = trigger_seconds - 7 * 86400
        end = trigger_seconds + 86400
        left = int(np.searchsorted(self.times_seconds, start, side="left"))
        right = int(np.searchsorted(self.times_seconds, end, side="right"))
        return (
            self.times_seconds[left:right],
            self.latitude[left:right],
            self.longitude[left:right],
            self.depth_km[left:right],
            self.magnitude[left:right],
        )


def grid_arrays(
    times_seconds: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    depth_km: np.ndarray,
    magnitude: np.ndarray,
    trigger_time: datetime,
    trigger_latitude: float,
    trigger_longitude: float,
    grid_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the three SmaAt-UNet maps and the raw next-day count map."""
    trigger_seconds = trigger_time.timestamp()
    relative_time = times_seconds - trigger_seconds
    relative_longitude = (longitude - trigger_longitude + 180.0) % 360.0 - 180.0
    relative_latitude = latitude - trigger_latitude
    spatial = (
        (relative_longitude >= -1.0)
        & (relative_longitude < 1.0)
        & (relative_latitude >= -1.0)
        & (relative_latitude < 1.0)
    )
    cell_size = 2.0 / grid_size
    x = np.floor((relative_longitude[spatial] + 1.0) / cell_size).astype(np.intp)
    y = np.floor((relative_latitude[spatial] + 1.0) / cell_size).astype(np.intp)
    times = relative_time[spatial]
    depths = depth_km[spatial]
    magnitudes = magnitude[spatial]

    daily_rate = np.zeros((7, grid_size, grid_size), dtype=np.float32)
    daily_mag = np.zeros_like(daily_rate)
    daily_depth_sum = np.zeros_like(daily_rate)
    daily_depth_count = np.zeros_like(daily_rate)
    input_mask = (times >= -7 * 86400) & (times <= 0)
    if np.any(input_mask):
        input_days = np.minimum(6, np.floor((times[input_mask] + 7 * 86400) / 86400)).astype(
            np.intp
        )
        iy, ix = y[input_mask], x[input_mask]
        np.add.at(daily_rate, (input_days, iy, ix), 1)
        np.maximum.at(daily_mag, (input_days, iy, ix), magnitudes[input_mask])
        np.add.at(daily_depth_sum, (input_days, iy, ix), depths[input_mask])
        np.add.at(daily_depth_count, (input_days, iy, ix), 1)

    daily_depth = np.divide(
        daily_depth_sum,
        daily_depth_count,
        out=np.zeros_like(daily_depth_sum),
        where=daily_depth_count > 0,
    )
    inputs = np.stack(
        (
            np.log1p(daily_rate.sum(axis=0)),
            daily_mag.max(axis=0),
            daily_depth.mean(axis=0),
        )
    ).astype(np.float32)

    target = np.zeros((1, grid_size, grid_size), dtype=np.uint16)
    target_mask = (times > 0) & (times <= 86400)
    if np.any(target_mask):
        np.add.at(target[0], (y[target_mask], x[target_mask]), 1)
    return inputs, target


def indexed_sample(
    index: CatalogueIndex, trigger_record: dict[str, object], grid_size: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    trigger_time = utc_time(str(trigger_record["trigger_time_utc"]))
    arrays = index.window(trigger_time)
    return grid_arrays(
        *arrays,
        trigger_time=trigger_time,
        trigger_latitude=float(trigger_record["latitude"]),
        trigger_longitude=float(trigger_record["longitude"]),
        grid_size=grid_size,
    )


def read_trigger_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def events_to_sample(events: list[Event], trigger: Event, grid_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Build the paper's three weekly input maps and next-day target map.

    The grid is 2 by 2 degrees, centred on an M4+ trigger, with 0.1-degree
    cells by default. Events below M2 or deeper than 40 km are excluded.
    """
    if trigger.magnitude < 4:
        raise ValueError("Trigger magnitude must be at least 4")

    selected = [event for event in events if event.magnitude >= 2 and 0 <= event.depth_km <= 40]
    times = np.asarray([event.time.timestamp() for event in selected], dtype=np.float64)
    inputs, target = grid_arrays(
        times,
        np.asarray([event.latitude for event in selected], dtype=np.float32),
        np.asarray([event.longitude for event in selected], dtype=np.float32),
        np.asarray([event.depth_km for event in selected], dtype=np.float32),
        np.asarray([event.magnitude for event in selected], dtype=np.float32),
        trigger.time,
        trigger.latitude,
        trigger.longitude,
        grid_size,
    )
    return inputs, np.log1p(target).astype(np.float32)


class SyntheticAftershockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Small clustered dataset for checking the complete training pipeline."""

    def __init__(self, size: int = 128, grid_size: int = 20, seed: int = 7):
        rng = np.random.default_rng(seed)
        inputs, targets = [], []
        yy, xx = np.mgrid[:grid_size, :grid_size]
        for _ in range(size):
            cx, cy = rng.uniform(5, grid_size - 5, size=2)
            width = rng.uniform(1.0, 2.8)
            spatial = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width**2))
            productivity = rng.uniform(2.0, 8.0)
            weekly_rate = rng.poisson(productivity * spatial).astype(np.float32)
            magnitude = np.where(weekly_rate > 0, 2 + rng.exponential(0.45, weekly_rate.shape), 0)
            magnitude = np.clip(magnitude, 0, 6.5).astype(np.float32)
            depth = np.where(weekly_rate > 0, rng.normal(9, 3, weekly_rate.shape), 0)
            depth = np.clip(depth, 0, 40).astype(np.float32)
            next_day = rng.poisson((0.18 * productivity + 0.12) * spatial).astype(np.float32)
            inputs.append(np.stack((np.log1p(weekly_rate), magnitude, depth)))
            targets.append(np.log1p(next_day[None]))
        self.inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def utc_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
