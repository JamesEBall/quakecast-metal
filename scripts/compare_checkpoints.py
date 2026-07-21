#!/usr/bin/env python3
"""Score checkpoints and their ensemble with sequence-level bootstrap intervals.

Information gain is a sum over 200 connected sequences, and a handful of
productive sequences carry most of it. A single number therefore hides how much
of any improvement is real. This resamples whole sequences to put an interval
around every comparison, and evaluates the rate-averaged ensemble alongside its
members.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from quakecast.demo import select_device
from quakecast.forecaster import build_model
from train_real import TensorDataset, poisson_log_likelihood


def predicted_rates(checkpoint_path: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("final_test_opened", False):
        raise SystemExit(f"Checkpoint says final test labels were opened: {checkpoint_path}")
    loss_name = metadata.get("loss", "log-mse")
    model = build_model(metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rates = []
    with torch.inference_mode():
        for features, _, _ in loader:
            prediction = model(features.to(device))
            rate = (
                torch.expm1(prediction).clamp_min(0)
                if loss_name == "log-mse"
                else torch.exp(prediction).clamp_max(1e6)
            )
            rates.append(rate.cpu().numpy())
    return np.concatenate(rates)


def sequence_scores(
    rates: np.ndarray, targets: np.ndarray, baseline: np.ndarray, components: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-sequence likelihood gain over persistence, and observed event counts."""
    model_ll = poisson_log_likelihood(torch.from_numpy(targets), torch.from_numpy(rates)).numpy()
    baseline_ll = poisson_log_likelihood(
        torch.from_numpy(targets), torch.from_numpy(baseline)
    ).numpy()
    delta = model_ll - baseline_ll
    observed = targets.sum(axis=(1, 2, 3))
    gains: dict[str, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)
    for value, component, events in zip(delta, components, observed, strict=True):
        gains[str(component)] += float(value)
        counts[str(component)] += float(events)
    return dict(gains), dict(counts)


def bootstrap_interval(
    gains: dict[str, float], counts: dict[str, float], draws: int, seed: int
) -> dict[str, float]:
    names = sorted(gains)
    gain_values = np.array([gains[name] for name in names])
    count_values = np.array([counts[name] for name in names])
    generator = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        picked = generator.integers(0, len(names), len(names))
        samples[draw] = gain_values[picked].sum() / max(count_values[picked].sum(), 1.0)
    return {
        "information_gain_per_observed_event": float(
            gain_values.sum() / max(count_values.sum(), 1.0)
        ),
        "bootstrap_low": float(np.percentile(samples, 2.5)),
        "bootstrap_high": float(np.percentile(samples, 97.5)),
        "bootstrap_std": float(samples.std()),
    }


def paired_difference(
    left: dict[str, float],
    right: dict[str, float],
    counts: dict[str, float],
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the left-minus-right gain on the same resampled sequences."""
    names = sorted(counts)
    left_values = np.array([left[name] for name in names])
    right_values = np.array([right[name] for name in names])
    count_values = np.array([counts[name] for name in names])
    generator = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        picked = generator.integers(0, len(names), len(names))
        total = max(count_values[picked].sum(), 1.0)
        samples[draw] = (left_values[picked].sum() - right_values[picked].sum()) / total
    difference = (left_values.sum() - right_values.sum()) / max(count_values.sum(), 1.0)
    return {
        "difference": float(difference),
        "bootstrap_low": float(np.percentile(samples, 2.5)),
        "bootstrap_high": float(np.percentile(samples, 97.5)),
        "probability_positive": float((samples > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, help="Checkpoint to compare the others against")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    validation_path = root / "processed" / "tensors" / "validation.npz"
    dataset = TensorDataset(validation_path)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    device = select_device()
    targets = dataset.targets.astype(np.float32)
    baseline = np.expm1(dataset.inputs[:, :1]) / 7.0
    components = dataset.component_ids

    rates = {path.stem: predicted_rates(path, loader, device) for path in args.checkpoints}
    if len(rates) > 1:
        rates["ensemble"] = np.mean(list(rates.values()), axis=0)

    report: dict[str, object] = {"author": "James Edward Ball", "final_test_opened": False}
    per_member: dict[str, dict[str, float]] = {}
    counts: dict[str, float] = {}
    for name, rate in rates.items():
        gains, counts = sequence_scores(rate, targets, baseline, components)
        per_member[name] = gains
        report[name] = {
            **bootstrap_interval(gains, counts, args.draws, args.seed),
            "forecast_observed_ratio": float(rate.sum() / max(targets.sum(), 1.0)),
        }

    reference = args.reference.stem if args.reference else None
    if reference and reference in per_member:
        comparisons = {}
        for name, gains in per_member.items():
            if name != reference:
                comparisons[name] = paired_difference(
                    gains, per_member[reference], counts, args.draws, args.seed
                )
        report["paired_against_" + reference] = comparisons

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
