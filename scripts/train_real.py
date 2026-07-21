#!/usr/bin/env python3
"""Train SmaAt-UNet on real catalogues without opening final-test labels.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from quakecast.demo import select_device
from quakecast.forecaster import (
    WRAPPER_NAME,
    ExponentialMovingAverage,
    RateForecaster,
    cosine_schedule,
)
from quakecast.catalogues import sha256


class TensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, path: Path, indexes: np.ndarray | None = None) -> None:
        archive = np.load(path)
        if "target_counts" not in archive.files:
            raise ValueError(f"Training package has no labels: {path}")
        self.inputs = archive["inputs"]
        self.targets = archive["target_counts"]
        self.component_ids = archive["component_ids"]
        self.trigger_ids = archive["trigger_ids"]
        self.catalogues = archive["catalogues"]
        self.indexes = np.arange(len(self.inputs)) if indexes is None else indexes

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indexes[position])
        features = torch.from_numpy(self.inputs[index])
        target_counts = torch.from_numpy(self.targets[index]).float()
        return features, target_counts, index


def sequence_sampler(dataset: TensorDataset, seed: int) -> WeightedRandomSampler:
    components = dataset.component_ids[dataset.indexes]
    sizes = Counter(components.tolist())
    weights = torch.tensor([1.0 / sizes[value] for value in components], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)


class CappedComponentSampler(torch.utils.data.Sampler[int]):
    """Draw at most `cap` distinct examples per connected sequence each epoch.

    Sequence-balanced weighting equalises sequences but resamples the same
    aftershock hundreds of times inside a productive one. Capping keeps the
    balancing while every drawn example within an epoch stays distinct.
    """

    def __init__(self, dataset: TensorDataset, cap: int, seed: int) -> None:
        components = dataset.component_ids[dataset.indexes]
        self.groups: list[np.ndarray] = []
        for value in sorted(set(components.tolist())):
            self.groups.append(np.flatnonzero(components == value))
        self.cap = cap
        self.generator = np.random.default_rng(seed)
        self.length = sum(min(len(group), cap) for group in self.groups)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        drawn = []
        for group in self.groups:
            take = min(len(group), self.cap)
            drawn.extend(self.generator.choice(group, size=take, replace=False).tolist())
        self.generator.shuffle(drawn)
        return iter(drawn)


def poisson_log_likelihood(target: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    rate = rate.clamp_min(1e-6)
    return (target * rate.log() - rate - torch.lgamma(target + 1)).sum(dim=(1, 2, 3))


def poisson_nll_from_rate(target: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    return (rate - target * rate.clamp_min(1e-8).log()).mean()


def multiscale_penalty(log_rate: torch.Tensor, target: torch.Tensor, scales: tuple[int, ...]) -> torch.Tensor:
    """Poisson loss on block-summed counts as well as individual cells.

    Cell-wise loss treats a forecast one cell away from an aftershock exactly
    like one on the far side of the grid. Scoring block sums too rewards
    putting rate in the right neighbourhood, which is what average precision
    over occupied cells measures.
    """
    rate = log_rate.exp()
    penalty = log_rate.new_zeros(())
    for scale in scales:
        area = scale * scale
        penalty = penalty + poisson_nll_from_rate(
            F.avg_pool2d(target, scale) * area, F.avg_pool2d(rate, scale) * area
        )
    return penalty / len(scales)


def calibration_penalty(log_rate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared log mismatch between forecast and observed events per example."""
    forecast = log_rate.exp().sum(dim=(1, 2, 3))
    observed = target.sum(dim=(1, 2, 3))
    return (torch.log1p(forecast) - torch.log1p(observed)).pow(2).mean()


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve for occupied cells.

    Threshold-free, and unlike the likelihood scores it is invariant to the
    clamp applied to zero-rate cells, so it measures ranking skill alone.
    """
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = ranked.sum()
    if positives == 0:
        return 0.0
    true_positives = np.cumsum(ranked)
    precision = true_positives / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def sequence_mean(values: np.ndarray, components: np.ndarray) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, component in zip(values, components, strict=True):
        grouped[str(component)].append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def validate(
    model: nn.Module,
    loader: DataLoader,
    dataset: TensorDataset,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    model.eval()
    squared_error, model_ll, baseline_ll = [], [], []
    floored_ll, smoothed_ll = [], []
    observed, forecast_totals, indexes = [], [], []
    cell_rates, cell_labels, cell_baseline = [], [], []
    spatial_tp = spatial_fp = spatial_fn = 0
    with torch.inference_mode():
        for features, target_counts, batch_indexes in loader:
            features = features.to(device)
            target_counts = target_counts.to(device)
            prediction = model(features)
            target_log = torch.log1p(target_counts)
            prediction_rate = (
                torch.expm1(prediction).clamp_min(0)
                if loss_name == "log-mse"
                else torch.exp(prediction).clamp_max(1e6)
            )
            comparable_log_prediction = torch.log1p(prediction_rate)
            baseline_rate = torch.expm1(features[:, :1]) / 7.0
            squared_error.extend(
                ((comparable_log_prediction - target_log) ** 2).mean(dim=(1, 2, 3)).cpu().tolist()
            )
            model_ll.extend(poisson_log_likelihood(target_counts, prediction_rate).cpu().tolist())
            baseline_ll.extend(poisson_log_likelihood(target_counts, baseline_rate).cpu().tolist())
            # Two less degenerate references. The frozen baseline predicts a
            # hard zero wherever the past week was empty, and the 1e-6 clamp
            # then charges 13.8 nats for every event that lands there.
            floored_ll.extend(
                poisson_log_likelihood(target_counts, baseline_rate.clamp_min(1e-3)).cpu().tolist()
            )
            uniform = baseline_rate.mean(dim=(1, 2, 3), keepdim=True)
            smoothed_ll.extend(
                poisson_log_likelihood(target_counts, 0.5 * baseline_rate + 0.5 * uniform)
                .cpu()
                .tolist()
            )
            cell_rates.append(prediction_rate.flatten().cpu().numpy())
            cell_labels.append((target_counts > 0).flatten().cpu().numpy())
            cell_baseline.append(baseline_rate.flatten().cpu().numpy())
            observed.extend(target_counts.sum(dim=(1, 2, 3)).cpu().tolist())
            forecast_totals.extend(prediction_rate.sum(dim=(1, 2, 3)).cpu().tolist())
            predicted_positive = prediction_rate >= 0.5
            observed_positive = target_counts > 0
            spatial_tp += int((predicted_positive & observed_positive).sum().item())
            spatial_fp += int((predicted_positive & ~observed_positive).sum().item())
            spatial_fn += int((~predicted_positive & observed_positive).sum().item())
            indexes.extend(batch_indexes.tolist())
    indexes_array = np.asarray(indexes, dtype=np.intp)
    components = dataset.component_ids[indexes_array]
    model_values = np.asarray(model_ll)
    baseline_values = np.asarray(baseline_ll)
    observations = np.asarray(observed)
    forecasts = np.asarray(forecast_totals)
    delta = model_values - baseline_values
    total_observed = max(float(observations.sum()), 1.0)
    precision = spatial_tp / max(spatial_tp + spatial_fp, 1)
    recall = spatial_tp / max(spatial_tp + spatial_fn, 1)
    metrics = {
        "log_mse_per_trigger": float(np.mean(squared_error)),
        "log_mse_per_sequence": sequence_mean(np.asarray(squared_error), components),
        "poisson_log_likelihood_model": float(model_values.sum()),
        "poisson_log_likelihood_seven_day_mean": float(baseline_values.sum()),
        "information_gain_per_observed_event": float(delta.sum() / total_observed),
        "sequence_mean_log_likelihood_gain": sequence_mean(delta, components),
        "observed_events": float(observations.sum()),
        "forecast_events": float(forecasts.sum()),
        "forecast_observed_ratio": float(forecasts.sum() / total_observed),
        "mean_absolute_number_error": float(np.mean(np.abs(forecasts - observations))),
        "spatial_precision_at_0_5": float(precision),
        "spatial_recall_at_0_5": float(recall),
        "spatial_csi_at_0_5": float(spatial_tp / max(spatial_tp + spatial_fp + spatial_fn, 1)),
        "spatial_false_alarm_ratio_at_0_5": float(spatial_fp / max(spatial_tp + spatial_fp, 1)),
        "sequences": float(len(set(components.tolist()))),
    }
    # Artefact-free companions to the frozen primary score. The headline number
    # is dominated by empty cells the persistence baseline scores at the clamp,
    # so research decisions should read these alongside it.
    rates_flat = np.concatenate(cell_rates)
    labels_flat = np.concatenate(cell_labels)
    baseline_flat = np.concatenate(cell_baseline)
    metrics["information_gain_per_event_floored_baseline"] = float(
        (model_values - np.asarray(floored_ll)).sum() / total_observed
    )
    metrics["information_gain_per_event_smoothed_baseline"] = float(
        (model_values - np.asarray(smoothed_ll)).sum() / total_observed
    )
    metrics["spatial_average_precision"] = average_precision(rates_flat, labels_flat)
    metrics["spatial_average_precision_persistence"] = average_precision(
        baseline_flat, labels_flat
    )
    metrics["occupied_cell_fraction"] = float(labels_flat.mean())
    most_productive = int(np.argmax(observations))
    metrics["maximum_observed_trigger_events"] = float(observations[most_productive])
    metrics["forecast_for_maximum_observed_trigger"] = float(forecasts[most_productive])
    metrics["maximum_trigger_forecast_observed_ratio"] = float(
        forecasts[most_productive] / max(observations[most_productive], 1)
    )
    catalogues = dataset.catalogues[indexes_array]
    for catalogue in sorted(set(catalogues.tolist())):
        mask = catalogues == catalogue
        catalogue_observed = max(float(observations[mask].sum()), 1.0)
        prefix = f"catalogue_{catalogue}"
        metrics[f"{prefix}_information_gain_per_event"] = float(delta[mask].sum() / catalogue_observed)
        metrics[f"{prefix}_forecast_observed_ratio"] = float(
            forecasts[mask].sum() / catalogue_observed
        )
        metrics[f"{prefix}_log_mse"] = float(np.mean(np.asarray(squared_error)[mask]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("log-mse", "poisson"), default="log-mse")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument(
        "--sequence-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Give every connected sequence equal sampling mass",
    )
    parser.add_argument(
        "--max-per-component",
        type=int,
        help="Non-replacement sequence balancing: cap examples drawn per sequence per epoch",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--multiscale-weight",
        type=float,
        default=0.0,
        help="Weight on Poisson loss over 2x2 and 4x4 block sums",
    )
    parser.add_argument(
        "--calibration-weight",
        type=float,
        default=0.0,
        help="Weight on squared log mismatch of total forecast events",
    )
    parser.add_argument("--gradient-clip", type=float, help="Clip global gradient norm")
    parser.add_argument("--scheduler", choices=("step", "cosine"), default="step")
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, help="Track and ship EMA weights")
    parser.add_argument(
        "--baseline-offset",
        action="store_true",
        help="Predict a log-multiplier on the seven-day persistence rate",
    )
    parser.add_argument("--width", type=int, default=64, help="Base channel count")
    parser.add_argument(
        "--bias-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the output layer at the mean training rate",
    )
    parser.add_argument(
        "--select-best",
        action="store_true",
        help="Ship the epoch with the best validation information gain, not the last",
    )
    parser.add_argument(
        "--calibration-band",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("LOW", "HIGH"),
        help="Epochs outside this forecast/observed band are ineligible for --select-best",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = args.root.expanduser().resolve()
    train_path = root / "processed" / "tensors" / "train.npz"
    validation_path = root / "processed" / "tensors" / "validation.npz"
    full_train_size = len(np.load(train_path)["inputs"])
    selected = None
    if args.max_train_examples and args.max_train_examples < full_train_size:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(full_train_size, args.max_train_examples, replace=False))
    train_data = TensorDataset(train_path, selected)
    validation_data = TensorDataset(validation_path)
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config={
                "author": "James Edward Ball",
                "architecture": "SmaAtUNet",
                "loss": args.loss,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "sequence_balanced": args.sequence_balanced,
                "max_per_component": args.max_per_component,
                "weight_decay": args.weight_decay,
                "multiscale_weight": args.multiscale_weight,
                "calibration_weight": args.calibration_weight,
                "gradient_clip": args.gradient_clip,
                "scheduler": args.scheduler,
                "warmup_epochs": args.warmup_epochs,
                "ema_decay": args.ema_decay,
                "baseline_offset": args.baseline_offset,
                "width": args.width,
                "select_best": args.select_best,
                "train_examples": len(train_data),
                "train_components": len(set(train_data.component_ids[train_data.indexes].tolist())),
                "validation_examples": len(validation_data),
                "validation_components": len(set(validation_data.component_ids.tolist())),
                "final_test_opened": False,
                "processed_manifest_sha256": sha256(root / "processed" / "processed-manifest.json"),
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("validation/*", step_metric="epoch")
    sampler: torch.utils.data.Sampler[int] | None = None
    if args.max_per_component:
        sampler = CappedComponentSampler(train_data, args.max_per_component, args.seed)
    elif args.sequence_balanced:
        sampler = sequence_sampler(train_data, args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
    )
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)

    device = select_device()
    model = RateForecaster(
        width=args.width,
        baseline_offset=args.baseline_offset,
        output_bias=(
            starting_bias(train_data, args.baseline_offset, args.loss)
            if args.bias_init
            else None
        ),
    ).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
        if args.scheduler == "step"
        else torch.optim.lr_scheduler.LambdaLR(
            optimiser, lambda step: cosine_schedule(step, total_steps, warmup_steps)
        )
    )
    averager = ExponentialMovingAverage(model, args.ema_decay) if args.ema_decay else None
    loss_fn: nn.Module = (
        nn.MSELoss()
        if args.loss == "log-mse"
        else nn.PoissonNLLLoss(log_input=True, full=False)
    )
    history = []
    best = {"score": -float("inf"), "epoch": 0, "state": None}
    low_band, high_band = args.calibration_band
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for features, target_counts, _ in train_loader:
            features, target_counts = features.to(device), target_counts.to(device)
            optimiser.zero_grad(set_to_none=True)
            prediction = model(features)
            target_for_loss = torch.log1p(target_counts) if args.loss == "log-mse" else target_counts
            loss = loss_fn(prediction, target_for_loss)
            if args.multiscale_weight:
                loss = loss + args.multiscale_weight * multiscale_penalty(
                    prediction, target_counts, (2, 4)
                )
            if args.calibration_weight:
                loss = loss + args.calibration_weight * calibration_penalty(
                    prediction, target_counts
                )
            loss.backward()
            if args.gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimiser.step()
            if args.scheduler == "cosine":
                scheduler.step()
            if averager:
                averager.update(model)
            loss_sum += loss.item() * len(features)
            seen += len(features)
        if args.scheduler == "step":
            scheduler.step()
        shipped = averager.shadow if averager else model
        metrics = validate(shipped, validation_loader, validation_data, device, args.loss)
        objective_name = "log_mse" if args.loss == "log-mse" else "poisson_nll"
        row = {
            "epoch": epoch,
            f"train_{objective_name}": loss_sum / max(seen, 1),
            "learning_rate": scheduler.get_last_lr()[0],
            **metrics,
        }
        history.append(row)
        score = metrics["information_gain_per_observed_event"]
        calibrated = low_band <= metrics["forecast_observed_ratio"] <= high_band
        if calibrated and score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in shipped.state_dict().items()},
            }
        print(json.dumps(row), flush=True)
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    f"train/{objective_name}": loss_sum / max(seen, 1),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    **{f"validation/{key}": value for key, value in metrics.items()},
                }
            )

    shipped = averager.shadow if averager else model
    selected_epoch = args.epochs
    if args.select_best and best["state"] is not None:
        shipped.load_state_dict(best["state"])
        selected_epoch = int(best["epoch"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "metadata": {
            "author": "James Edward Ball",
            "architecture": "SmaAtUNet",
            "model_wrapper": WRAPPER_NAME,
            "device": str(device),
            "loss": args.loss,
            "paper_loss": args.loss == "log-mse",
            "sequence_balanced": args.sequence_balanced,
            "max_per_component": args.max_per_component,
            "baseline_offset": args.baseline_offset,
            "width": args.width,
            "bias_init": args.bias_init,
            "ema_decay": args.ema_decay,
            "scheduler": args.scheduler,
            "weight_decay": args.weight_decay,
            "multiscale_weight": args.multiscale_weight,
            "calibration_weight": args.calibration_weight,
            "gradient_clip": args.gradient_clip,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "selected_epoch": selected_epoch,
            "train_examples": len(train_data),
            "validation_examples": len(validation_data),
            "final_test_opened": False,
        },
        "model_state_dict": shipped.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, args.output)
    summary = {
        **checkpoint["metadata"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": args.epochs,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "checkpoint": str(args.output),
        "selected_validation": history[selected_epoch - 1],
        "final_validation": history[-1],
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run:
        import wandb

        final_metrics = history[-1]
        for key, value in final_metrics.items():
            if key != "epoch":
                wandb_run.summary[f"final/{key}"] = value
        wandb_run.summary["governance/final_test_opened"] = False
        artifact = wandb.Artifact(
            name=f"{args.output.stem}-checkpoint",
            type="model",
            metadata={
                "author": "James Edward Ball",
                "loss": args.loss,
                "final_test_opened": False,
            },
        )
        artifact.add_file(str(args.output))
        artifact.add_file(str(args.output.with_suffix(".json")))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(json.dumps(summary, indent=2))


def starting_bias(dataset: TensorDataset, baseline_offset: bool, loss_name: str) -> float | None:
    """Initialise the output layer at the mean rate, or at persistence.

    Without this the first epoch's exp() forecast overshoots by four orders of
    magnitude and the run spends its early budget recovering.
    """
    if loss_name != "poisson":
        return None
    targets = dataset.targets[dataset.indexes].astype(np.float64)
    if baseline_offset:
        persistence = np.expm1(dataset.inputs[dataset.indexes][:, :1]) / 7.0
        scale = targets.sum() / max(persistence.sum() + 0.02 * targets[:, 0].size, 1e-6)
        return float(np.log(max(scale, 1e-6)))
    return float(np.log(max(targets.mean(), 1e-6)))


if __name__ == "__main__":
    main()
