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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from quakecast.demo import select_device
from quakecast.model import SmaAtUNet
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


def poisson_log_likelihood(target: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    rate = rate.clamp_min(1e-6)
    return (target * rate.log() - rate - torch.lgamma(target + 1)).sum(dim=(1, 2, 3))


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
    observed, forecast_totals, indexes = [], [], []
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
    sampler = sequence_sampler(train_data, args.seed) if args.sequence_balanced else None
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
    )
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)

    device = select_device()
    model = SmaAtUNet().to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
    loss_fn: nn.Module = (
        nn.MSELoss()
        if args.loss == "log-mse"
        else nn.PoissonNLLLoss(log_input=True, full=False)
    )
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for features, target_counts, _ in train_loader:
            features, target_counts = features.to(device), target_counts.to(device)
            optimiser.zero_grad(set_to_none=True)
            prediction = model(features)
            target_for_loss = torch.log1p(target_counts) if args.loss == "log-mse" else target_counts
            loss = loss_fn(prediction, target_for_loss)
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * len(features)
        scheduler.step()
        metrics = validate(model, validation_loader, validation_data, device, args.loss)
        objective_name = "log_mse" if args.loss == "log-mse" else "poisson_nll"
        row = {
            "epoch": epoch,
            f"train_{objective_name}": loss_sum / len(train_data),
            "learning_rate": scheduler.get_last_lr()[0],
            **metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    f"train/{objective_name}": loss_sum / len(train_data),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    **{f"validation/{key}": value for key, value in metrics.items()},
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "metadata": {
            "author": "James Edward Ball",
            "architecture": "SmaAtUNet",
            "device": str(device),
            "loss": args.loss,
            "paper_loss": args.loss == "log-mse",
            "sequence_balanced": args.sequence_balanced,
            "train_examples": len(train_data),
            "validation_examples": len(validation_data),
            "final_test_opened": False,
        },
        "model_state_dict": model.state_dict(),
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


if __name__ == "__main__":
    main()
