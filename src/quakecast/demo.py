"""Train and visualise a short synthetic SmaAt-UNet experiment.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import SyntheticAftershockDataset
from .model import SmaAtUNet


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("outputs/earthquake-demo.png"))
    args = parser.parse_args()

    torch.manual_seed(7)
    device = select_device()
    dataset = SyntheticAftershockDataset(size=args.samples)
    train_size = int(len(dataset) * 0.8)
    train, validation = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)

    model = SmaAtUNet().to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(features)

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for features, target in validation_loader:
                features, target = features.to(device), target.to(device)
                validation_loss += loss_fn(model(features), target).item() * len(features)
        row = {
            "epoch": epoch,
            "train_mse": train_loss / len(train),
            "validation_mse": validation_loss / len(validation),
        }
        history.append(row)
        print(json.dumps(row))

    features, target = validation[0]
    with torch.inference_mode():
        prediction = model(features[None].to(device)).cpu()[0, 0]
    observed = torch.expm1(target[0]).numpy()
    forecast = torch.expm1(prediction).clamp_min(0).numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    panels = (
        (features[0].numpy(), "Previous 7 days\nlog event count"),
        (observed, "Observed next day\nevent count"),
        (forecast, "SmaAt-UNet forecast\nexpected count"),
    )
    for axis, (image, title) in zip(axes, panels, strict=True):
        shown = axis.imshow(image, origin="lower", cmap="magma")
        axis.set_title(title)
        axis.set_xlabel("0.1 degree cells")
        fig.colorbar(shown, ax=axis, shrink=0.8)
    fig.suptitle("Synthetic pipeline check - research demonstration")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary = {
        "author": "James Edward Ball",
        "device": str(device),
        "parameters": parameter_count,
        "samples": len(dataset),
        "epochs": args.epochs,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "final_validation_mse": history[-1]["validation_mse"],
        "output": str(args.output),
        "forecast_grid": np.round(forecast, 5).tolist(),
        "recent_mean_depth_grid_km": np.round(features[2].numpy(), 2).tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
