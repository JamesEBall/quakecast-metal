"""Rate head that wraps SmaAt-UNet and optional training utilities.

The wrapper keeps the network output in log-rate space so every downstream
metric stays unchanged. Its one addition is an optional persistence offset:
the network then predicts a log-multiplier on the seven-day-mean rate that the
benchmark uses as its baseline, instead of predicting the rate from scratch.

Author: James Edward Ball
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from quakecast.model import SmaAtUNet

WRAPPER_NAME = "rate-forecaster"


class RateForecaster(nn.Module):
    """SmaAt-UNet log-rate forecaster with an optional persistence offset."""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        baseline_offset: bool = False,
        output_bias: float | None = None,
        floor: float = 0.02,
    ) -> None:
        super().__init__()
        self.core = SmaAtUNet(in_channels=in_channels, width=width)
        self.baseline_offset = baseline_offset
        # Softplus keeps the quiet-cell floor positive so log() stays finite.
        self.floor_raw = nn.Parameter(torch.tensor(math.log(math.expm1(floor))))
        if output_bias is not None:
            nn.init.constant_(self.core.output.bias, output_bias)
        if baseline_offset:
            # Start exactly at the persistence forecast, which already carries
            # the spatial structure. Zeroing the head without an offset instead
            # starts the model spatially flat, and it stays diffuse: a control
            # run held spatial CSI at 0.0 for two epochs because no cell ever
            # reached the 0.5 threshold.
            nn.init.zeros_(self.core.output.weight)

    def persistence_rate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x[:, :1]).clamp_min(0.0) / 7.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.core(x)
        if self.baseline_offset:
            floor = F.softplus(self.floor_raw)
            prediction = prediction + torch.log(self.persistence_rate(x) + floor)
        return prediction


def build_model(metadata: dict) -> nn.Module:
    """Rebuild the forecaster described by a checkpoint's metadata block.

    Checkpoints written before the wrapper existed hold a bare SmaAt-UNet.
    """
    if metadata.get("model_wrapper") != WRAPPER_NAME:
        return SmaAtUNet()
    return RateForecaster(
        width=int(metadata.get("width", 64)),
        baseline_offset=bool(metadata.get("baseline_offset", False)),
    )


class RateEnsemble(nn.Module):
    """Average member rates, returned in log space.

    Members are combined on the rate scale, not the log-rate scale, so the
    ensemble forecasts the mean expected count rather than a geometric mean.
    Downstream scoring exponentiates the output like any single model.
    """

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rates = torch.stack([member(x).exp() for member in self.members])
        return rates.mean(dim=0).clamp_min(1e-12).log()


class ExponentialMovingAverage:
    """Shadow copy of the weights, averaged over training steps."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)
        self.steps = 0

    def update(self, model: nn.Module) -> None:
        self.steps += 1
        # Warm up the average so early steps are not dominated by the init.
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        with torch.no_grad():
            for shadow, live in zip(
                self.shadow.state_dict().values(), model.state_dict().values(), strict=True
            ):
                if shadow.dtype.is_floating_point:
                    shadow.mul_(decay).add_(live.detach(), alpha=1 - decay)
                else:
                    shadow.copy_(live)


def cosine_schedule(step: int, total_steps: int, warmup_steps: int, floor: float = 0.01) -> float:
    """Linear warm-up then cosine decay to `floor` times the peak rate."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
