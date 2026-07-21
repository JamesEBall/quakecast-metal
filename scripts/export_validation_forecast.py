#!/usr/bin/env python3
"""Export one real validation forecast and update the MRI-style viewer.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from quakecast.data import CatalogueIndex, utc_time
from quakecast.demo import select_device
from quakecast.model import SmaAtUNet


def recent_mean_depth(
    index: CatalogueIndex,
    trigger_time: str,
    trigger_latitude: float,
    trigger_longitude: float,
) -> np.ndarray:
    times, latitude, longitude, depth, _ = index.window(utc_time(trigger_time))
    relative_time = times - utc_time(trigger_time).timestamp()
    relative_longitude = (longitude - trigger_longitude + 180.0) % 360.0 - 180.0
    relative_latitude = latitude - trigger_latitude
    mask = (
        (relative_time >= -7 * 86400)
        & (relative_time <= 0)
        & (relative_longitude >= -1)
        & (relative_longitude < 1)
        & (relative_latitude >= -1)
        & (relative_latitude < 1)
    )
    x = np.floor((relative_longitude[mask] + 1) / 0.1).astype(np.intp)
    y = np.floor((relative_latitude[mask] + 1) / 0.1).astype(np.intp)
    sums = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros((20, 20), dtype=np.float32)
    np.add.at(sums, (y, x), depth[mask])
    np.add.at(counts, (y, x), 1)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def update_viewer(template: Path, output: Path, data: dict[str, object]) -> None:
    fragment = template.read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"))
    fragment, replacements = re.subn(
        r'(<script type="application/json" id="ev-data">).*?(</script>)',
        lambda match: match.group(1) + payload + match.group(2),
        fragment,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError("Viewer template has no unique ev-data block")
    detail = (
        f"Real validation forecast - {html.escape(str(data['catalogue']).upper())} - "
        f"{html.escape(str(data['trigger_time_utc']))} - "
        f"M{float(data['trigger_magnitude']):.1f}. "
        "Depth is an exploratory allocation using the actual mean depth of recent events; "
        "the network forecasts a 2D surface rate."
    )
    fragment, replacements = re.subn(
        r'<div class="ev-note text-small">.*?</div>',
        f'<div class="ev-note text-small">{detail}</div>',
        fragment,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError("Viewer template has no unique note")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer-template", type=Path, required=True)
    parser.add_argument("--viewer-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    archive = np.load(root / "processed" / "tensors" / "validation.npz")
    target_totals = archive["target_counts"].sum(axis=(1, 2, 3))
    position = int(np.argmax(target_totals))
    features = torch.from_numpy(archive["inputs"][position : position + 1])
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SmaAtUNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    with torch.inference_mode():
        raw_prediction = model(features.to(device))
        prediction = (
            torch.expm1(raw_prediction).clamp_min(0)
            if checkpoint["metadata"].get("loss", "log-mse") == "log-mse"
            else torch.exp(raw_prediction).clamp_max(1e6)
        ).cpu()[0, 0].numpy()
    observed = archive["target_counts"][position, 0].astype(np.float32)
    catalogue = str(archive["catalogues"][position])
    trigger_time = str(archive["trigger_times_utc"][position])
    trigger_latitude = float(archive["trigger_latitude"][position])
    trigger_longitude = float(archive["trigger_longitude"][position])
    trigger_magnitude = float(archive["trigger_magnitude"][position])
    index = CatalogueIndex.from_normalized_csv(
        root / "processed" / "normalized" / f"{catalogue}.csv.gz"
    )
    depth = recent_mean_depth(index, trigger_time, trigger_latitude, trigger_longitude)
    result = {
        "author": "James Edward Ball",
        "source": "real validation example; final test unopened",
        "device": str(device),
        "trigger_id": str(archive["trigger_ids"][position]),
        "component_id": str(archive["component_ids"][position]),
        "catalogue": catalogue,
        "trigger_time_utc": trigger_time,
        "trigger_latitude": trigger_latitude,
        "trigger_longitude": trigger_longitude,
        "trigger_magnitude": trigger_magnitude,
        "observed_events": int(observed.sum()),
        "forecast_events": round(float(prediction.sum()), 4),
        "forecast_grid": np.round(prediction, 5).tolist(),
        "observed_grid": observed.astype(int).tolist(),
        "recent_mean_depth_grid_km": np.round(depth, 2).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    panels = (
        (np.expm1(archive["inputs"][position, 0]), "Previous 7 days", "magma"),
        (observed, "Observed next day", "magma"),
        (prediction, "SmaAt-UNet forecast", "magma"),
        (depth, "Recent mean depth (km)", "viridis"),
    )
    for axis, (values, title, colour) in zip(axes, panels, strict=True):
        shown = axis.imshow(values, origin="lower", cmap=colour)
        axis.set_title(title)
        axis.set_xlabel("0.1-degree cells")
        fig.colorbar(shown, ax=axis, shrink=0.76)
    axes[0].set_ylabel("0.1-degree cells")
    fig.suptitle(
        f"Validation example - {catalogue.upper()} - {trigger_time[:10]} - M{trigger_magnitude:.1f}"
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    update_viewer(args.viewer_template, args.viewer_output, result)
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("grid")}, indent=2))


if __name__ == "__main__":
    main()
