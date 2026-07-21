#!/usr/bin/env python3
"""Build paper-faithful 20x20 real-catalogue tensors from frozen splits.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quakecast.catalogues import iso_z, seal_processed_products, sha256, write_json
from quakecast.data import CatalogueIndex, indexed_sample, read_trigger_manifest


GRID_SIZE = 20


def metadata_arrays(records: list[dict[str, object]]) -> dict[str, np.ndarray]:
    return {
        "trigger_ids": np.asarray([record["trigger_id"] for record in records]),
        "component_ids": np.asarray([record["component_id"] for record in records]),
        "catalogues": np.asarray([record["catalogue"] for record in records]),
        "trigger_times_utc": np.asarray([record["trigger_time_utc"] for record in records]),
        "trigger_latitude": np.asarray([record["latitude"] for record in records], dtype=np.float32),
        "trigger_longitude": np.asarray([record["longitude"] for record in records], dtype=np.float32),
        "trigger_magnitude": np.asarray([record["magnitude"] for record in records], dtype=np.float32),
    }


def build_split(
    root: Path,
    split: str,
    indexes: dict[str, CatalogueIndex],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    records = read_trigger_manifest(root / "processed" / "splits" / f"{split}.jsonl")
    inputs = np.empty((len(records), 3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    targets = np.empty((len(records), 1, GRID_SIZE, GRID_SIZE), dtype=np.uint16)
    for position, record in enumerate(records):
        inputs[position], targets[position] = indexed_sample(indexes[str(record["catalogue"])], record)
    return records, inputs, targets


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def product(path: Path, root: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "arrays": {name: list(array.shape) for name, array in arrays.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    indexes = {}
    for catalogue in ("scedc", "ncedc", "geonet"):
        path = root / "processed" / "normalized" / f"{catalogue}.csv.gz"
        indexes[catalogue] = CatalogueIndex.from_normalized_csv(path)
        print(f"indexed {catalogue}: {len(indexes[catalogue].times_seconds):,} events")

    products = []
    statistics = {}
    for split in ("train", "validation", "test"):
        records, inputs, targets = build_split(root, split, indexes)
        metadata = metadata_arrays(records)
        if split == "test":
            input_path = root / "processed" / "tensors" / "test-inputs.npz"
            input_arrays = {"inputs": inputs, **metadata}
            write_npz(input_path, **input_arrays)
            products.append(product(input_path, root, input_arrays))

            label_path = root / "processed" / "sealed" / "test-labels.npz"
            label_arrays = {
                "target_counts": targets,
                "trigger_ids": metadata["trigger_ids"],
                "component_ids": metadata["component_ids"],
            }
            write_npz(label_path, **label_arrays)
            products.append(product(label_path, root, label_arrays))
        else:
            path = root / "processed" / "tensors" / f"{split}.npz"
            arrays = {"inputs": inputs, "target_counts": targets, **metadata}
            write_npz(path, **arrays)
            products.append(product(path, root, arrays))
        statistics[split] = {
            "examples": len(records),
            "components": len({record["component_id"] for record in records}),
            "catalogues": dict(sorted(Counter(str(record["catalogue"]) for record in records).items())),
            "input_events": int(np.expm1(inputs[:, 0]).round().sum()),
            "target_events": int(targets.sum()),
            "occupied_target_cells": int(np.count_nonzero(targets)),
            "empty_target_examples": int(np.count_nonzero(targets.sum(axis=(1, 2, 3)) == 0)),
            "finite_inputs": bool(np.isfinite(inputs).all()),
        }
        print(f"built {split}: {len(records):,} examples")

    test_input_ids = np.load(root / "processed" / "tensors" / "test-inputs.npz")["trigger_ids"]
    sealed_ids = np.load(root / "processed" / "sealed" / "test-labels.npz")["trigger_ids"]
    integrity = {
        "test_inputs_match_sealed_labels": bool(np.array_equal(test_input_ids, sealed_ids)),
        "test_inputs_contain_no_target_array": "target_counts"
        not in np.load(root / "processed" / "tensors" / "test-inputs.npz").files,
        "all_inputs_finite": all(value["finite_inputs"] for value in statistics.values()),
    }
    summary = {
        "metadata": {
            "author": "James Edward Ball",
            "created_utc": iso_z(datetime.now(timezone.utc)),
            "grid": "20x20 cells over 2x2 degrees, south-west inclusive and north-east exclusive",
            "input_window": "Seven non-overlapping 24-hour periods ending with and including trigger",
            "target_window": "24 hours immediately after trigger, trigger excluded",
            "channels": [
                "log1p(sum of seven daily M2+ count maps)",
                "maximum of seven daily maximum-magnitude maps",
                "mean of seven daily mean-depth maps in km; empty daily cells are zero",
            ],
            "target": "Raw M2+ event counts; transform in training only when required by the loss",
            "catalogue_isolation": "Each example uses events from its trigger catalogue only",
        },
        "splits": statistics,
        "integrity": integrity,
        "products": products,
    }
    if not all(integrity.values()):
        raise AssertionError(f"tensor integrity failed: {integrity}")
    write_json(root / "processed" / "tensor-summary.json", summary)
    seal_processed_products(root)
    print(json.dumps({"splits": statistics, "integrity": integrity}, indent=2))


if __name__ == "__main__":
    main()
