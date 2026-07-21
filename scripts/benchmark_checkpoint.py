#!/usr/bin/env python3
"""Evaluate one checkpoint on the frozen validation benchmark and record it.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quakecast.demo import select_device
from quakecast.forecaster import RateEnsemble, build_model
from train_real import TensorDataset, validate
from render_benchmark import read_attempts, render_markdown, render_svg


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One checkpoint, or several to score their rate-averaged ensemble",
    )
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--wandb-url")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--attempts", type=Path, default=Path("benchmarks/attempts.jsonl"))
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/benchmark-spec.json"))
    parser.add_argument("--graph", type=Path, default=Path("benchmarks/leaderboard.svg"))
    parser.add_argument("--table", type=Path, default=Path("benchmarks/leaderboard.md"))
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    root = args.root.expanduser().resolve()
    validation_path = root / "processed" / "tensors" / "validation.npz"
    manifest_path = root / "processed" / "processed-manifest.json"
    actual_validation_hash = file_sha256(validation_path)
    actual_manifest_hash = file_sha256(manifest_path)
    if actual_validation_hash != spec["validation_tensor_sha256"]:
        raise SystemExit("Validation tensor hash differs from the frozen benchmark specification")
    if actual_manifest_hash != spec["processed_manifest_sha256"]:
        raise SystemExit("Processed manifest hash differs from the frozen benchmark specification")

    attempts = read_attempts(args.attempts)
    if any(row["attempt_id"] == args.attempt for row in attempts):
        raise SystemExit(f"Attempt already exists: {args.attempt}")

    members, member_hashes, losses = [], [], set()
    for path in args.checkpoint:
        checkpoint_path = path.expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})
        if metadata.get("final_test_opened", False):
            raise SystemExit(f"Checkpoint metadata says final test labels were opened: {path}")
        losses.add(metadata.get("loss", "log-mse"))
        member = build_model(metadata)
        member.load_state_dict(checkpoint["model_state_dict"])
        members.append(member)
        member_hashes.append(file_sha256(checkpoint_path))
    if len(losses) != 1:
        raise SystemExit(f"Ensemble members disagree on the objective: {sorted(losses)}")
    loss_name = losses.pop()
    if loss_name not in {"log-mse", "poisson"}:
        raise SystemExit(f"Unsupported checkpoint loss: {loss_name}")
    if len(members) > 1 and loss_name != "poisson":
        raise SystemExit("Rate averaging is only defined for the Poisson objective")

    device = select_device()
    model = (members[0] if len(members) == 1 else RateEnsemble(members)).to(device)
    validation_data = TensorDataset(validation_path)
    loader = DataLoader(validation_data, batch_size=args.batch_size)
    metrics = validate(model, loader, validation_data, device, loss_name)
    repo_root = args.spec.resolve().parent.parent
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    record = {
        "attempt_id": args.attempt,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "author": "James Edward Ball",
        "description": args.description,
        "loss": loss_name,
        "git_commit": git_commit,
        "checkpoint_sha256": (
            member_hashes[0]
            if len(member_hashes) == 1
            else hashlib.sha256("".join(member_hashes).encode()).hexdigest()
        ),
        "validation_tensor_sha256": actual_validation_hash,
        "processed_manifest_sha256": actual_manifest_hash,
        "wandb_url": args.wandb_url,
        "checkpoint_members": [str(path) for path in args.checkpoint],
        "final_test_opened": False,
        "metrics": metrics,
    }
    with args.attempts.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    attempts.append(record)
    args.graph.write_text(render_svg(attempts))
    args.table.write_text(render_markdown(attempts))
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
