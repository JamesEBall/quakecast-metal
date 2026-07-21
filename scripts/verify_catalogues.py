"""Verify downloaded earthquake catalogues against the immutable manifest.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_file(path: Path, expected_year: int) -> tuple[int, list[str], int]:
    rows = 0
    event_ids = []
    timestamp_issues = 0
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split("|")]
            if len(fields) < 11:
                raise ValueError(f"Malformed event row in {path}: {line[:120]!r}")
            rows += 1
            event_ids.append(fields[0])
            if "/" in fields[1]:
                event_time = datetime.strptime(fields[1], "%Y/%m/%d %H:%M:%S.%f")
            else:
                event_time = datetime.fromisoformat(fields[1].replace("Z", "+00:00"))
            if event_time.year not in {expected_year, expected_year + 1}:
                timestamp_issues += 1
    return rows, event_ids, timestamp_issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest_path = root / "metadata" / "download-manifest.json"
    manifest = json.loads(manifest_path.read_text())

    errors = []
    catalogue_ids: dict[str, list[str]] = {}
    catalogue_rows: Counter[str] = Counter()
    catalogue_bytes: Counter[str] = Counter()
    timestamp_issues: Counter[str] = Counter()
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.exists():
            errors.append(f"missing: {record['path']}")
            continue
        actual_hash = sha256(path)
        if actual_hash != record["sha256"]:
            errors.append(f"checksum: {record['path']}")
        if path.stat().st_size != record["bytes"]:
            errors.append(f"bytes: {record['path']}")
        rows, ids, bad_times = inspect_file(path, record["year"])
        if rows != record["event_rows"]:
            errors.append(f"rows: {record['path']}")
        name = record["catalogue"]
        catalogue_ids.setdefault(name, []).extend(ids)
        catalogue_rows[name] += rows
        catalogue_bytes[name] += path.stat().st_size
        timestamp_issues[name] += bad_times

    summaries = {}
    for name, ids in catalogue_ids.items():
        summaries[name] = {
            "files": sum(1 for record in manifest["files"] if record["catalogue"] == name),
            "event_rows": catalogue_rows[name],
            "unique_event_ids": len(set(ids)),
            "duplicate_event_rows": len(ids) - len(set(ids)),
            "bytes": catalogue_bytes[name],
            "timestamp_range_issues": timestamp_issues[name],
        }

    unexpected_parts = [str(path.relative_to(root)) for path in root.rglob("*.part")]
    if unexpected_parts:
        errors.extend(f"partial: {path}" for path in unexpected_parts)
    result = {
        "metadata": {
            "author": "James Edward Ball",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path.relative_to(root)),
        },
        "catalogues": summaries,
        "errors": errors,
        "valid": not errors,
    }
    output = root / "metadata" / "validation-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
