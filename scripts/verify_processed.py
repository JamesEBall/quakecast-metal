#!/usr/bin/env python3
"""Verify every sealed normalized catalogue and split product.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((root / "processed" / "processed-manifest.json").read_text())
    errors = []
    source_manifest = root / manifest["metadata"]["source_manifest"]
    if not source_manifest.exists():
        errors.append(f"missing source manifest: {source_manifest}")
    elif sha256(source_manifest) != manifest["metadata"]["source_manifest_sha256"]:
        errors.append("source download manifest checksum changed")
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.exists():
            errors.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != item["bytes"]:
            errors.append(f"byte count changed: {item['path']}")
        if sha256(path) != item["sha256"]:
            errors.append(f"checksum changed: {item['path']}")
    listed = {item["path"] for item in manifest["files"]}
    actual = {
        str(path.relative_to(root))
        for path in (root / "processed").rglob("*")
        if path.is_file() and path.name != "processed-manifest.json"
    }
    for path in sorted(actual - listed):
        errors.append(f"unsealed processed file: {path}")
    for path in sorted(listed - actual):
        errors.append(f"manifest entry has no file: {path}")
    split_summary = json.loads((root / "processed" / "split-summary.json").read_text())
    if not split_summary["integrity"]["valid"]:
        errors.append("split-summary integrity status is false")
    result = {
        "valid": not errors,
        "sealed_files": len(manifest["files"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
