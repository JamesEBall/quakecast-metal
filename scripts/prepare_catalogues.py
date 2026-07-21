#!/usr/bin/env python3
"""Normalize downloaded catalogues and create frozen sequence-level splits.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from quakecast.catalogues import (
    TRIGGER_MAGNITUDE,
    assign_components,
    build_components,
    iso_z,
    normalize_catalogue,
    seal_processed_products,
    write_json,
    write_split_products,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Downloaded data folder")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    all_events = []
    summaries = []
    for catalogue in ("scedc", "ncedc", "geonet"):
        events, summary = normalize_catalogue(root, catalogue)
        all_events.extend(events)
        summaries.append(summary)
        print(f"{catalogue}: kept {len(events):,} events")

    triggers = [event for event in all_events if event.magnitude >= TRIGGER_MAGNITUDE]
    components = build_components(triggers)
    assignments = assign_components(components)
    split_summary = write_split_products(root, assignments)
    normalization = {
        "metadata": {
            "author": "James Edward Ball",
            "created_utc": iso_z(datetime.now(timezone.utc)),
            "filter": "SCEDC local earthquakes (eq/l or legacy le); NCEDC and GeoNet earthquake rows",
        },
        "catalogues": summaries,
        "totals": {
            "kept_events": sum(item["kept_events"] for item in summaries),
            "trigger_events_m4_plus": len(triggers),
        },
    }
    write_json(root / "processed" / "normalization-summary.json", normalization)
    seal_processed_products(root)
    print(json.dumps({"normalization": normalization["totals"], "splits": split_summary["triggers"]}, indent=2))


if __name__ == "__main__":
    main()
