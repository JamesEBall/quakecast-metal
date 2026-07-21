"""Download immutable yearly earthquake catalogues from authoritative FDSN services.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CATALOGUES = {
    "scedc": {
        "authority": "Southern California Earthquake Data Center",
        "endpoint": "https://service.scedc.caltech.edu/fdsnws/event/1/query",
        "parameters": {
            "catalog": "SCEDC",
            "minmag": "2.0",
            "mindepth": "0",
            "maxdepth": "40",
            "etype": "any",
            "format": "text",
            "orderby": "time-asc",
            "nodata": "204",
        },
        "citation": "SCEDC (2013), doi:10.7909/C3WD3xH1",
        "normalization_note": "Filter SCEDC ET codes to local/regional earthquakes after download.",
    },
    "ncedc": {
        "authority": "Northern California Earthquake Data Center",
        "endpoint": "https://service.ncedc.org/fdsnws/event/1/query",
        "parameters": {
            "catalog": "NCSS",
            "minmagnitude": "2.0",
            "mindepth": "0",
            "maxdepth": "40",
            "eventtype": "earthquake",
            "format": "text",
            "orderby": "time-asc",
            "nodata": "204",
        },
        "citation": "NCEDC (2014), doi:10.7932/NCEDC",
    },
    "geonet": {
        "authority": "GNS Science GeoNet",
        "endpoint": "https://service.geonet.org.nz/fdsnws/event/1/query",
        "parameters": {
            "minmagnitude": "2.0",
            "mindepth": "0",
            "maxdepth": "40",
            "eventtype": "earthquake",
            "format": "text",
            "orderby": "time-asc",
            "nodata": "204",
        },
        "citation": "GeoNet earthquake catalogue",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip() and not line.startswith(b"#"))


def download(url: str, destination: Path, attempts: int = 5) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "James-Edward-Ball-earthquake-research/0.1"})
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            partial.replace(destination)
            return
        except HTTPError as exc:
            if exc.code == 204:
                partial.write_bytes(b"")
                partial.replace(destination)
                return
            if attempt == attempts or exc.code < 500:
                raise
        except (TimeoutError, URLError):
            if attempt == attempts:
                raise
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}")


def query_url(config: dict[str, object], starttime: str, endtime: str) -> str:
    parameters = {
        **config["parameters"],
        "starttime": starttime,
        "endtime": endtime,
    }
    return f"{config['endpoint']}?{urlencode(parameters)}"


def download_split_fallback(config: dict[str, object], destination: Path, year: int) -> None:
    combined = destination.with_suffix(destination.suffix + ".part")

    def append_interval(output, start: datetime, end: datetime, label: str) -> None:
        temporary = destination.with_suffix(f".{label}.part")
        start_text = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_text = end.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            download(query_url(config, start_text, end_text), temporary)
        except HTTPError as exc:
            if exc.code != 413 or end - start <= timedelta(days=1):
                raise
            midpoint = start + (end - start) / 2
            append_interval(output, start, midpoint, label + "a")
            append_interval(output, midpoint, end, label + "b")
            return
        with temporary.open("rb") as source:
            while block := source.read(1024 * 1024):
                output.write(block)
        temporary.unlink()
        time.sleep(0.25)

    with combined.open("wb") as output:
        append_interval(
            output,
            datetime(year, 1, 1),
            datetime(year + 1, 1, 1),
            "split",
        )
        output.flush()
        os.fsync(output.fileno())
    combined.replace(destination)


def download_year(root: Path, name: str, year: int, force: bool = False) -> dict[str, object]:
    config = CATALOGUES[name]
    folder = root / "raw" / name
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{name}-{year}-m2-depth0-40.txt"
    used_split_fallback = False
    url = query_url(config, f"{year}-01-01T00:00:00", f"{year + 1}-01-01T00:00:00")
    if force or not destination.exists():
        try:
            download(url, destination)
        except HTTPError as exc:
            if exc.code != 413:
                raise
            print(f"{name} {year}: annual response too large; splitting the interval", flush=True)
            download_split_fallback(config, destination, year)
            used_split_fallback = True
        time.sleep(0.25)

    return {
        "catalogue": name,
        "authority": config["authority"],
        "citation": config["citation"],
        "year": year,
        "path": str(destination.relative_to(root)),
        "query": url,
        "split_fallback": used_split_fallback,
        "filters": {"minimum_magnitude": 2.0, "minimum_depth_km": 0, "maximum_depth_km": 40},
        "bytes": destination.stat().st_size,
        "event_rows": event_rows(destination),
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalogues", nargs="+", choices=CATALOGUES, default=list(CATALOGUES))
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.output.expanduser().resolve()
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.catalogues:
        for year in range(args.start_year, args.end_year + 1):
            record = download_year(root, name, year, force=args.force)
            records.append(record)
            print(f"{name} {year}: {record['event_rows']} events, {record['bytes']} bytes")

    manifest = {
        "metadata": {
            "author": "James Edward Ball",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "format": "FDSN event text, one immutable file per catalogue-year",
            "boundary_note": "Year-end queries meet at 00:00 UTC; deduplicate by source event ID after normalization.",
        },
        "files": records,
    }
    manifest_path = root / "metadata" / "download-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
