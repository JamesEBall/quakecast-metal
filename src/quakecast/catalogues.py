"""Catalogue normalization and leakage-resistant earthquake sequence splits.

Author: James Edward Ball
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, TextIO


INPUT_DAYS = 7
TARGET_DAYS = 1
FOOTPRINT_DEGREES = 2.0
TRIGGER_MAGNITUDE = 4.0
EMBARGO_DAYS = 30
CUTOFFS = (
    datetime(2022, 1, 1, tzinfo=timezone.utc),
    datetime(2024, 1, 1, tzinfo=timezone.utc),
)

NORMALIZED_FIELDS = (
    "catalogue",
    "event_id",
    "time_utc",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "event_type",
    "geographic_type",
    "source_file",
)


@dataclass(frozen=True, slots=True)
class Event:
    catalogue: str
    event_id: str
    time: datetime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    magnitude_type: str
    event_type: str
    geographic_type: str
    source_file: str

    @property
    def key(self) -> str:
        return f"{self.catalogue}:{self.event_id}"

    def normalized_row(self) -> dict[str, str | float]:
        return {
            "catalogue": self.catalogue,
            "event_id": self.event_id,
            "time_utc": iso_z(self.time),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth_km": self.depth_km,
            "magnitude": self.magnitude,
            "magnitude_type": self.magnitude_type,
            "event_type": self.event_type,
            "geographic_type": self.geographic_type,
            "source_file": self.source_file,
        }


@dataclass(frozen=True, slots=True)
class TriggerAssignment:
    event: Event
    component_id: str
    split: str

    def record(self) -> dict[str, object]:
        trigger = self.event.time
        longitude_bounds = longitude_box(self.event.longitude)
        return {
            "trigger_id": self.event.key,
            "component_id": self.component_id,
            "split": self.split,
            "catalogue": self.event.catalogue,
            "source_event_id": self.event.event_id,
            "trigger_time_utc": iso_z(trigger),
            "input_start_utc": iso_z(trigger - timedelta(days=INPUT_DAYS)),
            "input_end_utc": iso_z(trigger),
            "target_start_utc": iso_z(trigger),
            "target_end_utc": iso_z(trigger + timedelta(days=TARGET_DAYS)),
            "latitude": self.event.latitude,
            "longitude": self.event.longitude,
            "depth_km": self.event.depth_km,
            "magnitude": self.event.magnitude,
            "magnitude_type": self.event.magnitude_type,
            "box": {
                "min_latitude": self.event.latitude - FOOTPRINT_DEGREES / 2,
                "max_latitude": self.event.latitude + FOOTPRINT_DEGREES / 2,
                **longitude_bounds,
            },
        }


def iso_z(value: datetime) -> str:
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_longitude(value: float) -> float:
    """Return a longitude in [-180, 180), preserving antimeridian events."""
    return (value + 180.0) % 360.0 - 180.0


def longitude_distance(left: float, right: float) -> float:
    """Smallest angular distance between two longitudes in degrees."""
    return abs((left - right + 180.0) % 360.0 - 180.0)


def longitude_box(longitude: float) -> dict[str, float | bool]:
    half_width = FOOTPRINT_DEGREES / 2
    west = normalize_longitude(longitude - half_width)
    east = normalize_longitude(longitude + half_width)
    return {
        "west_longitude": west,
        "east_longitude": east,
        "crosses_antimeridian": west > east,
    }


def parse_time(value: str) -> datetime:
    value = value.strip()
    if "/" in value:
        parsed = datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")
        return parsed.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_event_line(catalogue: str, line: str, source_file: str) -> Event:
    fields = [field.strip() for field in line.split("|")]
    if catalogue == "scedc":
        if len(fields) < 13:
            raise ValueError(f"SCEDC row has {len(fields)} fields: {line[:120]!r}")
        event_type, geographic_type = fields[7], fields[8]
        magnitude_type, magnitude = fields[9], fields[10]
    else:
        if len(fields) < 14:
            raise ValueError(f"FDSN row has {len(fields)} fields: {line[:120]!r}")
        magnitude_type, magnitude = fields[9], fields[10]
        event_type, geographic_type = fields[13], ""
    return Event(
        catalogue=catalogue,
        event_id=fields[0],
        time=parse_time(fields[1]),
        latitude=float(fields[2]),
        longitude=normalize_longitude(float(fields[3])),
        depth_km=float(fields[4]),
        magnitude=float(magnitude),
        magnitude_type=magnitude_type,
        event_type=event_type,
        geographic_type=geographic_type,
        source_file=source_file,
    )


def keep_event(event: Event) -> tuple[bool, str]:
    if event.catalogue == "scedc":
        if event.event_type == "eq" and event.geographic_type == "l":
            return True, "local_earthquake"
        if event.event_type == "le":
            return True, "legacy_local_earthquake"
        return False, f"type={event.event_type or 'blank'};geo={event.geographic_type or 'blank'}"
    if event.event_type.lower() == "earthquake":
        return True, "earthquake"
    return False, f"type={event.event_type or 'blank'}"


def iter_raw_events(root: Path, catalogue: str) -> Iterator[Event]:
    raw_dir = root / "raw" / catalogue
    for path in sorted(raw_dir.glob("*.txt")):
        relative = str(path.relative_to(root))
        with path.open(encoding="utf-8", errors="strict") as stream:
            for line in stream:
                if not line.strip() or line.startswith("#"):
                    continue
                yield parse_event_line(catalogue, line, relative)


def deterministic_gzip_text(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_catalogue(root: Path, catalogue: str) -> tuple[list[Event], dict[str, object]]:
    events: list[Event] = []
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    malformed = 0
    output = root / "processed" / "normalized" / f"{catalogue}.csv.gz"
    with deterministic_gzip_text(output) as stream:
        writer = csv.DictWriter(stream, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        for event in iter_raw_events(root, catalogue):
            valid = (
                -90 <= event.latitude <= 90
                and -180 <= event.longitude < 180
                and 0 <= event.depth_km <= 40
                and event.magnitude >= 2
            )
            if not valid:
                malformed += 1
                continue
            keep, reason = keep_event(event)
            reasons[("kept:" if keep else "excluded:") + reason] += 1
            if not keep:
                continue
            if event.event_id in seen_ids:
                duplicate_ids += 1
                continue
            seen_ids.add(event.event_id)
            events.append(event)
            writer.writerow(event.normalized_row())
    events.sort(key=lambda event: (event.time, event.key))
    summary: dict[str, object] = {
        "catalogue": catalogue,
        "kept_events": len(events),
        "excluded_or_invalid_events": sum(
            count for reason, count in reasons.items() if reason.startswith("excluded:")
        )
        + malformed,
        "duplicate_source_ids_removed": duplicate_ids,
        "invalid_numeric_or_filter_rows": malformed,
        "trigger_events_m4_plus": sum(event.magnitude >= TRIGGER_MAGNITUDE for event in events),
        "time_start_utc": iso_z(events[0].time) if events else None,
        "time_end_utc": iso_z(events[-1].time) if events else None,
        "selection_counts": dict(sorted(reasons.items())),
        "output": str(output.relative_to(root)),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
    }
    return events, summary


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def footprints_overlap(left: Event, right: Event) -> bool:
    return (
        abs((left.time - right.time).total_seconds()) <= (INPUT_DAYS + TARGET_DAYS) * 86400
        and abs(left.latitude - right.latitude) <= FOOTPRINT_DEGREES
        and longitude_distance(left.longitude, right.longitude) <= FOOTPRINT_DEGREES
    )


def build_components(triggers: Iterable[Event]) -> list[list[Event]]:
    ordered = sorted(triggers, key=lambda event: (event.time, event.key))
    groups = UnionFind(len(ordered))
    active: deque[int] = deque()
    maximum_gap = timedelta(days=INPUT_DAYS + TARGET_DAYS)
    for current_index, current in enumerate(ordered):
        while active and current.time - ordered[active[0]].time > maximum_gap:
            active.popleft()
        for earlier_index in active:
            earlier = ordered[earlier_index]
            if (
                abs(current.latitude - earlier.latitude) <= FOOTPRINT_DEGREES
                and longitude_distance(current.longitude, earlier.longitude) <= FOOTPRINT_DEGREES
            ):
                groups.union(current_index, earlier_index)
        active.append(current_index)
    by_root: dict[int, list[Event]] = defaultdict(list)
    for index, event in enumerate(ordered):
        by_root[groups.find(index)].append(event)
    return sorted(
        by_root.values(),
        key=lambda component: (component[0].time, component[0].key),
    )


def nominal_split(time: datetime) -> str:
    if time < CUTOFFS[0]:
        return "train"
    if time < CUTOFFS[1]:
        return "validation"
    return "test"


def touches_embargo(time: datetime, embargo_days: int = EMBARGO_DAYS) -> bool:
    margin = timedelta(days=embargo_days)
    return any(cutoff - margin <= time < cutoff + margin for cutoff in CUTOFFS)


def assign_components(
    components: Iterable[list[Event]], embargo_days: int = EMBARGO_DAYS
) -> list[TriggerAssignment]:
    assignments: list[TriggerAssignment] = []
    for number, component in enumerate(components, start=1):
        component_id = f"sequence-{number:06d}"
        partitions = {nominal_split(event.time) for event in component}
        embargo = len(partitions) != 1 or any(
            touches_embargo(event.time, embargo_days) for event in component
        )
        split = "embargo" if embargo else partitions.pop()
        assignments.extend(
            TriggerAssignment(event=event, component_id=component_id, split=split)
            for event in component
        )
    return sorted(assignments, key=lambda item: (item.event.time, item.event.key))


def verify_assignments(assignments: list[TriggerAssignment]) -> dict[str, object]:
    component_splits: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        component_splits[assignment.component_id].add(assignment.split)
    split_components = [key for key, values in component_splits.items() if len(values) != 1]

    ordered = sorted(assignments, key=lambda item: (item.event.time, item.event.key))
    active: deque[TriggerAssignment] = deque()
    cross_split_overlaps = 0
    maximum_gap = timedelta(days=INPUT_DAYS + TARGET_DAYS)
    for current in ordered:
        while active and current.event.time - active[0].event.time > maximum_gap:
            active.popleft()
        for earlier in active:
            if footprints_overlap(current.event, earlier.event):
                if current.component_id != earlier.component_id:
                    raise AssertionError("overlapping footprints received different component IDs")
                if (
                    current.split != earlier.split
                    and current.split != "embargo"
                    and earlier.split != "embargo"
                ):
                    cross_split_overlaps += 1
        active.append(current)
    valid = not split_components and cross_split_overlaps == 0
    return {
        "valid": valid,
        "components_with_multiple_splits": split_components,
        "cross_partition_overlapping_footprints": cross_split_overlaps,
        "trigger_ids_unique": len({item.event.key for item in assignments}) == len(assignments),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def seal_processed_products(root: Path) -> None:
    """Bind every processed product to the immutable raw-download manifest."""
    manifest_path = root / "processed" / "processed-manifest.json"
    files = []
    for path in sorted((root / "processed").rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    raw_manifest = root / "metadata" / "download-manifest.json"
    write_json(
        manifest_path,
        {
            "metadata": {
                "author": "James Edward Ball",
                "created_utc": iso_z(datetime.now(timezone.utc)),
                "source_manifest": str(raw_manifest.relative_to(root)),
                "source_manifest_sha256": sha256(raw_manifest),
                "rule": "Any changed checksum creates a new experiment version.",
            },
            "files": files,
        },
    )


def write_split_products(root: Path, assignments: list[TriggerAssignment]) -> dict[str, object]:
    processed = root / "processed"
    records = [assignment.record() for assignment in assignments]
    trigger_path = processed / "triggers.csv.gz"
    flat_fields = [field for field in records[0] if field != "box"] if records else []
    with deterministic_gzip_text(trigger_path) as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in flat_fields})

    product_paths = [trigger_path]
    for split in ("train", "validation", "test", "embargo"):
        path = processed / "splits" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                if record["split"] == split:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        product_paths.append(path)

    counts = Counter(assignment.split for assignment in assignments)
    by_catalogue: dict[str, Counter[str]] = defaultdict(Counter)
    component_members: dict[str, list[TriggerAssignment]] = defaultdict(list)
    for assignment in assignments:
        by_catalogue[assignment.event.catalogue][assignment.split] += 1
        component_members[assignment.component_id].append(assignment)
    component_counts = Counter(members[0].split for members in component_members.values())
    component_sizes = sorted((len(members) for members in component_members.values()), reverse=True)
    integrity = verify_assignments(assignments)
    integrity["trigger_ids_unique"] = integrity["trigger_ids_unique"] and len(records) == len(
        {record["trigger_id"] for record in records}
    )
    summary = {
        "metadata": {
            "author": "James Edward Ball",
            "created_utc": iso_z(datetime.now(timezone.utc)),
            "policy": "M4+ trigger components formed by overlapping 8-day, 2x2-degree footprints",
            "trigger_magnitude": TRIGGER_MAGNITUDE,
            "input_days": INPUT_DAYS,
            "target_days": TARGET_DAYS,
            "footprint_degrees": FOOTPRINT_DEGREES,
            "embargo_days": EMBARGO_DAYS,
            "cutoffs_utc": [iso_z(value) for value in CUTOFFS],
        },
        "triggers": dict(sorted(counts.items())),
        "components": dict(sorted(component_counts.items())),
        "triggers_by_catalogue": {
            catalogue: dict(sorted(values.items()))
            for catalogue, values in sorted(by_catalogue.items())
        },
        "largest_component_triggers": component_sizes[0] if component_sizes else 0,
        "median_component_triggers": component_sizes[len(component_sizes) // 2]
        if component_sizes
        else 0,
        "integrity": integrity,
        "products": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in product_paths
        ],
    }
    write_json(processed / "split-summary.json", summary)
    if not all(bool(value) if isinstance(value, bool) else not value for value in integrity.values()):
        raise AssertionError(f"split integrity failed: {integrity}")
    return summary
