"""Tests for catalogue normalization and sequence splits.

Author: James Edward Ball
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from quakecast.catalogues import (
    Event,
    assign_components,
    build_components,
    keep_event,
    parse_event_line,
    verify_assignments,
)


def event(identifier: str, day: int, latitude: float = 0, longitude: float = 0) -> Event:
    return Event(
        catalogue="test",
        event_id=identifier,
        time=datetime(2021, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        latitude=latitude,
        longitude=longitude,
        depth_km=5,
        magnitude=4,
        magnitude_type="ml",
        event_type="earthquake",
        geographic_type="",
        source_file="fixture.txt",
    )


def test_parse_scedc_and_filter_local_earthquakes() -> None:
    row = "123|2020/01/02 03:04:05.120|34.1|-118.2|7.0|AUTH|SCEDC|eq|l|ml|4.2|AUTH|place"
    parsed = parse_event_line("scedc", row, "raw.txt")
    assert parsed.time.tzinfo == timezone.utc
    assert parsed.magnitude == 4.2
    assert keep_event(parsed) == (True, "local_earthquake")


def test_filter_scedc_quarry_and_regional_events() -> None:
    local = event("1", 0)
    quarry = replace(local, event_type="qb", geographic_type="l")
    regional = replace(local, event_type="eq", geographic_type="r")
    assert keep_event(quarry)[0] is False
    assert keep_event(regional)[0] is False


def test_overlapping_footprints_form_transitive_component() -> None:
    components = build_components([event("a", 0), event("b", 7), event("c", 14)])
    assert [[item.event_id for item in component] for component in components] == [["a", "b", "c"]]


def test_spatially_separate_triggers_remain_independent() -> None:
    components = build_components([event("a", 0), event("b", 1, latitude=3)])
    assert len(components) == 2


def test_antimeridian_triggers_share_a_component() -> None:
    components = build_components(
        [event("west", 0, longitude=179.5), event("east", 1, longitude=-179.5)]
    )
    assert len(components) == 1


def test_boundary_component_is_embargoed_as_a_unit() -> None:
    before = event("before", 0)
    before = replace(before, time=datetime(2021, 12, 29, tzinfo=timezone.utc))
    after = replace(before, event_id="after", time=datetime(2022, 1, 2, tzinfo=timezone.utc))
    assignments = assign_components(build_components([before, after]))
    assert {item.split for item in assignments} == {"embargo"}
    assert verify_assignments(assignments)["valid"] is True
