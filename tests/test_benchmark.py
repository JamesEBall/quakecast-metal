import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_attempt_log_is_unique_and_test_sealed() -> None:
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "benchmarks" / "attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ids = [row["attempt_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert all(row["author"] == "James Edward Ball" for row in rows)
    assert all(row["final_test_opened"] is False for row in rows)
    assert all("information_gain_per_observed_event" in row["metrics"] for row in rows)


def test_generated_artifacts_include_every_attempt() -> None:
    svg = (REPO_ROOT / "benchmarks" / "leaderboard.svg").read_text()
    table = (REPO_ROOT / "benchmarks" / "leaderboard.md").read_text()
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "benchmarks" / "attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        assert row["attempt_id"] in svg
        assert row["attempt_id"] in table
