#!/usr/bin/env python3
"""Render the public benchmark graph and table from the attempt log.

Author: James Edward Ball
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read_attempts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render_svg(attempts: list[dict]) -> str:
    width, height = 1080, 500
    left, right, top, bottom = 84, 42, 68, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    scores = [row["metrics"]["information_gain_per_observed_event"] for row in attempts]
    if scores:
        low = min(0.0, min(scores))
        high = max(scores)
        padding = max((high - low) * 0.18, 0.5)
        y_min, y_max = low - padding, high + padding
    else:
        y_min, y_max = 0.0, 1.0

    def x_at(index: int) -> float:
        inset = min(105, plot_width * 0.12)
        usable_width = plot_width - 2 * inset
        return left + inset + (usable_width / 2 if len(attempts) == 1 else index * usable_width / max(len(attempts) - 1, 1))

    def y_at(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    grid = []
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        y = y_at(value)
        grid.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        grid.append(f'<text class="tick" x="{left-14}" y="{y+5:.1f}" text-anchor="end">{value:.1f}</text>')

    points = []
    line_points = []
    for index, row in enumerate(attempts):
        score = row["metrics"]["information_gain_per_observed_event"]
        x, y = x_at(index), y_at(score)
        line_points.append(f"{x:.1f},{y:.1f}")
        number = html.escape(row["attempt_id"].split("-", 1)[0])
        objective = "log-MSE" if row["loss"] == "log-mse" else row["loss"].title()
        short = f"{number} · {html.escape(objective)}"
        points.extend([
            f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="7"><title>{html.escape(row["attempt_id"])}: {score:.3f}</title></circle>',
            f'<text class="value" x="{x:.1f}" y="{y-15:.1f}" text-anchor="middle">{score:.2f}</text>',
            f'<text class="attempt" x="{x:.1f}" y="{height-bottom+30}" text-anchor="middle">{short}</text>',
        ])
    polyline = f'<polyline class="series" points="{" ".join(line_points)}"/>' if line_points else ""
    empty = '<text class="empty" x="540" y="250" text-anchor="middle">No benchmark attempts recorded</text>' if not attempts else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">QuakeCast Metal validation benchmark progress</title>
<desc id="desc">Information gain per observed earthquake event by research attempt. Higher values are better.</desc>
<style>
  :root {{ color-scheme: light dark; }}
  svg {{ background: #fbfaf7; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  .title {{ fill: #172033; font-size: 24px; font-weight: 700; }}
  .subtitle, .tick, .axis-label, .attempt, .empty {{ fill: #596375; }}
  .subtitle {{ font-size: 14px; }} .tick {{ font-size: 12px; }} .axis-label {{ font-size: 13px; font-weight: 600; }}
  .grid {{ stroke: #d9dde5; stroke-width: 1; }} .axis {{ stroke: #7c8799; stroke-width: 1.5; }}
  .series {{ fill: none; stroke: #1769e0; stroke-width: 4; stroke-linejoin: round; stroke-linecap: round; }}
  .point {{ fill: #fbfaf7; stroke: #1769e0; stroke-width: 4; }}
  .value {{ fill: #0f55bd; font-size: 15px; font-weight: 700; }} .attempt {{ font-size: 12px; }}
  @media (prefers-color-scheme: dark) {{
    svg {{ background: #10141c; }} .title {{ fill: #eef3fb; }} .subtitle, .tick, .axis-label, .attempt, .empty {{ fill: #aeb8c7; }}
    .grid {{ stroke: #2c3441; }} .axis {{ stroke: #697588; }} .series {{ stroke: #68a0ff; }}
    .point {{ fill: #10141c; stroke: #68a0ff; }} .value {{ fill: #8bb8ff; }}
  }}
</style>
<text class="title" x="{left}" y="34">QuakeCast Metal benchmark</text>
<text class="subtitle" x="{left}" y="56">Fixed 2022-2023 validation set - Poisson information gain/event - higher is better</text>
{"".join(grid)}
<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>
<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>
<text class="axis-label" transform="translate(22 {top + plot_height/2}) rotate(-90)" text-anchor="middle">Information gain per observed event (nats)</text>
<text class="axis-label" x="{left + plot_width/2}" y="{height-20}" text-anchor="middle">Research attempt</text>
{polyline}{"".join(points)}{empty}
</svg>
'''


def render_markdown(attempts: list[dict]) -> str:
    lines = [
        "# Benchmark leaderboard",
        "",
        "Generated from `attempts.jsonl`. Higher information gain is better. Calibration is ideal near 1.0.",
        "",
        "| Attempt | Change | Information gain/event | Calibration | Spatial CSI | W&B |",
        "|---|---|---:|---:|---:|---|",
    ]
    ranked = sorted(attempts, key=lambda row: row["metrics"]["information_gain_per_observed_event"], reverse=True)
    for row in ranked:
        metrics = row["metrics"]
        wandb = f'[run]({row["wandb_url"]})' if row.get("wandb_url") else "-"
        description = row["description"].replace("|", "\\|")
        lines.append(
            f'| `{row["attempt_id"]}` | {description} | {metrics["information_gain_per_observed_event"]:.3f} | '
            f'{metrics["forecast_observed_ratio"]:.3f} | {metrics["spatial_csi_at_0_5"]:.3f} | {wandb} |'
        )
    lines.extend(["", "Metadata author: James Edward Ball.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, default=Path("benchmarks/attempts.jsonl"))
    parser.add_argument("--graph", type=Path, default=Path("benchmarks/leaderboard.svg"))
    parser.add_argument("--table", type=Path, default=Path("benchmarks/leaderboard.md"))
    args = parser.parse_args()
    attempts = read_attempts(args.attempts)
    args.graph.write_text(render_svg(attempts))
    args.table.write_text(render_markdown(attempts))


if __name__ == "__main__":
    main()
