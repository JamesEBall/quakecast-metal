# Research benchmark

This directory is the public lab notebook for QuakeCast Metal. Each line in
`attempts.jsonl` is one evaluated checkpoint on the fixed 2022-2023 validation
set. The chart and leaderboard are generated from that log.

The primary score is Poisson information gain per observed event against the
seven-day-mean persistence forecast. Higher is better. A useful attempt should
also improve rate calibration and spatial CSI. The 2024-2025 final test remains
sealed.

## Read the primary score with care

The primary metric is frozen so that every recorded attempt stays comparable,
but an audit showed it measures the reference at least as much as the model.
Three properties matter when interpreting any number in this log.

- **Most of the headline is a clamp artefact.** The persistence baseline
  predicts exactly zero wherever the previous week was empty, and the
  likelihood clamps those cells to 1e-6, charging 13.82 nats for every event
  that lands there. That accounts for 4.12 nats/event on its own. 29.8 percent
  of validation events fall in such cells, and 512 cells out of 206,400 supply
  86 percent of the reported gain. The model's own rate never approaches the
  clamp, so the penalty only ever falls on the baseline.
- **The score is concentrated and non-monotone.** The top 5 of 200 sequences
  contribute 66 percent of the gain, and the model is worse than persistence on
  46.5 percent of sequences. Removing the second-largest contributor *raises*
  the score. Attempts 001 and 002 are not statistically distinguishable.
- **The resolution floor is about half a nat.** The 95 percent sequence
  bootstrap interval on attempt 002's 3.994 is [2.60, 6.33]. Treat smaller
  differences as noise unless a paired bootstrap over repeated seeds says
  otherwise; `scripts/compare_checkpoints.py` computes both.

Every attempt therefore also records companions that the clamp cannot inflate:
information gain against floored and uniformly smoothed persistence baselines,
and average precision over occupied cells, which is threshold-free and
clamp-invariant. On attempt 002's weights those read 1.988, 2.268, and 0.228
against a persistence average precision of 0.145. Use them, not the headline,
to decide whether a change is real.

The full audit is in [metric-audit.md](metric-audit.md).

![Benchmark progress](leaderboard.svg)

The generated table is in [leaderboard.md](leaderboard.md).

## Record an attempt

Run this from the repository root after training a checkpoint:

```bash
uv run python scripts/benchmark_checkpoint.py \
  --root "/path/to/Earthquake Forecasting Data" \
  --checkpoint "/path/to/checkpoint.pt" \
  --attempt "003-short-name" \
  --description "The single change made in this attempt" \
  --wandb-url "https://wandb.ai/.../runs/..."
```

The command checks the frozen-data hashes, evaluates validation only, appends
the result, and regenerates both GitHub artifacts. Commit the log and generated
files together. Checkpoints belong in W&B artifacts rather than Git.

Metadata author: James Edward Ball.
