# Real-data baseline

Author: James Edward Ball

Date: 2026-07-21

Status: preliminary validation result

The first sequence-balanced SmaAt-UNet run used all 14,662 training triggers,
three Metal-accelerated epochs, and the paper's mean-squared-error objective on
log1p rate maps. It evaluated on 516 triggers from 200 independent 2022-2023
validation sequences. The 2024-2025 final-test labels remained unopened.

## Initial result

- validation log-MSE per sequence: 0.00365
- information gain against the seven-day-mean persistence forecast: +3.05 nats
  per observed event
- sequence-mean log-likelihood gain: +19.09
- training time on Apple M4 Metal: 76 seconds

The three-epoch Poisson comparison improved information gain to +3.99 nats per
event and sequence-mean likelihood gain to +23.68. It forecast 3,427 events
against 3,768 observed overall, while forecasting only 21.1 of the 347 events
in the most productive validation example. The full KPI history and checkpoint
are stored in the [tracked W&B run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/8wjptoxe).

These figures establish that the pipeline learns useful spatial information.
They do not establish a publishable forecasting result. The most productive
validation example contained 347 next-day events while the model forecast 8.67,
showing severe underprediction of an extreme sequence.

## Required before final testing

1. Compare log-MSE against Poisson and negative-binomial objectives.
2. Add previous-day persistence and ETAS baselines.
3. Report number and spatial calibration by region and sequence productivity.
4. Bootstrap score differences by connected sequence.
5. Freeze completeness-controlled magnitude thresholds using training data.
6. Select the checkpoint and thresholds on validation, then run the sealed test
   once.
