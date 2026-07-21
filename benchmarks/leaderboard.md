# Benchmark leaderboard

Generated from `attempts.jsonl`. Higher information gain is better. Calibration is ideal near 1.0.

| Attempt | Change | Information gain/event | Calibration | Spatial CSI | W&B |
|---|---|---:|---:|---:|---|
| `009-poisson-four-seed-ensemble` | Rate-averaged ensemble of the attempt 004 recipe at seeds 7, 1, 2 and 3 | 4.350 | 0.999 | 0.165 | - |
| `004-poisson-best-epoch-selection` | Poisson control at 12 epochs, best validation epoch selected inside a 0.5-2.0 calibration band (epoch 11 of 12) | 4.306 | 0.854 | 0.178 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/d6zm8g4f) |
| `010-poisson-capped-sequence-sampling` | Non-replacement sequence balancing: at most four distinct examples per connected sequence per epoch, replacing weighted resampling with replacement | 4.282 | 0.900 | 0.170 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/6vwak0x9) |
| `007-poisson-ema-weights` | Attempt 004 plus exponential moving average weights at decay 0.999; stabilises the end of training so the final epoch nearly matches a selected one | 4.267 | 0.871 | 0.162 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/sibqh5d3) |
| `005-poisson-mean-rate-bias-init` | Attempt 004 plus an output bias started at the mean training rate; regressed on every guardrail | 4.197 | 0.682 | 0.159 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/4pou2des) |
| `008-poisson-cosine-warmup` | Attempt 004 plus cosine decay with half-epoch warm-up; final epoch settles at 0.999 forecast/observed calibration | 4.196 | 0.842 | 0.169 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/5c1o3qez) |
| `002-poisson-sequence-balanced` | Three epochs with sequence-balanced sampling and Poisson objective | 3.994 | 0.909 | 0.160 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/8wjptoxe) |
| `006-poisson-persistence-offset` | Predict a log-multiplier on the seven-day persistence rate instead of the rate directly; anchors the model to persistence and loses spatial skill | 3.945 | 0.788 | 0.152 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/dhbayn7y) |
| `003-poisson-low-lr-eight-epochs` | Poisson, 8 epochs at learning rate 3e-4, sequence-balanced, final-epoch checkpoint | 3.683 | 1.866 | 0.127 | - |
| `001-log-mse-sequence-balanced` | Three epochs with sequence-balanced sampling and log-MSE objective | 3.051 | 0.442 | 0.129 | - |

Metadata author: James Edward Ball.
