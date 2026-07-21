# Benchmark leaderboard

Generated from `attempts.jsonl`. Higher information gain is better. Calibration is ideal near 1.0.

| Attempt | Change | Information gain/event | Calibration | Spatial CSI | W&B |
|---|---|---:|---:|---:|---|
| `009-poisson-four-seed-ensemble` | Rate-averaged ensemble of the attempt 004 recipe at seeds 7, 1, 2 and 3 | 4.350 | 0.999 | 0.165 | - |
| `004-poisson-best-epoch-selection` | Poisson control at 12 epochs, best validation epoch selected inside a 0.5-2.0 calibration band (epoch 11 of 12) | 4.306 | 0.854 | 0.178 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/d6zm8g4f) |
| `010-poisson-capped-sequence-sampling` | Non-replacement sequence balancing: at most four distinct examples per connected sequence per epoch, replacing weighted resampling with replacement | 4.282 | 0.900 | 0.170 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/6vwak0x9) |
| `007-poisson-ema-weights` | Attempt 004 plus exponential moving average weights at decay 0.999; stabilises the end of training so the final epoch nearly matches a selected one | 4.267 | 0.871 | 0.162 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/sibqh5d3) |
| `016-poisson-calibration-penalty` | Squared log mismatch on total forecast events at weight 0.05; too weak to bind, forecast mass still swings 0.48 to 2.91 across epochs | 4.265 | 1.038 | 0.170 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/k3swuimr) |
| `017-poisson-multiscale-loss-strong` | Multiscale block-sum Poisson loss at weight 1.0; average precision degrades further, closing the direction | 4.236 | 0.937 | 0.158 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/801319kb) |
| `014-poisson-capped-plus-cosine` | Capped sequence sampling combined with cosine decay; redundant, matches capped sampling alone | 4.216 | 1.010 | 0.158 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/e773wrkk) |
| `015-poisson-multiscale-loss` | Poisson loss on 2x2 and 4x4 block sums at weight 0.5 alongside cell-wise loss; does not improve average precision and destabilises calibration | 4.209 | 1.065 | 0.165 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/pfgcahzh) |
| `011-poisson-clip-and-weight-decay` | Gradient-norm clipping at 1.0 with weight decay 1e-5; no effect on any metric | 4.206 | 0.891 | 0.160 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/emu6s9oj) |
| `005-poisson-mean-rate-bias-init` | Attempt 004 plus an output bias started at the mean training rate; regressed on every guardrail | 4.197 | 0.682 | 0.159 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/4pou2des) |
| `008-poisson-cosine-warmup` | Attempt 004 plus cosine decay with half-epoch warm-up; final epoch settles at 0.999 forecast/observed calibration | 4.196 | 0.842 | 0.169 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/5c1o3qez) |
| `013-poisson-ema-cosine-ensemble` | Three-seed ensemble of averaged weights with cosine decay; the combination nearly removes the selected-versus-final gap | 4.182 | 0.861 | 0.167 | - |
| `012-poisson-width-96` | Base channel width 96 instead of 64, 9.0M parameters against 4.0M; no gain for 1.5x the training time | 4.172 | 0.716 | 0.175 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/8l5zuygo) |
| `002-poisson-sequence-balanced` | Three epochs with sequence-balanced sampling and Poisson objective | 3.994 | 0.909 | 0.160 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/8wjptoxe) |
| `006-poisson-persistence-offset` | Predict a log-multiplier on the seven-day persistence rate instead of the rate directly; anchors the model to persistence and loses spatial skill | 3.945 | 0.788 | 0.152 | [run](https://wandb.ai/james-ball-98-none/quakecast-metal/runs/dhbayn7y) |
| `003-poisson-low-lr-eight-epochs` | Poisson, 8 epochs at learning rate 3e-4, sequence-balanced, final-epoch checkpoint | 3.683 | 1.866 | 0.127 | - |
| `001-log-mse-sequence-balanced` | Three epochs with sequence-balanced sampling and log-MSE objective | 3.051 | 0.442 | 0.129 | - |

Metadata author: James Edward Ball.
