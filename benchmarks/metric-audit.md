# QuakeCast Metal — leakage and metric audit

Auditor: independent review agent
Date: 2026-07-21
Scope: read-only audit of temporal leakage, metric correctness, and benchmark integrity
Checkpoint audited: `/Users/clawd/Documents/Codex/2026-07-21/can/outputs/earthquake-real-poisson-wandb.pt` (loss = `poisson`)
Data root: `/Users/clawd/Documents/Codex/Earthquake Forecasting Data`

---

## 0. Sealed-data declaration

**Test labels exist at:**

```
/Users/clawd/Documents/Codex/Earthquake Forecasting Data/processed/sealed/test-labels.npz   (8,178 bytes)
```

They were **not opened**. No array in that file was loaded, decoded, summarised, or used. No number in this
report derives from it. `processed/tensors/test-inputs.npz` and `processed/splits/test.jsonl` were also never
loaded. The file's existence and size come from a directory listing only.

**One disclosure, for precision.** A catalogue-coverage QC check read
`processed/normalized/{geonet,ncedc,scedc}.csv.gz` — the shared normalized input product — and transiently
printed per-year event counts and magnitude percentiles for 2015–2025, which included 2024 and 2025 rows. That
is an aggregate over the raw catalogue, not a label: no trigger-centred grid, no per-example next-day count, and
no 2024–2025 trigger manifest was touched, so no test target was constructed. **All 2024–2025 figures have been
excluded from this report**; the coverage conclusion below rests only on 2015–2023. Flagging it so the record is
exact.

**Repository modifications between 14:38 and 14:42 are not mine.** This audit wrote only to its scratchpad and
to this file. See Appendix B for the timeline evidence — the scratchpad is shared with another agent that is
running a training queue.

---

## 1. Executive summary and severity ranking

| # | Finding | Severity |
|---|---|---|
| B5 | **The 3.99 headline information gain is ~57% a baseline-clamp artefact.** Honest value is 1.72–1.99 nats/event. | **CRITICAL** |
| B6 | Primary metric is extremely concentrated: top-5 of 200 sequences supply 66% of the gain; the model is *worse* than baseline on 46.5% of sequences. Metric is non-monotone under removal of top contributors. | **CRITICAL** |
| C11 | `--spec` is attacker-supplied, so the frozen-hash check is bypassable; nothing cross-checks `attempts.jsonl` hashes against the committed spec. | **IMPORTANT** |
| C10b | The `loss` string in checkpoint metadata is self-attested and changes the same weights' score from 3.994 to 1.471. | **IMPORTANT** |
| C11b | `torch.load(..., weights_only=False)` executes arbitrary pickle code before metrics are computed. | **IMPORTANT** |
| B8 | `maximum_trigger_forecast_observed_ratio` is an n=1 statistic; ratio varies 4x across the top-8 and the top-5 are all one sequence. | **IMPORTANT** |
| — | Train examples are 18x denser than validation (131.6 vs 7.3 events/example). Not a coverage artefact. | **IMPORTANT** |
| B7 | `spatial_csi_at_0_5` is near-optimal *for this model* by coincidence but is not scale-invariant across attempts. | **MINOR** |
| A1–A4 | **No temporal leakage found.** Splits are cleanly separated. | **PASS** |
| B9 | `sequence_mean_log_likelihood_gain` implementation matches its description exactly. | **PASS** |
| C10a | Tampered validation tensors *are* correctly rejected; metrics *are* independently re-derived. | **PASS** |

---

## A. Temporal leakage — clean

### A1. Train/validation are strictly separated, with margin

| Split | n | min trigger time | max trigger time |
|---|---|---|---|
| train | 14,662 | 1980-01-05T22:34:09Z | 2021-11-29T13:00:27Z |
| validation | 516 | 2022-02-05T14:46:36Z | 2023-12-01T07:43:27Z |

- **Smallest gap between any train trigger and any validation trigger: 68.074 days.** Train triggers at or after
  the validation minimum: 0. Validation triggers at or before the train maximum: 0.
- The claimed 30-day embargo holds with room to spare: the last train trigger sits 32.46 days before the
  2022-01-01 cutoff, the first validation trigger 35.62 days after it. Validation ends 30.68 days before the
  2024-01-01 cutoff.
- Mechanism is sound. `catalogues.py:328` `touches_embargo` sends any component within ±30 days of a cutoff to
  the `embargo` split, and `catalogues.py:340` sends any component spanning two nominal splits there too.

### A2. Input windows cannot overlap the other split's target day

Input window is `[t-7d, t]` (`catalogues.py:95-96`), target `(t, t+1d]` (`catalogues.py:97-98`). Counting all
516 × 14,662 pairs:

- train INPUT window vs validation TARGET day: **0 overlaps**
- validation INPUT window vs train TARGET day: **0 overlaps**
- full 8-day footprints, train vs validation: **0 overlaps**
- minimum separation between the end of any train footprint and the start of any validation footprint:
  **60.07 days**

The 68-day trigger gap exceeds the 8-day footprint span, so overlap is arithmetically impossible.

### A3. Component IDs are disjoint

3,952 train components, 200 validation components, **intersection empty**. Trigger-ID intersection is 0, and
trigger IDs are unique within each split. `catalogues.py:333-348` assigns one split per connected component, and
`verify_assignments` (`catalogues.py:351`) raises if overlapping footprints ever land in different components.

### A4. No event-level bleed

Checking time *and* location jointly (validation trigger inside a train example's input window **and** inside its
2°×2° box, and the reverse): **0 hits in both directions.**

One observation, not a leak: 513 of 516 validation boxes (99.4%) are geographically co-located with at least one
train trigger. This is unavoidable in seismology — the same faults rupture repeatedly. It is *not* exploitable
here because `data.py:91-101` builds the grid in **trigger-relative** coordinates, so the network never observes
absolute latitude/longitude and cannot memorise a static fault map. Worth stating explicitly in any write-up,
because a reviewer will ask.

---

## B. Metric correctness

### B5. CRITICAL — the headline information gain is mostly a clamp artefact

**This is the central finding of the audit.**

#### The mechanism

`train_real.py:90` computes the persistence baseline as `baseline_rate = expm1(features[:, :1]) / 7.0`. Channel 0
is `log1p(count over the past 7 days)`, so any cell with **zero** events in the prior week gets baseline rate
**exactly 0**. `poisson_log_likelihood` (`train_real.py:55-57`) then clamps to `1e-6`.

For a cell with `k` observed events and rate clamped to 1e-6, the log-likelihood is
`k·log(1e-6) − 1e-6 = −13.8155·k`. Every event landing in a previously-empty cell costs the baseline **13.82
nats**, which is not a modelling penalty — it is the arbitrary clamp constant.

**The clamp is one-sided.** The model's predicted rate has minimum **3.12e-4** and never once reaches the 1e-6
floor across all 206,400 validation cells. The clamp penalises only the baseline.

#### The magnitude

- 203,882 of 206,400 validation cells (**98.78%**) have baseline rate exactly 0.
- **1,124 of 3,768 observed validation events (29.83%)** land in those empty-history cells.
- Pure artefact term: `1,124 × 13.8155 / 3,768 = ` **4.121 nats/event**.

**That single term is 103% of the entire 3.994 headline.**

#### Cell-level decomposition (clamp 1e-6)

| Group | cells | events | Σ ΔLL | % of total | contribution to IG |
|---|---:|---:|---:|---:|---:|
| empty-history, target = 0 | 203,370 | 0 | −1,603.3 | −10.7% | −0.425 |
| **empty-history, target > 0** | **512** | **1,124** | **+13,004.7** | **+86.4%** | **+3.451** |
| has-history, target = 0 | 2,267 | 0 | −448.5 | −3.0% | −0.119 |
| has-history, target > 0 | 251 | 2,644 | +4,096.8 | +27.2% | +1.087 |
| **total** | 206,400 | 3,768 | +15,049.8 | 100% | **+3.994** |

**512 cells out of 206,400 — 0.25% of the grid — supply 86.4% of the reported gain.**

Note the empty-and-zero row: 203,370 cells contribute *negative* gain, because the model spends probability mass
there while the clamped baseline spends essentially none. The headline is the net of a large artefact credit
against a real calibration debit.

#### Re-scoring with a non-degenerate baseline

| Baseline variant | baseline LL | IG/event | % of headline |
|---|---:|---:|---:|
| **as shipped (clamp 1e-6)** | −24,881.3 | **3.994** | 100.0% |
| clamp 1e-4 | −19,725.2 | 2.626 | 65.7% |
| clamp 1e-3 | −17,320.6 | 1.988 | 49.8% |
| clamp 1e-2 | −16,567.5 | 1.788 | 44.8% |
| + per-example grid-mean floor | −16,503.1 | 1.771 | 44.3% |
| + global grid-mean floor (0.00358) | −16,397.0 | 1.742 | 43.6% |
| **+ optimal additive floor (0.00536)** | — | **1.716** | **43.0%** |

The last row is the *most charitable possible* smoothed persistence baseline — the floor is tuned to maximise the
baseline's own likelihood on validation, i.e. deliberately against the model. Even then the model retains
**1.716 nats/event**.

#### Verdict

> **The model has real skill, but roughly 57% of the reported 3.99 is a scoring artefact.
> The defensible number is 1.7–2.0 nats/event, and the honest single figure is ≈1.72.**

Two caveats for interpreting the existing bootstrap interval. First, this bias is **deterministic, not
stochastic** — correcting it shifts the whole sampling distribution down, it does not merely narrow it, so the
measured [2.60, 6.33] interval is centred on the wrong quantity rather than merely wide. Second, fitting the
floor on *training* data does not transfer (train-optimal floor 0.1 gives a nonsensical validation IG of 5.888)
because train examples are 18x denser than validation — see the density note below. Fit the baseline
per-evaluation-set, or use a principled ETAS background rate.

**Recommended fix.** Replace `baseline_rate = expm1(features[:, :1]) / 7.0` with a mixture that cannot assign
zero probability, e.g. `(1-ε)·persistence + ε·grid_mean` with ε declared once and fixed, or add a background
rate. Separately, in `poisson_log_likelihood`, either raise the clamp to a physically meaningful floor or assert
that no forecast reaches it — a clamp that binds is a silent scoring bug, not numerical hygiene. Report both the
clamped and smoothed numbers during the transition so the leaderboard stays comparable.

### B6. CRITICAL — the metric is dominated by a handful of sequences

`information_gain_per_observed_event = Σ(model_ll − baseline_ll) / Σ observed`. Grouping the 516 triggers into
their 200 components:

| rank | component | triggers | events | Σ ΔLL | % of total | cumulative |
|---|---|---:|---:|---:|---:|---:|
| 1 | sequence-004062 | 5 | 359 | 2,591.8 | 17.2% | 17.2% |
| 2 | sequence-004088 | 11 | 1,474 | 2,493.6 | 16.6% | 33.8% |
| 3 | sequence-004060 | 5 | 519 | 2,115.5 | 14.1% | 47.8% |
| 4 | sequence-004100 | 5 | 367 | 1,974.5 | 13.1% | 61.0% |
| 5 | sequence-004103 | 6 | 128 | 818.1 | 5.4% | 66.4% |

**Top-1 = 17.2%, top-5 = 66.4%, top-10 = 81.0%, top-20 = 88.9%.**

Per-sequence ΔLL distribution: p0 −13.74, p25 −1.26, **median 4.67**, p75 22.63, p90 61.03, p99 2,119.31, max
2,591.82, mean 75.25. The mean is 16x the median.

**93 of 200 sequences (46.5%) have negative gain** — the model is worse than even the degraded baseline on nearly
half of all independent validation sequences.

Dropping the largest contributor:

| | IG/event |
|---|---:|
| headline | 3.994 |
| drop top-1 | **3.654** |
| drop top-2 | 5.150 |
| drop top-3 | 5.543 |
| drop top-5 | 5.490 |

The metric is **non-monotone** under removal of its largest contributors — dropping the #2 sequence *raises* the
score to 5.150, because sequence-004088 carries 1,474 events (39% of the denominator) but only 16.6% of the
numerator. Numerator and denominator are dominated by different sequences. A statistic that moves in both
directions when you remove its biggest terms is not a stable ranking quantity.

Sequence-clustered bootstrap (20,000 resamples of the 200 components): mean 4.319, **95% CI [2.672, 6.346]**.
This independently reproduces the coordinator's measured [2.60, 6.33]. The interval spans a factor of 2.4, so the
benchmark cannot resolve differences below roughly 0.5–1.0 nats — consistent with the coordinator's assessment,
and it means the 3.051 → 3.994 improvement between attempts 001 and 002 is **not statistically distinguishable**.

**Recommended fix.** Publish the sequence-clustered CI alongside the point estimate in `attempts.jsonl`,
`leaderboard.md`, and the SVG, and treat overlapping intervals as ties. Add a trimmed variant (drop top and
bottom 5% of sequences) and report the fraction of sequences with positive gain — currently 53.5% — as a
guardrail. Consider promoting the median per-sequence gain to co-primary, since it is far more robust.

### B7. MINOR — CSI at 0.5 is defensible here but not scale-invariant

Predicted rate scale: mean 0.0166, median 0.00164, p90 0.0068, p99 0.261, p99.9 2.352, max 31.62. Observed
positive cells: 763 of 206,400 (0.370%). Cells predicted positive at 0.5: 1,221 (0.592%).

| threshold | pred+ | TP | FP | FN | precision | recall | CSI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 5,302 | 436 | 4,866 | 327 | 0.082 | 0.571 | 0.077 |
| 0.1 | 3,689 | 397 | 3,292 | 366 | 0.108 | 0.520 | 0.098 |
| 0.2 | 2,472 | 361 | 2,111 | 402 | 0.146 | 0.473 | 0.126 |
| 0.3 | 1,894 | 323 | 1,571 | 440 | 0.171 | 0.423 | 0.138 |
| 0.4 | 1,503 | 292 | 1,211 | 471 | 0.194 | 0.383 | 0.148 |
| **0.5** | **1,221** | **274** | **947** | **489** | **0.224** | **0.359** | **0.160** |
| 0.75 | 807 | 215 | 592 | 548 | 0.266 | 0.282 | 0.159 |
| 1.0 | 578 | 182 | 396 | 581 | 0.315 | 0.239 | 0.157 |
| 2.0 | 247 | 118 | 129 | 645 | 0.478 | 0.155 | 0.132 |

**The optimum is CSI 0.1615 at threshold 0.511 — so 0.5 captures 99.2% of attainable CSI.** Contrary to the
hypothesis in the brief, 0.5 is *not* an arbitrary slice for this model; it lands almost exactly on the peak.

But this is luck, not design, and it conceals the real problem: **0.5 is a fixed absolute rate applied to models
whose output scale is free.** The same weights scored under the `log-mse` rate transform give CSI 0.148 with a
3.1x smaller total forecast (see C10b). Any future attempt that changes the output scale — a different link
function, a calibration layer, a negative-binomial head — will be scored at a different point on its own CSI
curve, making the leaderboard column non-comparable. The curve is also flat near the peak (0.126–0.160 across
0.2–1.0), so CSI-at-0.5 has little discriminating power between attempts anyway.

**Recommended fix.** Adopt **AUPRC** as the threshold-free primary spatial score. Measured here:

- model AUPRC **0.2285** (prevalence 0.00370, so a **61.8x** lift over random)
- 7-day-mean baseline AUPRC **0.1463**

The model beats the baseline by 1.56x on a metric with no threshold and no clamp — this is the cleanest evidence
of genuine spatial skill in the whole audit, and it is unaffected by the B5 artefact. As a secondary, report CSI
at the **frequency-matched** threshold (predicted-positive count = observed-positive count = 763, giving
t = 0.787, precision = recall = 0.275, CSI = 0.1596), which is scale-invariant by construction. Keep CSI-at-0.5
only as a legacy column.

### B8. IMPORTANT — the maximum-trigger ratio is an n=1 statistic

`train_real.py:131-136` selects `argmax(observations)` — one example out of 516.

| rank | observed | forecast | ratio | component |
|---:|---:|---:|---:|---|
| 1 | 347 | 21.12 | 0.0609 | sequence-004088 |
| 2 | 340 | 38.53 | 0.1133 | sequence-004088 |
| 3 | 253 | 59.75 | 0.2361 | sequence-004088 |
| 4 | 206 | 49.52 | 0.2404 | sequence-004088 |
| 5 | 205 | 42.63 | 0.2079 | sequence-004088 |
| 6 | 153 | 10.85 | 0.0709 | sequence-004062 |
| 7 | 153 | 10.15 | 0.0664 | sequence-004060 |
| 8 | 150 | 13.01 | 0.0867 | sequence-004062 |

The reported 0.0609 is the *minimum* of the top eight. Ratios span 0.061–0.240, a **4.0x spread**, and ranks 1
and 2 differ by only 7 events out of 347 — a single event reclassification could flip which one is reported and
move the metric by 86%. Worse, **the top five all belong to the same component**, so this is not even a draw from
five independent sequences.

With n=1 there are zero degrees of freedom and no confidence interval is computable. The metric is a guardrail in
`benchmark-spec.json`, so it gates decisions while being essentially unmeasured.

Note the substantive finding survives: pooling the top-10 most productive examples gives
`Σforecast/Σobserved = 0.1546`, still a **6.5x underprediction** of extreme sequences. The problem is real; the
estimator is just bad.

**Recommended fix.** Replace with the pooled ratio over the top decile of observed productivity
(`Σ forecast / Σ observed`), which is stable and preserves the intent, and report it with a sequence-clustered
interval. Report the per-sequence ratio distribution rather than a single extremum.

### B9. PASS — `sequence_mean` matches its description

`train_real.py:60-64` groups values by `component_id`, means within each group, then means across groups.
Verified numerically:

- recomputed `sequence_mean(delta)` = **23.676657**
- reported `sequence_mean_log_likelihood_gain` = **23.676656**
- flat mean over all 516 triggers, for contrast = 29.166186

The implementation is correct as described. Two caveats on its usefulness: **134 of 200 sequences contain exactly
one trigger** (median size 1, max 53), so for two-thirds of the data "sequence mean" is just the trigger value and
the reweighting does less than it appears; and the metric is in raw nats per trigger, so it scales with sequence
productivity and inherits the same B5 clamp artefact and B6 concentration.

### Additional finding — train/validation density shift

| split | examples | events | events/example |
|---|---:|---:|---:|
| train | 14,662 | 1,930,125 | **131.64** |
| validation | 516 | 3,768 | **7.30** |

An **18x** gap. It is not a catalogue-coverage artefact: normalized annual event counts are stable through 2023
(GeoNet 5,503 in 2022 and 5,854 in 2023, against 9,189 in 2021 and 7,320 in 2019; magnitude p10 flat at 2.1), and
the catalogue mix is nearly identical across splits (GeoNet 78.9% of train vs 81.4% of validation). The gap is
within-catalogue and example-level: GeoNet averages 133.9 events/example in train vs 6.96 in validation.

The cause is that 1980–2021 contains several mega-sequences, each contributing many M4+ triggers whose next-day
windows all count the same dense aftershock cloud, whereas 2022–2023 happened to contain no such sequence. The
consequences are material: any preprocessing constant fitted on train transfers badly to validation (demonstrated
in B5), the training loss is dominated by mega-sequences even under sequence-balanced sampling, and validation
may simply be too quiet to measure extreme-sequence behaviour — which is exactly what B8 is trying to measure.
**The 2024–2025 test period may differ again in the same way, and this should be characterised from the trigger
manifest — without opening labels — before the seal is broken.**

---

## C. Benchmark integrity

### C10. Tamper rejection and metric re-derivation — both PASS

**Tampered validation set is correctly rejected.** Empirically verified: copying `validation.npz`, zeroing one
real event cell (target sum 3,768 → 3,767), and re-running `benchmark_checkpoint.py` produced

```
Validation tensor hash differs from the frozen benchmark specification
```

before any evaluation. The check at `benchmark_checkpoint.py:53-56` fires correctly. The live
`validation.npz` and `processed-manifest.json` hashes both match `benchmark-spec.json` exactly, so the
frozen set is currently intact. Incidentally, `np.savez_compressed` proved byte-reproducible here, which makes
hash pinning viable.

**Metrics are genuinely re-derived.** `benchmark_checkpoint.py:76` calls `validate()` and never reads
`checkpoint["history"]`. Verified by poisoning a checkpoint's history to claim IG = 999.0 and re-running:
`validate()` still returned 3.994096. Recomputation independently reproduces the recorded 3.994095 to six
decimals. **This is a genuine strength of the design.**

### C10b. IMPORTANT — the `loss` field is self-attested and changes the score

`benchmark_checkpoint.py:67` reads `loss_name = metadata.get("loss", "log-mse")` from the checkpoint, and
`train_real.py:84-88` uses it to choose the rate transform (`expm1` vs `exp`). The same weights:

| declared `loss` | IG/event | CSI | forecast/observed |
|---|---:|---:|---:|
| `poisson` | **3.994** | 0.160 | 0.909 |
| `log-mse` | **1.471** | 0.148 | 0.293 |

**A 2.7x swing driven by a string the checkpoint author writes.** Nothing verifies it against how the model was
actually trained. An author can try both and record the better one.

**Recommended fix.** Derive the transform from the model or evaluate under both and record both; at minimum,
store the training command line and W&B run config in the checkpoint and assert consistency.

### C11. IMPORTANT — paths to a non-frozen `attempts.jsonl` entry

Several. In rough order of severity:

1. **`--spec` is attacker-supplied** (`benchmark_checkpoint.py:42`). The frozen hashes live in the very file the
   caller chooses, so the integrity check verifies the data against whatever the caller says it should be.
   **Demonstrated:** with the tampered validation set from C10 plus a custom spec containing its hash, the run
   completed and wrote a full record. The record does carry the tampered hash
   (`22cb774e…` vs canonical `684128ed…`), so it is *detectable* — but nothing detects it automatically.
   **`tests/test_benchmark.py` never compares recorded hashes against `benchmark-spec.json`.**

2. **`attempts.jsonl` is an unsigned append-only text file** (`benchmark_checkpoint.py:95-96`, plain
   `open("a")`). Rows can be hand-written or edited. `render_benchmark.py` performs **no** validation — it reads
   the log and renders. The CI job (`.github/workflows/benchmark.yml`) runs `render_benchmark.py` then
   `git diff --exit-code`, which only proves the leaderboard is consistent with the log, **not that the log is
   legitimate**. A fabricated row propagates to the published SVG and table and passes CI.

3. **`torch.load(..., weights_only=False)`** (`benchmark_checkpoint.py:63`) executes arbitrary pickle code at
   load time, before metrics are computed. A crafted checkpoint could patch `validate`, NumPy, or Torch to
   return any metrics it likes. (Not exploited — reported from code inspection.)

4. **`final_test_opened` is self-attested** (`benchmark_checkpoint.py:65`). A model trained on test labels simply
   declares `False`. `tests/test_benchmark.py:17` asserts the field is `False`, which tests only that the claim
   was made.

5. **Nothing attests to the training set.** A checkpoint trained *on* `validation.npz` would score near-perfectly
   and be indistinguishable. Validation is designated for model selection, so this is partly by design, but the
   log records no train-set hash.

6. **`--attempts` is caller-supplied and the default is CWD-relative** (`Path("benchmarks/attempts.jsonl")`), so
   the destination depends on the working directory.

**Recommended fixes.** Resolve the spec from a module constant rather than a CLI argument, or pin its own SHA-256
in code. Add a test asserting every row's `validation_tensor_sha256` and `processed_manifest_sha256` equal the
committed spec's — this is a three-line test that closes the highest-value gap. Load checkpoints with
`weights_only=True`. Record a hash of `train.npz` and the training command in each record. Longer term, sign
records, or make CI recompute metrics from the checkpoint rather than trusting the appended row.

---

## Appendix A — reproduction

Reproduced on CPU with the repo venv; matches the recorded run to six decimals.

| quantity | recomputed | recorded |
|---|---:|---:|
| `poisson_log_likelihood_model` | −9,831.5 | −9,831.514 |
| `poisson_log_likelihood_seven_day_mean` | −24,881.3 | −24,881.266 |
| `information_gain_per_observed_event` | 3.994096 | 3.994095 |
| `sequence_mean_log_likelihood_gain` | 23.676657 | 23.676656 |
| `forecast_events` | 3,426.95 | 3,426.951 |

## Appendix B — provenance

This audit was carried out by an independent review agent against the frozen
2022-2023 validation set and modified no repository file. Its findings were
reproduced independently before being adopted: the floored-baseline information
gain and the average-precision figures quoted in the benchmark README were
recomputed from `scripts/train_real.py` and match to six decimal places.

Metadata author: James Edward Ball.
