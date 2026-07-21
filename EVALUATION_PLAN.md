# Catalogue and evaluation plan

Author: James Edward Ball

Status: proposed protocol
Date: 2026-07-21

## Objective

Evaluate whether SmaAt-UNet improves next-day, trigger-centred seismicity-rate
forecasts after an M4+ event. The forecast is the expected number of events in
each 0.1-degree cell during the following 24 hours.

The primary claim is conditional and retrospective: given the catalogue data
available in the experiment, does the model assign better-calibrated rates and
spatial probability than operationally meaningful baselines?

## Catalogue shortlist

### Tier 1 - initial experiment

1. Southern California SCSN/SCEDC standard catalogue - long record, annual
   files and searchable catalogue. QTM provides a high-resolution 2008-2017
   pretraining and ablation dataset.
2. Northern California NCSS/NCEDC standard catalogue - long record, CSV search
   and web services. Keep the catalogue definition and depth datum fixed.
3. New Zealand GeoNet - stable event IDs, modification timestamps, review
   status and FDSN/WFS access. This is useful for an external tectonic-region
   test.
4. Italy INGV ISIDe - FDSN access and a separate high-resolution 2016-2017
   Central Italy ML catalogue for pretraining and ablation.

### Tier 2 - external validation

5. Greece NOAIG - long annual archive and FDSN service. Check revision status
   and magnitude consistency before inclusion.
6. Japan NIED JUICE - high-resolution relocated 2001-2012 catalogue. Treat it
   as a separate historical domain because locations are relative and grid-edge
   artefacts are documented.

USGS ComCat is useful as a common ingestion fallback and for cross-identifying
events. Regional authoritative catalogues should remain the source of truth.

## Frozen data products

Every raw download receives:

- catalogue name, release/version and retrieval timestamp;
- source URL, query parameters and geographic bounds;
- SHA-256 digest and row count;
- original event ID, origin time, modification time and review status where
  available;
- original magnitude value and type;
- an immutable raw file plus a separate normalized table.

No catalogue refresh is permitted after the final test manifest is sealed.
Corrections become a new experiment version.

## Event normalization and quality control

1. Retain tectonic earthquakes only; remove blasts and non-seismic events.
2. Preserve original magnitude type. Harmonize magnitudes only with a mapping
   fitted on training data. Report results without harmonization as a
   sensitivity analysis.
3. Remove exact duplicates by source ID. Cross-catalogue duplicates are matched
   using origin time, hypocentral distance and magnitude tolerances, then
   manually inspect ambiguous M4+ triggers.
4. Keep depths from 0 to 40 km. Flag fixed, negative and poorly constrained
   depths rather than silently treating them as measurements.
5. Estimate time-varying magnitude of completeness for each region using
   training data only. Freeze the estimation method before validation.
6. Record catalogue regime changes such as network expansion, automated
   detection and magnitude-scale changes.

The paper's M2 threshold is below its reported completeness estimate for every
standard regional catalogue. We therefore run two named experiments:

- **Reproduction endpoint:** M2+ inputs and M2+ targets, matching the paper.
  Conclusions are catalogue-conditioned.
- **Completeness-controlled endpoint:** events above a region-and-period
  threshold fixed from training data. This is the primary scientific endpoint.

## Leakage-resistant sample grouping

The unit of independence is a connected earthquake sequence, not an individual
M4 trigger or map.

For every M4+ trigger, define its full example footprint as:

- time: trigger minus 7 days through trigger plus 1 day;
- space: the 2 by 2 degree forecast box;
- events: every catalogue event used by its input or target.

Build a graph where two examples are connected when they share any event or
their time-space footprints overlap. All examples in a connected component go
to the same partition. Apply a 30-day embargo around temporal boundaries and
repeat the analysis with 14- and 60-day embargoes.

This prevents:

- one aftershock sequence appearing in both train and test;
- a target event from one example appearing as an input event elsewhere across
  the split;
- adjacent M4 triggers from the same cascade being treated as independent;
- duplicate standard and high-resolution catalogue events crossing partitions.

## Proposed partitions

### Locked final experiment

- training: start of the selected modern regime through 2021-12-31;
- validation and all model selection: 2022-01-01 through 2023-12-31;
- final test: 2024-01-01 through 2025-12-31;
- boundary embargo: 30 days, applied by connected sequence.

The final test labels remain inaccessible to training, normalization,
completeness estimation, feature selection, threshold selection and early
stopping. One nominated person or sealed evaluation command should release the
scores once.

### Rolling-origin backtest

Run additional folds with two-year forecast windows. Each fold trains only on
earlier data and re-fits every learned preprocessing step. Aggregate fold
results by sequence, while retaining per-region results.

### Geographic generalization

Perform nested leave-one-region-out evaluation. Hyperparameters are chosen
using only the included regions; the held-out region is never used for model
selection. Southern and Northern California are treated as related domains and
also held out together in a sensitivity run.

## High-resolution catalogue policy

QTM 2008-2017 and Central Italy ML 2016-2017 may be used only in these declared
conditions:

1. historical pretraining followed by testing in strictly later standard
   catalogue periods;
2. within-period comparison where both training and evaluation use the same
   catalogue product, with sequence-grouped splits;
3. an ablation measuring the benefit of high-resolution data.

Never train on an ML catalogue and evaluate the same physical sequence in its
standard catalogue representation. Never merge a standard catalogue with its
high-resolution superset without event-level deduplication.

## Baselines

Every forecast is generated from exactly the same information cutoff:

1. previous-day persistence;
2. mean of the previous seven daily maps;
3. spatially smoothed historical seismicity fitted on training data;
4. a simple three-channel CNN with similar training procedure;
5. ETAS, calibrated using training data and the same magnitude threshold.

The network must beat persistence and seven-day mean to justify complexity. ETAS
is the substantive benchmark.

## Scores

### Primary

- Poisson log score and information gain per observed earthquake relative to
  ETAS and persistence;
- number calibration for total forecast events;
- conditional spatial likelihood or CSEP spatial test;
- sequence-clustered 95% confidence intervals for score differences.

### Secondary

- mean absolute error and root mean squared error;
- precision-recall AUC for occupied cells;
- critical success index and false alarm ratio at a threshold chosen once on
  validation data;
- reliability diagrams by forecast-rate decile;
- results stratified by region, trigger magnitude, sequence productivity,
  catalogue regime and magnitude threshold.

ROC AUC and raw accuracy are descriptive only because almost all cells are
empty. Statistical uncertainty is bootstrapped by connected sequence, never by
cell or individual trigger. Multiple model comparisons use a declared family
and adjusted p-values.

## Probabilistic output

Convert the model's log-rate output to a non-negative expected count before
scoring. Compare the paper's log-MSE training objective against Poisson negative
log-likelihood. Check overdispersion by sequence; if the Poisson variance is too
narrow, evaluate a negative-binomial or ensemble forecast while retaining a
Poisson-compatible rate for CSEP comparisons.

## Operational realism

Retrospective reviewed catalogues contain revisions that were unavailable at
forecast time. Label all initial results as retrospective. A later shadow test
should freeze the model and issue forecasts from real-time catalogue snapshots
with a declared latency, such as 10 or 30 minutes after the M4 trigger. Store
every issued forecast before its target window completes.

## Acceptance criteria

Before opening the final test set:

- all preprocessing and split-integrity tests pass;
- every test sequence is absent from training and validation;
- primary metrics, baselines and subgroup analyses are frozen;
- at least 200 independent test sequences are available overall;
- no region-level superiority claim is made with fewer than 50 independent
  sequences;
- the model shows positive information gain against both persistence baselines
  with a sequence-clustered 95% interval above zero;
- number and spatial calibration do not show systematic failure.

## Pipeline test plan

| Area | Test type | Required checks |
|---|---|---|
| Parsers | Unit | timestamps, magnitude types, missing/fixed depths, event types |
| Deduplication | Unit and integration | known duplicates merge; nearby distinct events remain separate |
| Gridding | Unit | boundary cells, coordinate orientation, trigger inclusion, target exclusion |
| Grouping | Property test | shared event or overlapping footprint always implies one split |
| Temporal split | Integration | no future rows enter fitted preprocessing or model training |
| Catalogue freeze | Integration | manifest hashes match raw files before every run |
| Forecasts | Unit | finite and non-negative rates; fixed 20 by 20 geometry |
| Metrics | Golden tests | reproduce hand-calculated Poisson and persistence examples |
| End to end | Smoke | one frozen mini-catalogue produces identical scores on CPU and Metal |

Target 100% branch coverage for split, grouping and leakage checks; 90% for
catalogue normalization and metrics; smoke coverage for model training.

## Sources

- Paper: https://doi.org/10.1186/s40623-025-02241-6
- SCEDC: https://scedc.caltech.edu/data/eq-catalogs.html
- Southern California QTM: https://scedc.caltech.edu/eq-catalogs/qtm.html
- NCEDC: https://www.ncedc.org/ncedc/catalog-search.html
- GeoNet: https://www.geonet.org.nz/data/types/eq_catalogue
- INGV data services: https://data.ingv.it/metadata/web_service_eng
- Greece NOAIG: https://www.gein.noa.gr/services/cat.html
- Japan JUICE: https://www.hinet.bosai.go.jp/topics/JUICE/
- Central Italy ML: https://doi.org/10.5281/zenodo.4736089
- USGS ComCat API: https://earthquake.usgs.gov/fdsnws/event/1/
- pyCSEP: https://docs.cseptesting.org/concepts/evaluations.html
