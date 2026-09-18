# MutantScope Evaluation Protocol

## Required comparisons

Evaluate at minimum:

1. the classical baseline;
2. the frozen ESM-2 mutation-aware MLP.

Optional zero-shot and parameter-efficient variants must use the same split manifest. Report compute and training differences; never compare models under different leakage conditions as if equivalent.

## Phase 2 validation baseline

The documented `phase2_ridge_baseline_v1` is selected on validation only from a fixed ridge-penalty grid. On 58,363 validation records it reports MAE 0.7061 kcal/mol, RMSE 0.9108 kcal/mol, Pearson 0.4688, and Spearman 0.4242; the training-mean control has MAE 0.7984 and RMSE 1.0303. These are development metrics, not test performance. The 58,333-record test partition remains unopened. See `docs/PHASE2_BASELINE.md` for exact features, provenance, and diagnostics.

## Phase 3 feature-cache audit

Phase 3 creates representations, not ddG predictions, so it introduces no MAE, RMSE, or correlation result. Completion requires exact coverage of all 272,372 training and 58,363 validation records, unchanged source/split provenance, label-free shards, finite float32 vectors, and the specified 3,840-dimensional feature layout. The independent audit also checks cached vectors against freshly encoded mutations and verifies residue indexing, special-token exclusion, padding invariance, and frozen encoder parameters. Record the evidence in `artifacts/phase3/verification_report.json`; a smoke cache or unit tests alone do not prove full completion. The held-out test partition remains deferred, with no Phase 3 test evaluation.

## Metrics

Report MAE, RMSE, Pearson correlation, Spearman correlation, sample counts, and protein/cluster counts. Where practical, report per-protein or grouped distributions so heavily measured proteins do not dominate the aggregate.

## Evaluation discipline

Fit scalers, imputers, clipping, and learned preprocessing on training data only. Select checkpoints, thresholds, and hyperparameters using validation only. Freeze the configuration, then evaluate the untouched test partition once. Save predictions, residuals, model version, split version, seeds, environment, and target convention.

## Diagnostics

Retain predicted-vs-observed scatter, residual plots, grouped performance, and baseline-vs-primary metric tables. Inspect sign/unit inversions, duplicate leakage, invalid alignment, extreme values, and underrepresented proteins.

## ThermoMutDB external validation

After the model, checkpoint, and thresholds are frozen, optionally evaluate once on a filtered ThermoMutDB release. Do not tune on its results. Report release, filters, condition policy, unit/sign conversion, counts, and overlap with every MegaScale partition. Separate overlapping records/proteins from the independent subset.

ThermoMutDB’s documented interpretation is positive values = stabilization and negative values = destabilization. Verify conversion before metrics. Because its conditions and composition differ from MegaScale, present it as a distribution-shift robustness check, not an interchangeable test score.

## Uncertainty

If intervals are deployed, report empirical coverage and interval width on held-out data. Do not display an uncertainty value without this evidence.
