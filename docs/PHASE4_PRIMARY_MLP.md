# Phase 4 — Frozen ESM MLP regression

## Scope and contract

All 14 pre-existing project Markdown files were read before implementation. Phase 4 trains a regression head on the verified Phase 3 development cache. The frozen encoder, eight-block 3,840-feature schema, MegaScale `v2_230420` targets (kcal/mol, positive stabilizing), and `megascale_cluster_split_v1` assignments remain fixed. Phase 5 is outside this work.

## Initial configuration

The head is `3840 → 256 → 128 → 1`, with GELU and dropout 0.1 after each hidden layer. Defaults: HuberLoss delta 1.0, AdamW learning rate 0.001 and weight decay 0.0001, batch size 512, seed 20260918, maximum 60 epochs. Checkpoint selection minimizes validation MAE. Early stopping uses eight stale epochs and a 0.0001 improvement threshold; the saved checkpoint retains the lowest actual MAE even if an improvement is smaller than that patience threshold.

Feature means and population standard deviations are fitted using training records only, with float64 streaming Welford updates. Scales below 1e-6 become 1.0. Raw ddG targets are not standardized. Source row identifiers only join labels to verified embeddings; they are not model inputs. Training batches use a deterministic global permutation each epoch. Evaluation disables dropout.

## Storage and environment

Phase 3 files are verified against their hashes, exact mutation keys and partition membership before producing label-free base-vector NumPy memmaps under `data/cache/phase4/development_v1/`. Batches reconstruct and standardize the feature blocks on demand, keeping working memory bounded. Encoder parameters are not present in the optimizer.

The full head fit used the approved PyTorch 2.12.1 CPU installation in `.venv-phase3-cpu`. CUDA had been blocked by Windows Smart App Control during Phase 3; a fresh Phase 4 probe subsequently succeeded with PyTorch 2.12.1+cu130 and the RTX 5070. A real GPU ESM forward pass matched CPU vectors within `1.430511474609375e-06`. No security settings were changed by this implementation.

Held-out extraction initially produced 32,768 rows on CPU using 16 threads. The remaining 25,565 rows were extracted on CUDA in about 38 seconds including verified-shard reuse and encoder initialization. During extraction, `nvidia-smi` reported 52% GPU utilization and 657 MiB device memory. Shard device fields, runtime history, and aggregate device counts preserve the mixed-device provenance. Cache keys depend on the model/precision/schema rather than hardware; the final verifier freshly checks test vectors against the same CPU encoder within float32 tolerance.

Shared dependency pins and Matplotlib 3.10.8 are in `requirements.txt`. Install these into the selected interpreter before evaluation. The head's frozen CPU training configuration is unchanged by the encoder-device migration. Configuration, package inventory, source/split hashes, checkpoint hashes, epoch history and elapsed time are recorded with the experiment. Model weights, large predictions and derived caches remain local. Cache `extraction_seconds` and evaluation `elapsed_seconds` describe the last invocation, including reuse checks; they do not total earlier interrupted invocations. The CPU-to-CUDA migration interrupted extraction before any completed evaluation report, then resumed the same frozen experiment. Plotting completion reused the fully extracted cache; no model/selection choice changed.

## Commands

```powershell
./.venv-phase3-cpu/Scripts/python.exe -m unittest discover -s tests -v
./.venv-phase3-cpu/Scripts/python.exe scripts/train_phase4_primary.py --smoke-rows 4096 --max-epochs 2 --artifacts-dir artifacts/phase4-smoke
./.venv-phase3-cpu/Scripts/python.exe scripts/train_phase4_primary.py
```

A smoke run never freezes a production selection. The full trainer refuses to overwrite an existing experiment; a new experiment needs a separate artifacts directory. The trainer filters selected partitions before interpreting targets and never loads test targets for training, normalization, early stopping or checkpoint selection.

## Completion gates

- Passing unit/integration tests and real-model smoke training.
- Complete training/validation coverage and independent verification of training-only statistics.
- Full-data head fitting, saved best checkpoint, validation history/predictions and frozen selection.
- Separate test feature extraction using that frozen contract; one held-out comparison of both the primary model and the already-selected ridge baseline.
- Verified predictions/residuals, MAE/RMSE/Pearson/Spearman, protein/cluster counts and grouped diagnostics; scatter/residual plots and comparison report.
- Documentation and status updated from actual outputs. Phase 5 was outside this phase; subsequent product integration is documented in `docs/PHASE5_PRODUCT.md`.

## Verified held-out comparison

The frozen test partition has 58,333 mutations across 63 proteins and 39 clusters. Its saved results are:

| Model | MAE (kcal/mol) | RMSE (kcal/mol) | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Ridge baseline | 0.7505 | 1.0052 | 0.4254 | 0.4117 |
| Frozen ESM MLP | 0.6390 | 0.9339 | 0.5661 | 0.5762 |

Primary protein-macro MAE is 0.6373, versus baseline 0.7521; primary cluster-macro MAE is 0.5770, versus baseline 0.6970. Per-protein primary MAE ranges from 0.3544 to 2.1027 kcal/mol, so the aggregate does not establish uniform accuracy for every protein. Predictions are point estimates; no uncertainty was calibrated. No hyperparameters or checkpoint were changed after opening test.

Evidence: `artifacts/phase4/selection.json`, `training_config.json`, `training_history.json`, `development_verification.json`, `test_metrics.json`, `test_sample.csv`, and `test_diagnostics.png`. Large local prediction/residual and per-protein/per-cluster files are Git-ignored. The full independent audit passed in `artifacts/phase4/verification_report.json`: exact source joins, training-only normalization, best-epoch selection, all validation/test predictions and residuals, every protein/cluster metric, and all five sample rows. Six fresh test re-encodings including boundary positions matched the mixed-device cache within `2.8014183044433594e-06`; the encoder remained frozen. The plot was visually inspected. All 26 tests passed at Phase 4 completion. Subsequent Phase 5 product integration is documented separately.

All 26 tests pass. A two-epoch smoke run on 4,096 records per split passed training/checkpoint reload in 33.77 seconds including full Phase 3 preparation; epochs took 0.3–0.5 seconds. Full fitting used all 272,372 training records, stopped at epoch 10 and selected epoch 2. Build time was 213.04 seconds; selected validation MAE 0.5995, RMSE 0.8217, Pearson 0.6281 and Spearman 0.6072. These are development metrics. The selection is frozen in `artifacts/phase4/selection.json`; independent development and full held-out audits both passed. No configuration or model selection changed after test opening.

After training, run `scripts/verify_phase4_artifacts.py --development-only`, then `scripts/evaluate_phase4_test.py`, and finally `scripts/verify_phase4_artifacts.py`, using the same CPU interpreter. The evaluator requires development verification, refuses to change the frozen contract after opening test, and refuses repeated completed evaluation. Interrupted feature extraction can resume only with identical model/configuration/source fingerprints. Verification reproduces the saved frozen predictions and does not select another model.
