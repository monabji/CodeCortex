# MutantScope Status

## Current state

Phases 0, 1, 2, 3, 4, and 5 are complete. The repository-wide implementation-contract audit and MegaScale source audit are complete. The source is pinned to Zenodo record 7992926 (`v2_230420`, CC BY 4.0, `Processed_K50_dG_datasets.zip`, publisher MD5 `f7e8c553efee734cf161ee6f2b0a09cf`); the local raw archive is checksum-verified and Git-ignored.

Validated preprocessing, cluster-safe splitting, and frozen ESM feature extraction are complete. Phase 4 trained the compact 1,016,321-parameter regression head on top of a pretrained ESM-2 encoder whose parameters remained frozen. The validation-selected epoch-2 MLP achieved the locally reported test MAE 0.6390 kcal/mol versus baseline 0.7505 on the same 58,333 held-out records. No model choices changed after opening test according to the local run metadata. The CUDA recheck succeeded during Phase 4; the final 25,565 test representations were extracted on RTX 5070 after 32,768 CPU-produced rows. Phase 5 FastAPI/Next.js product integration is complete and gated on the separately stored checkpoint and encoder artifacts; Phase 6 remains unstarted.

## Phase 0 progress

- [x] Read and reconciled every repository Markdown specification, including the architecture, data, ML, evaluation, phases, decision log, project scope, and prior status.
- [x] Pinned the permitted primary source: Zenodo record 7992926, release `v2_230420`, DOI `10.5281/zenodo.7992926`, open access under CC BY 4.0.
- [x] Recorded the source archive, expected contents, publisher MD5, requested-use registration notice, and source-level quality cautions in `docs/PHASE0_DATA_AUDIT.md`.
- [x] Implemented reproducible local acquisition with checksum verification in `scripts/acquire_megascale.ps1`; raw and derived data are Git-ignored.
- [x] Implemented a standard-library aggregate-only schema audit in `scripts/phase0_audit.py`; both it and the PowerShell acquisition script pass static validation.
- [x] Acquired the 1,013,678,473-byte source archive and verified its MD5 against the publisher value.
- [x] Inspected the actual archive membership and Dataset 2/3 CSV schema. The primary table has 776,298 rows, 479 `WT_name` proteins, and 133 `WT_cluster` groups; the complete aggregate report is `data/audit/megascale_v2_230420.audit.json`.
- [x] Verified the Phase 1 target contract: numeric `ddG_ML` is continuous kcal/mol, with positive values stabilizing. There are 607,839 numeric target values and 389,068 initial syntactically single-substitution candidates. Literal `-` targets are excluded; G0/G1/G2 are source concepts, not an inspected CSV field.

### Phase 0 evidence

- Source-audit record and invocation instructions: `docs/PHASE0_DATA_AUDIT.md`.
- Acquisition command: `./scripts/acquire_megascale.ps1`.
- Audit command after download: `python scripts/phase0_audit.py data/raw/Processed_K50_dG_datasets.zip`.
- The raw archive is intentionally absent from version control; only aggregate provenance/audit outputs may be committed.

## Phase 1 progress

- [x] Implemented strict single-substitution and amino-acid validation, including one-based bounds and mutant-residue checks.
- [x] Reconstructed each wild-type sequence from the source mutant `aa_seq`, then re-applied the mutation as a round-trip check.
- [x] Retained 389,068 valid records and wrote a local row-level exclusion manifest for 387,230 source rows: 288,384 non-single substitutions and 98,846 nonnumeric targets.
- [x] Implemented deterministic duplicate/conflict handling. This release had no duplicate or conflicting reconstructed-sequence/mutation keys.
- [x] Created `megascale_cluster_split_v1`: 130 cluster-safe groups assigned with seed `20260918` to 70/15/15 train/validation/test targets by row count.
- [x] Verified empty train/validation/test intersections for clusters, proteins, and reconstructed wild-type sequences; the independent verifier also confirmed retained-plus-excluded rows equal the source total.
- [x] Added unit tests for parsing, sequence reconstruction, mismatch rejection, duplicate policy, and deterministic split assignment.
- [x] Documented the policy, outputs, and safeguards in `docs/PHASE1_DATA_PIPELINE.md`, `docs/DATA_SPEC.md`, and `docs/DECISIONS.md`.

### Phase 1 evidence

- Builder: `python scripts/build_phase1_dataset.py`
- Independent verifier: `python scripts/verify_phase1_artifacts.py`
- Versioned split summary: `data/manifests/megascale_v2_230420_split_v1.json`
- Full policy and counts: `docs/PHASE1_DATA_PIPELINE.md`

## Phase 2 progress

- [x] Implemented a transparent 45-feature ridge-regression baseline using only mutation identities, normalized position, sequence length, and fixed biochemical substitution descriptors.
- [x] Fit feature scaling and all model weights on the 272,372-record training split only.
- [x] Selected the ridge penalty from a fixed six-value grid by validation MAE only; selected alpha: `0.0001`.
- [x] Saved validation predictions, residuals, per-protein diagnostics, model parameters, configuration, provenance, and a five-row human-checkable sample.
- [x] Independently verified that all 58,363 saved predictions belong exactly to the validation partition, recompute the recorded metrics, match the model feature schema, and leave the test partition unevaluated.
- [x] Recorded validation MAE 0.7061 kcal/mol, RMSE 0.9108 kcal/mol, Pearson 0.4688, and Spearman 0.4242. The training-mean control MAE is 0.7984 kcal/mol.
- [x] Added unit tests for feature construction, train-only scaling/model prediction, and metrics.
- [x] Confirmed with an independent project-health review that Phase 1 artifacts are valid and no Phase 3 work was present or started.

### Phase 2 evidence

- Training command: `python scripts/train_phase2_baseline.py`
- Independent artifact verifier: `python scripts/verify_phase2_baseline.py`
- Reproducibility and sample report: `docs/PHASE2_BASELINE.md`
- Validation metrics: `artifacts/phase2/validation_metrics.json`
- Five sample results: `artifacts/phase2/validation_sample.csv`

## Phase 3 progress

- [x] Re-read repository Markdown contracts and inspect current Phase 1/2 artifacts and hardware.
- [x] Implement immutable-revision frozen ESM-2 loading and per-batch residue/token alignment validation.
- [x] Implement residue-only mean pooling and the specified eight-block, 3,840-dimensional mutation representation.
- [x] Implement atomic, resumable, checksum-verified shards with model/schema/sequence keys and an exclusive builder lock.
- [x] Add independent provenance/coverage verification and live re-encoding/padding checks.
- [x] Pass 21 unit/integration tests (including the prior eight tests), covering finite corruption, repeated resume, orphan regeneration, provenance, the exclusive lock, and active-versus-shadowed package versions.
- [x] Download and publisher-checksum-verify the CUDA PyTorch wheel; installation succeeds but Smart App Control blocks `caffe2_nvrtc.dll` (WinError 4551 / Code Integrity event 3077).
- [x] Complete a real-model CPU smoke extraction of 16 records and independently verify its format/provenance (explicitly not a full-cache completion claim).
- [x] Benchmark approved CPU PyTorch: select 16 threads/batch 32, measured 86.28 records/second on mixed-length training sequences; save `artifacts/phase3/cpu_benchmark.json`.
- [x] Finish and independently verify all 330,735 training/validation records on CPU without downsampling; held-out test extraction is deferred.
- [x] Record aggregate cache provenance, environment, sample diagnostics, and live-encoder evidence.

### Phase 3 evidence

- Full policy and reproduction commands: `docs/PHASE3_ESM_FEATURES.md` and `README.md`.
- Builder: `./.venv-phase3-cpu/Scripts/python.exe scripts/build_phase3_features.py --device cpu --cpu-threads 16 --batch-size 32 --local-files-only --report artifacts/phase3/extraction_report.json`.
- Independent verifier: `./.venv-phase3-cpu/Scripts/python.exe scripts/verify_phase3_features.py --device cpu --local-files-only --report artifacts/phase3/verification_report.json`.
- Aggregate reports: `artifacts/phase3/extraction_report.json`, `artifacts/phase3/verification_report.json`, and `artifacts/phase3/cpu_benchmark.json`; human-checkable vector diagnostics: `artifacts/phase3/feature_sample.csv` (not ddG predictions).
- Exact train/validation coverage, file checksums, source/split fingerprints, label-free fields, finite float32 vectors, difference layout, and public-reader coverage all passed. Live verification re-encoded 12 mutations including first/last-residue examples; maximum absolute re-encoding error was `2.384185791015625e-06`, padding error `1.430511474609375e-06`, within the specified float32 tolerances. The encoder has 33,269,521 parameters and zero trainable parameters.
- CPU extraction used 16 threads/batch 32 and took 4,193.79 seconds (69.90 minutes) including setup/wild-type encoding. The local vector cache contains 2,663,064,398 bytes (2.66 GB; 2.48 GiB), excluding encoder weights/runtime downloads. Cache manifest SHA-256: `a7fa14638a94e6e979b570067df8c6fb5d9f55f2ffb858e6926af2bdcd1fb7b9`.
- All 24 shared dependency pins match the active CPU environment. The final verifier's `verification_runtime.packages` resolves active packages rather than shadowed inherited distributions; the original extraction metadata is retained unchanged. No security settings were changed, labels used, held-out test features extracted, or test performance reported.
- Phase 3 was committed and pushed as `17b2991`. Its status, evaluation, decision log and Phase 3 documentation are tracked; other local Markdown files are still covered by `.git/info/exclude`.

## Phase 4 progress

- [x] Read all 14 existing project Markdown files before implementation.
- [x] Implement the compact MLP, HuberLoss/AdamW trainer, training-only normalization, and validation checkpoint selection.
- [x] Validate Phase 3 hashes/mutation keys and prepare bounded, label-free memmaps.
- [x] Pass 26 tests, including normalization statistics, checkpoint prediction round trips, deferred test-target parsing, frozen/repeated evaluation guards, and saved-prediction precision consistency.
- [x] Complete real smoke training and checkpoint reload: two epochs on 4,096 records per split, 0.3–0.5 seconds per epoch (smoke-only metrics).
- [x] Fit the head on all 272,372 training records; early stopping at epoch 10 selects epoch 2 on 58,363 validation rows. Build time 213.04 seconds; validation MAE 0.5995, RMSE 0.8217, Pearson 0.6281, Spearman 0.6072. Selection/checkpoint are frozen; independent development audit passed (`artifacts/phase4/development_verification.json`).
- [x] Extract all 58,333 held-out representations and evaluate baseline/primary model in one frozen experiment. CPU produced 32,768 rows; CUDA produced 25,565 after a successful real-model device probe. The selection/checkpoint/baseline fingerprints remained unchanged.
- [x] Independently verify all predictions/residuals, grouped diagnostics, sample rows and cache joins, including six live test re-encodings (maximum absolute error `2.8014183044433594e-06`). Visually inspect the scatter/residual plot and document measured results.

Policy, configuration and reproduction commands: `docs/PHASE4_PRIMARY_MLP.md`. Subsequent Phase 5 product work is recorded below and in `docs/PHASE5_PRODUCT.md`.

### Phase 4 evidence

- Configuration/history/freeze: `artifacts/phase4/training_config.json`, `training_history.json`, `selection.json`; selected head has 1,016,321 trainable parameters, with no encoder parameters in its optimizer.
- Independent audits: `artifacts/phase4/development_verification.json` and `verification_report.json`. All 26 tests pass.
- Frozen held-out comparison: `artifacts/phase4/test_metrics.json`, 58,333 records/63 proteins/39 clusters. MLP MAE 0.63898, RMSE 0.93386, Pearson 0.56613, Spearman 0.57617; ridge MAE 0.75052, RMSE 1.00521, Pearson 0.42537, Spearman 0.41168. MLP test MAE is about 14.9% lower.
- Grouped primary macro MAE: protein 0.63731 versus baseline 0.75210; cluster 0.57702 versus baseline 0.69697. Individual primary protein MAEs range 0.35444–2.10274 kcal/mol.
- Human-checkable sample/plots: `artifacts/phase4/test_sample.csv`, `test_diagnostics.png`. Full validation/test predictions with residuals and protein/cluster tables remain local and Git-ignored, as do model weights and derived caches.
- Final checkpoint SHA-256: `9eded3ed44ece10615d9ab13aed79ef0324f87f1321653e0d429aa852d3a71ae`. The test guard records the immutable selection, model and input hashes and blocks another completed evaluation.
- Head fitting used CPU (213.04 seconds). Actual extractor sampling confirmed about 16 busy CPU cores. A later CUDA forward probe succeeded without changing security settings, matched CPU vectors within `1.430511474609375e-06`, and extraction sampling showed 52% GPU utilization/657 MiB. Remaining CUDA extraction took about 38 seconds. Runtime history and per-shard device provenance are retained; last-invocation timings exclude earlier interrupted work.
- Phase 4 changes are local and have not been committed or pushed. The new detailed Phase 4 document is locally excluded by `docs/`; tracked README/status/evaluation files also record the results.

## Phase 5 progress

The complete frontend now follows `Manu_protein_stability (5).html`: five anchor-linked sections, matching editorial styling and a fully connected clickable best-substitution residue map. API responses populate prediction, scan, evaluation and provenance views when the separately stored model artifacts are available. Reference example scores/opposite-sign labels, prototype contribution bars and placeholder contact details were replaced with scientifically accurate data/content. Five frontend tests pass; typecheck, lint, production build and all 49 backend tests pass. Desktop browser checks verify model-backed predictions when the local service is available, mismatch/stale-result handling, both scan scopes, 19-result dialogs, selection/re-prediction, cancellation and CSV download. Responsive CSS is implemented, but mobile visual verification is not claimed because the browser resize control did not change the actual viewport. Evidence: `artifacts/phase5/manu_ui_verification.json`.

- [x] Read all 15 existing project Markdown files and inspect model/source/runtime interfaces.
- [x] Implement frozen checkpoint deployment gates, exact feature construction and bounded memory inference caches.
- [x] Implement FastAPI health, provenance, single prediction and position scan endpoints with strict schemas and actionable errors.
- [x] Implement batched whole-protein jobs, progress, cancellation, queue capacity and result expiry.
- [x] Reproduce five saved validation predictions with real online inference; verify real first/last-position and 57-mutation protein scans.
- [x] Complete Next.js single, position and whole-protein views, provenance, ranked results/heatmap, full CSV export and same-origin API forwarding.
- [x] Pass all 49 backend tests (23 Phase 5), all five frontend tests, TypeScript checks, ESLint and production build.
- [x] Verify real browser flows, mismatch rejection, 19-position variants, 57-mutation scan, cancellation/input unlocking and successful CSV download; finalize documentation/evidence.

Phase 5 is complete. The selected Phase 4 model is reused without training or new model selection. CPU and CUDA deployment verifiers reproduce five saved validation predictions with maximum errors `1.7136335372924805e-7` and `1.0013580322265625e-5` kcal/mol, within the `2e-5` tolerance. Both verify boundary position scans, all 57 whole-protein variants and CSV contents. Reports: `artifacts/phase5/inference_verification.json`, `cuda_inference_verification.json`, and `product_verification.json`. A real 1,024-residue CUDA prediction also succeeded; a live 500-residue scan was cancelled after 132/9,500 variants. Current preview: `http://127.0.0.1:3010`, API `http://127.0.0.1:8010` (CUDA encoder, CPU head). Existing services on ports 3000/8000 were preserved. Phase 4/5 changes remain local and uncommitted.

## Next implementation sequence

1. Phase 5 is delivered; use the local preview or reproduce startup from `docs/PHASE5_PRODUCT.md`. No Phase 6 work is authorized by the current goal.
2. Any later explicitly requested research variants must preserve the split and disclose that the original test result is now known. ThermoMutDB remains reserved for post-development external validation with overlap/condition auditing.

## Evidence log

- 2026-09-18 — Phase 0 complete. Reviewed all repository Markdown specifications; acquired MegaScale Zenodo record 7992926 `v2_230420`; verified its 1,013,678,473-byte archive against MD5 `f7e8c553efee734cf161ee6f2b0a09cf`; and generated `data/audit/megascale_v2_230420.audit.json`. The Dataset 2/3 table contains 776,298 rows, 479 wild-type proteins, 133 clusters, 607,839 numeric `ddG_ML` values, and 389,068 initial exact-single-substitution candidates. Phase 1 must filter literal `-` targets, validate sequence/mutation alignment, deduplicate without crossing split boundaries, and use group-safe manifests; it must not infer G0/G1/G2 from a nonexistent CSV column.
- 2026-09-18 — Phase 1 complete. Built and independently verified a leakage-safe MegaScale pipeline. It retained 389,068 reconstructed, round-trip-validated single substitutions and excluded 387,230 rows with a row-level reason manifest. The seed-`20260918` cluster-safe split has 272,372 training, 58,363 validation, and 58,333 held-out test records; cluster, protein, and reconstructed wild-type sequence intersections are all empty. No Phase 2 model training or test evaluation was performed.
- 2026-09-18 — Phase 2 complete. Trained `phase2_ridge_baseline_v1` on the Phase 1 training split using only 45 inference-available tabular features. Selected alpha `0.0001` by validation MAE on 58,363 validation records: MAE 0.7061 kcal/mol, RMSE 0.9108 kcal/mol, Pearson 0.4688, Spearman 0.4242. The mean-prediction control MAE is 0.7984 kcal/mol. Independent verification confirmed exact validation-prediction coverage and no test-set evaluation. No Phase 3 work was started.

- 2026-09-18 — Phase 3 complete. Extracted and independently verified frozen `facebook/esm2_t12_35M_UR50D` revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`, layer 12, float32 site/global vectors and eight-block 3,840-wide features. All 330,735 MegaScale `v2_230420` development records under `megascale_cluster_split_v1`, seed `20260918`, are covered in 82 shards with 349 wild-type entries. Reports are `artifacts/phase3/extraction_report.json` and `artifacts/phase3/verification_report.json`; 21 tests pass. Full CPU build took 69.90 minutes and cached 2.66 GB. No labels or test records were used for feature construction; Phase 3 has no prediction metrics and Phase 4 remains unstarted.

- 2026-09-18 — Phase 4 complete. Read all 14 pre-existing Markdown files, implemented a 3840/256/128/1 frozen-feature MLP with GELU/dropout 0.1, HuberLoss/AdamW and train-only scaling. MegaScale `v2_230420`, `megascale_cluster_split_v1`, seed `20260918`, full 272,372 training/58,363 validation coverage. Early stopping at epoch 10 selected epoch 2: validation MAE 0.59949, RMSE 0.82170, Pearson 0.62810, Spearman 0.60720. Frozen held-out comparison on 58,333 mutations: primary MAE 0.63898 versus ridge 0.75052; full metrics in `artifacts/phase4/test_metrics.json`, independently reproduced in `verification_report.json`. All 26 tests pass. CPU-to-CUDA extraction migration preserved the frozen contract and all verified shards. Phase 5 remains unstarted.

- 2026-09-18 — Phase 5 complete in approximately 27 minutes. Read all 15 pre-existing Markdown contracts and delivered a frozen-checkpoint FastAPI service plus Next.js three-view application. All 49 backend tests and three frontend tests pass; typecheck, lint and production build pass. Real CPU/CUDA verifiers reproduce existing validation predictions and verify boundary/whole-protein scans and CSV. Live browser verifies all three workflows, errors, progress, cancellation, provenance and CSV attachment download. Checkpoint `9eded3ed44ece10615d9ab13aed79ef0324f87f1321653e0d429aa852d3a71ae`, MegaScale `v2_230420`, split `megascale_cluster_split_v1`, seed `20260918`, selected epoch 2 and existing held-out metrics are unchanged. No training, new scientific test evaluation, ESM fine-tuning or Phase 6/external-data work occurred. Changes are local, not committed/pushed.

Add dated entries here when milestones are complete. Each ML entry should link or name the data release, split manifest, configuration, seed, artifacts, and validation/test metrics.

- 2026-09-18 — Manu reference UI integration complete. Rebuilt the complete Next.js UI using the supplied HTML layout/palette/typefaces and five navigation sections. Connected prediction, scoped scans, best-by-position map, 19-substitution dialog, selected-result context/reuse, cancellation, CSV and live saved-model evaluation/provenance. Preserved the positive-stabilizing MegaScale contract and omitted prototype-only attributions/neutral categories and invented personal details. Final checks: 49 backend tests, five frontend tests, typecheck, lint and production build pass; desktop live-browser workflows pass. Mobile styles are implemented but actual mobile viewport verification was unavailable. Backend/model artifacts unchanged; no Phase 6 work or commit/push.

## Known constraints

- Initial scope is single substitutions only.
- Predictions are research estimates, not experimental or clinical advice.
- ESM full fine-tuning is deliberately out of scope for the first working system.
- ThermoMutDB is not currently part of training or model selection; it is reserved for post-development external validation after sign/unit harmonization and overlap auditing.
