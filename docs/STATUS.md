# MutantScope Status

## Current state

Phases 0, 1, 2, and 3 are complete. The repository-wide implementation-contract audit and MegaScale source audit are complete. The source is pinned to Zenodo record 7992926 (`v2_230420`, CC BY 4.0, `Processed_K50_dG_datasets.zip`, publisher MD5 `f7e8c553efee734cf161ee6f2b0a09cf`); the local raw archive is checksum-verified and Git-ignored.

Validated preprocessing, cluster-safe splitting, and a validation-only classical baseline are complete. Phase 3's full frozen ESM development cache has been extracted and independently verified: 272,372 training and 58,363 validation mutations, 349 unique wild types, 82 shards, and 3,840 features per mutation. No regression-head training, API, frontend, or held-out test evaluation has been implemented in Phase 3. Phase 4 has not been started.

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
- `docs/` is currently excluded through local `.git/info/exclude`; these status/spec updates are local. Reproduction instructions and aggregate reports are also available outside that excluded directory. Phase 3 changes have not been committed or pushed.

## Next implementation sequence

1. Phase 4, only when requested: train the cached-feature MLP with training-only normalization and validation-based early stopping, then evaluate the untouched test partition once after choices are frozen. Do not alter the Phase 1 split manifest.
2. Phase 5, only after Phase 4: package the selected model behind FastAPI and build the Next.js mutation and scan views.

## Evidence log

- 2026-09-18 — Phase 0 complete. Reviewed all repository Markdown specifications; acquired MegaScale Zenodo record 7992926 `v2_230420`; verified its 1,013,678,473-byte archive against MD5 `f7e8c553efee734cf161ee6f2b0a09cf`; and generated `data/audit/megascale_v2_230420.audit.json`. The Dataset 2/3 table contains 776,298 rows, 479 wild-type proteins, 133 clusters, 607,839 numeric `ddG_ML` values, and 389,068 initial exact-single-substitution candidates. Phase 1 must filter literal `-` targets, validate sequence/mutation alignment, deduplicate without crossing split boundaries, and use group-safe manifests; it must not infer G0/G1/G2 from a nonexistent CSV column.
- 2026-09-18 — Phase 1 complete. Built and independently verified a leakage-safe MegaScale pipeline. It retained 389,068 reconstructed, round-trip-validated single substitutions and excluded 387,230 rows with a row-level reason manifest. The seed-`20260918` cluster-safe split has 272,372 training, 58,363 validation, and 58,333 held-out test records; cluster, protein, and reconstructed wild-type sequence intersections are all empty. No Phase 2 model training or test evaluation was performed.
- 2026-09-18 — Phase 2 complete. Trained `phase2_ridge_baseline_v1` on the Phase 1 training split using only 45 inference-available tabular features. Selected alpha `0.0001` by validation MAE on 58,363 validation records: MAE 0.7061 kcal/mol, RMSE 0.9108 kcal/mol, Pearson 0.4688, Spearman 0.4242. The mean-prediction control MAE is 0.7984 kcal/mol. Independent verification confirmed exact validation-prediction coverage and no test-set evaluation. No Phase 3 work was started.

- 2026-09-18 — Phase 3 complete. Extracted and independently verified frozen `facebook/esm2_t12_35M_UR50D` revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`, layer 12, float32 site/global vectors and eight-block 3,840-wide features. All 330,735 MegaScale `v2_230420` development records under `megascale_cluster_split_v1`, seed `20260918`, are covered in 82 shards with 349 wild-type entries. Reports are `artifacts/phase3/extraction_report.json` and `artifacts/phase3/verification_report.json`; 21 tests pass. Full CPU build took 69.90 minutes and cached 2.66 GB. No labels or test records were used for feature construction; Phase 3 has no prediction metrics and Phase 4 remains unstarted.

Add dated entries here when milestones are complete. Each ML entry should link or name the data release, split manifest, configuration, seed, artifacts, and validation/test metrics.

## Known constraints

- Initial scope is single substitutions only.
- Predictions are research estimates, not experimental or clinical advice.
- ESM full fine-tuning is deliberately out of scope for the first working system.
- ThermoMutDB is not currently part of training or model selection; it is reserved for post-development external validation after sign/unit harmonization and overlap auditing.
