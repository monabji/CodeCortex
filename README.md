
# Protein Mutation Stability

This project estimates how a single amino-acid mutation may affect protein stability. Given a wild-type protein sequence and a mutation such as `V42A`, it predicts the expected change in stability as a continuous ddG value.

The project uses the public MegaScale protein-stability dataset. It validates sequences and mutation notation, keeps related proteins separated across training and evaluation splits, and contains code for a ridge baseline and a regression head trained on frozen ESM-2 features.

The ESM-2 encoder is a pretrained feature extractor. It is loaded with its parameters frozen; this repository does not fine-tune or train ESM-2. The trainable component is the 1,016,321-parameter MLP regression head described in `docs/PHASE4_PRIMARY_MLP.md`.

The goal is to help researchers prioritize single mutations for experimental testing. Predictions are computational estimates and are not a replacement for laboratory measurements.

## Local application (Phase 5, complete)

## Reproducibility status

The public repository contains the source code and aggregate experiment reports, but the trained MLP checkpoint, Phase 2 ridge model, processed records, and large ESM feature caches are stored outside Git. A fresh checkout therefore cannot reproduce the reported predictions without downloading the required data and rebuilding or obtaining the corresponding local artifacts.

The reported Phase 4 metrics are results from the authors' local run. They should be considered publicly reproducible only when the matching artifact bundle is available and its recorded hashes have been checked.

For the complete local reproduction workflow, see `docs/PHASE3_ESM_FEATURES.md`, `docs/PHASE4_PRIMARY_MLP.md`, and `docs/PHASE5_PRODUCT.md`.

FastAPI serves the verified Phase 4 model; Next.js now follows the supplied Manu HTML design: fixed editorial navigation, paper-grid cream/brown/blue styling, and prediction, mutation scan, evaluation, about and provenance sections. Single predictions, 19-substitution position scans, queued protein scans, progress/cancellation and full CSV downloads remain connected to the real API. The residue map shows the best actual substitution at each scanned position; clicking opens its 19-result dialog. The UI preserves kcal/mol, positive-stabilizing convention, model/data provenance, held-out results and scientific limitations. No confidence score or feature attribution is invented.

The current local preview is `http://127.0.0.1:3010`, using the CUDA encoder API at `http://127.0.0.1:8010`. Ports 3000/8000 belong to unrelated existing services and were preserved. To reproduce this preview, run these commands in separate terminals from the repository root:

```powershell
./.venv-phase3/Scripts/python.exe -m pip install -r requirements.txt
./.venv-phase3/Scripts/python.exe scripts/serve_phase5.py --device cuda --port 8010
```

```powershell
cd web
npm ci
$env:MUTANTSCOPE_API_URL = 'http://127.0.0.1:8010'
npm run dev -- --port 3010
```

For CPU, use `.venv-phase3-cpu/Scripts/python.exe` with `--device cpu`. Local encoder weights and the verified Phase 4 artifacts are required; missing/incompatible artifacts produce actionable unavailable responses. This is a localhost research prototype, not a publicly hosted service. For default-port and production-mode commands, API details, operational limits and verification evidence, see `docs/PHASE5_PRODUCT.md`.

All 54 backend tests and five frontend tests pass, as do TypeScript, ESLint and production build checks. Real CPU/CUDA checks reproduce saved validation predictions within tolerance; all three workflows, errors, cancellation, provenance and CSV download passed live browser checks. Reports are in `artifacts/phase5/`; the Manu redesign is recorded in `manu_ui_verification.json` and `docs/PHASE5_PRODUCT.md`. The optional Phase 6 frozen-ESM residual blend is packaged separately and improves held-out MAE from 0.63898 to 0.63797 kcal/mol; the original Phase 4 artifact remains preserved. See `docs/PHASE6_IMPROVEMENT.md`.

## Phase 6 frozen-ESM improvement (complete candidate)

Phase 6 experiments live under `artifacts/phase6/` and do not overwrite Phase 4/5. A label-free audit and augmented representation were tested on the CUDA GPU. The selected deployment is a validation-only 0.13 blend of the verified Phase 4 prediction and a small auxiliary residual head. On the untouched 58,333-row test partition it reports MAE 0.63797, RMSE 0.93231, Pearson 0.56772 and Spearman 0.57782, with a bootstrap MAE 95% interval of 0.63267–0.64352. Run the API with `--artifacts-dir artifacts/phase6/deployment` to use it; `scripts/serve_phase5.py` auto-selects it when present and falls back to Phase 4 otherwise. The frozen current deployment has now also been checked once against ThermoMutDB; see `artifacts/thermomutdb_external/report.json` and `docs/EVALUATION.md`. Fine-tuning ESM layers remains a separate follow-up experiment.

## Frozen ESM-2 features (Phase 3)

The feature pipeline uses frozen `facebook/esm2_t12_35M_UR50D` at revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`. It combines wild-type/mutant site and residue-mean vectors with signed and absolute differences into 3,840 features. It uses the existing cluster-held-out split, extracts training and validation only, and does not train a regression head.

Phase 3 is complete: all 272,372 training and 58,363 validation mutations were extracted without downsampling and independently verified (82 shards, 349 unique wild types, 2.66 GB of local vectors). The full CPU build took about 70 minutes. See `artifacts/phase3/extraction_report.json`, `artifacts/phase3/verification_report.json`, and `artifacts/phase3/feature_sample.csv` for aggregate provenance and vector diagnostics. Phase 3 itself produces no ddG predictions or test-set metrics. Active dependency versions are recorded in the verification report's `verification_runtime`.

On a fresh checkout, acquire and prepare the local data once:

```powershell
./scripts/acquire_megascale.ps1
python scripts/build_phase1_dataset.py
python scripts/verify_phase1_artifacts.py
```

The pinned Phase 3 environment targets Windows, CPython 3.14 and CUDA 13.0:

```powershell
python -m venv .venv-phase3
./.venv-phase3/Scripts/python.exe -m pip install -r requirements.txt
./.venv-phase3/Scripts/python.exe scripts/build_phase3_features.py --device cuda --report artifacts/phase3/extraction_report.json
./.venv-phase3/Scripts/python.exe scripts/verify_phase3_features.py --device cuda --report artifacts/phase3/verification_report.json
python -m unittest discover -s tests -v
```

The builder validates token indexing for every batch and resumes checksum-verified shards. The verifier checks exact partition coverage, label-free files, feature layout and freshly re-encoded samples. Source records, encoder weights, environments and vector caches stay local under Git-ignored directories; only small provenance reports/sample diagnostics belong in Git. A full development cache is required before Phase 4 training.

During Phase 3, Windows Smart App Control blocked the CUDA wheel's `caffe2_nvrtc.dll` (WinError 4551). A later recheck succeeded in Phases 4–5. The CPU reproduction path uses the existing approved PyTorch 2.12.1 CPU installation:

```powershell
python -m venv --system-site-packages .venv-phase3-cpu
./.venv-phase3-cpu/Scripts/python.exe -m pip install -r requirements.txt
./.venv-phase3-cpu/Scripts/python.exe scripts/benchmark_phase3_encoder.py --report artifacts/phase3/cpu_benchmark.json
./.venv-phase3-cpu/Scripts/python.exe scripts/build_phase3_features.py --device cpu --cpu-threads 16 --batch-size 32 --report artifacts/phase3/extraction_report.json
./.venv-phase3-cpu/Scripts/python.exe scripts/verify_phase3_features.py --device cpu --report artifacts/phase3/verification_report.json
```

The CPU path preserves the same encoder, precision and feature schema. The GPU loader error is explicit; extraction does not silently switch devices.

## Primary regression head (Phase 4, complete)

The head uses 3,840 frozen ESM features, hidden layers of 256/128 units, GELU and dropout 0.1, HuberLoss and AdamW. Feature means/scales are fitted on training records only. Validation MAE selects the checkpoint, with patience eight. Development vectors use disk-backed arrays to bound memory. The full fit stopped at epoch 10 and selected epoch 2: validation MAE 0.5995 kcal/mol, RMSE 0.8217, Pearson 0.6281 and Spearman 0.6072. The independent development and final artifact audits pass, along with all 26 Phase 4 tests. Phase 5 now integrates this selected model into the local application.

Both frozen models were compared on the same 58,333 test mutations from 63 proteins/39 clusters:

| Model | Test MAE (kcal/mol) | Test RMSE (kcal/mol) | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Ridge baseline | 0.7505 | 1.0052 | 0.4254 | 0.4117 |
| Frozen ESM MLP | 0.6390 | 0.9339 | 0.5661 | 0.5762 |

The MLP improves test MAE by about 14.9%; individual protein performance varies (primary protein MAE range 0.3544–2.1027). Reports, sample rows and scatter/residual plots are in `artifacts/phase4/test_metrics.json`, `verification_report.json`, `test_sample.csv`, and `test_diagnostics.png`. These are point estimates with no calibrated uncertainty.

From a verified full Phase 3 cache, run the trainer in a new experiment directory, verify development artifacts, then evaluate the frozen models once and audit the final outputs:

```powershell
./.venv-phase3-cpu/Scripts/python.exe scripts/train_phase4_primary.py
./.venv-phase3-cpu/Scripts/python.exe scripts/verify_phase4_artifacts.py --development-only
./.venv-phase3-cpu/Scripts/python.exe scripts/evaluate_phase4_test.py
./.venv-phase3-cpu/Scripts/python.exe scripts/verify_phase4_artifacts.py
```

The trainer requires the already-selected local Phase 2 ridge artifact and refuses to overwrite existing experiments. A smoke run uses `--smoke-rows 4096 --max-epochs 2 --artifacts-dir artifacts/phase4-smoke`; smoke selections cannot open test. The evaluator requires a frozen selection and independent development verification, fingerprints both selected models, and refuses repeated test evaluation. Interrupted test extraction can resume only under the same frozen contract. To inspect the completed experiment, run the final artifact verifier rather than retraining/re-evaluating.

Head fitting took 213 seconds on CPU. A subsequent CUDA recheck succeeded; test extraction preserved 32,768 CPU-produced rows and generated the remaining 25,565 on RTX 5070 in about 38 seconds. Shard devices and runtime history are recorded, and fresh CPU re-encoding verified the mixed-device cache. To extract on a working CUDA environment, install `requirements.txt` using `.venv-phase3/Scripts/python.exe`, then run that interpreter with `scripts/evaluate_phase4_test.py --device cuda` after freezing/verification. No system security settings were changed.
