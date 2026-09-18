
# Protein Mutation Stability Prediction

This project estimates how a single amino-acid mutation may affect protein stability. Given a wild-type protein sequence and a mutation such as `V42A`, it predicts the expected change in stability as a continuous ddG value.

The project uses the public MegaScale protein-stability dataset. It validates sequences and mutation notation, keeps related proteins separated across training and evaluation splits, and applies a reproducible baseline regression model using mutation and sequence-derived features.

The goal is to help researchers prioritize single mutations for experimental testing. Predictions are computational estimates and are not a replacement for laboratory measurements.

## Frozen ESM-2 features (Phase 3)

The feature pipeline uses frozen `facebook/esm2_t12_35M_UR50D` at revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`. It combines wild-type/mutant site and residue-mean vectors with signed and absolute differences into 3,840 features. It uses the existing cluster-held-out split, extracts training and validation only, and does not train a regression head.

Phase 3 is complete: all 272,372 training and 58,363 validation mutations were extracted without downsampling and independently verified (82 shards, 349 unique wild types, 2.66 GB of local vectors). The full CPU build took about 70 minutes; all 21 tests pass. See `artifacts/phase3/extraction_report.json`, `artifacts/phase3/verification_report.json`, and `artifacts/phase3/feature_sample.csv` for aggregate provenance and vector diagnostics. There are no new ddG predictions or test-set metrics, and Phase 4 has not been started. Active dependency versions are recorded in the verification report's `verification_runtime`.

On a fresh checkout, acquire and prepare the local data once:

```powershell
./scripts/acquire_megascale.ps1
python scripts/build_phase1_dataset.py
python scripts/verify_phase1_artifacts.py
```

The pinned Phase 3 environment targets Windows, CPython 3.14 and CUDA 13.0:

```powershell
python -m venv .venv-phase3
./.venv-phase3/Scripts/python.exe -m pip install -r requirements-phase3.txt
./.venv-phase3/Scripts/python.exe scripts/build_phase3_features.py --device cuda --report artifacts/phase3/extraction_report.json
./.venv-phase3/Scripts/python.exe scripts/verify_phase3_features.py --device cuda --report artifacts/phase3/verification_report.json
python -m unittest discover -s tests -v
```

The builder validates token indexing for every batch and resumes checksum-verified shards. The verifier checks exact partition coverage, label-free files, feature layout and freshly re-encoded samples. Source records, encoder weights, environments and vector caches stay local under Git-ignored directories; only small provenance reports/sample diagnostics belong in Git. A full development cache is required before Phase 4 training.

On this machine, Windows Smart App Control blocks the CUDA wheel's `caffe2_nvrtc.dll` (WinError 4551). The CPU path uses the existing approved PyTorch 2.12.1 CPU installation:

```powershell
python -m venv --system-site-packages .venv-phase3-cpu
./.venv-phase3-cpu/Scripts/python.exe -m pip install -r requirements-phase3-common.txt
./.venv-phase3-cpu/Scripts/python.exe scripts/benchmark_phase3_encoder.py --report artifacts/phase3/cpu_benchmark.json
./.venv-phase3-cpu/Scripts/python.exe scripts/build_phase3_features.py --device cpu --cpu-threads 16 --batch-size 32 --report artifacts/phase3/extraction_report.json
./.venv-phase3-cpu/Scripts/python.exe scripts/verify_phase3_features.py --device cpu --report artifacts/phase3/verification_report.json
```

The CPU path preserves the same encoder, precision and feature schema. The GPU loader error is explicit; extraction does not silently switch devices.
