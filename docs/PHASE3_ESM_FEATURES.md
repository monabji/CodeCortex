# Phase 3 — Frozen ESM-2 features

## Contract

The encoder is `facebook/esm2_t12_35M_UR50D` at immutable Hugging Face revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`. Use layer 12 (last hidden state), width 480, float32 computation/storage, evaluation mode, and no trainable encoder parameters. The source and Phase 1 split summary are SHA-256 fingerprinted. Targets are neither converted nor used by extraction, and they are absent from vector caches.

Biological positions are one-based. With ESM's leading CLS token, biological position 1 maps to token index 1 and position L to token index L. Every encoded batch verifies the actual token IDs, one token per residue, CLS/EOS placement, right padding, and absence of truncation. Site selection uses the residue array at `position - 1`; global vectors average residues only, excluding CLS, EOS and padding.

## Feature layout

The exact order is `s_w, s_m, s_m-s_w, abs(s_m-s_w), g_w, g_m, g_m-g_w, abs(g_m-g_w)`. Eight 480-wide blocks give 3,840 values per mutation. Each shard stores only four base vectors (`s_w, s_m, g_w, g_m`) and expands the differences on read. This preserves the complete feature schema and halves vector storage.

Each unique wild type is encoded once and its residue vectors reused across mutations. Mutant sequences are encoded in batches on the selected device. No dataset downsampling is used for the final development cache: all 272,372 training and 58,363 validation records are required. The held-out test partition remains deferred to Phase 4 after development choices are frozen; Phase 3 calculates no ddG predictions or model-performance metrics.

## Reproduce

```powershell
python -m venv .venv-phase3
./.venv-phase3/Scripts/python.exe -m pip install -r requirements.txt
./.venv-phase3/Scripts/python.exe scripts/build_phase3_features.py --device cuda --report artifacts/phase3/extraction_report.json
./.venv-phase3/Scripts/python.exe scripts/verify_phase3_features.py --device cuda --report artifacts/phase3/verification_report.json
python -m unittest discover -s tests -v
```

CUDA wheels are pinned for the local Windows/Python 3.14/RTX 5070 environment. Existing Phase 0–2 dependencies are untouched. `--device cpu` is supported if running with a suitable CPU PyTorch install; execution will be slower. Package versions and device information are captured in the extraction report.

If a large pip transfer stalls, `python scripts/acquire_phase3_torch.py` resumes the publisher-pinned wheel with HTTP byte ranges and verifies its SHA-256. Install the verified local wheel with `./.venv-phase3/Scripts/python.exe -m pip install data/cache/phase3/runtime/torch-2.12.1+cu130-cp314-cp314-win_amd64.whl`, then install the remaining requirements. The helper is specific to the documented Windows/Python 3.14 environment.

Before a full run, a limited smoke cache can be built with `--limit-per-split 128 --cache-dir data/cache/phase3/smoke_v1`. Verify it with the same cache directory and `--allow-smoke`. A limited cache never proves full Phase 3 completion.

## Cache and verification

The default local-only cache is `data/cache/phase3/esm2_site_global_v1/`. Its manifest records the encoder contract, schema, source/split hashes, partition counts, and SHA-256 for every vector file. A row key includes encoder revision, layer, numerical/tokenizer/pooling settings, schema, input sequences and position. Wild-type entries use the same encoder contract plus sequence.

Shards have 4,096 rows by default, keeping construction and reading memory bounded. Writes are atomic; rerunning the same command resumes complete shards. Resume rejects incompatible contracts, changed keys, invalid shapes, nonfinite vectors or checksum mismatches. An OS-held lock prevents concurrent builders and releases after a crash.

Checksums remain recorded across multiple interrupted resumes. A file with no recorded checksum (for example a shard completed just before a crash) is re-encoded instead of blindly trusted. The local encoder snapshot is pinned to the remote commit and its model/config/tokenizer files are fingerprinted; every subsequent load verifies those hashes.

`iter_cached_features` yields source-row IDs and 3,840-wide batches for the future head trainer. The source-row IDs are a join index, never predictive inputs. No normalization or regression-head training occurs in Phase 3.

The independent verifier proves exact train/validation coverage, no duplicate/cross-partition rows, label-free shard fields, provenance, finite shapes, correct differences, and public-reader coverage. It also freshly encodes sampled mutations, first/last-position examples and canonical-residue probes, checks batched versus unpadded results within float32 tolerance, and asserts the encoder remains frozen.

## Implementation evidence

Phase 3 is complete (2026-09-18): all 21 unit/integration tests pass, and the full development cache passes independent verification. GPU wheel acquisition and publisher SHA-256 verification succeeded, but Windows Smart App Control blocks its unsigned `caffe2_nvrtc.dll` (WinError 4551; Code Integrity event 3077). No system security settings were changed. A 16-record real-model CPU smoke check preceded the full run but is not used as full-completion evidence.

The approved CPU encoder was benchmarked without targets on training sequences. On AMD Ryzen 9 9955HX, 16 threads and batches of 32 achieved 86.28 sequences/second on the mixed-length benchmark; this setting beat the measured 8-thread and larger-batch candidates. See `artifacts/phase3/cpu_benchmark.json`. The full extraction uses the same frozen model and float32 schema on CPU:

```powershell
python -m venv --system-site-packages .venv-phase3-cpu
./.venv-phase3-cpu/Scripts/python.exe -m pip install -r requirements.txt
./.venv-phase3-cpu/Scripts/python.exe scripts/build_phase3_features.py --device cpu --cpu-threads 16 --batch-size 32 --local-files-only --report artifacts/phase3/extraction_report.json
./.venv-phase3-cpu/Scripts/python.exe scripts/verify_phase3_features.py --device cpu --local-files-only --report artifacts/phase3/verification_report.json
```

The CPU environment inherits the existing approved PyTorch 2.12.1 CPU installation; it does not use the blocked CUDA environment. Omit `--local-files-only` on the first run if the pinned local encoder snapshot is not already downloaded.

The full run encoded all 272,372 training and 58,363 validation mutations, with 349 unique wild types and 82 shards. Build time was 4,193.79 seconds (69.90 minutes), and local vector files occupy 2,663,064,398 bytes (2.66 GB; 2.48 GiB). Reports: `artifacts/phase3/extraction_report.json` and `artifacts/phase3/verification_report.json`. Their matching cache manifest SHA-256 is `a7fa14638a94e6e979b570067df8c6fb5d9f55f2ffb858e6926af2bdcd1fb7b9`.

The full audit passed exact row/partition coverage, every recorded checksum, source/split provenance, label-free fields, finite shapes, signed/absolute differences, and the public reader. It freshly re-encoded 12 mutations including first/last-residue examples; maximum absolute cached-versus-fresh error was `2.384185791015625e-06`, and padding error was `1.430511474609375e-06`, within float32 tolerances. The encoder's 33,269,521 parameters remain frozen. `artifacts/phase3/feature_sample.csv` contains 12 vector diagnostics, not predicted stability values.

All 24 shared package pins match the active CPU environment. A final reporting fix prevents shadowed global distribution metadata from replacing active environment versions. The original extraction report/cache manifest are preserved; use `verification_report.json`'s `verification_runtime.packages` for the corrected active-package inventory. The principal extraction versions (PyTorch 2.12.1+cpu, Transformers 4.57.6, NumPy 2.4.4) were already captured directly from the imported modules. No extraction data or vectors changed during this fix.

There is no Phase 3 ddG accuracy result: no head was trained, no normalization fitted, and no held-out test feature extraction/evaluation occurred during Phase 3. Subsequent Phase 4 work is recorded in `docs/PHASE4_PRIMARY_MLP.md` and `docs/STATUS.md`. This Phase 3 document was committed with the implementation; other local specifications may remain Git-excluded. README and small aggregate artifacts also carry the reproduction/completion evidence.
