# MutantScope Architecture

## System purpose

MutantScope is a research-support web application for estimating the stability effect of one protein substitution. It accepts a wild-type amino-acid sequence and mutation (for example, `V42A`), validates the request, produces a model estimate of ddG, and supports mutation scans. It does not replace experimental validation.

## Component map

```text
Next.js web client
  ├─ Single-mutation form and result view
  ├─ Position scan / whole-protein scan views
  └─ Method, limitation, and model-version display
             │ HTTPS / JSON
             ▼
FastAPI service
  ├─ Input and mutation validation
  ├─ Prediction and scan endpoints
  ├─ Model/feature cache lookup
  └─ Provenance and error responses
             │
             ▼
Inference package
  ├─ Frozen ESM-2 encoder (35M)
  ├─ Feature builder
  ├─ MLP regressor
  └─ Optional calibrated ensemble
             │
             ▼
Versioned artifacts
  ├─ checkpoint + configuration
  ├─ feature cache
  ├─ split manifest / metrics / plots
  └─ data and model provenance
```

## Online prediction flow

1. The client submits an uppercase canonical sequence and one substitution.
2. FastAPI checks amino-acid alphabet, one-based position bounds, and that the supplied wild-type residue matches the sequence.
3. The service creates the mutant sequence and obtains cached or newly computed frozen-ESM representations.
4. The feature builder creates the exact feature layout recorded with the checkpoint.
5. The regressor returns ddG in the unit/sign convention recorded with that model. The API returns the numeric prediction, model version, convention, and limitations.
6. If a validated uncertainty model is deployed, return its calibrated interval; otherwise omit uncertainty instead of inventing it.

## Scan flow

For a selected position, evaluate every permissible non-wild-type amino acid (up to 19). A protein-wide scan evaluates the same set at every position, batching requests to control memory and latency. The user interface must identify these as model estimates and surface missing/invalid positions explicitly.

## Offline training flow

```text
Raw MegaScale release
  → immutable source snapshot + provenance
  → schema/unit/sequence validation and filtering
  → protein/cluster-aware split manifest
  → baseline tabular features + classical model
  → frozen ESM embeddings and cache
  → mutation-aware feature matrices
  → MLP fitting / validation / early stopping
  → held-out test evaluation once
  → versioned deployment artifact
```

The feature cache is an offline-training optimization. Cache entries must be keyed by encoder version, sequence, preprocessing/tokenization settings, and representation layer. Inference caching must not mix incompatible model artifacts.

## Service boundary

Suggested endpoints:

- `GET /health` — service/model readiness.
- `GET /model-info` — model version, ddG convention, training-data version, and limitations.
- `POST /predict` — one sequence plus one mutation.
- `POST /scan/position` — one sequence plus one position.
- `POST /scan/protein` — one sequence, optionally with batching/job semantics for long sequences.

The API must reject multi-mutations in the MVP rather than silently approximating them.

## Operations and reproducibility

Separate raw data, derived data, feature caches, checkpoints, and user-provided sequences. Never commit restricted data or secrets. Each deployed checkpoint must point to a configuration, source dataset version, split manifest, seed, metrics, and feature schema sufficient to reproduce its reported evaluation.
