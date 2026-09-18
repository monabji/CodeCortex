# MutantScope Delivery Phases

## Phase 0 — Foundation and data audit

Acquire and inspect the permitted MegaScale release. Record version, terms, fields, target convention, quality controls, and reproducible environment.

## Phase 1 — Leakage-safe data pipeline

Implement sequence/mutation validation, mutant construction, deduplication, and protein/cluster-aware split manifests with automated intersection checks.

## Phase 2 — Classical baseline

Train a simple regression baseline using inference-available features and document reproducible validation metrics.

## Phase 3 — Frozen ESM features

Load frozen `facebook/esm2_t12_35M_UR50D`, verify token indexing, extract site/global representations, build signed/absolute differences, and create a versioned cache.

## Phase 4 — Primary MLP

Train the cached-feature head with HuberLoss, AdamW, validation early stopping, and training-only normalization. Evaluate baseline and primary model once on the frozen test manifest.

## Phase 5 — Product integration

Ship FastAPI validation/inference endpoints and Next.js views for single prediction, position scan, and protein scan. Show provenance, convention, and limitations.

## Phase 6 — Stretch work

Evaluate the optional zero-shot score and one LoRA/final-layer variant. Then, only after all development choices are frozen, evaluate filtered ThermoMutDB once as external validation. Harmonize signs/units and conditions, audit overlap, and never tune on the result. Full ESM fine-tuning is not a v1 phase.

## Working loop

For each phase, read the relevant specs, make a small tested change, record evidence in `STATUS.md`, and update `DECISIONS.md` when the scientific contract changes.
