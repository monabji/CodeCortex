# MutantScope — Repository Instructions

## Project

MutantScope is a research-support application that predicts the experimental stability effect of a **single amino-acid substitution**. Its primary machine-learning task is ΔΔG (ddG) regression; labels such as stabilizing, neutral, and destabilizing are derived presentation aids, not a replacement task.

## Read before significant work

- `docs/PROJECT_SPEC.md` — scope, users, product requirements, and limitations
- `docs/ML_SPEC.md` — model, feature, training, and inference rules
- `docs/DATA_SPEC.md` — dataset, validation, preprocessing, and leakage controls
- `docs/EVALUATION.md` — metrics and experimental protocol
- `docs/PHASES.md` — delivery order and definitions of done
- `ARCHITECTURE.md` — application components and data flows
- `docs/DECISIONS.md` — decisions that must not be silently reversed
- `STATUS.md` — current implementation state and next work

## Non-negotiable scientific rules

- The primary dataset is Tsuboyama/MegaScale. Treat its schema, measurement conventions, and licence as data-source facts to verify, never assumptions.
- Predict continuous ddG; do not substitute random mutation-level splitting or a classifier as the main model.
- Split by wild-type protein, and use protein-cluster-aware splits where available. Keep test proteins/cluster families completely held out from development.
- The initial encoder is frozen `facebook/esm2_t12_35M_UR50D`. Cache embeddings. Do not full-fine-tune ESM initially.
- Do not leak labels, duplicate proteins, near-identical protein families, preprocessing statistics, or test-set choices across splits.
- ThermoMutDB, if used, is external validation only after development; never use it for training, tuning, checkpoint, or threshold selection. Harmonize its sign/units and audit overlap with MegaScale.
- Never claim confidence/uncertainty unless it is computed and calibrated (for example, by an ensemble).
- Never present predictions as experimental, clinical, therapeutic, or safety conclusions.

## Working conventions

- Inspect actual data and package versions before implementing against a schema or API.
- Record data versions, split assignments, seeds, parameters, and metrics for every reported experiment.
- Update the relevant specification and `STATUS.md` when a material design decision changes.
- Prefer small, reproducible, tested steps. Do not silently alter targets, units, signs, split logic, or model architecture.

## Completion

Work is complete only after relevant tests/checks pass and the result is documented. For ML work, report the held-out protocol and metrics; do not report training metrics as final performance.
