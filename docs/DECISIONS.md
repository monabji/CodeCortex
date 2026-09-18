# MutantScope Decision Log

Add future entries with date, context, decision, alternatives, and consequences.

## D001 — Continuous ddG regression

Predict continuous experimental ddG; stability categories are derived UI aids.

## D002 — MegaScale/Tsuboyama primary dataset

Use the verified Tsuboyama/MegaScale release as the main training source; version and verify its schema, terms, units, and signs.

## D003 — Protein/cluster-held-out splits

Do not use random mutation-row splits as the principal result. Hold out whole proteins and, where possible, whole sequence clusters.

## D004 — Baseline first

Establish a classical regression baseline before ESM to validate data, splits, targets, and metrics.

## D005 — Frozen ESM-2 primary strategy

Use frozen `facebook/esm2_t12_35M_UR50D`, cached features, and a trainable regression head first.

## D006 — Local and global mutation-aware features

Use site/global wild-type and mutant vectors plus signed and absolute differences.

## D007 — Robust training defaults

Use HuberLoss, AdamW, validation early stopping, and training-only normalization as documented defaults.

## D008 — No full fine-tuning in v1

LoRA or final-layer adaptation are stretch experiments only; full ESM fine-tuning is out of scope initially.

## D009 — Calibrated uncertainty only

Do not display confidence unless produced by a defensible calibrated method and evaluated for coverage.

## D010 — FastAPI plus Next.js

Use FastAPI for validation/inference and Next.js for the interactive web client.

## D011 — ThermoMutDB external validation only

Use ThermoMutDB, if used, only after development as a frozen, one-time external robustness check. Never train, tune, select checkpoints, or choose thresholds with it. Harmonize its documented sign convention (positive stabilizing, negative destabilizing), units, and conditions; audit overlap with MegaScale before claiming independence.

## D012 — Research-support claims

Describe outputs as model estimates that prioritize experiments, not experimental, clinical, therapeutic, or safety conclusions.

## D013 — Deterministic cluster-safe Phase 1 split

- **Date:** 2026-09-18
- **Context:** MegaScale's Dataset 2/3 table contains mutant `aa_seq` values, 130 usable source clusters after single-substitution validation, and no explicit G0/G1/G2 column.
- **Decision:** Reconstruct and validate the wild type for every row; use reconstructed wild-type sequence plus mutation as the duplicate key; merge source clusters sharing an identical reconstructed wild type; and assign the resulting groups with a seed-`20260918`, row-count-weighted 70/15/15 train/validation/test split. Record and assert empty cluster, protein, and sequence intersections.
- **Alternatives:** Random mutation-row splitting; protein-only splitting without cluster protection; or target-informed balancing.
- **Consequences:** The test partition is family-held-out and must not guide Phase 2 model selection. Split row counts are approximately, rather than exactly, proportional because whole clusters are indivisible. Quality-group filtering remains explicitly deferred until a source-backed field or list is available.

## D014 — Ridge baseline with validation-only selection

- **Date:** 2026-09-18
- **Context:** Phase 2 needs a transparent regression baseline before adding a protein language model. The environment provides NumPy but not scikit-learn.
- **Decision:** Use a NumPy ridge regressor with 45 inference-available mutation/sequence descriptors. Fit feature scaling and model weights on training records only; choose the ridge penalty from a fixed grid by validation MAE; retain the test partition untouched.
- **Alternatives:** A random-row baseline; a model using protein IDs or source fields; an unregularized fit; or proceeding directly to ESM features.
- **Consequences:** The result is reproducible and interpretable but limited: it cannot use full sequence context beyond length and position. Its validation metrics are a baseline comparison, not a final held-out test claim.

## D015 — Frozen final-layer ESM-2 cache

- **Date:** 2026-09-18
- **Context:** Phase 3 requires mutation-aware site/global representations without training the encoder or using evaluation labels.
- **Decision:** Pin `facebook/esm2_t12_35M_UR50D` to revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`, use its 480-wide final layer in float32, validate biological/token alignment for every batch, and mean-pool residues only. Store four base vectors per mutation, expanding the prescribed signed/absolute difference blocks into 3,840 features on read. Extract all training/validation records under the existing split; defer test extraction and all regression-head training to Phase 4.
- **Alternatives:** Per-mutation full residue caches; duplicated eight-block storage; reduced datasets; unversioned model downloads; training ESM; or using test data for feature choices.
- **Consequences:** The full development feature cache fits locally without downsampling. Wild types are encoded once, shards resume atomically with checksums, and label-free provenance/coverage/live-encoder checks gate completion. GPU runtime and encoder weights remain local artifacts; small aggregate extraction/verification reports can be versioned.
