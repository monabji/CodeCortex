# MutantScope Decision Log

Add future entries with date, context, decision, alternatives, and consequences.

## D018 — Manu design with unchanged scientific output

- **Date:** 2026-09-18
- **Context:** The requested complete UI redesign references the supplied Manu HTML, whose scores, opposite-sign labels, attributions and personal details are illustrative.
- **Decision:** Implement its visual system and five-section navigation in Next.js while using real Phase 5 API responses. Select highest ΔΔG as the best substitution, provide actual residue dialogs, and keep the positive-stabilizing MegaScale convention. Use display-only numeric color bins without claiming calibrated neutral categories. Replace unavailable attribution with explicit model context; show actual saved metrics/provenance, not prototype values.
- **Alternatives:** Embed the disconnected static prototype; copy its fabricated predictions or silently invert scientific outputs.
- **Consequences:** The reference appearance is retained with truthful content. User HTML and frozen backend/model artifacts remain unchanged; no Phase 6 work is needed. About links are factual rather than invented personal contact details. Mobile CSS is implemented; mobile browser visual testing is explicitly unverified due to the preview resize limitation.

## D017 — Frozen Phase 4 product inference and bounded scan jobs

- **Date:** 2026-09-18
- **Context:** Phase 5 integrates the verified model into FastAPI and Next.js; whole-protein inference requires 19 substitutions per residue.
- **Decision:** Reuse the exact frozen epoch-2 checkpoint, normalizer and encoder schema, gate startup on artifact provenance/hashes, and expose numerical point estimates. Use batched inference, bounded process-memory caches, a single scan worker, progress/cancellation, queue limits and expiring results. Next.js forwards API calls from the same origin.
- **Alternatives:** Retraining during integration, unversioned prediction caches, synchronous unbounded protein scans, or displaying uncalibrated confidence.
- **Consequences:** The local product predicts with the evaluated Phase 4 model. Jobs/results are lost on restart and require one API worker. Phase 6 research variants remain deferred.

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

## D017 — Frozen local product deployment and bounded scans

- **Date:** 2026-09-18
- **Context:** Phase 5 must turn the verified Phase 4 model into a usable application without new research/model selection.
- **Decision:** Gate startup on the immutable checkpoint, selection, metrics and protected-source hashes. Serve one frozen encoder/head pair per localhost API process; use memory-bounded inference caches and one worker with two outstanding protein-scan tasks, eight retained jobs and one-hour terminal expiry. Expose progress/cancellation and complete CSV attachments. Next.js proxies same-origin API requests and shows numerical estimates, provenance and limitations without arbitrary neutral thresholds or uncertainty.
- **Alternatives:** Re-training during integration; synchronous unbounded protein scans; persistent user-sequence caches; distributed workers without durable shared state; or uncalibrated confidence displays.
- **Consequences:** Jobs/caches are process-local and lost on restart; cancellation waits for the active batch. Production/public multi-user hosting would require additional security/operations work. Phase 4 scientific results remain unchanged; Phase 6 remains unstarted. Live preview uses 8010/3010 to preserve existing services.

## D016 — Compact MLP with train-only scaling and frozen test gate

- **Date:** 2026-09-18
- **Context:** Phase 4 consumes 3,840 frozen ESM features; available RAM makes a fully expanded in-memory development matrix undesirable.
- **Decision:** Start with a 256/128-hidden-unit GELU/dropout 0.1 head, HuberLoss delta 1.0 and AdamW. Fit feature statistics on training only with streaming Welford updates; store base vectors in label-free memmaps and assemble batches on demand. Select the lowest validation MAE, with patience eight and minimum patience improvement 0.0001. Keep the original training split and raw target convention. Freeze checkpoint/configuration/baseline fingerprints before the single test comparison.
- **Alternatives:** ESM fine-tuning; random mutation splits; joint train/validation scaling; loading all expanded vectors into RAM; or tuning on held-out results.
- **Consequences:** The head trains independently of the encoder and preserves the Phase 3 contract. Epoch history and independent verification prove selection/scaling discipline; epoch 2 was selected, with stopping at epoch 10. Smoke runs remain separate from the full experiment. Both frozen models were evaluated on the same held-out manifest and independently reproduced. A successful CUDA recheck allowed device migration for remaining label-free extraction, with per-shard/runtime provenance and live cross-device verification; model choices remained fixed. Full evidence is in `artifacts/phase4/verification_report.json`.
