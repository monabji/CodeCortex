# MutantScope ML Specification

## Task

Predict continuous experimental ddG for a single amino-acid substitution. Preserve the source unit and sign convention through preprocessing, training, evaluation, API responses, and UI. Derived stability categories are optional display outputs, not the primary model target.

## Experiment ladder

### 0. Classical baseline

Start with inference-available features such as wild-type/mutant identities, substitution descriptors, normalized position, and sequence length. Use a regularized linear and/or tree-based regressor. This validates parsing, targets, splits, and metrics before adding a protein language model.

### 1. Primary model: frozen ESM transfer learning

Use frozen `facebook/esm2_t12_35M_UR50D`; train only a compact regression head. For wild-type and mutant sequences, extract the representation at the mutated residue and a mean-pooled sequence representation, excluding special/padding tokens after verifying token alignment.

With site vectors `s_w`, `s_m` and global vectors `g_w`, `g_m`, construct:

```text
s_w, s_m, s_m - s_w, abs(s_m - s_w),
g_w, g_m, g_m - g_w, abs(g_m - g_w)
```

Signed differences represent direction; absolute differences represent magnitude. Record the exact feature layout and dimension with every checkpoint.

## Training defaults

- Loss: `HuberLoss`.
- Optimizer: AdamW.
- Validation-based early stopping.
- Training-partition-only normalization/statistics.
- Log seeds, parameters, code/environment versions, and artifacts.

## Caching and indexing

Verify one-based biological residue positions against ESM token positions with tests before large extraction runs. Cache sequence or mutation features using keys containing encoder revision, layer, tokenizer/preprocessing settings, input sequence, and feature-schema version. Reject stale/incompatible cache entries.

## Optional and stretch experiments

A masked-language-model zero-shot score, such as log probability of mutant residue minus wild-type residue at the masked site, may be added only as a validation-tested ablation. After the frozen model works, try LoRA or only the final one/two ESM layers with a lower encoder learning rate. Do not full-fine-tune ESM initially.

ThermoMutDB is not a training source. If used, it is evaluated only after development choices are frozen, with explicit sign/unit harmonization and MegaScale overlap auditing.

## Uncertainty

Return a point estimate by default. Add intervals only with a defensible ensemble or calibrated method whose coverage and width are evaluated on held-out data. Dropout alone is not a calibrated confidence guarantee.
