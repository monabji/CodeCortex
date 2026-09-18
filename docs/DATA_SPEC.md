# MutantScope Data Specification

## Primary source

The primary training source is the Tsuboyama/MegaScale single-mutation stability release. Before use, record the exact version/download, licence, citation, access date, schema, quality filters, target unit, sign convention, and mutation notation. Do not assume undocumented fields or conventions.

## Logical record

Each retained example should be traceable to source record, protein/assay identifiers, wild-type sequence, one-based position, wild-type and mutant residues, measured ddG, unit/sign convention, quality fields, and provenance where available.

## Validation and preprocessing

1. Canonicalize case and validate the amino-acid alphabet.
2. Require exactly one substitution and confirm `sequence[position - 1]` matches the stated wild type.
3. Construct and re-check the mutant sequence.
4. Validate numeric targets, units, signs, missingness, and quality controls.
5. Apply a documented duplicate/conflict policy without crossing split boundaries.
6. Version the processed data and row-level exclusion manifest.

## Leakage-resistant splits

Random mutation-row splits are forbidden as the primary protocol. Assign whole wild-type proteins to train/validation/test, and assign whole sequence clusters/families together where available. Version the split manifest with group IDs, method, seed, counts, and intersection checks. Select architecture and hyperparameters on train/validation only; open the test partition once after selection.

## ThermoMutDB external validation

ThermoMutDB is reserved for **post-development external validation**. Never use it for training, feature selection, hyperparameter tuning, checkpoint selection, or threshold selection. Before use, record its exact release/version, access date, licence/terms, citation, schema, and extraction filters.

Harmonize its target with MegaScale explicitly: ThermoMutDB documents positive ΔΔG as stabilizing and negative ΔΔG as destabilizing, so verify whether sign conversion is needed for MutantScope’s declared convention. Confirm compatible units and restrict to comparable single substitutions and documented experimental-condition ranges.

Build an overlap report against MegaScale train, validation, and test manifests using protein identity/sequence, mutation, and source identifiers. Remove or separately flag overlaps before calling the subset independent. Report differences in assay conditions, coverage, protocols, and target distributions. ThermoMutDB is a robustness check, not a replacement for MegaScale splits.
