# MutantScope Project Specification

## Purpose

MutantScope is a research-support web application that estimates the stability effect of a single amino-acid substitution. A user supplies a wild-type protein sequence and a mutation such as `V42A`; the system returns a model-estimated ddG and supports mutation scans. It prioritizes experiments and does not replace wet-lab measurement, clinical judgment, or therapeutic decision-making.

## Primary users

- Protein-engineering and molecular-biology researchers prioritizing experiments.
- Students and reviewers inspecting an end-to-end protein ML workflow.

## MVP requirements

1. Validate a canonical amino-acid sequence and one substitution.
2. Return a real, versioned ddG prediction with unit/sign convention.
3. Support single-position scans and whole-protein substitution scans.
4. Display model/data provenance and concise scientific limitations.
5. Expose reproducible held-out evaluation, not only UI output.

## Non-goals for v1

- Multi-mutations, insertions/deletions, or structure-aware inference.
- Clinical, safety, efficacy, or therapeutic claims.
- Uncalibrated confidence scores.
- Claiming generalization beyond the documented held-out protocol.

## Product behavior

Reject invalid residues, out-of-range positions, wild-type residue mismatches, unsupported multi-mutations, and unavailable models with actionable errors. Do not silently repair scientific input. Stabilizing/neutral/destabilizing labels are derived presentation aids with a documented threshold and sign convention; ddG regression remains the primary task.

## Limitations

Sequence-only representations omit experimental context, folding conditions, structures, cofactors, complexes, and assay variation. Dataset coverage may be biased toward particular proteins and assays. Predictions prioritize experiments; they do not establish causality or substitute for wet-lab measurements.

## External validation policy

ThermoMutDB may be used only as a secondary external robustness check after the MegaScale model and all development choices are frozen. It is not a training, tuning, checkpoint-selection, or threshold-selection dataset. Reports must disclose its release, filters, condition compatibility, target sign/unit conversion, and overlap with MegaScale before calling the result independent.
