import assert from "node:assert/strict";
import test from "node:test";
import { canonicalSequence, estimateBand, orderedPredictions, positionSummaries, predictionCsv, predictionRequest, stabilityEffect, validateMutation, validatePosition } from "../lib/mutantscope.mjs";

test("canonicalizes case and outer paste whitespace, while rejecting internal symbols", () => {
  assert.equal(canonicalSequence(" acdv ").value, "ACDV");
  assert.match(canonicalSequence("   ").error, /Enter/);
  assert.match(canonicalSequence("ACD V").error, /canonical/);
  assert.match(canonicalSequence("ACD\nV").error, /canonical/);
});

test("validates a strict, sequence-matching single mutation", () => {
  assert.deepEqual(validateMutation("ACDV", " c2w ").value, { sequence: "ACDV", mutation: "C2W", position: 2, wildType: "C", mutant: "W" });
  assert.match(validateMutation("ACDV", "A2W").error, /not A/);
  assert.match(validateMutation("ACDV", "C2C").error, /differ/);
  assert.match(validateMutation("ACDV", "C2W D3A").error, /one substitution/);
});

test("validates one-based positions and produces deterministic scan exports", () => {
  assert.equal(validatePosition("ACDV", "4").value.position, 4);
  assert.match(validatePosition("ACDV", "0").error, /one-based/);
  const predictions = [{ mutation: "A1V", position: 1, wild_type: "A", mutant: "V", ddg_kcal_mol: -0.2, model_version: "v1", unit: "kcal/mol", positive_means: "stabilization" }, { mutation: "A1W", position: 1, wild_type: "A", mutant: "W", ddg_kcal_mol: 0.5, model_version: "v1", unit: "kcal/mol", positive_means: "stabilization" }];
  assert.deepEqual(orderedPredictions(predictions).map((item) => item.mutation), ["A1W", "A1V"]);
  assert.match(predictionCsv(predictions), /^mutation,position,wild_type,mutant,ddg_kcal_mol,model_version,unit,positive_means/);
  const valid = validateMutation("ACDV", "C2W").value;
  assert.deepEqual(predictionRequest(valid), { sequence: "ACDV", mutation: "C2W" });
});

test("uses the deployed positive-stabilizing convention for labels and numeric color bins", () => {
  assert.equal(stabilityEffect(0.1), "Stabilizing estimate");
  assert.equal(stabilityEffect(-0.1), "Destabilizing estimate");
  assert.equal(stabilityEffect(0), "Zero estimate");
  assert.deepEqual([0.6, 0.5, 0.001, 0, -0.001, -0.5, -0.6].map(estimateBand), ["rs", "rm", "rm", "rn", "um", "um", "us"]);
});

test("residue map selects the highest real ddG at each one-based position and retains all rows", () => {
  const rows = [
    { position: 2, wild_type: "C", mutation: "C2A", ddg_kcal_mol: -0.4 },
    { position: 1, wild_type: "A", mutation: "A1V", ddg_kcal_mol: 0.2 },
    { position: 2, wild_type: "C", mutation: "C2V", ddg_kcal_mol: -0.1 },
    { position: 1, wild_type: "A", mutation: "A1W", ddg_kcal_mol: 0.3 },
  ];
  const summary = positionSummaries(rows);
  assert.deepEqual(summary.map((site) => site.position), [1, 2]);
  assert.deepEqual(summary.map((site) => site.best.mutation), ["A1W", "C2V"]);
  assert.deepEqual(summary.map((site) => site.predictions.length), [2, 2]);
  assert.equal(summary[1].best.ddg_kcal_mol, -0.1); // Best among alternatives need not be stabilizing.
  assert.equal(rows[0].mutation, "C2A"); // Never mutate the server payload.
});
