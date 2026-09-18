export const AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY";
const mutationPattern = /^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$/;

export function canonicalSequence(value) {
  if (typeof value !== "string" || !value) return { value: null, error: "Enter a protein sequence." };
  // Preserve sequence content: only leading/trailing paste whitespace and case
  // are canonicalized. Internal whitespace/FASTA headers remain invalid.
  const sequence = value.trim().toUpperCase();
  if (!sequence) return { value: null, error: "Enter a protein sequence." };
  if (sequence.length > 1024) return { value: null, error: "Sequences may contain at most 1,024 residues." };
  if (![...sequence].every((residue) => AMINO_ACIDS.includes(residue))) {
    return { value: null, error: "Use only the 20 canonical amino-acid letters; spaces and line breaks are not accepted." };
  }
  return { value: sequence, error: null };
}

export function validateMutation(sequence, value) {
  const parsed = canonicalSequence(sequence);
  if (!parsed.value) return { value: null, error: parsed.error };
  const match = typeof value === "string" ? value.trim().toUpperCase().match(mutationPattern) : null;
  if (!match) return { value: null, error: "Use one substitution such as V42A (no spaces, insertions, deletions, or multiple mutations)." };
  const [, wildType, rawPosition, mutant] = match;
  const position = Number(rawPosition);
  if (position > parsed.value.length) return { value: null, error: `Position ${position} is outside this ${parsed.value.length}-residue sequence.` };
  if (parsed.value[position - 1] !== wildType) return { value: null, error: `The sequence has ${parsed.value[position - 1]} at position ${position}, not ${wildType}.` };
  if (wildType === mutant) return { value: null, error: "Wild-type and mutant residues must differ." };
  return { value: { sequence: parsed.value, mutation: `${wildType}${position}${mutant}`, position, wildType, mutant }, error: null };
}

export function validatePosition(sequence, value) {
  const parsed = canonicalSequence(sequence);
  if (!parsed.value) return { value: null, error: parsed.error };
  if (!/^[1-9][0-9]*$/.test(String(value))) return { value: null, error: "Enter a one-based whole-number position." };
  const position = Number(value);
  if (position > parsed.value.length) return { value: null, error: `Position ${position} is outside this ${parsed.value.length}-residue sequence.` };
  return { value: { sequence: parsed.value, position }, error: null };
}

export function orderedPredictions(predictions) {
  return [...(predictions ?? [])].sort((first, second) => second.ddg_kcal_mol - first.ddg_kcal_mol || first.mutation.localeCompare(second.mutation));
}

export function predictionRequest(value) {
  return { sequence: value.sequence, mutation: value.mutation };
}

// Display-only numeric bins. These are not calibrated biological classes.
// MegaScale convention: positive stabilizing, negative destabilizing.
export const ESTIMATE_BANDS = [
  { className: "rs", label: "Positive · above +0.50", range: "> +0.50 kcal/mol", color: "#6F9BB7" },
  { className: "rm", label: "Positive · up to +0.50", range: "> 0 to +0.50 kcal/mol", color: "#A8C8D8" },
  { className: "rn", label: "Zero estimate", range: "Exactly 0 kcal/mol", color: "#F1F1EF" },
  { className: "um", label: "Negative · down to −0.50", range: "−0.50 to < 0 kcal/mol", color: "#E9D8C5" },
  { className: "us", label: "Negative · below −0.50", range: "< −0.50 kcal/mol", color: "#5E442E" },
];

export function estimateBand(value) {
  if (value > 0.5) return "rs";
  if (value > 0) return "rm";
  if (value === 0) return "rn";
  if (value >= -0.5) return "um";
  return "us";
}

export function stabilityEffect(value) {
  return value > 0 ? "Stabilizing estimate" : value < 0 ? "Destabilizing estimate" : "Zero estimate";
}

export function positionSummaries(predictions) {
  const positions = new Map();
  for (const row of orderedPredictions(predictions)) {
    if (!positions.has(row.position)) positions.set(row.position, { position: row.position, wildType: row.wild_type, best: row, predictions: [] });
    positions.get(row.position).predictions.push(row);
  }
  return [...positions.values()].sort((first, second) => first.position - second.position);
}

export function predictionCsv(predictions) {
  const rows = orderedPredictions(predictions);
  return ["mutation,position,wild_type,mutant,ddg_kcal_mol,model_version,unit,positive_means", ...rows.map((item) =>
    [item.mutation, item.position, item.wild_type, item.mutant, item.ddg_kcal_mol, item.model_version, item.unit, item.positive_means].join(","))].join("\n") + "\n";
}
