# Phase 5 — Product integration

## Scope

FastAPI and Next.js expose single prediction, position scan and whole-protein scan with the verified Phase 4 checkpoint. The pinned encoder, 3,840-feature schema, training normalizer, raw ΔΔG convention and selected epoch remain fixed. A separately versioned Phase 6 residual/blend candidate is documented in `docs/PHASE6_IMPROVEMENT.md`; this document remains the Phase 5 compatibility record.

All 15 pre-existing repository Markdown files were read before implementation. Existing Phase 4 work and unrelated local HTML/JSON files are preserved.

## Local startup

From the repository root:

```powershell
./.venv-phase3-cpu/Scripts/python.exe -m pip install -r requirements.txt
./.venv-phase3-cpu/Scripts/python.exe scripts/serve_phase5.py
```

In a second terminal:

```powershell
cd web
npm ci
npm run dev
```

Open `http://localhost:3000`. Next.js forwards `/api/*` to `http://127.0.0.1:8000`. Set `MUTANTSCOPE_API_URL` before launching Next.js to override the API address. The API binds to localhost by default; its interactive schema is at `http://127.0.0.1:8000/docs`.

The verified preview uses ports 8010/3010 because unrelated services already occupy 8000/3000. To reproduce that arrangement, use these commands in separate terminals:

```powershell
./.venv-phase3/Scripts/python.exe scripts/serve_phase5.py --device cuda --port 8010
```

```powershell
cd web
$env:MUTANTSCOPE_API_URL = 'http://127.0.0.1:8010'
npm run dev -- --port 3010
```

Open `http://127.0.0.1:3010`. For a production-mode local launch, set the API address before `npm run build` (rewrites are compiled into the build), then use `npm run start -- --port 3010`. The delivered preview is the development server; the production build is separately verified. This is a localhost research prototype, not a public hosted deployment.

To use the local CUDA environment, install the same requirements using `.venv-phase3/Scripts/python.exe`, then launch that interpreter with `scripts/serve_phase5.py --device cuda`. The encoder computes float32 on the selected device; the small regression head stays on CPU. `--device auto` selects CUDA when available. Batch size, CPU threads, port and artifact directory are configurable CLI arguments. Explicit CUDA selection fails visibly when unavailable.

The checkpoint and encoder weights are Git-ignored. A fresh checkout must reproduce Phases 3–4 or supply the exact local experiment and pinned encoder snapshot. Startup verifies checkpoint/metric hashes, frozen selection, training-only normalization, encoder/schema provenance and protected Phase 4 source hashes. Missing or incompatible artifacts produce an unavailable health response and actionable 503 errors.

## API

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Model readiness: 200 ready, 503 unavailable. |
| `GET /model-info` | Model/checkpoint, encoder/data/split versions, convention, held-out metrics, limits and limitations. |
| `POST /predict` | `{sequence, mutation}` → one real ΔΔG prediction. |
| `POST /scan/position` | `{sequence, position}` → exactly 19 non-wild-type substitutions. |
| `POST /scan/protein` | `{sequence}` → 202 scan job with `19 × sequence_length` mutations. |
| `GET /scan/jobs/{job_id}` | Status/progress; all predictions returned only on successful completion. |
| `DELETE /scan/jobs/{job_id}` | Cancel a queued/running scan between inference batches. |
| `GET /scan/jobs/{job_id}/results.csv` | Download all completed results, sorted by descending ΔΔG; includes model/unit/sign metadata. Returns 409 before successful completion. |

Errors have `{error: {code, message, details?}}`: 422 invalid input, 404 missing/expired job, 429 full scan queue, and 503 unavailable model or inference failure. Schemas reject unknown fields and coerced integer positions.

Case and outer whitespace are canonicalized. Internal whitespace, FASTA headers, ambiguous/noncanonical residues, insertions/deletions, multiple substitutions and no-op mutations are rejected. Biological positions are one-based and must match the supplied wild-type residue. Maximum sequence length is 1,024; silent truncation is forbidden.

## Inference and operations

One encoder/checkpoint pair serves each API process, with serialized model access. Wild-type vectors have a 32 MiB LRU cache; predictions have an 8,192-entry LRU cache keyed by encoder/input/checkpoint identity. Caches stay in process memory; user sequences are not persisted. Mutants are batched, and the attention budget lowers batch size for longer sequences. Feature order and saved training-only normalization exactly match Phase 4. Both encoder and head remain frozen in evaluation mode.

Protein scans run one at a time, allow two outstanding tasks, and retain at most eight jobs. Cancelled queued tasks continue counting toward capacity until the worker consumes them; repeated cancellation cannot create an unbounded executor queue. Terminal jobs expire after an hour and older terminal jobs may be evicted at capacity. Server restart loses jobs. Cancellation finishes the active batch before stopping. Use one API worker; shared durable job storage would be required for multiple workers or a distributed deployment.

The client shows numerical ΔΔG in kcal/mol (positive stabilizing), model/data provenance, held-out results and limitations. There is no confidence score, calibrated interval or arbitrary classification threshold. Predictions support experimental prioritization, not experimental/clinical/therapeutic conclusions.

## Verification

```powershell
./.venv-phase3-cpu/Scripts/python.exe -m unittest discover -s tests -v
./.venv-phase3-cpu/Scripts/python.exe scripts/verify_phase5.py
cd web
npm test
npm run typecheck
npm run lint
npm run build
```

All 49 backend tests pass, including 23 Phase 5 tests; all five frontend tests, typecheck, lint and production build pass. Backend coverage includes strict validation, exact features/normalization, bounded caches, position/protein coverage, progress/cancellation/capacity/expiry, schema/error/readiness behavior, CSV export and artifact gates.

The real verifier reproduces five saved validation predictions within `2e-5` kcal/mol, verifies first/last-residue position scans, all 57 substitutions for a three-residue sequence and complete CSV contents. CPU maximum error is `1.7136335372924805e-7`; CUDA maximum error is `1.0013580322265625e-5`. Evidence: `artifacts/phase5/inference_verification.json` and `cuda_inference_verification.json`. CUDA reproduction command:

```powershell
./.venv-phase3/Scripts/python.exe scripts/verify_phase5.py --device cuda --report artifacts/phase5/cuda_inference_verification.json
```

Live browser checks passed: actual single prediction, residue mismatch rejection with stale result cleared, 19 unique position substitutions, complete 57-result scan with ranked table/position heatmap, provenance/convention/limitations, cancellation of a 9,500-variant job after 132 processed variants with inputs unlocked, and a successful full CSV attachment download. A real 1,024-residue CUDA prediction succeeded in approximately 1.05 seconds on the local RTX 5070; this is one observed request, not a general latency guarantee. The UI shows the top 100 ranked variants for long scans; CSV contains every result. Consolidated evidence is `artifacts/phase5/product_verification.json`.

These are deployment compatibility checks, not training/tuning or another scientific test evaluation. The Phase 4 checkpoint and held-out results remain unchanged. Phase 5 is complete locally; Phase 6 has its own frozen candidate artifacts and verification report. Changes are uncommitted.

## Manu reference UI revision

The supplied `Manu_protein_stability (5).html` is the visual reference, preserved unchanged. Its final editorial topbar, cream grid background, brown/blue palette, Source Serif 4/Source Sans 3/IBM Plex Mono typefaces, card treatments and five vertically stacked anchor-linked sections are implemented in Next.js. Google Fonts load at runtime with local fallback families; builds do not depend on downloading fonts. Header/card/nav rules adapt to narrow widths and support focus indicators, skip navigation, keyboard-operable residue buttons, a native modal dialog, Escape dismissal and reduced-motion preferences.

Connection mapping:

| Reference area | Real connected behavior |
| --- | --- |
| Prediction input/result | Strict validation and `POST /predict`; no fabricated initial result; input changes clear stale predictions. |
| Entire protein/single position scope | Existing job and position endpoints; whole scans expose progress/cancel/full CSV. |
| Sequence map and residue dialog | Groups returned rows by one-based position; highest ΔΔG is best under our convention. Click opens the actual 19 returned substitutions, never generated example values. |
| Ranked mutations/context | Select rows or dialog substitutions to inspect exact mutation/model metadata; “Use in prediction” fills the real form. The magnitude bar visualizes a score, not feature attribution. |
| Evaluation/provenance | Live `/model-info` supplies saved held-out metrics, counts, split/seed, encoder revision, checkpoint, convention and runtime. |
| About | Factual project overview, existing repository/data links; no invented biography/email/location. |

The prototype uses the opposite sign interpretation, illustrative contribution bars and unperformed external-validation labels. These are not scientific output: positive remains stabilizing, negative destabilizing, no neutral threshold is asserted, attributions are explicitly unavailable and ThermoMutDB is marked not performed. Five map colors use numeric display bins: above +0.50, positive through +0.50, exactly zero, negative down to −0.50, and below −0.50 kcal/mol. They are not calibrated biological classes. Best alternatives can still have negative scores. The supplied default `V42A` is out of bounds for its 24-residue sequence, so the connected example uses the valid `V14A` instead.

The redesign passed five frontend tests, typecheck, lint, production build and all 49 backend tests. Desktop browser checks verified real A1V and example V14A predictions; mismatch rejection/stale-result clearing; position 20's 19 alternatives; modal count/selection/re-prediction; complete ACD 57-result scan, three best-by-position map entries and CSV attachment download; cancellation of a 500-residue job after 376/9,500 variants with inputs unlocked and no partial map. Evaluation, About and provenance anchors were inspected. A transient development error occurred while the stylesheet was being replaced; final production build and fresh-load checks pass. Responsive rules are implemented, but mobile visual verification is not claimed: the preview browser accepted a resize request without changing its actual 1,280-pixel viewport. Evidence: `artifacts/phase5/manu_ui_verification.json`; the original Phase 5 deployment reports remain historical evidence.
