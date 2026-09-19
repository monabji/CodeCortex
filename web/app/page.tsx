"use client";

import { FormEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { canonicalSequence, ESTIMATE_BANDS, estimateBand, orderedPredictions, positionSummaries, predictionRequest, stabilityEffect, validateMutation, validatePosition } from "../lib/mutantscope.mjs";

type Prediction = { mutation: string; position: number; wild_type: string; mutant: string; ddg_kcal_mol: number; model_version: string; unit: "kcal/mol"; positive_means: "stabilization" };
type ModelInfo = { model_version: string; unit: "kcal/mol"; positive_means: "stabilization"; encoder: { model_id: string; revision: string; layer: number }; dataset: { name: string; release: string; doi: string }; feature_schema: string; checkpoint_sha256: string; split_version: string; split_seed: number; selected_epoch: number; test_metrics: { mae: number; rmse: number; pearson: number; spearman: number; count: number; proteins: number; clusters: number }; limits: { max_residues: number }; limitations: string[]; runtime: { device: string; batch_size: number } };
type ScanJob = { job_id: string; status: "queued" | "running" | "completed" | "failed" | "cancelled"; completed: number; total: number; sequence_length: number; predictions: Prediction[]; error?: string | null };
type PositionSummary = { position: number; wildType: string; best: Prediction; predictions: Prediction[] };
type Hover = { site: PositionSummary; x: number; y: number };
const navigation = [["prediction", "prediction"], ["scan", "mutation scan"], ["evaluation", "evaluation"], ["about", "about"], ["provenance", "provenance"]] as const;
const exampleSequence = "MKTIIALSYIFCLVFADYKDDDDK";
const limitations = ["Sequence-only estimates omit structure, folding conditions, cofactors, complexes and assay variation.", "Predictions prioritize experimental measurements; they are not experimental, clinical or therapeutic conclusions.", "Accuracy varies across proteins and datasets. No calibrated uncertainty or confidence interval is available."];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...init, headers: { "content-type": "application/json", ...init?.headers }, cache: "no-store" });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const details = body?.error?.details;
    const detail = Array.isArray(details) ? details.map((item: { field?: string; message?: string }) => `${item.field ? item.field + ": " : ""}${item.message ?? "Invalid field."}`).join(" ") : undefined;
    throw new Error(detail || body?.error?.message || `Request failed (${response.status}). Check the API connection and try again.`);
  }
  return body as T;
}
function number(value: number) { return value.toFixed(3); }
function signed(value: number) { return `${value >= 0 ? "+" : "−"}${number(Math.abs(value))}`; }
function effectClass(value: number) { return value > 0 ? "stable" : value < 0 ? "unstable" : "neutral"; }
function effectLabel(value: number) { return stabilityEffect(value).replace(" estimate", ""); }
function CardHead({ title, help, children }: { title: string; help?: string; children?: ReactNode }) {
  return <div className="cardhead"><div><h2 className="title">{title}</h2>{help && <div className="help">{help}</div>}</div>{children}</div>;
}
function ErrorMessage({ message }: { message: string | null }) { return message ? <div className="error-box" role="alert">{message}</div> : null; }
function Empty({ children }: { children: ReactNode }) { return <div className="empty-state">{children}</div>; }

export default function Home() {
  const [sequence, setSequence] = useState(exampleSequence);
  const [mutation, setMutation] = useState("V14A");
  const [position, setPosition] = useState("14");
  const [scope, setScope] = useState<"whole" | "single">("whole");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [positionRows, setPositionRows] = useState<Prediction[]>([]);
  const [job, setJob] = useState<ScanJob | null>(null);
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [modalSite, setModalSite] = useState<PositionSummary | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [service, setService] = useState<"loading" | "ready" | "unavailable">("loading");
  const [serviceError, setServiceError] = useState<string | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"prediction" | "scan" | null>(null);
  const [activePage, setActivePage] = useState("prediction");
  const dialog = useRef<HTMLDialogElement>(null);

  const refresh = useCallback(async () => {
    try {
      await request<{ model_ready: boolean }>("/health");
      setInfo(await request<ModelInfo>("/model-info"));
      setService("ready"); setServiceError(null);
    } catch (error) {
      setService("unavailable"); setInfo(null);
      setServiceError(error instanceof Error ? error.message : "The model service is unavailable.");
    }
  }, []);
  useEffect(() => { const timer = window.setTimeout(() => { void refresh(); }, 0); return () => window.clearTimeout(timer); }, [refresh]);
  useEffect(() => {
    let frame = 0;
    const update = () => {
      let current: string = "prediction";
      for (const [id] of navigation) {
        const section = document.getElementById(id);
        if (section && section.getBoundingClientRect().top <= 160) current = id;
      }
      setActivePage(current); frame = 0;
    };
    const onScroll = () => { if (!frame) frame = window.requestAnimationFrame(update); };
    onScroll(); window.addEventListener("scroll", onScroll, { passive: true }); window.addEventListener("resize", onScroll);
    return () => { window.cancelAnimationFrame(frame); window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);

  const jobId = job?.job_id;
  const jobStatus = job?.status;
  useEffect(() => {
    if (!jobId || !jobStatus || !["queued", "running"].includes(jobStatus)) return;
    let disposed = false;
    let timer: number;
    const poll = async () => {
      try {
        const next = await request<ScanJob>(`/scan/jobs/${jobId}`);
        if (disposed) return;
        setJob(next);
        if (["queued", "running"].includes(next.status)) timer = window.setTimeout(() => { void poll(); }, 1000);
      } catch (error) {
        if (disposed) return;
        const message = error instanceof Error ? error.message : "Could not update scan progress.";
        setScanError(message);
        setJob((current) => current?.job_id === jobId ? { ...current, status: "failed", error: message } : current);
      }
    };
    timer = window.setTimeout(() => { void poll(); }, 500);
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [jobId, jobStatus]);
  useEffect(() => {
    const element = dialog.current;
    if (modalSite && element && !element.open) element.showModal();
    if (!modalSite && element?.open) element.close();
  }, [modalSite]);

  const scanActive = jobStatus === "queued" || jobStatus === "running";
  const inputsLocked = busy !== null || scanActive;
  const sequenceCheck = canonicalSequence(sequence);
  const rows = useMemo(() => orderedPredictions(scope === "single" ? positionRows : jobStatus === "completed" ? job?.predictions : []) as Prediction[], [scope, positionRows, jobStatus, job]);
  const sites = useMemo(() => positionSummaries(rows) as PositionSummary[], [rows]);
  const updateSequence = (value: string) => {
    setSequence(value); setPrediction(null); setPositionRows([]); setJob(null); setSelected(null); setModalSite(null); setHover(null); setPredictionError(null); setScanError(null);
  };
  const ready = () => service === "ready";
  async function runPrediction(event: FormEvent) {
    event.preventDefault(); setPrediction(null); setPredictionError(null);
    const checked = validateMutation(sequence, mutation);
    if (!checked.value) { setPredictionError(checked.error); return; }
    if (!ready()) { setPredictionError(serviceError || "The model service is not ready. Refresh the connection below."); return; }
    setBusy("prediction");
    try { setPrediction(await request<Prediction>("/predict", { method: "POST", body: JSON.stringify(predictionRequest(checked.value)) })); }
    catch (error) { setPredictionError(error instanceof Error ? error.message : "Prediction failed."); }
    finally { setBusy(null); }
  }
  async function runScan(event: FormEvent) {
    event.preventDefault(); setScanError(null); setSelected(null); setHover(null); setModalSite(null);
    const checked = scope === "single" ? validatePosition(sequence, position) : canonicalSequence(sequence);
    if (!checked.value) { setScanError(checked.error); return; }
    if (!ready()) { setScanError(serviceError || "The model service is not ready. Refresh the connection above."); return; }
    setBusy("scan");
    try {
      if (scope === "single") {
        setPositionRows([]);
        const body = validatePosition(sequence, position).value;
        const result = await request<{ predictions: Prediction[] }>("/scan/position", { method: "POST", body: JSON.stringify(body) });
        const ranked = orderedPredictions(result.predictions) as Prediction[];
        setPositionRows(ranked); setSelected(ranked[0] ?? null);
      } else {
        setJob(null);
        setJob(await request<ScanJob>("/scan/protein", { method: "POST", body: JSON.stringify({ sequence: sequenceCheck.value }) }));
      }
    } catch (error) { setScanError(error instanceof Error ? error.message : "Scan failed."); }
    finally { setBusy(null); }
  }
  async function cancelScan() {
    if (!job) return;
    try { setJob(await request<ScanJob>(`/scan/jobs/${job.job_id}`, { method: "DELETE" })); }
    catch (error) { setScanError(error instanceof Error ? error.message : "Could not cancel the scan."); }
  }
  const chooseScope = (next: "whole" | "single") => { setScope(next); setSelected(null); setHover(null); setScanError(null); };
  const openSite = (site: PositionSummary) => { setHover(null); setSelected(site.best); setModalSite(site); };
  const populateMutation = (row: Prediction) => { setMutation(row.mutation); setPrediction(null); setPredictionError(null); setModalSite(null); document.getElementById("prediction")?.scrollIntoView({ behavior: "smooth" }); };
  const sequenceHint = sequenceCheck.value ? `${sequenceCheck.value.length} residues · canonical amino acids only · maximum 1,024` : sequenceCheck.error;

  return <div className="app">
    <a className="skip-link" href="#main-content">Skip to content</a>
    <header className="topbar">
      <a className="brand" href="#prediction" aria-label="Manu Protein Stability home"><b>Manu</b><i aria-hidden="true" /><span>Protein Stability</span></a>
      <nav className="nav" aria-label="Primary navigation">{navigation.slice(0, 4).map(([id, label]) => <a key={id} href={`#${id}`} className={activePage === id ? "active" : ""} aria-current={activePage === id ? "location" : undefined}>{label}</a>)}</nav>
      <a className={`source-link ${activePage === "provenance" ? "active" : ""}`} href="#provenance" aria-current={activePage === "provenance" ? "location" : undefined}>provenance</a>
    </header>
    <main className="main" id="main-content">
      <section id="prediction" className="page">
        <h1>Protein stability prediction</h1>
        <p className="subtitle">Estimate the stability effect of one amino-acid substitution from a wild-type protein sequence.</p>
        <div className="prediction-layout">
          <form className="card input-card" onSubmit={runPrediction}>
            <CardHead title="Prediction input" help="Enter a canonical sequence and one substitution."><span className="section-kicker">Input</span></CardHead>
            <label htmlFor="predictionSequence">Wild-type sequence</label>
            <textarea id="predictionSequence" className="mono" placeholder="Paste amino-acid sequence…" value={sequence} onChange={(event) => updateSequence(event.target.value)} disabled={inputsLocked} spellCheck={false} aria-invalid={!sequenceCheck.value} aria-describedby="prediction-sequence-help" />
            <div id="prediction-sequence-help" className={sequenceCheck.value ? "field-help" : "field-error"}>{sequenceHint}</div>
            <label className="spaced-label" htmlFor="mutation">Mutation</label>
            <input id="mutation" className="mono" value={mutation} onChange={(event) => { setMutation(event.target.value); setPrediction(null); setPredictionError(null); }} disabled={inputsLocked} placeholder="e.g. V42A" spellCheck={false} />
            <div className="action-row"><span className="small">Single substitutions only.</span><button className="btn" disabled={inputsLocked} type="submit">{busy === "prediction" ? "Predicting…" : "Run prediction"}</button></div>
            <ErrorMessage message={predictionError} />
            <div className="result-footer section-actions"><span className={`service-status ${service}`}><span className="service-dot" />{service === "loading" ? "Connecting to model…" : service === "ready" ? "Model ready" : "Model unavailable"}</span>{service !== "ready" && <button className="text-button" type="button" onClick={() => void refresh()}>Refresh connection</button>}<button className="text-button" type="button" disabled={inputsLocked} onClick={() => { updateSequence(exampleSequence); setMutation("V14A"); setPosition("14"); }}>Load example</button></div>
            {serviceError && <p className="field-error">{serviceError}</p>}
          </form>
          <div className="card result-card" aria-live="polite">
            <div><CardHead title="Prediction result" help="Versioned model estimate">{prediction && <span className={`pill ${effectClass(prediction.ddg_kcal_mol)}`}>{effectLabel(prediction.ddg_kcal_mol)}</span>}</CardHead>
              <div className="section-kicker">Predicted ΔΔG</div><div className="result-number">{prediction ? signed(prediction.ddg_kcal_mol) : "—"}</div><div className="result-unit">kcal/mol</div>
              {!prediction && <p className="help">{busy === "prediction" ? "Computing with frozen ESM features and the trained regression head…" : "Run a prediction to see the actual model estimate."}</p>}
              <div className="result-rule" /><div className="result-meta"><div className="meta-box"><div className="meta-label">Mutation</div><div className="meta-value">{prediction?.mutation ?? "Not predicted"}</div></div><div className="meta-box"><div className="meta-label">Model</div><div className="meta-value">{prediction?.model_version ?? info?.model_version ?? "Unavailable"}</div></div></div>
            </div><div className="note"><b>Model estimate; not experimental measurement.</b><br />No calibrated confidence interval is available.</div>
          </div>
        </div>
        <div className="card"><CardHead title="ΔΔG interpretation" help="Sign convention used throughout Manu."><span className="section-kicker">Interpretation</span></CardHead>
          <div className="notice mono">MegaScale ddG_ML · kcal/mol · positive means stabilization</div>
          <div className="grid3 interpretation-grid"><div><span className="pill stable">Positive ΔΔG</span><div className="small">Predicted stabilizing effect</div></div><div><span className="pill neutral">Zero estimate</span><div className="small">No predicted change; not a calibrated neutral category</div></div><div><span className="pill unstable">Negative ΔΔG</span><div className="small">Predicted destabilizing effect</div></div></div>
          <div className="note">The model preserves the experimental source convention. Sequence-only representations omit experimental conditions, folding context, structures, cofactors, complexes and assay variation. Validate predictions experimentally.</div>
        </div>
      </section>

      <section id="scan" className="page">
        <h1>Mutation Scan</h1><p className="subtitle">Explore predicted stability effects across possible single-residue substitutions.</p>
        <form className="card" onSubmit={runScan}><CardHead title="Scan configuration" help="Run a position-specific or whole-protein substitution scan." />
          <label htmlFor="scanSequence">Wild-type sequence</label><textarea id="scanSequence" className="mono scan-sequence" value={sequence} onChange={(event) => updateSequence(event.target.value)} disabled={inputsLocked} spellCheck={false} aria-invalid={!sequenceCheck.value} aria-describedby="scan-sequence-help" />
          <div id="scan-sequence-help" className={sequenceCheck.value ? "field-help" : "field-error"}>{sequenceHint} · shared with prediction input</div>
          <div className="scanbar"><div className="grow"><label id="scope-label">Scope</label><div className="scope" role="group" aria-labelledby="scope-label"><button type="button" className={scope === "whole" ? "active" : ""} aria-pressed={scope === "whole"} disabled={inputsLocked} onClick={() => chooseScope("whole")}>Entire protein</button><button type="button" className={scope === "single" ? "active" : ""} aria-pressed={scope === "single"} disabled={inputsLocked} onClick={() => chooseScope("single")}>Single position</button></div></div>
            {scope === "single" && <div className="position-input"><label htmlFor="position">Position (one-based)</label><input id="position" className="mono" value={position} onChange={(event) => { setPosition(event.target.value); setPositionRows([]); setSelected(null); setHover(null); setScanError(null); }} disabled={inputsLocked} inputMode="numeric" /></div>}
            <button className="btn" disabled={inputsLocked} type="submit">{busy === "scan" ? "Scanning…" : scanActive ? "Scan in progress…" : "Run scan"}</button>
          </div><ErrorMessage message={scanError} />
          {scope === "whole" && job && <div className="job-progress" aria-live="polite"><div className="section-actions"><strong>{job.status === "completed" ? "Protein scan complete" : `Scan ${job.status}`}</strong><span className="mono small">{job.completed.toLocaleString()} / {job.total.toLocaleString()} substitutions processed</span>{scanActive && <button className="text-button" type="button" onClick={() => void cancelScan()}>Cancel scan</button>}</div><progress max={job.total} value={job.completed} />{job.error && <p className="field-error">{job.error}</p>}</div>}
          {scope === "single" && positionRows.length > 0 && <div className="note" role="status">Position scan complete · 19 unique alternatives · wild-type excluded.</div>}
        </form>
        <div className="card map-card"><CardHead title="Sequence stability map" help="Each scanned position is colored by its best predicted single-residue substitution."><span className="pill neutral">Hover a residue</span></CardHead>
          {sites.length > 0 ? <div className="mapbox"><div className="map">{sites.map((site) => <button type="button" key={site.position} className={`res ${estimateBand(site.best.ddg_kcal_mol)}`} aria-label={`Position ${site.position} ${site.wildType}; best ${site.best.mutation}, ${signed(site.best.ddg_kcal_mol)} kcal/mol; open 19 substitutions`} onClick={() => openSite(site)} onMouseEnter={(event) => setHover({ site, x: Math.max(8, Math.min(event.clientX + 16, window.innerWidth - 235)), y: Math.max(8, Math.min(event.clientY + 16, window.innerHeight - 140)) })} onMouseLeave={() => setHover(null)} onFocus={(event) => { const bounds = event.currentTarget.getBoundingClientRect(); setHover({ site, x: Math.max(8, Math.min(bounds.left, window.innerWidth - 235)), y: Math.max(8, Math.min(bounds.bottom + 8, window.innerHeight - 140)) }); }} onBlur={() => setHover(null)}><span>{site.wildType}</span><small>{site.position}</small></button>)}</div></div> : <Empty>Run a scan to populate the map with real predictions. Unscanned positions are not assigned a color or score.</Empty>}
          <div className="click-hint">Hover for the best predicted substitution · Click a residue to open its 19-substitution scan.</div>
          <div className="legend-title">Color key · best predicted ΔΔG at each position</div><div className="legend-grid">{ESTIMATE_BANDS.map((band) => <div key={band.className} className="legend-item"><span className="swatch" style={{ background: band.color }} /><div><strong>{band.label}</strong><span>{band.range}</span></div></div>)}</div>
          <div className="legend-note">Positive = stabilizing · negative = destabilizing. Color ranges are display-only numeric bins, not calibrated biological classes. “Best” means highest predicted ΔΔG among alternatives; even the best alternative may be destabilizing.</div>
        </div>
        <div className="scanbottom">
          <div className="card"><CardHead title="Best predicted mutations" help="Actual model outputs, ranked from highest to lowest ΔΔG." />
            {rows.length ? <PredictionTable predictions={rows} onSelect={setSelected} selectedMutation={selected?.mutation} /> : <Empty>Ranked substitutions appear after a successful scan.</Empty>}
            {scope === "whole" && jobStatus === "completed" && <div className="action-row"><a className="btn secondary" href={`/api/scan/jobs/${jobId}/results.csv`} download="mutantscope-protein-scan.csv">Download CSV</a></div>}
          </div>
          <div className="card"><CardHead title="Why this prediction?" help="Model context for the selected mutation." />
            {selected ? <><div className="notice"><b className="mono">{selected.mutation}</b> · <span className="mono">{signed(selected.ddg_kcal_mol)} kcal/mol</span><br />{stabilityEffect(selected.ddg_kcal_mol)}</div><div className="feature"><div className="ftop"><span>Prediction magnitude</span><span className="mono">{signed(selected.ddg_kcal_mol)} kcal/mol</span></div><div className="track"><div className={`bar ${selected.ddg_kcal_mol < 0 ? "neg" : ""}`} style={{ width: `${Math.min(100, Math.abs(selected.ddg_kcal_mol) / Math.max(...rows.map((row) => Math.abs(row.ddg_kcal_mol)), 0.001) * 100)}%` }} /></div></div>
              <dl className="model-details"><div><dt>Position</dt><dd>{selected.position} · one-based</dd></div><div><dt>Substitution</dt><dd>{selected.wild_type} → {selected.mutant}</dd></div><div><dt>Model version</dt><dd>{selected.model_version}</dd></div><div><dt>Features</dt><dd>3,840 frozen ESM site/global/difference features</dd></div></dl><div className="action-row"><button className="btn secondary" type="button" disabled={inputsLocked} onClick={() => populateMutation(selected)}>Use in prediction</button></div></> : <Empty>Select a result or residue to inspect its mutation and model context.</Empty>}
            <div className="interp"><b>Interpretation:</b> The regressor uses learned sequence representations. Feature-level attribution has not been computed, so this view does not invent side-chain contribution bars or mechanistic explanations. The magnitude bar only visualizes the prediction.</div>
          </div>
        </div><div className="card"><div className="notice"><b>Model estimate; not experimental measurement.</b> Predictions prioritize experiments and do not establish causality or replace wet-lab measurements.</div><div className="note">MegaScale convention: positive stabilizing · negative destabilizing · single substitutions only.</div></div>
      </section>

      <section id="evaluation" className="page"><h1>Evaluation</h1><p className="subtitle">Reproducible held-out evaluation for the selected model and frozen development choices.</p>
        <div className="stats"><Stat label="Evaluation split" value="Cluster-held-out" /><Stat label="Primary selection metric" value="MAE · kcal/mol" /><Stat label="External validation" value="Not performed" /></div>
        {info ? <><div className="metrics-grid"><Stat label="Test MAE · kcal/mol" value={number(info.test_metrics.mae)} /><Stat label="Test RMSE · kcal/mol" value={number(info.test_metrics.rmse)} /><Stat label="Pearson correlation" value={number(info.test_metrics.pearson)} /><Stat label="Spearman correlation" value={number(info.test_metrics.spearman)} /></div><p className="note">{info.test_metrics.count.toLocaleString()} mutations · {info.test_metrics.proteins} proteins · {info.test_metrics.clusters} clusters. These are saved Phase 4 results, not scores computed from your sequence.</p></> : <div className="card"><Empty>Verified model metrics will load when the API is available.</Empty></div>}
        <div className="card evaluation-protocol"><CardHead title="Evaluation protocol" help="Keep test data separate from training, tuning and checkpoint selection." /><div className="grid2"><div className="notice"><b>Primary evaluation</b><p>Protein/cluster-aware MegaScale test partition. The checkpoint was selected using validation MAE; normalization was fitted on training data only. The frozen model was evaluated once.</p>{info && <div className="mono small">{info.split_version}<br />Seed {info.split_seed} · selected epoch {info.selected_epoch}</div>}</div><div className="notice"><b>External robustness check</b><p>ThermoMutDB remains reserved for later external validation. It has not been used for training, tuning or reported evaluation.</p><span className="pill neutral">Phase 6 not started</span></div></div><div className="note">Held-out MAE is an aggregate error, not a confidence interval for individual predictions. Generalization beyond this protocol is not established.</div></div>
      </section>

      <section id="about" className="page"><h1>About</h1><p className="subtitle">The story behind Manu, and the project that powers it.</p>
        <div className="card"><CardHead title="Behind the project" help="A sequence-based research-support application." /><div className="notice">Manu is the protein-stability interface for CodeCortex. It combines a pretrained ESM-2 encoder used as a frozen feature extractor with a regression head trained on validated MegaScale single-substitution measurements.</div></div>
        <div className="card"><CardHead title="Why Manu" help="The motivation for this tool." /><p>Make sequence-based stability estimates quick to explore and easy to inspect, without requiring a structural model. Researchers and students can compare single substitutions, explore position scans, and prioritize variants for experimental measurement.</p><div className="note">It supports canonical sequences of up to 1,024 residues and one amino-acid substitution at a time. Multi-mutations, insertions/deletions and clinical conclusions are outside the current scope.</div></div>
        <div className="card"><CardHead title="Project links" help="Source code, scientific data and model transparency." /><div className="grid3"><a className="stat link-card" href="https://github.com/monabji/CodeCortex" target="_blank" rel="noreferrer"><div className="statlabel">GitHub</div><div className="statvalue">CodeCortex ↗</div></a><a className="stat link-card" href={info ? `https://doi.org/${info.dataset.doi}` : "https://doi.org/10.5281/zenodo.7992926"} target="_blank" rel="noreferrer"><div className="statlabel">Dataset</div><div className="statvalue">MegaScale ↗</div></a><a className="stat link-card" href="#provenance"><div className="statlabel">Transparency</div><div className="statvalue">Provenance ↓</div></a></div></div>
      </section>

      <section id="provenance" className="page"><h1>Provenance</h1><p className="subtitle">Model, data and limitations that make Manu predictions reproducible and interpretable.</p>
        <div className="card"><CardHead title="Model" help="Versioned prediction configuration." />{info ? <><div className="grid2"><Stat label="Model identifier" value={info.model_version} /><Stat label="Output" value="ΔΔG · kcal/mol" /></div><dl className="model-details"><div><dt>Encoder</dt><dd>{info.encoder.model_id}</dd></div><div><dt>Encoder revision</dt><dd>{info.encoder.revision}</dd></div><div><dt>Representation layer</dt><dd>{info.encoder.layer} · frozen float32</dd></div><div><dt>Feature schema</dt><dd>{info.feature_schema}</dd></div><div><dt>Checkpoint SHA-256</dt><dd>{info.checkpoint_sha256}</dd></div><div><dt>Inference runtime</dt><dd>ESM encoder: {info.runtime.device.toUpperCase()} · regression head: CPU</dd></div><div><dt>Sequence limit</dt><dd>{info.limits.max_residues.toLocaleString()} residues · never truncated</dd></div></dl></> : <Empty>Model provenance is unavailable until the API is ready.</Empty>}</div>
        <div className="card"><CardHead title="Data and limitations" help="Source release and documented scientific boundaries." />{info && <div className="notice"><b>{info.dataset.name} {info.dataset.release}</b><br /><a href={`https://doi.org/${info.dataset.doi}`} target="_blank" rel="noreferrer">DOI {info.dataset.doi}</a><br /><span className="mono small">{info.split_version} · seed {info.split_seed}</span></div>}<ul className="limitations">{(info?.limitations ?? limitations).map((item) => <li key={item}>{item}</li>)}</ul><div className="note">User sequences and inference caches are not persisted by this service. Scan jobs are memory-only, expire after an hour and are lost on API restart. One worker, two outstanding scans and eight retained jobs bound service resources.</div></div>
        <div className="card sign-card"><CardHead title="Sign convention" help="Preserved from the MegaScale ddG_ML target." /><div className="code">ΔΔG &gt; 0 → stabilizing estimate<br />ΔΔG &lt; 0 → destabilizing estimate<br />Unit: kcal/mol</div><div className="note">No calibrated neutral threshold, confidence score or prediction interval is deployed. Reference-design example scores and opposite-sign labels have not been copied into scientific output.</div></div>
      </section>
    </main>
    {hover && <div className="hover-card" role="tooltip" style={{ display: "block", left: hover.x, top: hover.y }}><div className="hc-title">Position {hover.site.position} · {hover.site.wildType}</div><div className="hc-row">Best: {hover.site.best.mutation}</div><div className="hc-row">ΔΔG: {signed(hover.site.best.ddg_kcal_mol)} kcal/mol</div><div className="hc-row">Click for all 19 substitutions</div></div>}
    <dialog ref={dialog} className="scan-modal" aria-labelledby="modalTitle" onCancel={() => setModalSite(null)} onClick={(event) => { if (event.target === event.currentTarget) setModalSite(null); }}>
      {modalSite && <div className="scan-panel"><div className="scan-panel-head"><div><h2 id="modalTitle">Position {modalSite.position} · {modalSite.wildType}</h2><div className="help">Wild-type residue {modalSite.wildType} · 19 alternative amino-acid substitutions</div></div><button type="button" className="close-btn" aria-label="Close position scan" onClick={() => setModalSite(null)}>×</button></div><div className="notice"><b>Best predicted substitution:</b> <span className="mono">{modalSite.best.mutation} · {signed(modalSite.best.ddg_kcal_mol)} kcal/mol</span></div><div className="section-kicker modal-kicker">19 possible substitutions</div><div className="sub-grid">{modalSite.predictions.map((row) => <button type="button" className="sub-row" key={row.mutation} onClick={() => { setSelected(row); setModalSite(null); }}><span className="mutation">{row.mutation}</span><span className="sub-ddg">{signed(row.ddg_kcal_mol)} kcal/mol</span><span className={`pill ${effectClass(row.ddg_kcal_mol)}`}>{stabilityEffect(row.ddg_kcal_mol)}</span></button>)}</div><div className="note">Actual versioned model outputs. Positive means stabilizing; negative means destabilizing. Select a substitution to inspect it in the model-context card.</div></div>}
    </dialog>
  </div>;
}

function Stat({ label, value }: { label: string; value: string }) { return <div className="stat"><div className="statlabel">{label}</div><div className="statvalue">{value}</div></div>; }
function PredictionTable({ predictions, onSelect, selectedMutation }: { predictions: Prediction[]; onSelect: (row: Prediction) => void; selectedMutation?: string }) {
  const shown = predictions.slice(0, 100);
  return <div className="table-wrap"><table className="table"><thead><tr><th>Mutation</th><th>ΔΔG · kcal/mol</th><th>Effect</th></tr></thead><tbody>{shown.map((row) => <tr key={row.mutation} className={selectedMutation === row.mutation ? "selected-row" : ""}><td><button className="text-button mono" type="button" onClick={() => onSelect(row)} aria-label={`Inspect mutation ${row.mutation}`}>{row.mutation}</button></td><td className="mono">{signed(row.ddg_kcal_mol)}</td><td><span className={`pill ${effectClass(row.ddg_kcal_mol)}`}>{stabilityEffect(row.ddg_kcal_mol)}</span></td></tr>)}</tbody></table>{predictions.length > shown.length && <div className="note">Showing the top 100 of {predictions.length.toLocaleString()} substitutions. Download CSV for every result; the residue map retains all positions.</div>}</div>;
}
