import { useState, useRef, useEffect } from "react";

interface PipelineResult {
  disease: string;
  abstracts: number;
  disease_genes: number;
  drug_candidates: number;
  kg_status: string;
  final_report: string;
}

interface LogLine {
  id: number;
  text: string;
}

function parseRanks(report: string): { rank: string; fields: Record<string, string> }[] {
  const blocks = report.split(/\n---\n/).filter(Boolean);
  return blocks
    .filter(b => /RANK\s+\d+/i.test(b))
    .map(block => {
      const lines = block.trim().split("\n");
      const fields: Record<string, string> = {};
      let currentKey = "";
      for (const line of lines) {
        const match = line.match(/^-\s+([^:]+):\s*(.*)/);
        if (match) {
          currentKey = match[1].trim();
          fields[currentKey] = match[2].trim();
        } else if (currentKey && line.trim()) {
          fields[currentKey] += " " + line.trim();
        }
      }
      const rankLine = lines.find(l => /RANK\s+\d+/i.test(l)) || "RANK ?";
      const rank = rankLine.match(/RANK\s+(\d+)/i)?.[1] ?? "?";
      return { rank, fields };
    });
}

const PHASE_PATTERNS = [
  { pattern: /PubMed Node|Fetching abstracts/i, label: "Fetching Abstracts", icon: "📚" },
  { pattern: /NER Node|PubTator3|LLM.*classify/i, label: "Extracting Genes", icon: "🧬" },
  { pattern: /DGIdb/i, label: "Querying DGIdb", icon: "💊" },
  { pattern: /KG Node|Neo4j/i, label: "Building Knowledge Graph", icon: "🗄️" },
  { pattern: /Reasoning Node/i, label: "Reasoning", icon: "🤖" },
];

export default function Home() {
  const [disease, setDisease] = useState("Pneumonia");
  const [maxResults, setMaxResults] = useState(300);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState("");
  const [phase, setPhase] = useState("");
  const [phaseIcon, setPhaseIcon] = useState("");
  const [activeRank, setActiveRank] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);
  const logCounter = useRef(0);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const runPipeline = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    setLogs([]);
    setPhase("Warming up models");
    setPhaseIcon("🔥");
    setActiveRank(0);

    try {
      const res = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ disease, max_results: maxResults }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      if (!res.body) throw new Error("No stream body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          try {
            const item = JSON.parse(line.slice(5).trim());
            if (item.type === "log") {
              const text: string = item.text;
              // Detect phase
              for (const p of PHASE_PATTERNS) {
                if (p.pattern.test(text)) {
                  setPhase(p.label);
                  setPhaseIcon(p.icon);
                  break;
                }
              }
              setLogs(prev => [...prev.slice(-300), { id: logCounter.current++, text }]);
            } else if (item.type === "result") {
              setResult(item as PipelineResult);
              setPhase("Complete");
              setPhaseIcon("✅");
            } else if (item.type === "error") {
              setError(item.text);
            }
          } catch {}
        }
      }
    } catch (err: any) {
      setError(err.message || "Failed to connect to pipeline");
    } finally {
      setLoading(false);
    }
  };

  const ranks = result ? parseRanks(result.final_report) : [];

  const scoreColor = (score: string) => {
    const n = parseInt(score);
    if (n >= 70) return "#00d68f";
    if (n >= 40) return "#f0b429";
    return "#ff4d4d";
  };

  return (
    <>
      <div className="page">
        {/* Header */}
        <header className="header">
          <div className="header-left">
            <div className="header-eyebrow">AI-Powered Research Tool</div>
            <h1 className="header-title">
              Drug <em>Repurposing</em><br />Intelligence
            </h1>
            <p className="header-desc">
              Mines PubMed literature to identify gene dysregulation patterns,
              queries DGIdb for drug interactions, and proposes repurposing
              candidates backed by evidence scores.
            </p>
          </div>
        </header>

        {/* Controls */}
        <div className="controls">
          <div className="field">
            <label>Target Disease</label>
            <input
              value={disease}
              onChange={e => setDisease(e.target.value)}
              placeholder="e.g. Alzheimer's Disease"
              disabled={loading}
            />
          </div>
          <div className="field">
            <label>Abstracts</label>
            <input
              type="number"
              value={maxResults}
              onChange={e => setMaxResults(Number(e.target.value))}
              min={50} max={500} step={50}
              disabled={loading}
            />
          </div>
          <button className="run-btn" onClick={runPipeline} disabled={loading}>
            {loading ? "Running…" : "Run Pipeline →"}
          </button>
        </div>

        {/* Phase indicator */}
        {loading && phase && (
          <div className="phase-bar">
            <div className="phase-dot" />
            <span style={{fontSize:16}}>{phaseIcon}</span>
            <span>{phase}</span>
          </div>
        )}

        {/* Live log terminal */}
        {logs.length > 0 && (
          <div className="log-wrap">
            <div className="log-header">
              <div className="log-dot" style={{background:"#ff5f57"}}/>
              <div className="log-dot" style={{background:"#febc2e"}}/>
              <div className="log-dot" style={{background:"#28c840"}}/>
              <span className="log-title">pipeline.log — live output</span>
            </div>
            <div className="log-body" ref={logRef}>
              {logs.map(l => {
                const t = l.text;
                let cls = "log-line";
                if (/✅|Kept|Ready|complete|done/i.test(t)) cls += " success";
                else if (/⚠️|warning|timeout/i.test(t)) cls += " warn";
                else if (/⏭️|skipping/i.test(t)) cls += " skip";
                else if (/❌|error|failed/i.test(t)) cls += " err";
                else if (/Node\]|Phase|🧬|📚|💊|🗄️|🤖|🔥/i.test(t)) cls += " phase";
                return <div key={l.id} className={cls}>{t}</div>;
              })}
            </div>
          </div>
        )}

        {/* Error */}
        {error && <div className="error-box">⚠ {error}</div>}

        {/* Results */}
        {result && (
          <>
            {/* Stats */}
            <div className="stats-grid" style={{marginTop:8}}>
              {[
                { label: "Abstracts", value: result.abstracts },
                { label: "Gene Pairs", value: result.disease_genes },
                { label: "Drug Candidates", value: result.drug_candidates },
                { label: "KG Status", value: result.kg_status === "done" ? "✓" : "✗" },
              ].map(s => (
                <div className="stat-card" key={s.label}>
                  <div className="stat-value">{s.value}</div>
                  <div className="stat-label">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Candidates */}
            {ranks.length > 0 && (
              <>
                <h2 className="section-title">Top Repurposing Candidates</h2>
                <p className="section-sub">for <strong style={{color:"var(--accent)"}}>{result.disease}</strong> — ranked by multi-factor evidence score</p>

                <div className="rank-tabs">
                  {ranks.map((r, i) => (
                    <button
                      key={i}
                      className={`rank-tab ${activeRank === i ? "active" : ""}`}
                      onClick={() => setActiveRank(i)}
                    >
                      #{r.rank} {r.fields["Drug Name"] || "—"}
                    </button>
                  ))}
                </div>

                {ranks[activeRank] && (() => {
                  const { fields } = ranks[activeRank];
                  const scoreRaw = fields["Pre-computed Score"] || fields["Pre-Computed Score"] || "";
                  const scoreNum = scoreRaw.match(/\d+/)?.[0] ?? "?";
                  const approved = /FDA Approved/i.test(fields["Current Approved Use"] || "");
                  const direction = (fields["Gene in " + result.disease] || "").toLowerCase();
                  const geneRole =fields[`Gene's Role in ${result.disease}`] || fields["Gene's Role"] || "—";
                  const knownRisks = fields[`Known Risks in ${result.disease}`] || fields["Known Risks"] || "—";
                  const matchText = fields["Direction Match"] || "";
                  const matchCls = /PERFECT/i.test(matchText) ? "match-perfect" :
                                   /PARTIAL/i.test(matchText) ? "match-partial" : "match-none";
                  const pmids = (fields["Supporting PMIDs"] || "").split(",").map(p => p.trim()).filter(Boolean);
                  const evidenceCount = pmids.length;
                  const sColor = scoreNum !== "?" ? scoreColor(scoreNum) : "#fff";

                  return (
                    <div className="candidate-card">
                      <div className="card-top">
                        <div>
                          <div className="drug-name">{fields["Drug Name"] || "Unknown Drug"}</div>
                          <span className={`drug-badge ${approved ? "badge-approved" : "badge-exp"}`}>
                            {approved ? "✓ FDA Approved" : "⚗ Experimental"}
                          </span>
                        </div>
                        <div className="score-circle" style={{borderColor: sColor, color: sColor}}>
                          <div className="score-num">{scoreNum}</div>
                          <div className="score-lbl">SCORE</div>
                        </div>
                      </div>

                      <div className="card-body">
                        <div className="field-block">
                          <div className="field-key">Target Gene</div>
                          <div className="field-val">
                            <span className="gene-chip">{fields["Target Gene"] || "—"}</span>
                          </div>
                        </div>
                        <div className="field-block">
                          <div className="field-key">Gene in {result.disease}</div>
                          <div className="field-val">
                            {direction.includes("up") ?
                              <span className="direction-up">▲ Upregulated</span> :
                              direction.includes("down") ?
                              <span className="direction-down">▼ Downregulated</span> :
                              <span>{fields["Gene in " + result.disease] || "—"}</span>
                            }
                          </div>
                        </div>
                        <div className="field-block">
                          <div className="field-key">Drug Action</div>
                          <div className="field-val" style={{fontFamily:"var(--font-mono)", fontSize:13}}>{fields["Drug→Gene Interaction"] || fields["Drug Action"] || "—"}</div>
                        </div>
                        <div className="field-block">
                          <div className="field-key">Direction Match</div>
                          <div className={`field-val ${matchCls}`} style={{fontSize:13}}>{matchText || "—"}</div>
                        </div>
                        <div className="field-block full">
                            <div className="field-key">Gene's Role</div>
                            <div className="field-val">{geneRole}</div>
                        </div>
                        <div className="field-block full">
                          <div className="field-key">Treatment Hypothesis</div>
                          <div className="field-val">{fields["Treatment Hypothesis"] || "—"}</div>
                        </div>
                        <div className="field-block full">
                          <div className="field-key">Known Risks</div>
                          <div className="field-val">{knownRisks}</div>
                        </div>
                        <div className="field-block full">
                          <div className="field-key">Recommended Next Step</div>
                          <div className="field-val">{fields["Recommended Next Step"] || "—"}</div>
                        </div>
                        {pmids.length > 0 && (
                          <div className="field-block full">
                            <div className="field-key">Supporting PMIDs</div>
                            <div className="pmid-list">
                              {pmids.map(p => (
                                <a key={p} href={`https://pubmed.ncbi.nlm.nih.gov/${p}`}
                                   target="_blank" rel="noreferrer"
                                   className="pmid-chip" style={{textDecoration:"none", cursor:"pointer"}}>
                                  {p}
                                </a>
                              ))}
                            </div>
                          </div>
                        )}
                        <div className="field-block">
                          <div className="field-key">Evidence Strength</div>
                          <div className="field-val">{fields["Evidence Strength"] || "—"}</div>
                        </div>
                        <div className="field-block">
                        <div className="field-key">Evidence Count</div>
                            <div className="field-val" style={{fontFamily:"var(--font-mono)"}}>
                            {evidenceCount} papers
                            </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </>
            )}

            {ranks.length === 0 && result.final_report && (
              <div style={{background:"var(--surface)", border:"1px solid var(--border)", borderRadius:12, padding:24}}>
                <div className="field-key" style={{marginBottom:12}}>Final Report</div>
                <pre style={{fontFamily:"var(--font-mono)", fontSize:12, color:"var(--text)", whiteSpace:"pre-wrap", lineHeight:1.7}}>
                  {result.final_report}
                </pre>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
