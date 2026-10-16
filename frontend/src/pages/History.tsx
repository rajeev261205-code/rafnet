import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import type { HistoryPage } from "../types";
import { VerdictBadge } from "../components/Verdict";

export default function History(): React.JSX.Element {
  const [data, setData] = useState<HistoryPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    api.history(100, 0).then(setData).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <div className="soc-panel p-6 text-red-300">History unavailable: {error}</div>;
  if (!data) return <div className="soc-panel p-6" style={{ color: "#7d93b2" }}>Loading history…</div>;

  const items = data.items.filter((r) =>
    query.trim() === "" ? true : r.audio_name.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <div>
      <h1 className="mb-5 text-2xl font-bold">History</h1>
      <div className="soc-panel p-4">
        <label htmlFor="history-filter" className="mono mb-2 block text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
          Filter by file name
        </label>
        <input
          id="history-filter"
          className="soc-input max-w-sm"
          placeholder="Filter by file name…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="mono mt-2 text-xs" style={{ color: "#7d93b2" }}>
          {data.total} total analyses
        </div>
      </div>
      <div className="soc-panel mt-4 overflow-x-auto">
        {items.length === 0 ? (
          <div className="p-6" style={{ color: "#7d93b2" }}>
            No analyses yet. Upload audio or start Live Detection.
          </div>
        ) : (
          <table className="soc-table w-full">
            <thead>
              <tr>
                <th>File</th>
                <th>Prediction</th>
                <th>Verdict</th>
                <th>Confidence</th>
                <th>Risk</th>
                <th>Model</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id}>
                  <td className="mono">
                    <Link to={`/history/${r.id}`} style={{ color: "#22d3ee" }}>
                      {r.audio_name}
                    </Link>
                    <div className="text-xs" style={{ color: "#7d93b2" }}>
                      {r.id.slice(0, 8)}
                    </div>
                  </td>
                  <td className={`pred-${r.prediction} font-semibold`}>
                    {r.prediction}
                    {r.mock ? <span className="mono text-xs" style={{ color: "#fbbf24" }}> ·MOCK</span> : null}
                    {r.input_lossy ? <span className="mono text-xs" style={{ color: "#fbbf24" }}> ·LOSSY{r.input_codec ? `(${r.input_codec})` : ""}</span> : null}
                  </td>
                  <td>{r.verdict ? <VerdictBadge label={r.verdict.label} /> : <span className="mono text-xs" style={{ color: "#7d93b2" }}>—</span>}</td>
                  <td className="mono">{(r.confidence * 100).toFixed(1)}%</td>
                  <td className="mono">
                    {r.risk_score} / {r.risk_level}
                  </td>
                  <td className="mono text-xs" style={{ color: "#7d93b2" }}>
                    {r.model_version}
                  </td>
                  <td className="mono text-xs" style={{ color: "#7d93b2" }}>
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
