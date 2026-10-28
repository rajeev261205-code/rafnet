import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { ThreatRecord, VerdictLabel } from "../types";
import { VerdictBadge } from "../components/Verdict";

export default function Threats(): React.JSX.Element {
  const [items, setItems] = useState<ThreatRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = (): void => {
    api.threats().then((r) => setItems(r.items)).catch((e: Error) => setError(e.message));
  };

  useEffect(load, []);

  const triage = (id: string, status: string): void => {
    if (busyId !== null) return;
    setBusyId(id);
    setError(null);
    api.setThreatStatus(id, status).then(load).catch((e: Error) => setError(e.message)).finally(() => setBusyId(null));
  };

  if (error) return <div className="soc-panel p-6 text-red-300">Threats unavailable: {error}</div>;
  if (!items) return <div className="soc-panel p-6" style={{ color: "#7d93b2" }}>Loading threats…</div>;

  return (
    <div>
      <h1 className="mb-5 text-2xl font-bold">Threats</h1>
      <div className="soc-panel overflow-x-auto">
        {items.length === 0 ? (
          <div className="p-6" style={{ color: "#7d93b2" }}>
            Not available — no HIGH/CRITICAL analyses yet.
          </div>
        ) : (
          <table className="soc-table w-full">
            <thead>
              <tr>
                <th>Threat</th>
                <th>Severity</th>
                <th>Prediction</th>
                <th>Verdict</th>
                <th>Status</th>
                <th>Triage</th>
              </tr>
            </thead>
            <tbody>
              {items.map((t) => (
                <tr key={t.id}>
                  <td className="mono">
                    {t.id}
                    <div className="text-xs" style={{ color: "#7d93b2" }}>
                      analysis {t.analysis_id.slice(0, 8)} · {t.threat_type}
                    </div>
                  </td>
                  <td className="font-semibold" style={{ color: "#f87171" }}>
                    {t.severity}
                  </td>
                  <td className="mono">{(t.confidence * 100).toFixed(1)}%</td>
                  <td>{t.verdict_label ? <VerdictBadge label={t.verdict_label as VerdictLabel} /> : <span className="mono" style={{ color: "#7d93b2" }}>—</span>}</td>
                  <td className="mono">{t.status}</td>
                  <td>
                    <div className="flex gap-2">
                      {(["acknowledged", "closed"] as const).map((s) => (
                        <button
                          key={s}
                          className="soc-btn"
                          style={{ padding: "0.25rem 0.6rem", fontSize: "0.75rem" }}
                          disabled={busyId === t.id}
                          onClick={() => triage(t.id, s)}
                        >
                          {busyId === t.id ? "…" : s}
                        </button>
                      ))}
                    </div>
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
