import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { CallSession, VerdictLabel } from "../types";
import { VerdictBadge } from "../components/Verdict";

export default function Sessions(): React.JSX.Element {
  const [items, setItems] = useState<CallSession[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.calls().then((r) => setItems(r.items)).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <div className="soc-panel p-6 text-red-300">Sessions unavailable: {error}</div>;
  if (!items) return <div className="soc-panel p-6" style={{ color: "#7d93b2" }}>Loading sessions…</div>;

  return (
    <div>
      <h1 className="mb-5 text-2xl font-bold">Call Sessions</h1>
      <div className="soc-panel overflow-x-auto">
        {items.length === 0 ? (
          <div className="p-6" style={{ color: "#7d93b2" }}>
            Not available — no call sessions yet. Each analysis derives one session.
          </div>
        ) : (
          <table className="soc-table w-full">
            <thead>
              <tr>
                <th>Session</th>
                <th>Prediction</th>
                <th>Verdict</th>
                <th>Duration</th>
                <th>Risk</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {items.map((c) => (
                <tr key={c.id}>
                  <td className="mono">
                    {c.id}
                    <div className="text-xs" style={{ color: "#7d93b2" }}>
                      {c.caller} → {c.receiver}
                    </div>
                  </td>
                  <td className={`pred-${c.prediction} font-semibold`}>{c.prediction}</td>
                  <td>{c.verdict_label ? <VerdictBadge label={c.verdict_label as VerdictLabel} /> : <span className="mono" style={{ color: "#7d93b2" }}>—</span>}</td>
                  <td className="mono">{c.duration_seconds.toFixed(2)} s</td>
                  <td className="mono">
                    {c.risk_score} / {c.risk_level}
                  </td>
                  <td className="mono">{c.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
