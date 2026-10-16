// Network Monitor: the backend collects no network telemetry, so this page
// states that plainly. It also shows measured per-analysis inference latency
// (processing_time_ms, a real backend measurement) explicitly labeled as
// inference latency — not network telemetry.
import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { HistoryPage } from "../types";
import { DataTable, EmptyState, Panel, Skeleton } from "../components/ui";

export default function Network(): React.JSX.Element {
  const [data, setData] = useState<HistoryPage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .history(20, 0)
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error)
    return (
      <Panel title="Network Monitor">
        <div className="text-sm text-red-300">Network data unavailable: {error}</div>
      </Panel>
    );
  if (!data)
    return (
      <Panel title="Network Monitor">
        <Skeleton lines={5} />
      </Panel>
    );

  return (
    <div>
      <h1 className="mb-5 text-2xl font-bold">Network Monitor</h1>
      <Panel title="Network telemetry">
        <EmptyState
          title="Not available — no network telemetry collected"
          detail="The backend runs audio-only inference and stores no packet-loss, jitter, RTT or bitrate telemetry. Nothing on this page is estimated; network monitoring will appear here only when a real collector is connected."
        />
      </Panel>
      <Panel
        title="Measured inference latency (backend processing_time_ms — not network)"
        className="mt-4"
      >
        <div className="overflow-x-auto">
          <DataTable
            rows={data.items.map((r) => ({ ...r }))}
            emptyNote="Not available — no analyses yet. Latency is measured per analysis once audio is uploaded."
            columns={[
              {
                key: "audio_name",
                header: "File",
                render: (r) => <span className="mono">{r.audio_name}</span>,
              },
              {
                key: "processing_time_ms",
                header: "Measured latency",
                render: (r) => <span className="mono">{r.processing_time_ms.toFixed(1)} ms</span>,
              },
              {
                key: "audio_duration_seconds",
                header: "Audio length",
                render: (r) => <span className="mono">{r.audio_duration_seconds.toFixed(2)} s</span>,
              },
              {
                key: "created_at",
                header: "Time",
                render: (r) => (
                  <span className="mono text-xs" style={{ color: "#7d93b2" }}>
                    {new Date(r.created_at).toLocaleString()}
                  </span>
                ),
              },
            ]}
          />
        </div>
      </Panel>
    </div>
  );
}
