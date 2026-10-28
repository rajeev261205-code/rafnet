import { useState } from "react";
import { API_BASE_URL, WS_URL, api } from "../services/api";
import { SUPABASE_CONFIGURED, useAuth } from "../lib/auth";

export default function Settings(): React.JSX.Element {
  const { mode, userLabel } = useAuth();
  const [clearing, setClearing] = useState(false);
  const [cleared, setCleared] = useState<string | null>(null);
  const [clearError, setClearError] = useState<string | null>(null);

  const clearAll = async (): Promise<void> => {
    if (
      !window.confirm(
        "Delete all analysis data (history, threats, call sessions)? This cannot be undone."
      )
    ) {
      return;
    }
    setClearing(true);
    setCleared(null);
    setClearError(null);
    try {
      const r = await api.clearAnalyses();
      setCleared(`Deleted ${r.deleted} ${r.deleted === 1 ? "analysis" : "analyses"}.`);
    } catch (e) {
      setClearError(e instanceof Error ? e.message : "Clear failed");
    } finally {
      setClearing(false);
    }
  };
  return (
    <div>
      <h1 className="mb-5 text-2xl font-bold">Settings</h1>
      <div className="soc-panel max-w-2xl p-6">
        <div className="mono mb-3 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
          Connection (from frontend/.env.local)
        </div>
        <dl className="mono space-y-2 text-sm">
          <div className="flex justify-between gap-4">
            <dt style={{ color: "#7d93b2" }}>VITE_API_URL</dt>
            <dd>{API_BASE_URL}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt style={{ color: "#7d93b2" }}>VITE_WS_URL</dt>
            <dd>{WS_URL}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt style={{ color: "#7d93b2" }}>Auth mode</dt>
            <dd>{mode === "dev" ? "DEV MODE login" : "Supabase email auth"}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt style={{ color: "#7d93b2" }}>Supabase configured</dt>
            <dd>{String(SUPABASE_CONFIGURED)}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt style={{ color: "#7d93b2" }}>Signed in as</dt>
            <dd>{userLabel ?? "—"}</dd>
          </div>
        </dl>
        <p className="mt-4 text-xs" style={{ color: "#7d93b2" }}>
          To use real email auth, set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in
          frontend/.env.local (anon key only — never a service_role key) and restart the
          frontend. Risk thresholds, rate limits, and CORS are backend settings in the
          repo-root .env.
        </p>
      </div>
      <div className="soc-panel mt-4 max-w-2xl p-6">
        <div className="mono mb-3 text-xs uppercase tracking-widest" style={{ color: "#f87171" }}>
          Danger zone
        </div>
        <p className="text-sm" style={{ color: "#7d93b2" }}>
          Delete all analysis data visible to you (history, threats, call sessions).
          Anonymous local rows are removed in dev mode; with auth enabled only your
          own rows are removed.
        </p>
        <button className="raf-btn raf-btn-danger mt-3" disabled={clearing} onClick={() => void clearAll()}>
          {clearing ? "Clearing…" : "Clear all analysis data"}
        </button>
        {cleared && <div className="mono mt-3 text-sm" style={{ color: "#34d399" }}>{cleared}</div>}
        {clearError && <div className="mt-3 text-sm text-red-300">{clearError}</div>}
      </div>
    </div>
  );
}
