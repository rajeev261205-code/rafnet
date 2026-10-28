// RAFNet console shell: sidebar (shield logo, MAIN CONSOLE + SYSTEM nav,
// backend-status footer), top status chips bound to real backend data, and a
// radial glow driven by the latest analysis (red = HIGH/CRITICAL risk,
// amber = uncertain verdict, cyan = normal). Responsive with a mobile drawer.
import { useEffect, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useConsoleStats } from "../lib/useConsoleStats";
import { API_BASE_URL } from "../services/api";
import { Chip } from "./ui";

const MAIN_NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/live", label: "Live Detection" },
  { to: "/analyze", label: "Analyze" },
  { to: "/sessions", label: "Call Sessions" },
  { to: "/models", label: "Model Intelligence" },
  { to: "/fusion", label: "Fusion & Gateways" },
  { to: "/network", label: "Network Monitor" },
  { to: "/history", label: "History" },
  { to: "/threats", label: "Threats" },
];

const SYSTEM_NAV = [{ to: "/settings", label: "Settings" }];

const PAGE_TITLES: Record<string, string> = {
  "/": "Overview · RAFNet",
  "/live": "Live Detection · RAFNet",
  "/analyze": "Analyze · RAFNet",
  "/history": "History · RAFNet",
  "/threats": "Threats · RAFNet",
  "/sessions": "Call Sessions · RAFNet",
  "/models": "Model Intelligence · RAFNet",
  "/fusion": "Fusion & Gateways · RAFNet",
  "/network": "Network Monitor · RAFNet",
  "/settings": "Settings · RAFNet",
};

import type { HealthResponse } from "../types";

function SidebarBody({
  health,
  offline,
  onNav,
}: {
  health: HealthResponse | null;
  offline: boolean;
  onNav?: () => void;
}): React.JSX.Element {
  const { mode } = useAuth();
  const backendLabel = offline
    ? "OFFLINE"
    : health
      ? `${health.status.toUpperCase()} · ${health.model}`
      : "CONNECTING";
  const backendColor = offline ? "#f87171" : health ? "#34d399" : "#fbbf24";
  return (
    <div className="flex h-full flex-col p-4">
      <div className="mb-6 flex items-center gap-3 px-2">
        <img src="/shield.svg" alt="RAFNet shield" width={36} height={36} />
        <div>
          <div className="text-lg font-bold tracking-wide" style={{ color: "#22d3ee" }}>
            RAFNet
          </div>
          <div className="mono text-xs" style={{ color: "#7d93b2", letterSpacing: "0.2em" }}>
            VOICE SECURITY
          </div>
        </div>
      </div>

      <div className="raf-navsection px-2 pb-2">Main Console</div>
      <nav className="flex flex-col gap-1">
        {MAIN_NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            onClick={onNav}
            className={({ isActive }) => `raf-navlink${isActive ? " active" : ""}`}
          >
            {n.label}
          </NavLink>
        ))}
      </nav>

      <div className="raf-navsection px-2 pb-2 pt-5">System</div>
      <nav className="flex flex-col gap-1">
        {SYSTEM_NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            onClick={onNav}
            className={({ isActive }) => `raf-navlink${isActive ? " active" : ""}`}
          >
            {n.label}
          </NavLink>
        ))}
      </nav>

      <div className="flex-1" />
      <div className="mono mt-6 border-t px-2 pt-4 text-xs" style={{ borderColor: "#1b2a44" }}>
        <div style={{ color: backendColor }}>● BACKEND {backendLabel}</div>
        <div className="mt-1 truncate" style={{ color: "#7d93b2" }}>
          {API_BASE_URL}
        </div>
        <div className="mt-1" style={{ color: "#7d93b2" }}>
          {mode === "dev" ? "DEV MODE login" : "Supabase auth"}
        </div>
      </div>
    </div>
  );
}

export default function Layout({ children }: { children: React.ReactNode }): React.JSX.Element {
  const { userLabel, signOut } = useAuth();
  const { health, dashboard, threatCount, offline } = useConsoleStats();
  const navigate = useNavigate();
  const location = useLocation();
  const [drawer, setDrawer] = useState(false);

  useEffect(() => {
    document.title = PAGE_TITLES[location.pathname] ?? "RAFNet | Voice Security";
  }, [location.pathname]);

  useEffect(() => {
    setDrawer(false);
  }, [location.pathname]);

  const onSignOut = async (): Promise<void> => {
    await signOut();
    navigate("/login");
  };

  const modelLoaded = !offline && health?.model_loaded === true;
  // "AI Active" only for the REAL model: mock reports model_loaded=true too,
  // so mock must explicitly suppress it (it already gets the amber Mock chip).
  const modelReady = modelLoaded && health?.mock_mode !== true;
  const modelDown = !offline && !!health && !modelLoaded;
  const latest = dashboard?.recent?.[0];
  const glow =
    latest && (latest.risk_level === "HIGH" || latest.risk_level === "CRITICAL")
      ? "glow-threat"
      : latest && latest.prediction === "UNCERTAIN"
        ? "glow-uncertain"
        : "glow-normal";

  return (
    <div className={`soc-grid-bg console-glow ${glow} min-h-screen`}>
      <div className="relative z-10 flex min-h-screen">
        <aside className="raf-sidebar raf-sidebar-desktop shrink-0">
          <SidebarBody health={health} offline={offline} />
        </aside>

        {drawer && (
          <div className="raf-sidebar-mobile fixed inset-0 z-40" role="dialog" aria-label="Navigation">
            <div className="absolute inset-0 bg-black/70" onClick={() => setDrawer(false)} />
            <aside className="raf-sidebar absolute inset-y-0 left-0">
              <SidebarBody health={health} offline={offline} onNav={() => setDrawer(false)} />
            </aside>
          </div>
        )}

        <div className="flex min-w-0 flex-1 flex-col">
          <header
            className="flex flex-wrap items-center gap-2 border-b px-5 py-3"
            style={{ borderColor: "#1b2a44", background: "rgba(7,13,24,0.9)" }}
          >
            <button
              className="raf-btn raf-btn-ghost raf-hamburger"
              style={{ padding: "0.3rem 0.7rem" }}
              onClick={() => setDrawer(true)}
              aria-label="Open navigation"
            >
              ☰
            </button>
            {offline ? (
              <Chip color="red">System: offline</Chip>
            ) : (
              <Chip color={health?.status === "ok" ? "emerald" : "amber"}>
                System: {health ? health.status.toUpperCase() : "…"}
              </Chip>
            )}
            <Chip color="cyan">Analyses: {dashboard ? dashboard.total_analyses : "…"}</Chip>
            <Chip color={threatCount ? "red" : "muted"}>
              Threats: {threatCount ?? "…"}
            </Chip>
            <Chip color="muted">Model: {dashboard ? dashboard.model_version : "…"}</Chip>
            {health?.mock_mode && !offline && <Chip color="amber">Mock mode — not a real prediction</Chip>}
            {modelDown && (
              <Chip color="red">
                Model not loaded{health?.error ? ` — ${health.error.slice(0, 120)}` : ""}
              </Chip>
            )}
            <span className="flex-1" />
            <span className="mono text-xs" style={{ color: "#7d93b2" }}>
              {userLabel ?? ""}
            </span>
            <button className="soc-btn" style={{ padding: "0.3rem 0.8rem" }} onClick={() => void onSignOut()}>
              Logout
            </button>
            {modelReady && <Chip color="emerald">⚡ AI Active</Chip>}
          </header>

          {offline && (
            <div className="mx-5 mt-4 rounded-lg border border-red-900 bg-red-950/40 p-3 text-sm text-red-200">
              Backend is offline at the configured API URL. Start it with{" "}
              <span className="mono">start-local.ps1</span> — no data below is live until it
              reconnects.
            </div>
          )}

          <main className="min-w-0 flex-1 p-5">{children}</main>
        </div>
      </div>
    </div>
  );
}
