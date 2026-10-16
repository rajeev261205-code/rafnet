// Polls the real backend for top-bar / shell data: health, dashboard
// aggregates and threat count. All values come from the backend;
// while unreachable the shell reports OFFLINE instead of guessing.
import { useEffect, useState } from "react";
import type { DashboardResponse, HealthResponse } from "../types";
import { api } from "../services/api";

export interface ConsoleStats {
  health: HealthResponse | null;
  dashboard: DashboardResponse | null;
  threatCount: number | null;
  offline: boolean;
}

export function useConsoleStats(pollMs = 15000): ConsoleStats {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [threatCount, setThreatCount] = useState<number | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async (): Promise<void> => {
      try {
        const [h, d, t] = await Promise.all([
          api.health(),
          api.dashboard(),
          api.threats(),
        ]);
        if (!alive) return;
        setHealth(h);
        setDashboard(d);
        setThreatCount(t.items.length);
        setOffline(false);
      } catch {
        if (alive) setOffline(true);
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), pollMs);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [pollMs]);

  return { health, dashboard, threatCount, offline };
}
