import { useEffect, useState } from "react";
import type { HealthResponse } from "../types";
import { api } from "../services/api";

export function useBackendHealth(pollMs = 15000): {
  health: HealthResponse | null;
  offline: boolean;
} {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async (): Promise<void> => {
      try {
        const h = await api.health();
        if (alive) {
          setHealth(h);
          setOffline(false);
        }
      } catch {
        if (alive) setOffline(true);
      }
    };
    void poll();
    const t = window.setInterval(() => void poll(), pollMs);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [pollMs]);

  return { health, offline };
}
