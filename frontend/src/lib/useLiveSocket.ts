import { useEffect, useRef, useState } from "react";
import { WS_URL } from "../services/api";
import type { LiveEvent } from "../types";

export type WsStatus = "LIVE" | "RECONNECTING" | "OFFLINE";

// WebSocket hook for /ws/live with auto-reconnect (1s, 2s, 4s ... capped 10s).
// Keeps the latest real events; never fabricates any.
export function useLiveSocket(active: boolean): {
  status: WsStatus;
  events: LiveEvent[];
  mockMode: boolean | null;
} {
  const [status, setStatus] = useState<WsStatus>("OFFLINE");
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [mockMode, setMockMode] = useState<boolean | null>(null);
  const retryRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!active) return;
    let closed = false;

    const connect = (): void => {
      if (closed) return;
      setStatus((s) => (s === "LIVE" ? s : retryRef.current === 0 ? "RECONNECTING" : "RECONNECTING"));
      let ws: WebSocket;
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        schedule();
        return;
      }
      wsRef.current = ws;
      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("LIVE");
      };
      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data as string) as LiveEvent;
          if (data.event === "system_status" && typeof data.mock_mode === "boolean") {
            setMockMode(data.mock_mode);
          }
          if (data.event === "ping") return;
          setEvents((prev) => [data, ...prev].slice(0, 50));
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        if (!closed) {
          setStatus("RECONNECTING");
          schedule();
        }
      };
      ws.onerror = () => {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      };
    };

    const schedule = (): void => {
      if (closed) return;
      retryRef.current += 1;
      const delay = Math.min(1000 * 2 ** (retryRef.current - 1), 10000);
      setStatus("RECONNECTING");
      timerRef.current = window.setTimeout(connect, delay);
    };

    connect();
    // If the first connect never opens, onclose may never fire; treat a
    // still-non-LIVE socket as OFFLINE after 12s of trying.
    const offlineTimer = window.setTimeout(() => {
      if (!closed && wsRef.current?.readyState !== WebSocket.OPEN) setStatus("OFFLINE");
    }, 12000);

    return () => {
      closed = true;
      window.clearTimeout(offlineTimer);
      if (timerRef.current) window.clearTimeout(timerRef.current);
      try {
        wsRef.current?.close();
      } catch {
        /* ignore */
      }
      setStatus("OFFLINE");
    };
  }, [active]);

  return { status, events, mockMode };
}
