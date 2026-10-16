// Shared RAFNet console components: Panel, Chip, Badge, Button, Tabs,
// Gauge, ProgressBar, DataTable, Modal, EmptyState, Skeleton.
// All values rendered by these components come from backend/browser data;
// they never invent content themselves.
import type { ButtonHTMLAttributes, ReactNode } from "react";

/* ---------- Panel ---------- */
export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}): React.JSX.Element {
  return (
    <section className={`raf-panel p-5 ${className}`}>
      {(title ?? right) && (
        <div className="mb-4 flex flex-wrap items-center gap-3">
          {title && <h2 className="raf-panel-title">{title}</h2>}
          <span className="flex-1" />
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

/* ---------- Chip ---------- */
const CHIP_COLORS: Record<string, string> = {
  cyan: "#22d3ee",
  emerald: "#34d399",
  red: "#f87171",
  amber: "#fbbf24",
  muted: "#7d93b2",
};

export function Chip({
  color = "cyan",
  children,
}: {
  color?: keyof typeof CHIP_COLORS;
  children: ReactNode;
}): React.JSX.Element {
  return (
    <span className="raf-chip" style={{ color: CHIP_COLORS[color] }}>
      {children}
    </span>
  );
}

/* ---------- Badge ---------- */
export function Badge({
  tone = "muted",
  children,
}: {
  tone?: "ok" | "warn" | "bad" | "info" | "muted";
  children: ReactNode;
}): React.JSX.Element {
  return <span className={`raf-badge raf-badge-${tone}`}>{children}</span>;
}

/* ---------- Button ---------- */
type ButtonVariant = "primary" | "danger" | "warn" | "ghost";

export function Button({
  variant = "primary",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }): React.JSX.Element {
  const cls =
    variant === "primary"
      ? "raf-btn"
      : variant === "danger"
        ? "raf-btn raf-btn-danger"
        : variant === "warn"
          ? "raf-btn raf-btn-warn"
          : "raf-btn raf-btn-ghost";
  return <button className={cls} {...rest} />;
}

/* ---------- Tabs ---------- */
export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: T; label: string }[];
  active: T;
  onChange: (id: T) => void;
}): React.JSX.Element {
  return (
    <div className="raf-tabs" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={t.id === active}
          className={`raf-tab${t.id === active ? " active" : ""}`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* ---------- Gauge (SVG, circular) ---------- */
export function Gauge({
  value,
  label,
  sub,
  size = 190,
}: {
  value: number; // 0-100
  label: string;
  sub?: string;
  size?: number;
}): React.JSX.Element {
  const clamped = Math.max(0, Math.min(100, value));
  const color = clamped >= 65 ? "#f87171" : clamped >= 35 ? "#fbbf24" : "#34d399";
  const r = 80;
  const c = 2 * Math.PI * r;
  // 270-degree arc with the gap at the bottom (rotate 135deg, sweep 75% of circle).
  const track = `${0.75 * c} ${0.25 * c}`;
  const shown = `${(clamped / 100) * 0.75 * c} ${c}`;
  return (
    <svg viewBox="0 0 200 200" width={size} height={size} role="img" aria-label={`${label} ${clamped}`}>
      <circle
        cx="100"
        cy="100"
        r={r}
        fill="none"
        stroke="#1b2a44"
        strokeWidth="14"
        strokeDasharray={track}
        transform="rotate(135 100 100)"
      />
      <circle
        cx="100"
        cy="100"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeLinecap="round"
        strokeDasharray={shown}
        transform="rotate(135 100 100)"
        style={{ filter: `drop-shadow(0 0 8px ${color})` }}
      />
      <text x="100" y="96" textAnchor="middle" fill="#dbe7f5" fontSize="38" fontWeight="700">
        {clamped.toFixed(clamped >= 10 ? 0 : 1)}
        <tspan fontSize="16">%</tspan>
      </text>
      <text
        x="100"
        y="122"
        textAnchor="middle"
        fill="#7d93b2"
        fontSize="12"
        className="mono"
        style={{ textTransform: "uppercase", letterSpacing: "0.15em" }}
      >
        {label}
      </text>
      {sub && (
        <text x="100" y="142" textAnchor="middle" fill="#7d93b2" fontSize="11" className="mono">
          {sub}
        </text>
      )}
      {/* end gauge */}
    </svg>
  );
}

/* ---------- ProgressBar ---------- */
export function ProgressBar({
  value,
  color = "#22d3ee",
}: {
  value: number; // 0-100
  color?: string;
}): React.JSX.Element {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className="raf-progress" role="progressbar" aria-valuenow={clamped} aria-valuemin={0} aria-valuemax={100}>
      <div style={{ width: `${clamped}%`, background: color }} />
    </div>
  );
}

/* ---------- DataTable ---------- */
export interface Column<T> {
  key: string;
  header: string;
  render?: (row: T) => ReactNode;
}

export function DataTable<T extends { id: string }>({
  columns,
  rows,
  emptyNote,
}: {
  columns: Column<T>[];
  rows: T[];
  emptyNote: string;
}): React.JSX.Element {
  if (rows.length === 0) {
    return (
      <div className="p-6 text-sm" style={{ color: "#7d93b2" }}>
        {emptyNote}
      </div>
    );
  }
  return (
    <table className="raf-table w-full">
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c.key}>{c.header}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id}>
            {columns.map((c) => (
              <td key={c.key}>{c.render ? c.render(row) : String((row as unknown as Record<string, unknown>)[c.key] ?? "—")}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ---------- Modal ---------- */
export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}): React.JSX.Element {
  return (
    <div
      className="raf-modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="raf-modal raf-panel p-6" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-center gap-3">
          <h2 className="text-lg font-bold">{title}</h2>
          <span className="flex-1" />
          <Button variant="ghost" onClick={onClose} aria-label="Close">
            ✕
          </Button>
        </div>
        {children}
      </div>
    </div>
  );
}

/* ---------- EmptyState ---------- */
export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: ReactNode;
}): React.JSX.Element {
  return (
    <div className="flex flex-col items-start gap-2 p-6">
      <div className="mono text-sm uppercase tracking-widest" style={{ color: "#7d93b2" }}>
        {title}
      </div>
      {detail && <p className="max-w-2xl text-sm" style={{ color: "#7d93b2" }}>{detail}</p>}
      {action}
    </div>
  );
}

/* ---------- Skeleton ---------- */
export function Skeleton({ lines = 3 }: { lines?: number }): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="raf-skeleton" style={{ height: "1rem", opacity: 1 - i * 0.15 }} />
      ))}
    </div>
  );
}
