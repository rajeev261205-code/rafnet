export default function RiskGauge({ score }: { score: number }): React.JSX.Element {
  const clamped = Math.max(0, Math.min(100, score));
  const angle = (clamped / 100) * 180;
  const rad = (Math.PI * (180 - angle)) / 180;
  const cx = 100;
  const cy = 90;
  const r = 70;
  const nx = cx + r * Math.cos(rad);
  const ny = cy - r * Math.sin(rad);
  const color = clamped >= 85 ? "#f87171" : clamped >= 65 ? "#f87171" : clamped >= 35 ? "#fbbf24" : "#34d399";

  const arc = (from: number, to: number): string => {
    const a0 = ((180 - from) * Math.PI) / 180;
    const a1 = ((180 - to) * Math.PI) / 180;
    return `M ${cx + r * Math.cos(a0)} ${cy - r * Math.sin(a0)} A ${r} ${r} 0 0 1 ${cx + r * Math.cos(a1)} ${cy - r * Math.sin(a1)}`;
  };

  return (
    <svg viewBox="0 0 200 110" className="w-full max-w-62">
      <path d={arc(0, 100)} fill="none" stroke="#1b2a44" strokeWidth="14" strokeLinecap="round" />
      <path d={arc(0, clamped)} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round" />
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth="3" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="6" fill={color} />
      <text x={cx} y={cy + 2} textAnchor="middle" fill="#dbe7f5" fontSize="18" fontWeight="700" className="mono">
        {clamped}
      </text>
    </svg>
  );
}
