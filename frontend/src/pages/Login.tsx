import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { SUPABASE_CONFIGURED, useAuth } from "../lib/auth";

export default function Login(): React.JSX.Element {
  const { mode, signIn, signUp, devLogin, isAuthed } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    document.title = "Sign in · RAFNet";
  }, []);

  useEffect(() => {
    if (isAuthed) navigate("/", { replace: true });
  }, [isAuthed, navigate]);

  if (isAuthed) {
    return <></>;
  }

  const submit = async (fn: (e: string, p: string) => Promise<string | null>): Promise<void> => {
    setBusy(true);
    setError(null);
    const err = await fn(email, password);
    setBusy(false);
    if (err) setError(err);
    else navigate("/", { replace: true });
  };

  return (
    <div className="soc-grid-bg flex min-h-screen items-center justify-center p-6">
      <div className="soc-panel w-full max-w-md p-8">
        <h1 className="text-2xl font-bold tracking-wide" style={{ color: "#22d3ee" }}>
          SOC LOGIN
        </h1>
        <p className="mono mt-1 text-xs" style={{ color: "#7d93b2" }}>
          RAFNet — voice security console
        </p>

        {mode === "dev" && (
          <div className="mt-4 rounded-lg border border-amber-900 bg-amber-950/30 p-3 text-xs text-amber-200">
            DEV MODE login — Supabase keys are not configured, so any operator name signs
            in locally. Set VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY for real email auth.
          </div>
        )}
        {SUPABASE_CONFIGURED && (
          <div className="mono mt-4 text-xs" style={{ color: "#7d93b2" }}>
            Supabase email auth
          </div>
        )}

        {mode === "supabase" ? (
          <div className="mt-4 flex flex-col gap-3">
            <label htmlFor="login-email" className="mono text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
              Email
            </label>
            <input
              id="login-email"
              className="soc-input"
              placeholder="Email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <label htmlFor="login-password" className="mono text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
              Password
            </label>
            <input
              id="login-password"
              className="soc-input"
              placeholder="Password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            {error && <div className="text-sm text-red-300">{error}</div>}
            <div className="flex gap-3">
              <button className="soc-btn flex-1" disabled={busy} onClick={() => void submit(signIn)}>
                Sign in
              </button>
              <button className="soc-btn flex-1" disabled={busy} onClick={() => void submit(signUp)}>
                Sign up
              </button>
            </div>
          </div>
        ) : (
          <div className="mt-4 flex flex-col gap-3">
            <label htmlFor="login-name" className="mono text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
              Operator name
            </label>
            <input
              id="login-name"
              className="soc-input"
              placeholder="Operator name (e.g. soc-operator)"
              autoComplete="username"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <button
              className="soc-btn"
              onClick={() => {
                devLogin(name);
                navigate("/", { replace: true });
              }}
            >
              Enter console (dev mode)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
