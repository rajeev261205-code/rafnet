import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";
import { setAuthTokenGetter } from "../services/api";

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

export const SUPABASE_CONFIGURED = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

interface AuthState {
  mode: "supabase" | "dev";
  session: Session | null;
  devUser: string | null;
  userLabel: string | null;
  isAuthed: boolean;
  signIn: (email: string, password: string) => Promise<string | null>;
  signUp: (email: string, password: string) => Promise<string | null>;
  devLogin: (name: string) => void;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

const DEV_KEY = "rafnet-dev-user";

export function AuthProvider({ children }: { children: React.ReactNode }): React.JSX.Element {
  const client = useMemo<SupabaseClient | null>(
    () =>
      SUPABASE_CONFIGURED
        ? createClient(SUPABASE_URL as string, SUPABASE_ANON_KEY as string)
        : null,
    [],
  );
  const [session, setSession] = useState<Session | null>(null);
  const [devUser, setDevUser] = useState<string | null>(() =>
    SUPABASE_CONFIGURED ? null : localStorage.getItem(DEV_KEY),
  );

  useEffect(() => {
    if (!client) return;
    client.auth.getSession().then(({ data }) => setSession(data.session));
    const { data: sub } = client.auth.onAuthStateChange((_ev, s) => setSession(s));
    return () => sub.subscription.unsubscribe();
  }, [client]);

  useEffect(() => {
    setAuthTokenGetter(() => session?.access_token ?? null);
  }, [session]);

  const value: AuthState = {
    mode: client ? "supabase" : "dev",
    session,
    devUser,
    userLabel: client ? (session?.user?.email ?? null) : devUser,
    isAuthed: client ? session !== null : devUser !== null,
    signIn: async (email, password) => {
      if (!client) return "Supabase is not configured";
      const { error } = await client.auth.signInWithPassword({ email, password });
      return error ? error.message : null;
    },
    signUp: async (email, password) => {
      if (!client) return "Supabase is not configured";
      const { error } = await client.auth.signUp({ email, password });
      return error ? error.message : null;
    },
    devLogin: (name) => {
      const label = name.trim() || "soc-operator";
      localStorage.setItem(DEV_KEY, label);
      setDevUser(label);
    },
    signOut: async () => {
      if (client) await client.auth.signOut();
      localStorage.removeItem(DEV_KEY);
      setDevUser(SUPABASE_CONFIGURED ? null : null);
      setSession(null);
    },
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
