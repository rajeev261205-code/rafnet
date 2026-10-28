import { Navigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

export default function RequireAuth({
  children,
}: {
  children: React.JSX.Element;
}): React.JSX.Element {
  const { isAuthed } = useAuth();
  if (!isAuthed) return <Navigate to="/login" replace />;
  return children;
}
