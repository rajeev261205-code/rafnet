import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import RequireAuth from "./components/RequireAuth";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import LiveSOC from "./pages/LiveSOC";
import AnalyzePage from "./pages/AnalyzePage";
import History from "./pages/History";
import AnalysisDetail from "./pages/AnalysisDetail";
import Threats from "./pages/Threats";
import Sessions from "./pages/Sessions";
import Models from "./pages/Models";
import Fusion from "./pages/Fusion";
import Network from "./pages/Network";
import Settings from "./pages/Settings";

function Shell({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <RequireAuth>
      <Layout>{children}</Layout>
    </RequireAuth>
  );
}

export default function App(): React.JSX.Element {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<Shell><Dashboard /></Shell>} />
      <Route path="/live" element={<Shell><LiveSOC /></Shell>} />
      <Route path="/analyze" element={<Shell><AnalyzePage /></Shell>} />
      <Route path="/history" element={<Shell><History /></Shell>} />
      <Route path="/history/:id" element={<Shell><AnalysisDetail /></Shell>} />
      <Route path="/threats" element={<Shell><Threats /></Shell>} />
      <Route path="/sessions" element={<Shell><Sessions /></Shell>} />
      <Route path="/models" element={<Shell><Models /></Shell>} />
      <Route path="/fusion" element={<Shell><Fusion /></Shell>} />
      <Route path="/network" element={<Shell><Network /></Shell>} />
      <Route path="/settings" element={<Shell><Settings /></Shell>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
