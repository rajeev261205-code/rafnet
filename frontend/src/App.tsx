import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AnalyzePage from './pages/AnalyzePage';
import Dashboard from './pages/Dashboard';
import LiveSOC from './pages/LiveSOC';

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-slate-900 text-white">
        <nav className="bg-slate-800 border-b border-slate-700 p-4 flex gap-6">
          <Link to="/" className="font-bold text-blue-400">AEGIS SOC</Link>
          <Link to="/dashboard" className="hover:text-blue-300">Dashboard</Link>
          <Link to="/analyze" className="hover:text-blue-300">Analyze</Link>
          <Link to="/live" className="hover:text-blue-300">Live Stream</Link>
        </nav>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/live" element={<LiveSOC />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
