import React, { useState, useEffect } from 'react';
import { analysisService } from '../services/api';
import { AnalysisResult } from '../types';

const Dashboard: React.FC = () => {
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    analysisService.getHistory()
      .then(setHistory)
      .finally(() => setLoading(false));
  }, []);

  const stats = {
    total: history.length,
    synthetic: history.filter(h => h.prediction === 'SYNTHETIC').length,
    real: history.filter(h => h.prediction === 'REAL').length,
    highRisk: history.filter(h => h.risk_score > 70).length,
  };

  return (
    <div className="p-8 bg-slate-900 text-white min-h-screen">
      <h1 className="text-3xl font-bold mb-6">SOC Command Center</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <StatCard label="Total Analyses" value={stats.total} color="text-blue-400" />
        <StatCard label="Synthetic" value={stats.synthetic} color="text-red-400" />
        <StatCard label="Authentic" value={stats.real} color="text-green-400" />
        <StatCard label="High Risk" value={stats.highRisk} color="text-yellow-400" />
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-slate-700 text-slate-300">
            <tr>
              <th className="p-4">File</th>
              <th className="p-4">Prediction</th>
              <th className="p-4">Risk Score</th>
              <th className="p-4">Confidence</th>
              <th className="p-4">Date</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item) => (
              <tr key={item.id} className="border-t border-slate-700 hover:bg-slate-750">
                <td className="p-4">{item.audio_name}</td>
                <td className="p-4">
                  <span className={`px-2 py-1 rounded text-xs font-bold ${
                    item.prediction === 'SYNTHETIC' ? 'bg-red-900 text-red-200' : 'bg-green-900 text-green-200'
                  }`}>
                    {item.prediction}
                  </span>
                </td>
                <td className="p-4 font-mono">{item.risk_score}</td>
                <td className="p-4">{(item.confidence * 100).toFixed(1)}%</td>
                <td className="p-4 text-slate-400 text-sm">{new Date(item.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {history.length === 0 && !loading && (
          <div className="p-8 text-center text-slate-500">No analysis history available.</div>
        )}
      </div>
    </div>
  );
};

const StatCard = ({ label, value, color }: any) => (
  <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
    <p className="text-slate-400 text-sm">{label}</p>
    <p className={`text-3xl font-bold ${color}`}>{value}</p>
  </div>
);

export default Dashboard;
