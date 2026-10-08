import React, { useState } from 'react';
import { analysisService } from '../services/api';
import { AnalysisResult } from '../types';

const AnalyzePage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await analysisService.analyzeAudio(file);
      setResult(res);
    } catch (e) {
      alert('Error analyzing audio');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 bg-slate-900 text-white min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Forensic Audio Analysis</h1>
      <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
        <input 
          type="file" 
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block w-full text-sm text-slate-400 mb-4"
        />
        <button 
          onClick={handleUpload}
          disabled={loading || !file}
          className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-semibold transition-all"
        >
          {loading ? 'Analyzing...' : 'Start Forensic Analysis'}
        </button>
      </div>

      {result && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
            <p className="text-slate-400 text-sm">Prediction</p>
            <p className={`text-2xl font-bold ${result.prediction === 'SYNTHETIC' ? 'text-red-500' : 'text-green-500'}`}>
              {result.prediction}
            </p>
          </div>
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
            <p className="text-slate-400 text-sm">Risk Score</p>
            <p className="text-2xl font-bold text-yellow-500">{result.risk_score}/100</p>
          </div>
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
            <p className="text-slate-400 text-sm">Confidence</p>
            <p className="text-2xl font-bold text-blue-400">{(result.confidence * 100).toFixed(2)}%</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyzePage;
