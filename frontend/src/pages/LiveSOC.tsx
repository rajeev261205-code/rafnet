import React, { useState, useEffect } from 'react';

const LiveSOC: React.FC = () => {
  const [events, setEvents] = useState<any[]>([]);
  const [status, setStatus] = useState('Disconnected');

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/live');

    ws.onopen = () => setStatus('Connected');
    ws.onclose = () => setStatus('Disconnected');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setEvents(prev => [data, ...prev].slice(0, 10));
    };

    return () => ws.close();
  }, []);

  return (
    <div className="p-8 bg-slate-900 text-white min-h-screen">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Live Detection Stream</h1>
        <span className={`px-3 py-1 rounded-full text-xs ${status === 'Connected' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
          {status}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 h-96 overflow-y-auto font-mono text-sm">
          <h2 className="text-slate-400 mb-4 uppercase text-xs font-bold">Event Log</h2>
          {events.map((e, i) => (
            <div key={i} className="mb-2 p-2 bg-slate-900 rounded border-l-4 border-blue-500">
              <span className="text-slate-500">[{new Date(e.timestamp).toLocaleTimeString()}]</span> {e.event}: {e.status}
            </div>
          ))}
          {events.length === 0 && <p className="text-slate-600">Waiting for stream events...</p>}
        </div>
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex items-center justify-center">
          <div className="text-center">
            <div className="w-24 h-24 rounded-full border-4 border-blue-500 border-t-transparent animate-spin mx-auto mb-4"></div>
            <p className="text-slate-400">Monitoring live audio streams...</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveSOC;
