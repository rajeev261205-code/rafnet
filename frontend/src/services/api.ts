const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const analysisService = {
  async analyzeAudio(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error('Analysis failed');
    return response.json();
  },

  async getHistory(): Promise<any[]> {
    const response = await fetch(`${API_BASE_URL}/api/history`);
    if (!response.ok) throw new Error('Failed to fetch history');
    return response.json();
  },

  async checkHealth(): Promise<{status: string}> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  }
};
