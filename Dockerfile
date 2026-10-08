# Aegis Voice SOC backend — persistent Python host (Railway/Render/Fly/VM).
# Full ML image: install torch CPU + transformers + librosa on top.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOCK_INFERENCE=false \
    RAFNET_DEVICE=cpu

WORKDIR /app
COPY requirements-soc.txt ./
RUN pip install --no-cache-dir -r requirements-soc.txt

# Optional full model runtime (uncomment for real inference image):
# RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
#  && pip install --no-cache-dir transformers librosa scikit-learn scipy pyyaml soundfile

COPY src ./src
COPY experiments ./experiments
COPY supabase ./supabase

EXPOSE 8000
CMD ["sh", "-c", "cd src && uvicorn rafnet.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
