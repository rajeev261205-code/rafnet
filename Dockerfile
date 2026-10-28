# RAFNet backend — full real-inference image (Railway/Render/Fly/VM).
#
# RAM REQUIREMENTS (honest, measured): torch CPU + WavLM-base-plus (94M params,
# ~360 MB fp32) + app peaks well above 512 MB. Do NOT run this on a 512 MB
# instance (it will OOM). Minimum: 1 GB RAM (Render Starter); recommended 2 GB+.
# The SOC-only image (requirements-soc.txt, no torch) is the lightweight
# alternative for small instances — but it cannot run the real model.
FROM python:3.11-slim

# Production is the default for the container: authentication is required and
# startup fails closed unless SUPABASE_URL + a Supabase key + a non-wildcard
# SOC_CORS_ORIGINS are supplied. For a throwaway local demo, override with
# `-e RAFNET_ENV=development -e SOC_REQUIRE_AUTH=false`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOCK_INFERENCE=false \
    RAFNET_DEVICE=cpu \
    RAFNET_ENV=production \
    SOC_REQUIRE_AUTH=true \
    HF_HUB_OFFLINE=0

WORKDIR /app

# 1. PyTorch CPU first (separate index — PyPI wheels would pull CUDA builds).
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Full ML runtime (scikit-learn==1.9.1 matches the training pickles, so the
#    SOC adapter's compat shims retire in this image).
COPY requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY src ./src
COPY experiments ./experiments
COPY supabase ./supabase

# 3. Bake WavLM weights into the image so cold starts don't re-download
#    (~400 MB; Render disks are ephemeral, so first-boot download would recur
#    on every restart without this).
RUN python -c "from transformers import WavLMModel; WavLMModel.from_pretrained('microsoft/wavlm-base-plus')"

# Render injects $PORT (seen live: 10000); default 8000 elsewhere.
EXPOSE 8000
CMD ["sh", "-c", "cd src && uvicorn rafnet.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
