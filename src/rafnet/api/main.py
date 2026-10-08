from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import logging
import os

# Legacy JSON-frame stack needs torch; the SOC stack (rafnet.soc) does not.
# Guard the import so the app still boots (SOC-only) on machines without the
# heavy ML runtime — e.g. mock-mode dev or frontend integration work.
try:
    from rafnet.api.routes import router, initialize_service, http_exception_handler, generic_exception_handler
    from rafnet.api.service import InferenceService
    from rafnet.config.schema import RAFNetConfig
    from rafnet.inference import LatencyTarget

    _LEGACY_AVAILABLE = True
    _LEGACY_ERROR: str | None = None
except ImportError as _legacy_exc:  # noqa: BLE001 — torch/scipy missing
    router = None  # type: ignore[assignment]
    initialize_service = None  # type: ignore[assignment]
    http_exception_handler = None  # type: ignore[assignment]
    generic_exception_handler = None  # type: ignore[assignment]
    InferenceService = None  # type: ignore[assignment, misc]
    RAFNetConfig = None  # type: ignore[assignment, misc]
    LatencyTarget = None  # type: ignore[assignment, misc]
    _LEGACY_AVAILABLE = False
    _LEGACY_ERROR = str(_legacy_exc)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration from environment
def get_config():  # -> RAFNetConfig when legacy stack is available
    """Get configuration from environment."""
    if not _LEGACY_AVAILABLE:
        raise RuntimeError(f"Legacy inference stack unavailable: {_LEGACY_ERROR}")
    config = RAFNetConfig()
    
    # Override from environment
    config.device = os.getenv("RAFNET_DEVICE", "cpu")
    config.classical.model_type = os.getenv("RAFNET_CLASSICAL_MODEL", "logistic")
    config.quantum.fallback_mode = os.getenv("RAFNET_QUANTUM_FALLBACK", "classical")
    
    return config


def get_latency_target():  # -> LatencyTarget when legacy stack is available
    """Get latency targets from environment."""
    if not _LEGACY_AVAILABLE:
        raise RuntimeError(f"Legacy inference stack unavailable: {_LEGACY_ERROR}")
    return LatencyTarget(
        preprocessing=float(os.getenv("LATENCY_PREPROCESSING", "5.0")),
        audio_encoder=float(os.getenv("LATENCY_AUDIO_ENCODER", "600.0")),
        network_encoder=float(os.getenv("LATENCY_NETWORK_ENCODER", "10.0")),
        gateway1=float(os.getenv("LATENCY_GATEWAY1", "10.0")),
        classical=float(os.getenv("LATENCY_CLASSICAL", "20.0")),
        quantum=float(os.getenv("LATENCY_QUANTUM", "20.0")),
        gateway2=float(os.getenv("LATENCY_GATEWAY2", "5.0")),
        calibration=float(os.getenv("LATENCY_CALIBRATION", "5.0")),
        decision_gate=float(os.getenv("LATENCY_DECISION_GATE", "2.0")),
        total=float(os.getenv("LATENCY_TOTAL", "700.0")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAFNet API...")

    if _LEGACY_AVAILABLE:
        config = get_config()
        latency_target = get_latency_target()
        audio_window = float(os.getenv("AUDIO_WINDOW_SECONDS", "1.0"))
        max_queue = int(os.getenv("MAX_QUEUE_SIZE", "100"))

        # Initialize inference service
        await initialize_service(
            config=config,
            device=config.device,
            latency_target=latency_target,
            audio_window_seconds=audio_window,
            max_queue_size=max_queue,
        )
    else:
        logger.warning("Legacy inference stack unavailable (%s); running SOC-only", _LEGACY_ERROR)

    # SOC model warms lazily on first request via get_soc_service(); nothing to start.
    logger.info("RAFNet API started (legacy=%s)", "on" if _LEGACY_AVAILABLE else "off")

    yield

    # Shutdown
    logger.info("Shutting down RAFNet API...")
    if _LEGACY_AVAILABLE:
        from rafnet.api.service import get_inference_service
        service = get_inference_service()
        await service.shutdown()
    logger.info("RAFNet API shut down")


app = FastAPI(
    title="RAFNet API",
    description="""
    RAFNet (Real-time Audio-Fusion Network) API for synthetic speech detection.
    
    This API provides real-time inference for detecting synthetic vs real speech
    using a multi-modal pipeline combining audio and network telemetry.
    
    ## Pipeline
    ```
    Audio → WavLM + Handcrafted → Audio Encoder
    Network Telemetry → Network Encoder
                    ↓
              Gateway 1 (Fusion)
                    ↓
         Classical Branch (XGBoost/Logistic)
                    ↓
         Quantum Branch (VQC + Classical Head)
                    ↓
              Gateway 2 (Attention Fusion)
                    ↓
         Calibration (Temperature/Platt/Isotonic)
                    ↓
         Decision Gate (τ_low/τ_high)
                    ↓
         Decision: REAL / UNCERTAIN / SYNTHETIC
    ```
    
    ## Latency Targets
    - Total: 700ms (CPU) / 100ms (GPU)
    - Audio Encoder: 600ms (CPU) / 30ms (GPU)
    - Other stages: <20ms each
    
    ## Authentication
    Currently no authentication required. Add API key middleware for production.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS — configured via CORS_ORIGINS (comma-separated). Default locks to local
# dev origins; set CORS_ORIGINS=* explicitly only for throwaway local testing.
def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins


_cors_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse

try:
    from rafnet.soc.limits import MaxBodySizeMiddleware

    app.add_middleware(MaxBodySizeMiddleware)
except Exception as exc:  # noqa: BLE001 — never break boot on hardening
    logger.warning("Body-size guard not mounted: %s", exc)

# Exception handlers
app.add_exception_handler(404, lambda r, e: JSONResponse(status_code=404, content={"error": "Not found", "detail": str(e)}))
app.add_exception_handler(500, lambda r, e: JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(e)}))

# Include routes — legacy /api/v1 only when torch stack is present.
if _LEGACY_AVAILABLE and router is not None:
    app.include_router(router, prefix="/api/v1")
else:
    logger.warning("Legacy /api/v1 routes disabled: %s", _LEGACY_ERROR)

# Aegis Voice SOC routes (file-upload analysis, dashboard, model info).
# Torch-independent; always mounted when importable.
try:
    from rafnet.soc.routes import router as soc_router
    from rafnet.soc.websocket import ws_router

    app.include_router(soc_router)
    app.include_router(ws_router)
except Exception as exc:  # noqa: BLE001
    logger.warning("SOC router not mounted: %s", exc)


def custom_openapi():
    """Custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RAFNet API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes (for future use)
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        },
        "BearerToken": {
            "type": "http",
            "scheme": "bearer",
        },
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Health check without prefix (for load balancers)
@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Simple health check for load balancers."""
    if not _LEGACY_AVAILABLE:
        # SOC-only boot: report SOC model state instead of the legacy engine.
        try:
            from rafnet.soc.routes import get_soc_service

            svc = get_soc_service()
            return {"status": "ok" if svc.model.is_loaded() else "degraded"}
        except Exception:  # noqa: BLE001
            return {"status": "degraded"}
    from rafnet.api.service import get_inference_service
    service = get_inference_service()
    if service.is_ready():
        return {"status": "ok"}
    return {"status": "degraded"}


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "name": "RAFNet API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/healthz",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rafnet.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )