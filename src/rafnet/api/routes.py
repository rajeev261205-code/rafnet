from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from datetime import datetime

from rafnet.api.schemas import (
    PredictRequest, PredictResponse, HealthResponse,
    ModelInfoResponse, VersionResponse, ErrorResponse,
    PredictRequest as BatchPredictRequest,
)
from rafnet.api.service import InferenceService, get_inference_service, initialize_service
from rafnet.config.schema import RAFNetConfig
from rafnet.inference import LatencyTarget

logger = logging.getLogger(__name__)

router = APIRouter()


def get_service() -> "InferenceService":
    """Dependency to get inference service."""
    return get_inference_service()


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Run inference on audio and network data",
    description="Submit audio frame and network telemetry for real-time classification.",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Service not ready"},
        504: {"model": ErrorResponse, "description": "Inference timeout"},
    },
)
async def predict(
    request: PredictRequest,
    service: InferenceService = Depends(get_service),
) -> PredictResponse:
    """
    Run single inference on audio and network telemetry.
    
    Args:
        request: Audio frame and network telemetry
        
    Returns:
        Prediction with probabilities, decision, and latency info
    """
    try:
        logger.info(f"Prediction request: frame_id={request.audio.frame_id}")
        result = await service.predict(request)
        logger.info(f"Prediction complete: decision={result.decision_label}, latency={result.total_latency_ms:.1f}ms")
        return result
    
    except RuntimeError as e:
        logger.error(f"Service not ready: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready"
        )
    except TimeoutError as e:
        logger.error(f"Inference timeout: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout"
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=List[PredictResponse],
    status_code=status.HTTP_200_OK,
    summary="Run batch inference",
    description="Submit multiple audio frames and network telemetry for batch classification.",
)
async def predict_batch(
    requests: List[PredictRequest],
    service: InferenceService = Depends(get_service),
) -> List[PredictResponse]:
    """Run batch inference on multiple samples."""
    try:
        return await service.batch_predict(requests)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready"
        )
    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check service health and component status.",
)
async def health(service: InferenceService = Depends(get_service)) -> HealthResponse:
    """Health check endpoint."""
    return service.get_health()


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model information",
    description="Get model configuration and component details.",
)
async def model_info(service: InferenceService = Depends(get_service)) -> ModelInfoResponse:
    """Get model configuration and metadata."""
    return service.get_model_info()


@router.get(
    "/version",
    response_model=VersionResponse,
    status_code=status.HTTP_200_OK,
    summary="API version",
    description="Get API and model version information.",
)
async def version(service: InferenceService = Depends(get_service)) -> VersionResponse:
    """Get version information."""
    return service.get_version()


@router.get(
    "/latency",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Latency statistics",
    description="Get latency statistics for each pipeline stage.",
)
async def latency_stats(service: InferenceService = Depends(get_service)) -> dict:
    """Get latency statistics for all pipeline stages."""
    return service.get_latency_stats()


@router.get(
    "/log",
    response_model=List[dict],
    status_code=status.HTTP_200_OK,
    summary="Prediction log",
    description="Get recent prediction history.",
)
async def prediction_log(
    limit: int = 100,
    service: InferenceService = Depends(get_service),
) -> List[dict]:
    """Get recent prediction log."""
    return service.get_prediction_log(limit=limit)


# Admin endpoints for model fitting
@router.post(
    "/admin/fit-classical",
    status_code=status.HTTP_200_OK,
    summary="Fit classical branch",
    description="Fit classical branch on training embeddings.",
)
async def fit_classical(
    embeddings: List[List[float]],
    labels: List[int],
    service: InferenceService = Depends(get_service),
) -> dict:
    """Fit classical branch on embeddings."""
    try:
        import numpy as np
        service.fit_classical(np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int64))
        return {"status": "success", "message": "Classical branch fitted"}
    except Exception as e:
        logger.exception(f"Fit classical error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/admin/fit-quantum",
    status_code=status.HTTP_200_OK,
    summary="Fit quantum branch PCA",
    description="Fit quantum branch PCA on training embeddings.",
)
async def fit_quantum(
    embeddings: List[List[float]],
    service: InferenceService = Depends(get_service),
) -> dict:
    """Fit quantum branch PCA."""
    try:
        import numpy as np
        service.fit_quantum(np.array(embeddings, dtype=np.float32))
        return {"status": "success", "message": "Quantum branch PCA fitted"}
    except Exception as e:
        logger.exception(f"Fit quantum error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/admin/fit-calibration",
    status_code=status.HTTP_200_OK,
    summary="Fit calibration",
    description="Fit calibration on validation data.",
)
async def fit_calibration(
    train_probs: List[float],
    train_labels: List[int],
    val_probs: List[float],
    val_labels: List[int],
    service: InferenceService = Depends(get_service),
) -> dict:
    """Fit calibration on validation data."""
    try:
        import numpy as np
        service.fit_calibration(
            np.array(train_probs, dtype=np.float32),
            np.array(train_labels, dtype=np.int64),
            np.array(val_probs, dtype=np.float32),
            np.array(val_labels, dtype=np.int64),
        )
        return {"status": "success", "message": "Calibration fitted"}
    except Exception as e:
        logger.exception(f"Fit calibration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/admin/optimize-thresholds",
    status_code=status.HTTP_200_OK,
    summary="Optimize decision thresholds",
    description="Optimize decision thresholds on validation data.",
)
async def optimize_thresholds(
    train_probs: List[float],
    train_labels: List[int],
    val_probs: List[float],
    val_labels: List[int],
    service: InferenceService = Depends(get_service),
) -> dict:
    """Optimize decision thresholds on validation data."""
    try:
        import numpy as np
        service.optimize_thresholds(
            np.array(train_probs, dtype=np.float32),
            np.array(train_labels, dtype=np.int64),
            np.array(val_probs, dtype=np.float32),
            np.array(val_labels, dtype=np.int64),
        )
        return {"status": "success", "message": "Decision thresholds optimized"}
    except Exception as e:
        logger.exception(f"Optimize thresholds error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow(),
        ).model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow(),
        ).model_dump(),
    )