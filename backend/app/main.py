import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import torch
import uuid
from datetime import datetime

from .ml.model_adapter import RAFNetAdapter
from .services.risk_engine import RiskEngine
from .schemas.analysis import AnalysisResult
from .db.supabase import supabase
from .websocket.manager import manager
from .core.config import settings
from .core.logging import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production API for RAFNet",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model = RAFNetAdapter(device=settings.DEVICE)
risk_engine = RiskEngine()

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    try:
        content = await file.read()
        audio_np = np.frombuffer(content, dtype=np.float32)
        
        dummy_network = {"packet_loss": 0.01, "jitter": 0.002, "rtt": 0.03, "bitrate": 64000}
        result = model.predict(audio_np, dummy_network)
        
        risk = risk_engine.calculate_risk(
            prediction=result["decision_label"],
            confidence=result["confidence"]
        )
        
        analysis = AnalysisResult(
            id=str(uuid.uuid4()),
            user_id="system_user",
            audio_name=file.filename,
            prediction=result["decision_label"],
            confidence=result["confidence"],
            risk_score=risk["score"],
            model_version=result["model_version"],
            processing_time_ms=result["total_latency_ms"],
            audio_duration_seconds=len(audio_np)/16000,
            created_at=datetime.utcnow()
        )
        
        supabase.save_analysis(analysis.model_dump())
        logger.info(f"Analysis complete: {analysis.id} - {analysis.prediction}")
        return analysis
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    return supabase.get_history("system_user")

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
            await manager.broadcast({
                "event": "heartbeat",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "active"
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
