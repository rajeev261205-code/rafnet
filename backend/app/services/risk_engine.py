from typing import Dict
import numpy as np

class RiskEngine:
    def __init__(self, low_thresh=0.3, high_thresh=0.7):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def calculate_risk(self, prediction: str, confidence: float, anomalies: bool = False) -> Dict:
        # Prediction base weight
        weights = {"REAL": 0, "UNCERTAIN": 50, "SYNTHETIC": 100}
        base_score = weights.get(prediction, 50) * confidence
        
        # Add penalty for telemetry anomalies
        penalty = 15 if anomalies else 0
        final_score = min(100, base_score + penalty)
        
        # Determine level
        if final_score < 20: level = "LOW"
        elif final_score < 40: level = "MEDIUM"
        elif final_score < 60: level = "MEDIUM"
        elif final_score < 80: level = "HIGH"
        else: level = "CRITICAL"
        
        return {
            "score": int(final_score),
            "level": level
        }
