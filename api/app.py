import os
import json
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Posture Correction API",
    description="Predicts good vs bad posture from upper-body landmark features. "
                "Uses an XGBoost model exported to ONNX for fast inference.",
    version="1.0.0",
)

# Globals loaded at startup
session = None
feature_names = None
input_name = None
model_info = None


@app.on_event("startup")
def load_model():
    """Load ONNX model and feature names when the server starts."""
    global session, feature_names, input_name, model_info

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "posture_model.onnx")
    features_path = os.path.join(base_dir, "models", "feature_names.json")
    summary_path = os.path.join(base_dir, "models", "training_summary.json")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Run training first.")

    if not os.path.exists(features_path):
        raise RuntimeError(f"Feature names not found at {features_path}. Run training first.")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(features_path) as f:
        feature_names = json.load(f)

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            model_info = json.load(f)

    print(f"Model loaded successfully.")
    print(f"Features ({len(feature_names)}): {feature_names}")


class PostureInput(BaseModel):
    """Input: dictionary of feature name to float value."""
    features: Dict[str, float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "neck_incl_L": 20.5,
                    "neck_incl_R": 21.3,
                    "neck_incl_avg": 20.9,
                    "head_forward_z": -0.82,
                    "nose_above_shoulder": 0.35,
                    "shoulder_y_diff": 0.02,
                    "shoulder_width": 0.45,
                    "ear_y_diff": 0.01,
                    "ear_shoulder_ratio_L": 0.93,
                    "ear_shoulder_ratio_R": 0.94,
                    "head_droop_L": 122.0,
                    "head_droop_R": 119.0,
                    "eye_ear_y_diff": -0.03,
                }
            }
        }


class PostureBatchInput(BaseModel):
    """Input: list of feature dictionaries for batch prediction."""
    samples: list[Dict[str, float]]


class PostureOutput(BaseModel):
    prediction: int
    label: str
    confidence: float
    latency_ms: float


class PostureBatchOutput(BaseModel):
    predictions: list[PostureOutput]
    total_latency_ms: float
    avg_latency_ms: float


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    model_accuracy: float = None
    model_f1: float = None


@app.get("/health", response_model=HealthOutput)
def health():
    """Check if the API and model are ready."""
    return HealthOutput(
        status="healthy" if session is not None else "not ready",
        model_loaded=session is not None,
        n_features=len(feature_names) if feature_names else 0,
        model_accuracy=model_info.get("test_acc") if model_info else None,
        model_f1=model_info.get("test_f1") if model_info else None,
    )


@app.get("/features")
def get_features():
    """Return the list of expected feature names."""
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"feature_names": feature_names, "n_features": len(feature_names)}


@app.post("/predict", response_model=PostureOutput)
def predict(input_data: PostureInput):
    """Predict posture from a single set of features."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate all features are present
    missing = [f for f in feature_names if f not in input_data.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}",
        )

    # Build input array in correct order
    values = [input_data.features[name] for name in feature_names]
    input_array = np.array([values], dtype=np.float32)

    # Inference with timing
    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_array})
    latency_ms = (time.perf_counter() - start) * 1000

    prediction = int(outputs[0][0])

    # Get confidence
    confidence = 1.0
    if len(outputs) > 1:
        probs = outputs[1]
        if isinstance(probs, list):
            confidence = float(probs[0].get(prediction, 1.0))
        elif isinstance(probs, np.ndarray) and probs.ndim == 2:
            confidence = float(probs[0][prediction])

    return PostureOutput(
        prediction=prediction,
        label="good" if prediction == 1 else "bad",
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=PostureBatchOutput)
def predict_batch(input_data: PostureBatchInput):
    """Predict posture for multiple samples at once."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(input_data.samples) == 0:
        raise HTTPException(status_code=400, detail="Empty sample list")

    if len(input_data.samples) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 samples per batch")

    # Validate features in first sample
    missing = [f for f in feature_names if f not in input_data.samples[0]]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features in sample 0: {missing}",
        )

    # Build batch input
    batch = []
    for sample in input_data.samples:
        values = [sample[name] for name in feature_names]
        batch.append(values)
    input_array = np.array(batch, dtype=np.float32)

    # Inference
    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_array})
    total_latency = (time.perf_counter() - start) * 1000

    predictions_raw = outputs[0].flatten()

    # Get probabilities
    probs_raw = None
    if len(outputs) > 1:
        probs_raw = outputs[1]

    # Build response
    results = []
    for i in range(len(input_data.samples)):
        pred = int(predictions_raw[i])

        confidence = 1.0
        if probs_raw is not None:
            if isinstance(probs_raw, list):
                confidence = float(probs_raw[i].get(pred, 1.0))
            elif isinstance(probs_raw, np.ndarray) and probs_raw.ndim == 2:
                confidence = float(probs_raw[i][pred])

        results.append(PostureOutput(
            prediction=pred,
            label="good" if pred == 1 else "bad",
            confidence=round(confidence, 4),
            latency_ms=round(total_latency / len(input_data.samples), 3),
        ))

    return PostureBatchOutput(
        predictions=results,
        total_latency_ms=round(total_latency, 3),
        avg_latency_ms=round(total_latency / len(input_data.samples), 3),
    )