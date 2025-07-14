from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from typing import List
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

app = FastAPI(title="News Classification API")

# Define Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('api_predictions_total', 'Total predictions by category', ['category'])

# CRITICAL FIX: Initialize global variables to prevent "name not defined" errors
model = None
vectorizer = None

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    category: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, vectorizer
    print("Starting model loading...")
    
    mlflow.set_tracking_uri("http://mlflow:5001")
    try:
        print("Attempting to load model from MLflow...")
        model = mlflow.sklearn.load_model("models:/news_classifier/Production")
        vectorizer = mlflow.sklearn.load_model("models:/news_classifier_vectorizer/Production")
        print("Model and vectorizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Setting model and vectorizer to None - predictions will return 503")
        # CRITICAL FIX: Explicitly set to None when loading fails
        model = None
        vectorizer = None
    
    print("Startup complete")

@app.get("/")
async def root():
    return {"message": "News Classification API is running"}

@app.get("/health")
async def health():
    # Enhanced health check showing model status
    model_status = "loaded" if model is not None and vectorizer is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "service": "news-classification-api"
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    start_time = time.time()
    
    # CRITICAL FIX: Check if model is loaded before using
    if model is None or vectorizer is None:
        REQUEST_COUNT.labels('POST', '/predict', 503).inc()
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. MLflow connection failed during startup. Check MLflow service."
        )
    
    try:
        # Transform input text
        text_vector = vectorizer.transform([input_data.text])
        # Get prediction and probability
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        confidence = float(np.max(probabilities))
        
        # Record metrics
        REQUEST_COUNT.labels('POST', '/predict', 200).inc()
        REQUEST_LATENCY.labels('POST', '/predict').observe(time.time() - start_time)
        PREDICTION_COUNT.labels(prediction).inc()
        
        return PredictionResponse(
            category=prediction,
            confidence=confidence
        )
    except Exception as e:
        REQUEST_COUNT.labels('POST', '/predict', 500).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(input_data: BatchTextInput):
    start_time = time.time()
    
    # CRITICAL FIX: Check if model is loaded before using
    if model is None or vectorizer is None:
        REQUEST_COUNT.labels('POST', '/batch-predict', 503).inc()
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. MLflow connection failed during startup. Check MLflow service."
        )
    
    try:
        # Transform input texts
        text_vectors = vectorizer.transform(input_data.texts)
        # Get predictions and probabilities
        predictions = model.predict(text_vectors)
        probabilities = model.predict_proba(text_vectors)
        
        # Record metrics
        REQUEST_COUNT.labels('POST', '/batch-predict', 200).inc()
        REQUEST_LATENCY.labels('POST', '/batch-predict').observe(time.time() - start_time)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            confidence = float(np.max(prob))
            PREDICTION_COUNT.labels(pred).inc()
            results.append({
                "category": pred,
                "confidence": confidence
            })
        
        return results
    except Exception as e:
        REQUEST_COUNT.labels('POST', '/batch-predict', 500).inc()
        raise HTTPException(status_code=500, detail=str(e))