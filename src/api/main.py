from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from typing import List
import numpy as np

app = FastAPI(title="News Classification API")

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
    mlflow.set_tracking_uri("http://mlflow:5000")
    model = mlflow.sklearn.load_model("models:/news_classifier/Production")
    vectorizer = mlflow.sklearn.load_model("models:/news_classifier_vectorizer/Production")

# Add these imports at the top
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Add these metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('api_predictions_total', 'Total predictions by category', ['category'])

# Add a metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Update the predict endpoint to record metrics
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    try:
        # Record request latency
        with REQUEST_LATENCY.labels('POST', '/predict').time():
            # Transform input text
            text_vector = vectorizer.transform([input_data.text])
            
            # Get prediction and probability
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            confidence = float(np.max(probabilities))
            
            # Record prediction category
            PREDICTION_COUNT.labels(prediction).inc()
            
            # Record successful request
            REQUEST_COUNT.labels('POST', '/predict', 200).inc()
            
            return PredictionResponse(
                category=prediction,
                confidence=confidence
            )
    except Exception as e:
        # Record failed request
        REQUEST_COUNT.labels('POST', '/predict', 500).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(input_data: BatchTextInput):
    try:
        # Transform input texts
        text_vectors = vectorizer.transform(input_data.texts)
        
        # Get predictions and probabilities
        predictions = model.predict(text_vectors)
        probabilities = model.predict_proba(text_vectors)
        
        return [
            {
                "category": pred,
                "confidence": float(np.max(prob))
            }
            for pred, prob in zip(predictions, probabilities)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))