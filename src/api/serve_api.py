from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="News Classification API")

# Global variables
model = None
vectorizer = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float
    model_type: str

@app.on_event("startup")
async def load_model():
    global model, vectorizer

    try:
        # Define paths
        model_path = Path("/app/models/neural_network.pkl")
        vectorizer_path = Path("/app/models/vectorizer.pkl")

        # Fallback to local folder if needed
        if not model_path.exists():
            model_path = Path("models/neural_network.pkl")
        if not vectorizer_path.exists():
            vectorizer_path = Path("models/vectorizer.pkl")

        # Load model
        with model_path.open("rb") as f:
            model = pickle.load(f)
        logger.info(f"✅ Loaded Neural Network model from {model_path}")

        # Load vectorizer
        with vectorizer_path.open("rb") as f:
            vectorizer = pickle.load(f)
        logger.info(f"✅ Loaded vectorizer from {vectorizer_path}")

    except Exception as e:
        logger.error(f"❌ Failed to load model or vectorizer: {e}")

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model and vectorizer else "degraded",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_type": type(model).__name__ if model else None,
        "accuracy": "55.2%"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded")

    try:
        # Vectorize input
        text_vector = vectorizer.transform([request.text])

        # Predict
        prediction = model.predict(text_vector)[0]

        # The model returns string labels directly!
        # Just clean it up
        category = str(prediction).lower().strip()
        
        # Handle variations in category names
        category_mapping = {
            'sports': 'sport',
            'technology': 'tech',
            'wellness': 'health',
            'health': 'health'
        }
        
        category = category_mapping.get(category, category)

        # Confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vector)[0]
            confidence = float(max(probs))
        else:
            # Default confidence based on your model's accuracy
            confidence = 0.552

        return PredictionResponse(
            category=category,
            confidence=confidence,
            model_type="neural_network"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "News Classification API",
        "model_status": "Neural Network (55.2% accuracy)" if model else "Not loaded",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions", 
            "/docs": "API documentation"
        }
    }

@app.get("/models")
async def list_models():
    """List available models"""
    import os
    models_dir = Path("/app/models")
    if models_dir.exists():
        files = os.listdir(models_dir)
        return {
            "models_directory": str(models_dir),
            "files": files,
            "loaded_model": "neural_network.pkl" if model else None
        }
    return {"error": "Models directory not found"}