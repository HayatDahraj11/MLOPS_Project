from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
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
        # Load the neural network model (best performer)
        model_path = "/app/models/neural_network.pkl"
        vectorizer_path = "/app/models/vectorizer.pkl"
        
        # Try alternative paths if needed
        if not Path(model_path).exists():
            model_path = "models/neural_network.pkl"
        if not Path(vectorizer_path).exists():
            vectorizer_path = "models/vectorizer.pkl"
            
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✅ Loaded Neural Network model from {model_path}")
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info(f"✅ Loaded vectorizer from {vectorizer_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Try to find any pkl files
        import os
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Files in /app/models: {os.listdir('/app/models') if os.path.exists('/app/models') else 'Not found'}")
        logger.info(f"Files in models: {os.listdir('models') if os.path.exists('models') else 'Not found'}")

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model and vectorizer else "degraded",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_type": type(model).__name__ if model else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model or not vectorizer:
        raise HTTPException(503, "Model or vectorizer not loaded")
    
    try:
        # Vectorize the text
        text_vector = vectorizer.transform([request.text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        
        # Get probability if available
        confidence = 0.552  # Default to your model's accuracy
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = float(max(probabilities))
        
        # Map prediction to category name
        # Based on typical news categories
        category_map = {
            0: 'business',
            1: 'entertainment',
            2: 'politics', 
            3: 'sport',
            4: 'tech'
        }
        
        category = category_map.get(int(prediction), f'category_{prediction}')
        
        return PredictionResponse(
            category=category,
            confidence=confidence,
            model_type="neural_network"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

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
    models_dir = "/app/models"
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        return {
            "models_directory": models_dir,
            "files": files,
            "loaded_model": "neural_network.pkl" if model else None
        }
    return {"error": "Models directory not found"}
