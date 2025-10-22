"""
FastAPI ML Prediction Service
Minimal version with basic UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import sys
sys.path.insert(0, '/app/src')
from model import predict
import uvicorn

app = FastAPI(title="ML Prediction API", version="1.0")

# Request/Response models
class PredictionInput(BaseModel):
    daily_time_spent: float = Field(..., ge=0, example=68.95)
    age: int = Field(..., ge=18, le=100, example=35)
    area_income: float = Field(..., ge=0, example=61833.90)
    daily_internet_usage: float = Field(..., ge=0, example=256.09)
    male: int = Field(..., ge=0, le=1, example=0)

class PredictionOutput(BaseModel):
    prediction: int
    will_click: bool
    confidence: str
    message: str

# Basic HTML UI
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Prediction API</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        pre { background: #333; color: #0f0; padding: 15px; border-radius: 5px; overflow-x: auto; }
        a { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <h1>ML Prediction API</h1>
    <p>Ad Click Prediction Service</p>
    
    <h2>Endpoints:</h2>
    
    <div class="endpoint">
        <strong>GET /</strong> - This page
    </div>
    
    <div class="endpoint">
        <strong>GET /health</strong> - Health check
    </div>
    
    <div class="endpoint">
        <strong>POST /predict</strong> - Make prediction
    </div>
    
    <div class="endpoint">
        <strong>GET /docs</strong> - Interactive API documentation
    </div>
    
    <h2>Example Request:</h2>
    <pre>curl -X POST "http://localhost:5555/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "daily_time_spent": 68.95,
    "age": 35,
    "area_income": 61833.90,
    "daily_internet_usage": 256.09,
    "male": 0
  }'</pre>
    
    <p><a href="/docs">→ Go to Interactive Docs</a></p>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    """Main page"""
    return HTML

@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy", "service": "ml-api"}

@app.post("/predict", response_model=PredictionOutput)
def make_prediction(input_data: PredictionInput):
    """Make prediction"""
    try:
        features = [
            input_data.daily_time_spent,
            input_data.age,
            input_data.area_income,
            input_data.daily_internet_usage,
            input_data.male
        ]
        
        prediction, probability = predict(features)
        
        return {
            "prediction": prediction,
            "will_click": bool(prediction),
            "confidence": f"{probability:.1%}",
            "message": "Will click ad" if prediction else "Will not click ad"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5555)