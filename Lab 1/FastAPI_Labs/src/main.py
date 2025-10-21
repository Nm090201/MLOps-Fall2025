from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

# California housing dataset has 8 features:
# 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
# 'Population', 'AveOccup', 'Latitude', 'Longitude'
class CaliforniaHousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class CaliforniaHousingResponse(BaseModel):
    response: float  # housing price prediction is continuous

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=CaliforniaHousingResponse)
async def predict_california(data: CaliforniaHousingData):
    try:
        features = [[
            data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
            data.Population, data.AveOccup, data.Latitude, data.Longitude
        ]]

        prediction = predict_data(features)  # should return array-like
        return CaliforniaHousingResponse(response=float(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
