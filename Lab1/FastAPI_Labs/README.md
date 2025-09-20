# California Housing Price Prediction API – FastAPI  

This project demonstrates how to build a **FastAPI application** that predicts the **median house value** using the **California Housing dataset** from `sklearn`.  
A **Linear Regression** model is trained on the dataset, and the API exposes an endpoint to make predictions based on housing features.  

---

## 📊 Dataset  

- **Source**: `sklearn.datasets.fetch_california_housing()`  
- The dataset contains information on housing in California districts with the following features:  

  - `MedInc` – Median income in block group (in tens of thousands).  
  - `HouseAge` – Median house age in block group.  
  - `AveRooms` – Average number of rooms per household.  
  - `AveBedrms` – Average number of bedrooms per household.  
  - `Population` – Block group population.  
  - `AveOccup` – Average number of household members.  
  - `Latitude` – Block group latitude.  
  - `Longitude` – Block group longitude.  

Target:  
- `MedHouseVal` – Median house value (in $100,000 units).  

---

## 🧠 Model  

- **Algorithm**: Linear Regression (`sklearn.linear_model.LinearRegression`)  
- The model is trained on the California housing dataset.  

---

## 🌐 API Endpoints  

### `POST /predict`  

**Input JSON format:**  

```json
{
  "MedInc": 5.0,
  "HouseAge": 20,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 1000,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
