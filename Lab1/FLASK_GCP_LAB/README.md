# Iris Flower Prediction API – Flask  

This project demonstrates how to build a simple **Flask API** for predicting the species of an Iris flower using the **Iris dataset** from `sklearn`.  
The model is trained with a **RandomForestClassifier**, and the API exposes an endpoint to get predictions based on input flower measurements.  

---

## 📊 Dataset  

- **Source**: `sklearn.datasets.load_iris()`  
- The dataset contains 150 samples of iris flowers with 4 features:  
  - `sepal_length`  
  - `sepal_width`  
  - `petal_length`  
  - `petal_width`  

Target labels correspond to flower species:  
- **0 → setosa**  
- **1 → versicolor**  
- **2 → virginica**  

---

## 🧠 Model  

- **Algorithm**: RandomForestClassifier (from `sklearn.ensemble`)  
- The model is trained on the Iris dataset and serialized for use in the API.  

---

## 🌐 API Endpoints  

### `POST /predict`  

**Input JSON format:**  

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
