# predict.py
import pickle
import numpy as np

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
target_names = data["target_names"]

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_idx = model.predict(features)[0]
    return target_names[pred_idx]
