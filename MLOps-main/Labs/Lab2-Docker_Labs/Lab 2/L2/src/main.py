# main.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = "house_price_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found! Please train it first.")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        bedrooms = float(request.form["bedrooms"])
        age = float(request.form["age"])

        features = np.array([[area, bedrooms, age]])
        prediction = model.predict(features)[0]
        return render_template("index.html", prediction_text=f"Estimated House Price: ${prediction:,.2f}")
    except Exception as e:
        return jsonify({"error": str(e)})

# Optional: REST API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    features = np.array([[data["area"], data["bedrooms"], data["age"]]])
    prediction = model.predict(features)[0]
    return jsonify({"predicted_price": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
