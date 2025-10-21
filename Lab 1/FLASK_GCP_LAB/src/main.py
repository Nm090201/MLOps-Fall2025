# main.py
from flask import Flask, request, jsonify
from predict import predict_iris
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "Iris Prediction API is running! Use POST /predict to get predictions."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Send a POST request with JSON to get predictions."

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    try:
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Invalid input, must provide sepal_length, sepal_width, petal_length, petal_width'}), 400

    prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
