import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import base64
import os


def load_data():
    df = pd.read_csv("/opt/airflow/data/file.csv")
    df = df.dropna()
    data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    serialized = pickle.dumps(data)
    return base64.b64encode(serialized).decode('ascii')


def scale_features(data_b64):
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    artifacts = pickle.dumps({"scaled_data": scaled_data, "scaler": scaler})
    return base64.b64encode(artifacts).decode('ascii')


def find_optimal_clusters(data_b64):
    data_bytes = base64.b64decode(data_b64)
    artifacts = pickle.loads(data_bytes)
    scaled_data = artifacts["scaled_data"]
    scaler = artifacts["scaler"]
    sse = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow if kl.elbow else 3
    print(f"Optimal clusters: {optimal_k}")
    result = pickle.dumps({"optimal_k": optimal_k, "scaled_data": scaled_data, "scaler": scaler})
    return base64.b64encode(result).decode('ascii')


def train_and_save_model(data_b64, filename):
    data_bytes = base64.b64decode(data_b64)
    artifacts = pickle.loads(data_bytes)
    optimal_k = artifacts["optimal_k"]
    scaled_data = artifacts["scaled_data"]
    scaler = artifacts["scaler"]
    model = KMeans(n_clusters=optimal_k, init="random", n_init=10, max_iter=300, random_state=42)
    model.fit(scaled_data)
    output_path = f"/opt/airflow/model/{filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({"model": model, "scaler": scaler, "optimal_k": optimal_k}, f)
    print(f"Model saved with {optimal_k} clusters to {output_path}")
    return filename


def predict_test_data(filename):
    model_path = f"/opt/airflow/model/{filename}"
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    optimal_k = artifacts["optimal_k"]
    test_df = pd.read_csv("/opt/airflow/data/test.csv")
    test_scaled = scaler.transform(test_df)
    prediction = model.predict(test_scaled)[0]
    print(f"Test prediction: Cluster {prediction} (out of {optimal_k} clusters)")
    return int(prediction)