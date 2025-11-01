"""
Simple Wine Quality Classification with MLflow
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ========== LOAD DATA ==========
print("Loading data...")
white = pd.read_csv("data/winequality-white.csv", sep=";")
red = pd.read_csv("data/winequality-red.csv", sep=",")

red['is_red'] = 1
white['is_red'] = 0
data = pd.concat([red, white])
data.columns = data.columns.str.replace(' ', '_')

# High quality = score >= 7
data['quality'] = (data.quality >= 7).astype(int)
print(f"✓ Loaded {len(data)} samples")

# ========== SPLIT DATA ==========
X = data.drop('quality', axis=1)
y = data.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ========== TRAIN MODEL ==========
print("\nTraining model...")
mlflow.set_experiment("wine-quality")

with mlflow.start_run():
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, predictions)
    
    # Log to MLflow
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("auc", auc)
    mlflow.sklearn.log_model(model, "model")
    
    run_id = mlflow.active_run().info.run_id
    print(f"✓ Model AUC: {auc:.4f}")
    print(f"✓ Run ID: {run_id}")

# ========== REGISTER MODEL ==========
print("\nRegistering model...")
model_name = "wine_quality"
mlflow.register_model(f"runs:/{run_id}/model", model_name)
print(f"✓ Model registered as '{model_name}'")

# ========== TEST LOADING ==========
print("\nTesting model loading...")
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
test_auc = roc_auc_score(y_test, loaded_model.predict_proba(X_test)[:,1])
print(f"✓ Loaded model AUC: {test_auc:.4f}")

print("\n" + "="*50)
print("DONE! 🎉")
print("="*50)
print("View results: mlflow ui --port 5001")
print(f"Serve model: mlflow models serve -m runs:/{run_id}/model -p 5002")
print("="*50)