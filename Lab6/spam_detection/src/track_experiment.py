import mlflow
import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import os
from joblib import dump

# Setup
mlflow.set_tracking_uri("./mlruns")
experiment_name = f"Spam_Experiment_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
experiment_id = mlflow.create_experiment(experiment_name)

# Generate data
X, y = make_classification(n_samples=800, n_features=5, n_classes=2, random_state=42)

with mlflow.start_run(experiment_id=experiment_id, run_name="Logistic_Regression"):
    
    # Log parameters
    mlflow.log_params({
        "dataset": "Email Spam",
        "samples": X.shape[0],
        "features": X.shape[1]
    })
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    })
    
    # Save model
    os.makedirs('models', exist_ok=True)
    dump(model, f'models/{experiment_id}.joblib')
    
    print(f"Experiment logged with ID: {experiment_id}")