"""
Iris flower classification using Logistic Regression and MLflow tracking.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import sys

def train_model(C=1.0, max_iter=100):
    """Train and log iris classification model."""
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run():
        # Train model
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log to MLflow
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Save model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        print(f"Logistic Regression (C={C}, max_iter={max_iter})")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        
        return model

if __name__ == "__main__":
    mlflow.set_experiment("iris-classification")
    
    # Get params from command line or use defaults
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    train_model(C, max_iter)