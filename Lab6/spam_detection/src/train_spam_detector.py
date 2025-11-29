import mlflow, datetime, os, pickle
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    
    timestamp = args.timestamp
    
    # Generate synthetic spam/not-spam data
    X, y = make_classification(
        n_samples=1000,
        n_features=5,  # 5 email features (word counts, links, etc.)
        n_informative=4,
        n_classes=2,
        random_state=42
    )
    
    # Save data
    os.makedirs('data', exist_ok=True)
    with open('data/emails.pickle', 'wb') as f:
        pickle.dump(X, f)
    with open('data/labels.pickle', 'wb') as f:
        pickle.dump(y, f)
    
    # MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    experiment_name = f"Spam_Detection_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    experiment_id = mlflow.create_experiment(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="Spam_Detector"):
        mlflow.log_params({
            "dataset": "Email Spam",
            "emails": X.shape[0],
            "features": X.shape[1]
        })
        
        # Train model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mlflow.log_metrics({
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        })
        
        # Save model
        os.makedirs('models', exist_ok=True)
        dump(model, f'spam_model_{timestamp}.joblib')
        print(f"Model trained and saved: spam_model_{timestamp}.joblib")