import os, json, argparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import make_classification
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    
    timestamp = args.timestamp
    
    # Load model
    try:
        model = joblib.load(f'spam_model_{timestamp}.joblib')
        print(f"Model loaded: spam_model_{timestamp}.joblib")
    except:
        raise ValueError('Failed to load model')
    
    # Generate test data
    X_test, y_test = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=4,
        n_classes=2,
        random_state=99
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    with open(f'metrics/{timestamp}_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation complete: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")