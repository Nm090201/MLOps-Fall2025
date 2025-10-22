import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Base paths
DATA_PATH = Path("/opt/airflow/data/advertising.csv")
MODEL_PATH = Path("/opt/airflow/model")


def load_data():
    """Load advertising data from CSV."""
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def data_preprocessing(data):
    """
    Preprocess data: select features, split, and scale.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Preprocessing data...")
    
    # Select numeric features and target
    feature_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 
                    'Daily Internet Usage', 'Male']
    
    X = data[feature_cols]
    y = data['Clicked on Ad']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


def build_model(data, filename="model.pkl"):
    """
    Train logistic regression model and save to disk.
    
    Args:
        data: tuple of (X_train, X_test, y_train, y_test)
        filename: name of the model file
    """
    X_train, X_test, y_train, y_test = data
    
    print("Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save model
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_file = MODEL_PATH / filename
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_file}")
    return test_score


def load_model(filename="model.pkl"):
    """Load a saved model from disk."""
    model_file = MODEL_PATH / filename
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_file}")
    return model


if __name__ == '__main__':
    # Test the pipeline
    data = load_data()
    processed_data = data_preprocessing(data)
    build_model(processed_data)
    print("Pipeline completed successfully!")