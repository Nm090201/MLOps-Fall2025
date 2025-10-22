"""
ML Model Training and Prediction
Simple logistic regression for ad click prediction
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = '/opt/airflow/data/advertising.csv'
MODEL_PATH = '/opt/airflow/model/model.pkl'
SCALER_PATH = '/opt/airflow/model/scaler.pkl'

def train_pipeline():
    """Train model and save to disk"""
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape}")
    
    # Select features
    features = ['Daily Time Spent on Site', 'Age', 'Area Income', 
                'Daily Internet Usage', 'Male']
    X = df[features]
    y = df['Clicked on Ad']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    # Save
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    pickle.dump(scaler, open(SCALER_PATH, 'wb'))
    print("Model saved!")
    
    return {'accuracy': accuracy, 'f1_score': f1}

def predict(features):
    """Make prediction on new data"""
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return int(prediction), float(probability)

if __name__ == '__main__':
    train_pipeline()