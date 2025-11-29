import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
VERSION_FILE_NAME = os.getenv('VERSION_FILE_NAME')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from google.cloud import storage
import joblib
from datetime import datetime

# Function to download the California Housing dataset
def download_data():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    features = pd.DataFrame(housing.data, columns=housing.feature_names)
    target = pd.Series(housing.target)
    return features, target

# Function to split data into train and test sets
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train a Linear Regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to get current model version from GCS
def get_model_version(bucket_name, version_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(version_file_name)
    
    if blob.exists():
        version_as_string = blob.download_as_text()
        version = int(version_as_string)
    else:
        version = 0
    return version

# Function to update model version in GCS
def update_model_version(bucket_name, version_file_name, version):
    if not isinstance(version, int):
        raise ValueError("Version must be an integer")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(version))
        return True
    except Exception as e:
        print(f"Error updating model version: {e}")
        return False

# Function to ensure folder exists in GCS
def ensure_folder_exists(bucket, folder_name):
    blob = bucket.blob(f"{folder_name}/")
    if not blob.exists():
        blob.upload_from_string('')
        print(f"Created folder: {folder_name}")

# Function to save model to GCS
def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    ensure_folder_exists(bucket, "trained_models")
    
    blob = bucket.blob(blob_name)
    blob.upload_from_filename('model.joblib')

# Main function
def main():
    # Get and update version
    current_version = get_model_version(BUCKET_NAME, VERSION_FILE_NAME)
    new_version = current_version + 1
    
    # Download and preprocess data
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train and evaluate model
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Model MSE: {mse:.4f}')
    print(f'Model R² Score: {r2:.4f}')
    
    # Save model with version and timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"trained_models/housing_model_v{new_version}_{timestamp}.joblib"
    save_model_to_gcs(model, BUCKET_NAME, blob_name)
    print(f"Model saved to gs://{BUCKET_NAME}/{blob_name}")
    
    # Update version in GCS
    if update_model_version(BUCKET_NAME, VERSION_FILE_NAME, new_version):
        print(f"Model version updated to {new_version}")
        print(f"MODEL_VERSION_OUTPUT: {new_version}")
    else:
        print("Failed to update model version")

if __name__ == "__main__":
    main()