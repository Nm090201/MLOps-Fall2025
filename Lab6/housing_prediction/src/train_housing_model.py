import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from google.cloud import storage
import joblib
from datetime import datetime

# Generate simple housing data
def download_data():
    """Load California Housing dataset"""
    from sklearn.datasets import fetch_california_housing
    
    housing = fetch_california_housing()
    features = pd.DataFrame(housing.data, columns=housing.feature_names)
    target = pd.Series(housing.target, name='house_price')
    
    return features, target

# Split the data
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Train simple model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Save to GCS
def save_model_to_gcs(model, bucket_name, blob_name, project_id):
    try:
        joblib.dump(model, "housing_model.joblib")
        print(f"Model saved locally as housing_model.joblib")
        
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        # Check if bucket exists
        if not bucket.exists():
            raise ValueError(f"Bucket {bucket_name} does not exist in project {project_id}!")
        
        blob = bucket.blob(blob_name)
        blob.upload_from_filename('housing_model.joblib')
        print(f"Model uploaded successfully to gs://{bucket_name}/{blob_name}")
        
    except Exception as e:
        print(f"Error saving model to GCS: {e}")
        raise

# Main pipeline
def main():
    # Get data
    print("Downloading data...")
    X, y = download_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Preprocess
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    
    # Save to cloud
    print("Saving model to GCS...")
    project_id = "github-actions-ml-pipeline"
    bucket_name = "github-action-labb"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"housing_models/model_{timestamp}.joblib"
    
    save_model_to_gcs(model, bucket_name, blob_name, project_id)
    print(f"✓ Model saved to gs://{bucket_name}/{blob_name}")
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
