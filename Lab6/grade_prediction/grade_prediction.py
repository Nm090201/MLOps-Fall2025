import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from google.cloud import storage
import joblib
from datetime import datetime

# Generate simple student data
def download_data():
    """Create synthetic student grade data"""
    from sklearn.datasets import make_regression
    
    # Create simple dataset: study hours -> grade
    X, y = make_regression(
        n_samples=200,
        n_features=3,
        noise=10,
        random_state=42
    )
    
    # Scale to realistic values
    y = (y - y.min()) / (y.max() - y.min()) * 100  # Grades 0-100
    
    feature_names = ['study_hours', 'attendance_rate', 'previous_grade']
    features = pd.DataFrame(X, columns=feature_names)
    target = pd.Series(y, name='final_grade')
    
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
def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "grade_model.joblib")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename('grade_model.joblib')

# Main pipeline
def main():
    # Get data
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    
    # Save to cloud
    bucket_name = "github-action-labb"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"grade_models/model_{timestamp}.joblib"
    
    save_model_to_gcs(model, bucket_name, blob_name)
    print(f"Model saved to gs://{bucket_name}/{blob_name}")

if __name__ == "__main__":
    main()
