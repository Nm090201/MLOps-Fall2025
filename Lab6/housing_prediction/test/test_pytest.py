import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from unittest.mock import patch, MagicMock
from src.train_housing_model import download_data, preprocess_data, train_model
from src.train_housing_model import get_model_version, update_model_version
from src.train_housing_model import ensure_folder_exists, save_model_to_gcs
from google.cloud import storage


# ----------------- Test Download ----------------- #
# Test the download_data function to ensure it correctly downloads and returns data
def test_download_data():
    X, y = download_data()
    
    # Check if the data is downloaded correctly and matches expected formats
    assert isinstance(X, pd.DataFrame)  # X should be a DataFrame
    assert isinstance(y, pd.Series)     # y should be a Series
    assert not X.empty                  # X should not be empty
    assert not y.empty                  # y should not be empty
    assert X.shape[0] == y.shape[0]     # The number of rows in X and y should be the same

# ----------------- Test Preprocess ----------------- #
# Test the preprocess_data function to ensure it correctly preprocesses the data
def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Assert that the preprocessing splits the data correctly
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]  # Rows in train and test should total original rows
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]  # Rows in train and test labels should total original labels
    assert X_train.shape[1] == X.shape[1]  # Number of features should remain the same

# ----------------- Test Train model ----------------- #
# Test the train_model function to ensure it correctly trains the model
def test_train_model():
    # Generate sample data for testing (housing features)
    X = pd.DataFrame({
        'MedInc': [8.3, 8.3, 7.3, 5.6, 3.8],
        'HouseAge': [41.0, 21.0, 52.0, 52.0, 52.0],
        'AveRooms': [6.98, 6.24, 8.29, 5.82, 4.98],
        'AveBedrms': [1.02, 0.97, 1.07, 1.07, 1.04],
        'Population': [322.0, 2401.0, 496.0, 558.0, 565.0],
        'AveOccup': [2.55, 2.11, 2.80, 2.55, 2.18],
        'Latitude': [37.88, 37.86, 37.85, 37.85, 37.85],
        'Longitude': [-122.23, -122.22, -122.24, -122.25, -122.25]
    })
    y = pd.Series([4.526, 3.585, 3.521, 3.413, 3.422])
    
    # Train the model using the sample data
    model = train_model(X, y)
    
    # Assertions to verify the model is trained correctly
    assert isinstance(model, LinearRegression)  # Check if the returned model is of the correct type
    assert hasattr(model, 'predict')            # Ensure the model has a predict method

# ----------------- Test Model versioning ----------------- #
# This function tests the get_model_version function responsible for retrieving the version of the model stored in Google Cloud Storage.
def test_get_model_version():
    # Patch the GCP storage client to prevent actual network operations during the test.
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()  # Create a mock bucket object.
        mock_blob = MagicMock()    # Create a mock blob object to represent the file in the storage.

        # Configure mock objects to return other mocks when methods are called.
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Set the test inputs for actual function calls
        bucket_name = "bucket-test"
        version_file_name = "version.txt"

        # Simulate the scenario where the version file exists in the storage
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = '1'  # Simulate blob returning version '1' as text
        version = get_model_version(bucket_name, version_file_name)

        # Check if the correct version is retrieved and the corresponding methods are called on the mock
        assert version == 1
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_called_once()
        
        # Reset mocks to clear call history before the next test
        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()
        
        # Test scenario where the version file does not exist
        mock_blob.exists.return_value = False
        version = get_model_version(bucket_name, version_file_name)

        # Ensure it handles the absence of the version file correctly
        assert version == 0
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_not_called()

# ----------------- Test Update Model version ----------------- #
# This function tests the update_model_version function that updates the version of the model stored in Google Cloud Storage.
def test_update_model_version():
    # Patch the GCP storage client to prevent actual network operations during the test.
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()  # Create a mock bucket object.
        mock_blob = MagicMock()    # Create a mock blob object to represent the file in the storage.

        # Configure mock objects to return other mocks when methods are called.
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        bucket_name = 'bucket-test'
        version_file_name = 'version.txt'
        new_version = 2
        
        # Test successful update of the model version
        result = update_model_version(bucket_name, version_file_name, new_version)
        # Assert the function returns true indicating success
        assert result == True
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))
        
        # Reset mocks to clear call history for further tests
        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()
        
        # Test error handling with an invalid version (not an integer)
        with pytest.raises(ValueError):
            update_model_version(bucket_name, version_file_name, 'invalid_version')
        
        # Simulate an exception during the blob upload to test error handling
        mock_blob.upload_from_string.side_effect = Exception("Upload failed")
        result = update_model_version(bucket_name, version_file_name, new_version)
        # Assert the function returns false indicating failure
        assert result == False
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))


# ----------------- Test Ensure Folder Exists ----------------- #
# Test ensure_folder_exists function to verify it correctly ensures the presence of a folder in the storage
def test_ensure_folder_exists():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        folder_name = "trained_models"
        
        # When folder does not exist
        mock_blob.exists.return_value = False
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_called_once_with('')
        
        # Reset the mock for the next test
        mock_blob.reset_mock()
        
        # When folder exists
        mock_blob.exists.return_value = True
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_not_called()

# ----------------- Test Save model to GCS ----------------- #
# This function tests the 'save_model_to_gcs' function to ensure it correctly saves a model to Google Cloud Storage.
def test_save_model_to_gcs():
    # Create a mock LinearRegression model for testing purposes. This represents the model that you want to save.
    model = LinearRegression()
    
    # 'patch' is used to temporarily replace the 'google.cloud.storage.Client' class with a mock, so that no real network operations are performed.
    with patch('google.cloud.storage.Client') as mock_storage_client:
        # Create a mock bucket object. This mock simulates the bucket where the model will be stored in GCS.
        mock_bucket = MagicMock()
        # Create a mock blob object. This simulates the file or object within the GCS bucket.
        mock_blob = MagicMock()
        
        # Set up the return values for when the storage client, bucket, and blob are called.
        # This ensures that when the save_model_to_gcs function tries to interact with GCS, it uses these mock objects instead.
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Configure the mock blob to simulate a scenario where the blob does not already exist in the GCS bucket.
        mock_blob.exists.return_value = False
        
        # Call the function 'save_model_to_gcs' with the mock model and specific bucket and blob names.
        # This is the function you are testing, which should perform the actual saving of the model.
        save_model_to_gcs(model, 'bucket-test', 'housing_model_blob-test')
        
        # Ensure the storage client was initialized exactly once.
        mock_storage_client.assert_called_once()
        # Check that the bucket was fetched exactly once with the specified name.
        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        
        # Assert that the blob method was called exactly twice (once for checking existence, once for uploading).
        # This line checks that the blob method is called with the correct parameters at least once.
        assert mock_bucket.blob.call_count == 2
        # These calls ensure that both blob invocations were with the expected names.
        mock_bucket.blob.assert_any_call('trained_models/')
        mock_bucket.blob.assert_any_call('housing_model_blob-test')
        # Verify that the blob's 'upload_from_filename' method was called once with the filename 'model.joblib' to upload the model.
        mock_blob.upload_from_filename.assert_called_once_with('model.joblib')