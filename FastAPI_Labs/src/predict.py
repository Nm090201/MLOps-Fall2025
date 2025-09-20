import joblib

def predict_data(X):
    """
    Predict the median house value for the input data.
    Args:
        X (list or numpy.ndarray): Input features for prediction 
                                   [[MedInc, HouseAge, AveRooms, AveBedrms,
                                     Population, AveOccup, Latitude, Longitude]]
    Returns:
        y_pred (numpy.ndarray): Predicted house values.
    """
    # Load trained California Housing model
    model = joblib.load("../model/california_model.pkl")
    
    # Make prediction
    y_pred = model.predict(X)
    return y_pred
