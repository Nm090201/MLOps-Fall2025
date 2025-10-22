from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

if __name__ == '__main__':
    # Load dataset
    wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='auto')
    X, y = wine.data, wine.target.astype(float)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate and save
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results = {
        'rmse': float(mse ** 0.5),
        'r2_score': float(r2_score(y_test, y_pred))
    }
    
    joblib.dump(model, 'wine_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training complete! RMSE: {results['rmse']:.4f}, R²: {results['r2_score']:.4f}")