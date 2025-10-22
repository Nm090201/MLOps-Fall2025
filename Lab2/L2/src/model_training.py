import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == '__main__':
    wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='auto')
    X, y = wine.data, wine.target.astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape=(11,), activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    model.save('wine_model.keras')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model saved!")