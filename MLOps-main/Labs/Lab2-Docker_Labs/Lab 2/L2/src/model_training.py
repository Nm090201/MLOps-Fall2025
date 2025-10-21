# model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# --- Generate synthetic dataset ---
np.random.seed(42)
n_samples = 200

# Features: area (sq ft), bedrooms, age of house
area = np.random.randint(800, 4000, n_samples)
bedrooms = np.random.randint(1, 5, n_samples)
age = np.random.randint(1, 30, n_samples)

# Target: price (in thousands)
price = (area * 0.3) + (bedrooms * 50) - (age * 10) + np.random.randint(2000, 10000, n_samples)
price = price.astype(float)

# Create DataFrame
data = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "age": age,
    "price": price
})

# --- Train/Test Split ---
X = data[["area", "bedrooms", "age"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate ---
score = model.score(X_test, y_test)
print(f"Model trained successfully! R² score: {score:.3f}")

# --- Save Model ---
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'house_price_model.pkl'")
