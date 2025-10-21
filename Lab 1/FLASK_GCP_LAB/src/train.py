# train.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and target names
with open("iris_model.pkl", "wb") as f:
    pickle.dump({"model": model, "target_names": iris.target_names}, f)

print("Model trained and saved as iris_model.pkl")
