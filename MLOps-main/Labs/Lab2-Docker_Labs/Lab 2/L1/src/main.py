# main.py

# ✅ Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

if __name__ == '__main__':
    # 🌸 Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names

    # 🌸 Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🌸 Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 🌸 Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print("✅ Model training complete!\n")
    print(f"🔹 Accuracy on test set: {accuracy * 100:.2f}%\n")
    print("🔹 Confusion Matrix:")
    print(conf_matrix, "\n")
    print("🔹 Classification Report:")
    print(report)

    # 🌸 Make a prediction for a sample
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)
    predicted_class = class_names[prediction[0]]
    print(f"🌸 Predicted class for sample {sample} → {predicted_class}\n")

    # 🌸 Save the model
    joblib.dump(model, 'iris_model.pkl')
    print("Model saved as 'iris_model.pkl'")
