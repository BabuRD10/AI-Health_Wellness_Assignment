import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent


# Load the dataset from Question A
data = pd.read_csv(
    BASE_DIR / "question_A" / "diabetes.csv"
)


# Separate input features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]


# Use the same train/test split as Question A
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=3012,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=3012
)


# Train the model
model.fit(X_train, y_train)


# Test the model
predictions = model.predict(X_test)


# Calculate performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)


# Display results
print("Random Forest model trained successfully.")

print("\nModel Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# Save the trained model
model_path = BASE_DIR / "question_B" / "diabetes_model.pkl"

joblib.dump(model, model_path)


print("\nSaved model as:")
print(model_path)