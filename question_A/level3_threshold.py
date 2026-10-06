import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score



data = pd.read_csv("question_A/diabetes.csv")


X = data.drop("Outcome", axis=1)
y = data["Outcome"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=3012,
    stratify=y
)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LogisticRegression(
    max_iter=1000,
    random_state=3012
)

model.fit(X_train_scaled, y_train)



train_probabilities = model.predict_proba(X_train_scaled)[:, 1]
test_probabilities = model.predict_proba(X_test_scaled)[:, 1]



thresholds = np.arange(0.01, 1.00, 0.01)

selected_threshold = None
selected_train_recall = None

selected_threshold = None
selected_train_recall = None

for threshold in thresholds[::-1]:
    train_predictions = (train_probabilities >= threshold).astype(int)

    recall = recall_score(
        y_train,
        train_predictions,
        zero_division=0
    )

    if recall >= 0.90:
        selected_threshold = threshold
        selected_train_recall = recall
        break


test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nSelected Threshold:", selected_threshold)

print("Training Recall at Selected Threshold:",
      selected_train_recall)

print("\nTest Results at Selected Threshold:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)

print("\nTest Accuracy Percentage:",
      test_accuracy * 100, "%")

print("Test Precision Percentage:",
      test_precision * 100, "%")

print("Test Recall Percentage:",
      test_recall * 100, "%")