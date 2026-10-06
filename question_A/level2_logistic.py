import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split



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



X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))



mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std



def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def compute_loss(y, predictions):
    epsilon = 1e-15

    predictions = np.clip(
        predictions,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y * np.log(predictions)
        + (1 - y) * np.log(1 - predictions)
    )

    return loss



def train_logistic_regression(
    X,
    y,
    learning_rate=0.01,
    epochs=5000
):

   
    weights = np.zeros(X.shape[1])

   
    bias = 0

    for epoch in range(epochs):

       
        z = np.dot(X, weights) + bias

        
        predictions = sigmoid(z)

      
        dw = np.dot(
            X.T,
            (predictions - y)
        ) / len(y)

        db = np.mean(predictions - y)

        
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    
    final_predictions = sigmoid(
        np.dot(X, weights) + bias
    )

    final_loss = compute_loss(
        y,
        final_predictions
    )

    return weights, bias, final_loss



weights, bias, final_loss = train_logistic_regression(
    X_train,
    y_train
)


print("\nModel training completed.")
print("Final Training Loss:", final_loss)
print("Number of weights:", len(weights))
print("Bias:", bias)



test_scores = sigmoid(
    np.dot(X_test, weights) + bias
)


test_predictions = (
    test_scores >= 0.5
).astype(int)


print("\nPredictions completed.")
print(
    "First 10 predictions:",
    test_predictions[:10]
)



accuracy = np.mean(
    test_predictions == y_test
)


print(
    "\nFrom-scratch Logistic Regression Accuracy:",
    accuracy
)

print(
    "Accuracy percentage:",
    accuracy * 100,
    "%"
)



true_positive = np.sum(
    (y_test == 1) &
    (test_predictions == 1)
)

true_negative = np.sum(
    (y_test == 0) &
    (test_predictions == 0)
)

false_positive = np.sum(
    (y_test == 0) &
    (test_predictions == 1)
)

false_negative = np.sum(
    (y_test == 1) &
    (test_predictions == 0)
)


print("\nConfusion Matrix:")
print("True Negative:", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)
print("True Positive:", true_positive)



feature_names = X.columns

weight_pairs = list(
    zip(feature_names, weights)
)


weight_pairs.sort(
    key=lambda x: abs(x[1]),
    reverse=True
)


print("\nTop 3 Feature Weights:")

for feature, weight in weight_pairs[:3]:
    print(
        feature,
        ":",
        weight
    )