import os

import pandas as pd
from sklearn.metrics import accuracy_score


# Paths
predictions_path = "output/predictions.csv"
labels_path = "hidden/test_labels.csv"


# Check that the prediction file exists
if not os.path.exists(predictions_path):
    print("FAIL: predictions.csv was not found.")
    exit(1)


# Load predictions and hidden labels
predictions = pd.read_csv(predictions_path)
labels = pd.read_csv(labels_path)


# Check that the required column exists
if "churn" not in predictions.columns:
    print("FAIL: predictions.csv must contain a 'churn' column.")
    exit(1)


# Check that the number of predictions is correct
if len(predictions) != len(labels):
    print("FAIL: Incorrect number of predictions.")
    exit(1)


# Check that predictions contain only 0 or 1
if not predictions["churn"].isin([0, 1]).all():
    print("FAIL: Predictions must be either 0 or 1.")
    exit(1)


# Calculate accuracy
score = accuracy_score(
    labels["churn"],
    predictions["churn"]
)

print(f"Accuracy: {score:.2%}")


# Determine pass/fail
if score >= 0.75:
    print("PASS")
else:
    print("FAIL")
