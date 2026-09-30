Customer Churn Prediction
Objective

Build a machine-learning model that predicts whether a customer will churn.

You are provided with two datasets:

/data/train.csv — training data containing the target variable
/data/test.csv — test data without the target variable
Dataset

The input features are:

age — customer's age
monthly_spend — customer's monthly spending
support_calls — number of customer support calls
tenure_months — number of months the customer has been subscribed

The target variable is:

churn — 0 means the customer did not churn, and 1 means the customer churned.
Requirements

Your solution must:

Load /data/train.csv.
Load /data/test.csv.
Train a binary classification model using the training data.
Generate one churn prediction for every row in the test dataset.
Save the predictions to:

/output/predictions.csv

The output file must contain exactly one column:

churn

The number of predictions must equal the number of rows in /data/test.csv.

Evaluation

Your predictions will be compared against hidden test labels.

The evaluation metric is classification accuracy.

A solution must achieve at least 75% accuracy to pass.

Constraints
Use Python.
Your solution must run in the provided environment.
The complete solution must finish within 5 minutes.
Do not modify the provided datasets.