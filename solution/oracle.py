import os

import pandas as pd
from sklearn.linear_model import LogisticRegression


# Load the training and test data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# Separate features and target
X_train = train.drop(columns=["churn"])
y_train = train["churn"]


# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)


# Generate predictions
predictions = model.predict(test)


# Create the required output format
output = pd.DataFrame({
    "churn": predictions
})


# Create output directory
os.makedirs("output", exist_ok=True)


# Save predictions
output.to_csv("output/predictions.csv", index=False)

print("Predictions saved successfully!")
print(output)
