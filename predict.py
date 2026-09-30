import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Features and target
features = [
    "age",
    "monthly_spend",
    "support_calls",
    "tenure_months",
]

X_train = train[features]
y_train = train["churn"]

X_test = test[features]

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Save exactly one column: churn
output = pd.DataFrame({"churn": predictions})
output.to_csv("output/predictions.csv", index=False)

print(f"Generated {len(predictions)} predictions.")
print("Saved predictions to output/predictions.csv")
