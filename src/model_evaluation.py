import os
import pandas as pd
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load test data
test_data = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# Load trained model
model = pickle.load(open("./models/model.pkl", "rb"))

# Prediction
y_pred = model.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

# Store metrics
metrics_dict = {
    "acc": acc,
    "precision": pre,
    "recall": recall,
    "f1_score": f1score
}

# Create reports directory
os.makedirs("reports", exist_ok=True)

# Save metrics
with open("reports/metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)

print("Model evaluation completed successfully.")
print(metrics_dict)
