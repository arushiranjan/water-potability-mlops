import pandas as pd
import os
import pickle
import yaml

from sklearn.ensemble import RandomForestClassifier

# Load parameters
n_estimators = yaml.safe_load(open("params.yaml", "r"))["model_building"]["n_estimators"]

train_data = pd.read_csv("./data/processed/train_processed.csv")

# X_train = train_data.iloc[:, 0:-1].values
# y_train = train_data.iloc[:, -1].values
X_train = train_data.drop(columns=['Potability'])
y_train = train_data['Potability']

clf = RandomForestClassifier(n_estimators=n_estimators)

clf.fit(X_train, y_train)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
pickle.dump(clf, open("models/model.pkl", "wb"))

print("Model building completed successfully.")
