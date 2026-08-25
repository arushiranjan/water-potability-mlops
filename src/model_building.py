import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("./data/processed/train_processed.csv")

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
pickle.dump(clf, open("models/model.pkl", "wb"))

print("Model building completed successfully.")
