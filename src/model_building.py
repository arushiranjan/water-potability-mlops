import pandas as pd
import os
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

def load_params(path: str) -> int:
    try: 
        with open(path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error ocurred while loading parameters from {path}: {e}")


def load_data(path: str) -> pd.DataFrame:
    try: 
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error while loading data from {path}: {e}")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = data.drop(columns=['Potability'])
    y = data['Potability']
    return (X,y)

# X_train = train_data.iloc[:, 0:-1].values
# y_train = train_data.iloc[:, -1].values

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        return clf.fit(X, y)
    except Exception as e:
        raise Exception(f"Error training model: {e}")


def save_model(model: RandomForestClassifier, filepath: str):

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {filepath}: {e}")

def main():
    try:

        n_estimators = load_params("params.yaml")
        data_path = "./data/processed/train_processed.csv"
        save_path = "models/model.pkl"
        data = load_data(data_path)
        X_train, y_train = prepare_data(data)
        model = train_model(X_train, y_train, n_estimators)
        save_model(model, save_path)
        print("Model building completed successfully.")

    except Exception as e:
        raise Exception(f"Error in model_building: {e}")

if __name__ == "__main__":
    main()
