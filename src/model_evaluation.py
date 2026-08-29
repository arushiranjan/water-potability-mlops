import os
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from dvclive import Live


# X_test = test_data.iloc[:, 0:-1].values
# y_test = test_data.iloc[:, -1].values
def load_data(path: str) -> pd.DataFrame:
    try: 
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error while loading data from {path}: {e}")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = data.drop(columns=['Potability'])
    y = data['Potability']
    return (X,y)


def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")


def evaluate(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    try:
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

        # save in dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric("acc", acc)
            live.log_metric("precision", pre)
            live.log_metric("recall", recall)
            live.log_metric("fl_score", f1score)
            
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error while evaluating: {e}")

def save_metrics(metrics_dict: dict, filepath: str) -> None:
    try:
        with open(filepath, "w") as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}")


def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluate(model, X_test, y_test)

        save_metrics(metrics, metrics_path)
        print("Model evaluation completed successfully.")

    except Exception as e:
        raise Exception(f"Error in model evaluation: {e}")

if __name__=="__main__":
    main()


