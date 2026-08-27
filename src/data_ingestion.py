import os
import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import yaml

load_dotenv()

# Load parameters
def load_params(filepath: str) -> float:
    try: 
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params["data_ingestion"]["test_size"]
    except Exception as e:
        raise Exception(f"Error ocurred while loading parameters from {filepath}: {e}")

def load_data(dataset: str, filename: str) -> pd.DataFrame:
    try:
        api = KaggleApi()
        api.authenticate()
        os.makedirs("data/raw", exist_ok=True)

        api.dataset_download_file(dataset, filename, path="data/raw", force=True)

        # Read downloaded CSV
        data = pd.read_csv(os.path.join("data", "raw", filename))
        return data
    except Exception as e:
            raise Exception(f"Error ocurred while loading data: {e}")
    

def split_data(data: pd.DataFrame, test_size:float):
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except ValueError as e:
        raise ValueError(f"Error splitting data: {e}")


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    try: 
        # Save train/test data
        data_path = os.path.join("data", "raw")
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

        print("Data ingestion completed successfully.")
    except Exception as e:
        raise Exception(f"Error ocurred while saving data: {e}")


def main():

    test_size = load_params("params.yaml")

    # Download dataset from Kaggle
    KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
    DATASET_NAME = os.getenv("DATASET_NAME")
    RAW_FILE_NAME = os.getenv("RAW_FILE_NAME")

    if not KAGGLE_API_TOKEN:
        raise ValueError("KAGGLE_API_TOKEN not found")
    if not DATASET_NAME:
        raise ValueError("DATASET_NAME not found")
    if not RAW_FILE_NAME:
        raise ValueError("RAW_FILE_NAME not found")

    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

    data = load_data(DATASET_NAME, RAW_FILE_NAME)

    train_data, test_data = split_data(data, test_size)

    save_data(train_data, test_data)

if __name__=="__main__":
    main()