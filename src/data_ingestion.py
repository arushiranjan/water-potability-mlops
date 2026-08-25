import os
import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import yaml

load_dotenv()

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)
test_size = params["data_ingestion"]["test_size"]

# Download dataset from Kaggle
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
DATASET_NAME = os.getenv("DATASET_NAME")
if not KAGGLE_API_TOKEN:
    raise ValueError("KAGGLE_API_TOKEN not found")
if not DATASET_NAME:
    raise ValueError("DATASET_NAME not found")

os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

api = KaggleApi()
api.authenticate()


RAW_FILE_NAME = os.getenv("RAW_FILE_NAME")
if not RAW_FILE_NAME:
    raise ValueError("RAW_FILE_NAME not found")
api.dataset_download_file(DATASET_NAME, RAW_FILE_NAME, path="data/raw", force=True)


# Read downloaded CSV
data = pd.read_csv(os.path.join("data", "raw", RAW_FILE_NAME))

# Train-test split
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# Save train/test data
data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

print("Data ingestion completed successfully.")