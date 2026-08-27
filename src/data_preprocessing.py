import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error while loading data from {filepath}: {e}")

def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
        return df
    except Exception as e:
        raise Exception(f"Error while filling missing values with median: {e}")


def save_data(train_processed: pd.DataFrame, test_processed:pd.DataFrame):
    try:
        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)
        train_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    except Exception as e:
        raise Exception(f"Error while saving processed data: {e}")


def main():
    try:
        train_data = load_data("./data/raw/train.csv")
        test_data = load_data("./data/raw/test.csv")

        train_processed_data = fill_missing_with_median(train_data)
        test_processed_data = fill_missing_with_median(test_data)

        save_data(train_processed_data, test_processed_data)

        print("Data preprocessing completed successfully.")

    except Exception as E:
        raise Exception(f"An error ocurred: {e}")

if __name__ == "__main__":
    main()