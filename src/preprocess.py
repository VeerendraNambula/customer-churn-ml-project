from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"


def load_and_clean_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def encode_features(df: pd.DataFrame):
    if "Churn" not in df.columns:
        raise ValueError("Training data must contain a 'Churn' target column.")

    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        invalid = sorted(df.loc[y.isna(), "Churn"].dropna().astype(str).unique())
        raise ValueError(f"Unexpected Churn labels: {invalid}")

    X = df.drop(columns=["Churn"])
    return pd.get_dummies(X, drop_first=True, dtype=int), y.astype(int)


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_preprocessed_data(path=DATA_PATH):
    return train_test_split_data(*encode_features(load_and_clean_data(path)))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
