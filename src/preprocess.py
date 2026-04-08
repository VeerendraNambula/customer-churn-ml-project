import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/telco_churn.csv"

def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop customerID (identifier, not a feature)
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges type
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Strip spaces in categorical columns just in case
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def encode_features(df: pd.DataFrame):
    # Target
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Features
    X = df.drop(columns=["Churn"])

    # One‑hot encode all categorical columns (drop_first to reduce multicollinearity)
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_preprocessed_data(path: str = DATA_PATH):
    df = load_and_clean_data(path)
    X, y = encode_features(df)
    return train_test_split_data(X, y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)