import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .preprocess import DATA_PATH, load_and_clean_data
except ImportError:  # Support running as: python src/predict.py
    from preprocess import DATA_PATH, load_and_clean_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logreg.pkl"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def prepare_input(input_csv: str, feature_columns=None):
    df = pd.read_csv(input_csv)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    encoded = pd.get_dummies(df, drop_first=True, dtype=int)
    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded


def main(input_csv: str, output_csv: str):
    model = load_model()
    X_new = prepare_input(input_csv, getattr(model, "feature_names_in_", None))
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    output = pd.read_csv(input_csv).copy()
    output["churn_probability"] = probabilities
    output["churn_predicted"] = predictions
    output.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new telecom customers for churn risk.")
    parser.add_argument("--input", required=True, help="CSV containing customer feature columns")
    parser.add_argument("--output", required=True, help="Destination CSV for predictions")
    args = parser.parse_args()
    main(args.input, args.output)
