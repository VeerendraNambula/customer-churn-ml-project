import argparse
import pandas as pd
import joblib
from preprocess import load_and_clean_data, encode_features

MODEL_PATH = "models/logreg.pkl"

def load_model(path: str = MODEL_PATH):
    return joblib.load(path)

def prepare_input(input_csv: str):
    # Load original full data to get structure
    df_full = load_and_clean_data()
    if "Churn" in df_full.columns:
        df_full = df_full.drop(columns=["Churn"])

    # Load new customers
    df_new = pd.read_csv(input_csv)
    
    if "customerID" in df_new.columns:
        df_new = df_new.drop(columns=["customerID"])
        
    # Convert TotalCharges to numeric, fill NaNs with 0 to prevent dropping predictive rows
    df_new["TotalCharges"] = pd.to_numeric(df_new["TotalCharges"], errors="coerce").fillna(0)
    for col in df_new.select_dtypes(include="object").columns:
        df_new[col] = df_new[col].str.strip()

    # Merge to align dummies seamlessly
    df_new["_is_new"] = 1
    df_concat = pd.concat([df_full, df_new], ignore_index=True)

    # One-hot encoding
    X_encoded = pd.get_dummies(df_concat, drop_first=True)

    # Filter back out the new instances & drop helper column
    X_new = X_encoded[X_encoded["_is_new"] == 1].copy()
    X_new = X_new.drop(columns=["_is_new"])

    return X_new

def main(input_csv: str, output_csv: str):
    model = load_model()
    X_new = prepare_input(input_csv)
    probs = model.predict_proba(X_new)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = pd.read_csv(input_csv).copy()
    out["churn_probability"] = probs
    out["churn_predicted"] = preds

    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV with new customers")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    args = parser.parse_args()

    main(args.input, args.output)