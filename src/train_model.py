from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .preprocess import get_preprocessed_data
except ImportError:  # Support running as: python src/train_model.py
    from preprocess import get_preprocessed_data

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
LOGREG_PATH = MODEL_DIR / "logreg.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"


def train_logistic_regression(X_train, y_train):
    """Train a scaled logistic-regression pipeline.

    Scaling is important for distance/gradient-based optimisation when input
    features have very different numeric ranges. Keeping the scaler inside the
    pipeline ensures inference applies the exact same transformation.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    return model.fit(X_train, y_train)


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n=== {name} ===\n{report}")
    print(f"ROC-AUC: {auc:.4f}")
    return {"classification_report": report, "roc_auc": auc}


def main():
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logreg = train_logistic_regression(X_train, y_train)
    evaluate_model(logreg, X_test, y_test, name="Logistic Regression")
    joblib.dump(logreg, LOGREG_PATH)

    forest = train_random_forest(X_train, y_train)
    evaluate_model(forest, X_test, y_test, name="Random Forest")
    joblib.dump(forest, RF_PATH)
    print(f"\nModels saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
