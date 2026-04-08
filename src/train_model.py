import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import get_preprocessed_data

MODEL_DIR = "models"
LOGREG_PATH = f"{MODEL_DIR}/logreg.pkl"
RF_PATH = f"{MODEL_DIR}/random_forest.pkl"

import os
os.makedirs(MODEL_DIR, exist_ok=True)

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"  # helps with imbalance
    )
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_preprocessed_data()

    logreg = train_logistic_regression(X_train, y_train)
    evaluate_model(logreg, X_test, y_test, name="Logistic Regression")
    joblib.dump(logreg, LOGREG_PATH)

    rf = train_random_forest(X_train, y_train)
    evaluate_model(rf, X_test, y_test, name="Random Forest")
    joblib.dump(rf, RF_PATH)

    print(f"\nModels saved to {MODEL_DIR}/")