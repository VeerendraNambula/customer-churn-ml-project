import pandas as pd
import pytest

from src.preprocess import encode_features, load_and_clean_data
from src.predict import prepare_input


def test_load_and_clean_data_drops_identifier_and_coerces_charges(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame({
        "customerID": ["a", "b", "c"],
        "TotalCharges": ["10.5", " ", "20"],
        "Contract": [" Month-to-month ", "One year", " Two year "],
        "Churn": ["Yes", "No", "No"],
    }).to_csv(path, index=False)

    result = load_and_clean_data(path)
    assert "customerID" not in result.columns
    assert len(result) == 2
    assert result["Contract"].tolist() == ["Month-to-month", "Two year"]


def test_encode_features_returns_binary_target_and_encoded_columns():
    frame = pd.DataFrame({
        "TotalCharges": [10.0, 20.0, 30.0, 40.0],
        "Contract": ["Month", "Year", "Month", "Year"],
        "Churn": ["No", "Yes", "No", "Yes"],
    })
    X, y = encode_features(frame)
    assert set(y.tolist()) == {0, 1}
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)


def test_encode_features_rejects_unknown_target_label():
    frame = pd.DataFrame({"Contract": ["Month"], "Churn": ["Maybe"]})
    with pytest.raises(ValueError, match="Unexpected Churn labels"):
        encode_features(frame)


def test_prepare_input_aligns_to_training_features_without_training_csv(tmp_path):
    path = tmp_path / "new_customers.csv"
    pd.DataFrame({
        "customerID": ["new-1"],
        "TotalCharges": ["12.50"],
        "Contract": ["One year"],
        "InternetService": ["Fiber optic"],
    }).to_csv(path, index=False)

    result = prepare_input(path, ["TotalCharges", "Contract_One year", "InternetService_No"])
    assert result.columns.tolist() == [
        "TotalCharges", "Contract_One year", "InternetService_No"
    ]
    assert result.iloc[0].tolist() == [12.5, 1, 0]



def test_logistic_regression_uses_scaling_and_supports_prediction():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.train_model import train_logistic_regression

    X_train = pd.DataFrame({
        "monthly_charges": [20, 30, 40, 50, 60, 70, 80, 90],
        "tenure": [1, 2, 3, 4, 8, 9, 10, 12],
    })
    y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    model = train_logistic_regression(X_train, y_train)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert model.feature_names_in_.tolist() == X_train.columns.tolist()
    assert model.predict(X_train).shape == (len(X_train),)
    assert model.predict_proba(X_train).shape == (len(X_train), 2)
