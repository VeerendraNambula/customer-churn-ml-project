# Customer Churn Prediction — Project Report

## Business objective

Customer churn can reduce recurring revenue and increase acquisition costs. This project explores a supervised-learning workflow for estimating churn risk from customer and subscription attributes. Scores could help prioritize further investigation and retention outreach; they should not be treated as decisions on their own.

## Dataset and scope

The code expects the public Telco Customer Churn dataset at `data/telco_churn.csv`. The dataset is not committed to this repository. Download it separately from the source linked in the root README and review its license and permitted use.

This is a single-dataset learning exercise, not a production churn platform. It does not establish that the model generalizes to other providers or industries.

## Methodology

The preprocessing workflow removes `customerID`, converts `TotalCharges` to numeric, removes rows with missing converted charges, trims whitespace in categorical columns, maps `Churn` to binary labels, and one-hot encodes categorical predictors with `drop_first=True`. It then creates an 80/20 stratified train/test split with `random_state=42`.

Two classifiers are trained:

- Logistic Regression with `max_iter=1000`.
- Random Forest with 200 estimators, `random_state=42`, and `class_weight="balanced"`.

The training script prints a classification report and ROC-AUC for each model and saves both models under `models/`. The prediction script currently loads the Logistic Regression artifact.

## Results

No verified benchmark output is stored in this repository. Run `python src/train_model.py` with the dataset to generate the classification report and ROC-AUC. No performance score is claimed here because results have not been captured and independently checked.

## Risks and limitations

- **Dataset scope:** Results reflect one public telecom dataset and may not transfer to another customer population.
- **Class imbalance:** Accuracy alone is insufficient; precision, recall, F1, and ROC-AUC should be reviewed together.
- **Decision threshold:** Prediction uses a fixed 0.5 threshold. A business threshold should be selected using validation data and the costs of false positives and false negatives.
- **Feature consistency:** Training and inference must use the same feature definitions and encoded column order. The current prediction workflow uses the training CSV to align one-hot encoded columns, so that dataset must remain available.
- **Model lifecycle:** Cross-validation, tuning, calibration, drift monitoring, and automated retraining are not implemented.
- **Business use:** Predictions are risk estimates, not explanations or guarantees. Retention actions should be reviewed for fairness and customer impact.

## Reproducibility

Install dependencies with `pip install -r requirements.txt`, place the dataset at `data/telco_churn.csv`, and follow the root README for exploratory analysis, training, and batch prediction.
