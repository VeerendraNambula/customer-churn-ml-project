# Customer Churn Prediction – Telco

## 1. Business Problem

A telecom company is experiencing customer churn and wants to identify which active customers are most likely to leave soon.  
By predicting churn, the business can target at-risk customers with retention offers (discounts, better support, personalized plans) and reduce revenue loss. [web:14][web:25]

## 2. Project Objective

- Build a machine learning model that predicts whether a customer will churn (Yes/No).
- Achieve at least:
  - AUC-ROC ≥ 0.80 on the test set.
  - Recall for churned customers ≥ 0.70.
- Deliver:
  - Reusable data preprocessing pipeline.
  - Trained model artifact.
  - Simple interface (script or API) for scoring new customers.
  - A short report for business stakeholders explaining insights and recommendations.

## 3. Dataset

- Source: Telco Customer Churn dataset (Kaggle). [web:14][web:15]
- Each row represents a customer; each column contains customer attributes such as:
  - Demographics: gender, senior citizen, tenure. [web:21][web:25]
  - Services: phone service, multiple lines, internet, online security, streaming, etc. [web:14][web:25]
  - Account and billing: contract type, paperless billing, payment method, monthly charges, total charges. [web:15][web:27]
  - Target variable: `Churn` (Yes = customer left, No = customer stayed). [web:14][web:21]

## 4. Requirements, Assumptions, Constraints

### 4.1 Requirements
- Historical labeled data with churn information.
- Tools:
  - Python 3.10+ (Pandas, NumPy, scikit-learn, Matplotlib/Seaborn).
  - SQL database (optional; e.g., SQLite/PostgreSQL for practice).
  - Jupyter Notebook for exploration.

### 4.2 Assumptions
- The churn labels in the dataset are correct and reflect actual customer behavior. [web:21]
- The relationships in this dataset are representative of real-world telco customers.
- All input features used at training time are available at prediction time.

### 4.3 Constraints
- Dataset is limited to one company/time period and may not generalize to all telcos. [web:25]
- No real-time streaming data in this project (batch scoring only).
- Single-node, local training (no distributed training).

## 5. Project Structure

```text
customer-churn-ml-project/
│
├─ data/
│  └─ telco_churn.csv          # Kaggle dataset (not tracked in git ideally)
│
├─ notebooks/
│  ├─ 01_eda.ipynb             # Data exploration & quality checks
│  └─ 02_modeling.ipynb        # Feature engineering & model training
│
├─ src/
│  ├─ preprocess.py            # Cleaning, encoding, train/test split
│  ├─ train_model.py           # Training & saving models
│  └─ predict.py               # Loading model & predicting churn
│
├─ reports/
│  └─ project_report.md        # Business objectives, risks, results
│
├─ README.md                   # Overview, objectives, setup, usage
└─ requirements.txt            # Python dependencies
```

## 6. How to Run (placeholder)

1. Create virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run notebooks in order:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_modeling.ipynb`
3. Train and save model:
   ```bash
   python src/train_model.py
   ```
4. Score new customers:
   ```bash
   python src/predict.py --input path/to/customers.csv --output predictions.csv
   ```

## 7. Risks, Assumptions, and Constraints (Business View)

| Type        | Item                                  | Description |
|------------|----------------------------------------|-------------|
| Assumption | Accurate labels                        | Churn flag reflects whether the customer really left. [web:21] |
| Assumption | Stable data                            | Feature distributions do not change drastically in short term. |
| Constraint | Limited history                        | Dataset covers a single telco and limited time period. [web:25] |
| Risk       | Model overfitting                      | Model may perform well on this dataset but poorly in production. |
| Risk       | Actionability                          | Business might not act on predictions, limiting impact. |
| Risk       | Data drift                             | Over time, customer behavior may change, degrading performance. |

## 8. Next Steps

- Perform EDA and data quality checks in `01_eda.ipynb`.
- Engineer features and train multiple models in `02_modeling.ipynb`.
- Create a simple CLI/API in `src/` for scoring.
- Write `reports/project_report.md` as a non-technical summary for stakeholders.