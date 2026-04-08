import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/telco_churn.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nFirst 5 rows:")
print(df.head())

# 1) Fix TotalCharges type
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing values AFTER TotalCharges fix:")
print(df.isnull().sum())

# 2) Drop rows where TotalCharges is NaN (very few rows, safe to drop)
df = df.dropna(subset=["TotalCharges"])

print("\nShape after dropping missing TotalCharges:", df.shape)

# 3) Basic target distribution (churn rate)
print("\nChurn value counts:")
print(df["Churn"].value_counts())

print("\nChurn rate (%):")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# Ensure we work on the cleaned df
df_clean = df.copy()

# 1) Numeric summary grouped by Churn
print("\nNumeric summary by Churn (tenure, MonthlyCharges, TotalCharges):")
print(df_clean.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean())

# 2) Churn rate by contract type
print("\nChurn rate by Contract:")
print(
    df_clean.groupby("Contract")["Churn"]
    .value_counts(normalize=True)
    .rename("proportion")
    .mul(100)
    .round(2)
)

# 3) Optional: quick plots (run locally)
# Churn countplot
sns.countplot(data=df_clean, x="Churn")
plt.title("Churn distribution")
plt.show()

# Churn by Contract
sns.countplot(data=df_clean, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

# tenure distribution by churn
sns.kdeplot(data=df_clean, x="tenure", hue="Churn", common_norm=False)
plt.title("Tenure distribution by Churn")
plt.show() 