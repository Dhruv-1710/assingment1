# ====================================
# Task 1 : Data Cleaning & Preprocessing
# ====================================

import pandas as pd
import numpy as np

# ------------------------------------
# 1. Data Ingestion
# ------------------------------------

df = pd.read_excel("customers-100.xlsx")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nDataset Shape")
print(df.shape)


# ------------------------------------
# 2. Duplicate Detection
# ------------------------------------

duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

print("Duplicates Removed")


# ------------------------------------
# 3. Column Management
# ------------------------------------

df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

print("\nClean Column Names")
print(df.columns)


# ------------------------------------
# 4. Missing Value Analysis
# ------------------------------------

print("\nMissing Values")
print(df.isna().sum())

missing_percent = (df.isna().sum() / len(df)) * 100
print("\nMissing Percentage")
print(missing_percent)


# Drop columns with >70% missing values

threshold = 0.7 * len(df)

df = df.dropna(axis=1, thresh=threshold)

print("\nColumns after dropping high missing columns")
print(df.columns)


# ------------------------------------
# 5. Missing Value Imputation
# ------------------------------------

# Numerical columns → Median

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# Categorical columns → Mode

cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


print("\nMissing Values After Cleaning")
print(df.isna().sum())


# ------------------------------------
# 6. Data Type Correction
# ------------------------------------

# Convert date columns

for col in df.columns:
    if "date" in col or "time" in col:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# Convert numeric values stored as text

for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", "").str.replace("$", ""),
                errors="ignore"
            )
        except:
            pass


print("\nUpdated Data Types")
print(df.dtypes)


# ------------------------------------
# 7. Format Standardization
# ------------------------------------

# Clean text columns

for col in cat_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()


# Standardize categorical values

df.replace({
    "m": "male",
    "f": "female",
    "usa": "united states",
    "us": "united states",
    "uk": "united kingdom"
}, inplace=True)


# ------------------------------------
# 8. Outlier Detection (Optional)
# ------------------------------------

for col in num_cols:
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])


# ------------------------------------
# 9. Validation Checks
# ------------------------------------

print("\nFinal Dataset Shape")
print(df.shape)

print("\nRemaining Missing Values")
print(df.isna().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

assert df.duplicated().sum() == 0


# ------------------------------------
# 10. Save Clean Dataset
# ------------------------------------

df.to_csv(r"D:\cleaned_customers_data.csv", index=False)

print("\nData Cleaning Completed Successfully")