import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("SHAPE:", df.shape)
print("\nDTYPES:\n", df.dtypes)
print("\nNULLS:\n", df.isnull().sum())
print("\nDUPLICATE customerID:", df['customerID'].duplicated().sum())
print("\nTotalCharges sample (checking numeric parse issues):")
tc_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Non-numeric TotalCharges count:", tc_numeric.isnull().sum())
print(df[tc_numeric.isnull()][['customerID','tenure','MonthlyCharges','TotalCharges']])
print("\nChurn value counts:\n", df['Churn'].value_counts())
print("\ntenure describe:\n", df['tenure'].describe())
print("\nMonthlyCharges describe:\n", df['MonthlyCharges'].describe())
print("\nSeniorCitizen unique:", df['SeniorCitizen'].unique())
print("\nContract value counts:\n", df['Contract'].value_counts())
print("\nPaymentMethod value counts:\n", df['PaymentMethod'].value_counts())
print("\nInternetService value counts:\n", df['InternetService'].value_counts())
for col in ['MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
    print(f"\n{col}:\n", df[col].value_counts())
