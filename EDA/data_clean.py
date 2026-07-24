"""
02_clean.py — Customer Retention & Churn Analysis
Dataset: Telco Customer Churn (Kaggle - blastchar/telco-customer-churn)
Purpose: Clean the raw snapshot export and engineer the features needed for
         cohort/retention, churn-driver, and CLV analysis.

IMPORTANT DATASET CHARACTERISTIC (disclosed throughout all deliverables):
This is a SNAPSHOT dataset — one row per customer, no signup date and no
transaction history. There is no way to build a true calendar-based cohort
("customers who joined in Jan 2021"). Industry-standard workaround for this
exact, widely-used dataset: (1) build TENURE-BASED cohorts (group by months-
on-book) to see how churn risk changes over the customer lifecycle, and
(2) fit a Kaplan-Meier survival curve treating `tenure` as duration and
`Churn` as the event indicator — this reconstructs a genuine retention curve
from cross-sectional data, which is the standard technique when no
longitudinal signup-date data exists.
"""
import pandas as pd
import numpy as np

RAW_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
OUT_PATH = 'data/cleaned_telco_churn.csv'
LOG_PATH = 'outputs/cleaning_log.txt'

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
log(f"[LOAD] Raw shape: {df.shape}")

# ------------------------------------------------------------------
# 2. DUPLICATE CHECK
# ------------------------------------------------------------------
dup = df['customerID'].duplicated().sum()
log(f"[DUPLICATES] customerID duplicates found: {dup} (none expected — one row per customer)")

# ------------------------------------------------------------------
# 3. FIX TotalCharges (stored as text; 11 blank values)
#    All 11 blanks occur at tenure == 0, i.e. brand-new customers who
#    haven't been billed a full month yet. Correct treatment: impute as 0,
#    NOT drop (dropping would silently remove real, very-new customers from
#    every downstream KPI, which would bias churn/tenure analysis).
# ------------------------------------------------------------------
before_blank = (df['TotalCharges'].str.strip() == '').sum()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
log(f"[TOTAL CHARGES] Converted to numeric; {before_blank} blank values (all at tenure=0) "
    f"imputed as 0 — these are brand-new customers not yet billed")

# ------------------------------------------------------------------
# 4. READABILITY FIXES
# ------------------------------------------------------------------
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
df['Churn'] = df['Churn'].str.strip()
df['ChurnFlag'] = (df['Churn'] == 'Yes').astype(int)
log("[READABILITY] SeniorCitizen mapped 0/1 -> No/Yes; added numeric ChurnFlag (0/1) for modeling/DAX")

# ------------------------------------------------------------------
# 5. FEATURE ENGINEERING — TENURE COHORTS
#    Standard bucketing used across published analyses of this dataset.
# ------------------------------------------------------------------
def tenure_cohort(t):
    if t <= 6: return '0-6 months'
    elif t <= 12: return '6-12 months'
    elif t <= 24: return '12-24 months'
    elif t <= 48: return '24-48 months'
    else: return '48-72 months'

df['TenureCohort'] = df['tenure'].apply(tenure_cohort)
cohort_order = ['0-6 months','6-12 months','12-24 months','24-48 months','48-72 months']
log(f"[COHORTS] Created 5 tenure-based cohorts (dataset has no signup date, so calendar "
    f"cohorts aren't possible — this is the standard workaround for this dataset)")

# ------------------------------------------------------------------
# 6. FEATURE ENGINEERING — SERVICE ADOPTION
# ------------------------------------------------------------------
addon_services = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies']
df['ServiceCount'] = df[addon_services].apply(lambda row: (row == 'Yes').sum(), axis=1)
df['HasInternet'] = df['InternetService'] != 'No'
df['HasMultipleLines'] = df['MultipleLines'] == 'Yes'
log(f"[SERVICE ADOPTION] Created ServiceCount (0-6 add-on services subscribed), "
    f"HasInternet, HasMultipleLines flags")

# ------------------------------------------------------------------
# 7. FEATURE ENGINEERING — CONTRACT / PAYMENT RISK FLAGS
# ------------------------------------------------------------------
df['IsMonthToMonth'] = df['Contract'] == 'Month-to-month'
df['IsAutoPay'] = df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)'])
df['IsElectronicCheck'] = df['PaymentMethod'] == 'Electronic check'
log("[RISK FLAGS] IsMonthToMonth, IsAutoPay, IsElectronicCheck created for churn-driver analysis")

# ------------------------------------------------------------------
# 8. FEATURE ENGINEERING — CUSTOMER LIFETIME VALUE (CLV) PROXIES
#    - CLV_ToDate: actual historical value already billed (TotalCharges).
#    - CLV_ProjectedAnnual: MonthlyCharges * 12, a simple forward run-rate
#      proxy — NOT a discounted/probabilistic CLV model, disclosed as such.
# ------------------------------------------------------------------
df['CLV_ToDate'] = df['TotalCharges']
df['CLV_ProjectedAnnual'] = df['MonthlyCharges'] * 12
log("[CLV] CLV_ToDate = actual historical TotalCharges (real). "
    "CLV_ProjectedAnnual = MonthlyCharges x 12, a simple run-rate proxy "
    "(not a probabilistic/discounted CLV model) — disclosed in report")

# ------------------------------------------------------------------
# 9. OUTLIER CHECK (MonthlyCharges, tenure)
#    Both are bounded, plausible business ranges (tenure 0-72 months,
#    MonthlyCharges £18.25-£118.75) — no outlier treatment needed, this is
#    a clean, already-curated snapshot extract.
# ------------------------------------------------------------------
log(f"[OUTLIERS] tenure range [{df['tenure'].min()}, {df['tenure'].max()}], "
    f"MonthlyCharges range [{df['MonthlyCharges'].min():.2f}, {df['MonthlyCharges'].max():.2f}] "
    f"— both plausible, no outlier treatment required")

# ------------------------------------------------------------------
# 10. FINAL CHECKS & SAVE
# ------------------------------------------------------------------
log(f"\n[FINAL SHAPE] {df.shape}")
log(f"[OVERALL CHURN RATE] {df['ChurnFlag'].mean()*100:.2f}% ({df['ChurnFlag'].sum()} of {len(df)} customers)")
log(f"[NULLS CHECK] {df.isnull().sum().sum()} total nulls remaining")

df.to_csv(OUT_PATH, index=False)
log(f"\n[SAVED] Cleaned dataset written to {OUT_PATH}")

with open(LOG_PATH, 'w') as f:
    f.write('\n'.join(log_lines))
