"""
03_eda.py — Churn Driver Analysis, Retention Curve, CLV Trends
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = ["#1B4F72", "#2E86C1", "#5DADE2", "#F39C12", "#CA6F1E", "#909497"]
CHURN_COLORS = {"No": "#2E86C1", "Yes": "#CB4335"}
plt.rcParams['figure.figsize'] = (11, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

df = pd.read_csv('data/cleaned_telco_churn.csv')
cohort_order = ['0-6 months','6-12 months','12-24 months','24-48 months','48-72 months']
df['TenureCohort'] = pd.Categorical(df['TenureCohort'], categories=cohort_order, ordered=True)

kpi_lines = []
def kpi(msg):
    print(msg)
    kpi_lines.append(str(msg))

# ==========================================================================
# HEADLINE KPIs
# ==========================================================================
total_customers = len(df)
churned = df['ChurnFlag'].sum()
churn_rate = df['ChurnFlag'].mean()
retention_rate = 1 - churn_rate
avg_tenure = df['tenure'].mean()
avg_tenure_churned = df.loc[df['Churn']=='Yes','tenure'].mean()
avg_tenure_retained = df.loc[df['Churn']=='No','tenure'].mean()
avg_monthly_churned = df.loc[df['Churn']=='Yes','MonthlyCharges'].mean()
avg_monthly_retained = df.loc[df['Churn']=='No','MonthlyCharges'].mean()
total_clv_at_risk = df.loc[df['Churn']=='Yes','CLV_ProjectedAnnual'].sum()
avg_clv_todate = df['CLV_ToDate'].mean()

kpi("=== HEADLINE RETENTION & CHURN KPIs ===")
kpi(f"Total Customers:                {total_customers:,}")
kpi(f"Churned Customers:              {churned:,}")
kpi(f"Overall Churn Rate:             {churn_rate*100:.2f}%")
kpi(f"Overall Retention Rate:         {retention_rate*100:.2f}%")
kpi(f"Avg Tenure (all customers):     {avg_tenure:.1f} months")
kpi(f"Avg Tenure — Churned:           {avg_tenure_churned:.1f} months")
kpi(f"Avg Tenure — Retained:          {avg_tenure_retained:.1f} months")
kpi(f"Avg Monthly Charges — Churned:  £{avg_monthly_churned:.2f}")
kpi(f"Avg Monthly Charges — Retained: £{avg_monthly_retained:.2f}")
kpi(f"Annual Revenue At Risk (churned customers' projected annual value): £{total_clv_at_risk:,.0f}")
kpi(f"Avg CLV-to-date per customer:   £{avg_clv_todate:,.2f}")

# ==========================================================================
# 1. KAPLAN-MEIER RETENTION / SURVIVAL CURVE
#    Standard technique to reconstruct a genuine retention curve from
#    cross-sectional tenure+churn data (no signup dates available).
# ==========================================================================
kmf = KaplanMeierFitter()
kmf.fit(durations=df['tenure'], event_observed=df['ChurnFlag'], label='All Customers')

fig, ax = plt.subplots()
kmf.plot_survival_function(ax=ax, color=PALETTE[0], linewidth=2.5, ci_show=True)
ax.set_title('Customer Retention Curve (Kaplan-Meier Survival Estimate)')
ax.set_xlabel('Tenure (Months)')
ax.set_ylabel('Retention Probability')
ax.set_ylim(0, 1.02)
ax.legend().remove()
fig.tight_layout()
fig.savefig('visuals/01_retention_curve_overall.png', dpi=150)
plt.close(fig)

# Retention curve split by Contract type — the single strongest churn driver
fig, ax = plt.subplots()
for i, contract in enumerate(df['Contract'].unique()):
    mask = df['Contract'] == contract
    kmf.fit(durations=df.loc[mask,'tenure'], event_observed=df.loc[mask,'ChurnFlag'], label=contract)
    kmf.plot_survival_function(ax=ax, color=PALETTE[i], linewidth=2.5, ci_show=False)
ax.set_title('Retention Curve by Contract Type')
ax.set_xlabel('Tenure (Months)')
ax.set_ylabel('Retention Probability')
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig('visuals/02_retention_curve_by_contract.png', dpi=150)
plt.close(fig)

kpi(f"\n=== RETENTION CURVE (Kaplan-Meier) — key checkpoints ===")
kmf.fit(durations=df['tenure'], event_observed=df['ChurnFlag'])
for month in [6, 12, 24, 48, 72]:
    if month <= df['tenure'].max():
        surv = kmf.survival_function_at_times(month).values[0]
        kpi(f"Retention probability at month {month}: {surv*100:.1f}%")

# ==========================================================================
# 2. CHURN RATE BY TENURE COHORT (retention driver #1: lifecycle stage)
# ==========================================================================
cohort_churn = df.groupby('TenureCohort', observed=True)['ChurnFlag'].agg(['mean','count'])
fig, ax = plt.subplots()
bars = ax.bar(cohort_churn.index.astype(str), cohort_churn['mean']*100, color=PALETTE[0])
for i, (rate, n) in enumerate(zip(cohort_churn['mean'], cohort_churn['count'])):
    ax.text(i, rate*100+1, f'{rate*100:.1f}%\n(n={n})', ha='center', fontsize=9)
ax.set_title('Churn Rate by Tenure Cohort (Customer Lifecycle Stage)')
ax.set_ylabel('Churn Rate (%)')
ax.set_xlabel('Tenure Cohort')
ax.set_ylim(0, cohort_churn['mean'].max()*100 + 12)
fig.tight_layout()
fig.savefig('visuals/03_churn_by_tenure_cohort.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 3. CHURN DRIVERS — CONTRACT TYPE
# ==========================================================================
contract_churn = df.groupby('Contract')['ChurnFlag'].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=contract_churn.values*100, y=contract_churn.index, palette=PALETTE, ax=ax)
ax.set_title('Churn Rate by Contract Type')
ax.set_xlabel('Churn Rate (%)')
fig.tight_layout()
fig.savefig('visuals/04_churn_by_contract.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 4. CHURN DRIVERS — PAYMENT METHOD
# ==========================================================================
payment_churn = df.groupby('PaymentMethod')['ChurnFlag'].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=payment_churn.values*100, y=payment_churn.index, palette=PALETTE, ax=ax)
ax.set_title('Churn Rate by Payment Method')
ax.set_xlabel('Churn Rate (%)')
fig.tight_layout()
fig.savefig('visuals/05_churn_by_payment.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 5. CHURN DRIVERS — INTERNET SERVICE TYPE
# ==========================================================================
internet_churn = df.groupby('InternetService')['ChurnFlag'].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=internet_churn.values*100, y=internet_churn.index, palette=PALETTE, ax=ax)
ax.set_title('Churn Rate by Internet Service Type')
ax.set_xlabel('Churn Rate (%)')
fig.tight_layout()
fig.savefig('visuals/06_churn_by_internet.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 6. SERVICE ADOPTION VS CHURN (retention driver: add-on stickiness)
# ==========================================================================
service_churn = df.groupby('ServiceCount')['ChurnFlag'].mean()
fig, ax = plt.subplots()
ax.bar(service_churn.index.astype(str), service_churn.values*100, color=PALETTE[0])
ax.set_title('Churn Rate by Number of Add-On Services Subscribed')
ax.set_xlabel('Number of Add-On Services (of 6 possible)')
ax.set_ylabel('Churn Rate (%)')
fig.tight_layout()
fig.savefig('visuals/07_churn_by_service_count.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 7. COHORT x CONTRACT RETENTION MATRIX (heatmap "cohort table")
# ==========================================================================
matrix = df.pivot_table(index='TenureCohort', columns='Contract', values='ChurnFlag',
                          aggfunc='mean', observed=True) * 100
fig, ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix, annot=True, fmt='.1f', cmap='Reds', ax=ax, cbar_kws={'label':'Churn Rate (%)'})
ax.set_title('Churn Rate Matrix: Tenure Cohort x Contract Type')
ax.set_ylabel('')
fig.tight_layout()
fig.savefig('visuals/08_cohort_contract_matrix.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 8. MONTHLY CHARGES DISTRIBUTION — CHURNED VS RETAINED
# ==========================================================================
fig, ax = plt.subplots()
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette=CHURN_COLORS, ax=ax, alpha=0.4)
ax.set_title('Monthly Charges Distribution: Churned vs Retained')
ax.set_xlabel('Monthly Charges (£)')
fig.tight_layout()
fig.savefig('visuals/09_monthly_charges_dist.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 9. CUSTOMER LIFETIME VALUE TRENDS (CLV to date by cohort/churn status)
# ==========================================================================
clv_by_cohort = df.groupby(['TenureCohort','Churn'], observed=True)['CLV_ToDate'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=clv_by_cohort, x='TenureCohort', y='CLV_ToDate', hue='Churn',
            palette=CHURN_COLORS, ax=ax)
ax.set_title('Average Customer Lifetime Value (To Date) by Tenure Cohort')
ax.set_ylabel('Avg CLV To Date (£)')
ax.set_xlabel('Tenure Cohort')
fig.tight_layout()
fig.savefig('visuals/10_clv_by_cohort.png', dpi=150)
plt.close(fig)

# ==========================================================================
# 10. DEMOGRAPHIC CHURN DRIVERS — SENIOR CITIZEN, PARTNER, DEPENDENTS
# ==========================================================================
demo_churn = pd.DataFrame({
    'SeniorCitizen': df.groupby('SeniorCitizen')['ChurnFlag'].mean(),
}).T
fig, axes = plt.subplots(1, 3, figsize=(14,5))
for ax, col in zip(axes, ['SeniorCitizen','Partner','Dependents']):
    vals = df.groupby(col)['ChurnFlag'].mean()*100
    sns.barplot(x=vals.index, y=vals.values, palette=PALETTE, ax=ax)
    ax.set_title(f'Churn by {col}')
    ax.set_ylabel('Churn Rate (%)')
fig.tight_layout()
fig.savefig('visuals/11_demographic_churn.png', dpi=150)
plt.close(fig)

kpi(f"\n=== TOP CHURN DRIVERS (highest-risk segments) ===")
kpi(f"Month-to-month contract churn rate: {contract_churn.iloc[0]*100:.1f}% "
    f"vs Two year: {contract_churn.get('Two year',0)*100:.1f}%")
kpi(f"Electronic check payment churn rate: {payment_churn.get('Electronic check',0)*100:.1f}% "
    f"vs Bank transfer (automatic): {payment_churn.get('Bank transfer (automatic)',0)*100:.1f}%")
kpi(f"Fiber optic churn rate: {internet_churn.get('Fiber optic',0)*100:.1f}% "
    f"vs DSL: {internet_churn.get('DSL',0)*100:.1f}%")
kpi(f"0 add-on services churn rate: {service_churn.get(0,0)*100:.1f}% "
    f"vs 5-6 services: {service_churn.reindex([5,6]).mean()*100:.1f}%")

with open('outputs/kpi_summary.txt', 'w') as f:
    f.write('\n'.join(kpi_lines))

print("\nAll visuals saved to visuals/. KPI summary saved to outputs/kpi_summary.txt")
