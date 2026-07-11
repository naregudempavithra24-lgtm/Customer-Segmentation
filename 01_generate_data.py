"""
Step 1: Generate a realistic synthetic customer dataset.
Combines demographics + behavioral (RFM-style) purchase data.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

# ---- Demographics ----
age = np.random.normal(40, 13, N).clip(18, 75).astype(int)
gender = np.random.choice(['Female', 'Male', 'Other'], N, p=[0.49, 0.48, 0.03])
annual_income = np.random.lognormal(mean=10.8, sigma=0.45, size=N).clip(15000, 250000)
tenure_years = np.random.exponential(3, N).clip(0.1, 15)
region = np.random.choice(['North', 'South', 'East', 'West'], N, p=[0.28, 0.24, 0.26, 0.22])

# ---- Behavioral: build 4 latent "true" customer archetypes with noise ----
# Archetype weights per customer (soft assignment used only to generate correlated behavior)
archetype = np.random.choice(
    ['champion', 'loyal_low_value', 'at_risk', 'new_bargain'],
    N, p=[0.18, 0.32, 0.25, 0.25]
)

recency = np.zeros(N)     # days since last purchase (lower = more recent)
frequency = np.zeros(N)   # purchases per year
monetary = np.zeros(N)    # total annual spend
discount_affinity = np.zeros(N)  # 0-1, how much they rely on discounts
online_ratio = np.zeros(N)       # 0-1 share of purchases made online

for i, a in enumerate(archetype):
    if a == 'champion':
        recency[i] = np.random.exponential(8)
        frequency[i] = np.random.normal(24, 5)
        monetary[i] = np.random.normal(3200, 600)
        discount_affinity[i] = np.random.beta(2, 6)
        online_ratio[i] = np.random.beta(5, 3)
    elif a == 'loyal_low_value':
        recency[i] = np.random.exponential(20)
        frequency[i] = np.random.normal(14, 4)
        monetary[i] = np.random.normal(900, 200)
        discount_affinity[i] = np.random.beta(3, 4)
        online_ratio[i] = np.random.beta(3, 3)
    elif a == 'at_risk':
        recency[i] = np.random.exponential(120) + 60
        frequency[i] = np.random.normal(4, 2)
        monetary[i] = np.random.normal(600, 250)
        discount_affinity[i] = np.random.beta(3, 3)
        online_ratio[i] = np.random.beta(3, 4)
    else:  # new_bargain
        recency[i] = np.random.exponential(15)
        frequency[i] = np.random.normal(6, 2)
        monetary[i] = np.random.normal(300, 100)
        discount_affinity[i] = np.random.beta(6, 2)
        online_ratio[i] = np.random.beta(6, 2)

recency = recency.clip(0, 400)
frequency = frequency.clip(1, 40)
monetary = monetary.clip(50, 8000)
discount_affinity = discount_affinity.clip(0, 1)
online_ratio = online_ratio.clip(0, 1)

avg_order_value = monetary / frequency
preferred_category = np.random.choice(
    ['Electronics', 'Fashion', 'Grocery', 'Home & Living', 'Beauty'],
    N, p=[0.22, 0.24, 0.22, 0.17, 0.15]
)

df = pd.DataFrame({
    'CustomerID': [f'CUST{i:05d}' for i in range(1, N + 1)],
    'Age': age,
    'Gender': gender,
    'AnnualIncome': annual_income.round(0),
    'TenureYears': tenure_years.round(1),
    'Region': region,
    'Recency': recency.round(0),
    'Frequency': frequency.round(0),
    'Monetary': monetary.round(2),
    'AvgOrderValue': avg_order_value.round(2),
    'DiscountAffinity': discount_affinity.round(2),
    'OnlineRatio': online_ratio.round(2),
    'PreferredCategory': preferred_category,
})

df.to_csv('/home/claude/segmentation/customer_data.csv', index=False)
print(df.shape)
print(df.head())
print(df.describe())
