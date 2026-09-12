"""
Generates a realistic synthetic telecom customer-churn dataset.
Column design mirrors the well-known "Telco Customer Churn" dataset structure,
so the project generalizes to real data if you swap this file out later.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

tenure = np.random.randint(0, 72, N)
monthly_charges = np.round(np.random.normal(65, 25, N).clip(18, 120), 2)
contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20])
tech_support = np.random.choice(["Yes", "No"], N, p=[0.35, 0.65])
online_security = np.random.choice(["Yes", "No"], N, p=[0.35, 0.65])
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N
)
num_support_calls = np.random.poisson(1.5, N)
total_charges = np.round(monthly_charges * np.maximum(tenure, 1) * np.random.uniform(0.9, 1.0, N), 2)

# Build churn probability from realistic risk factors, then sample labels.
risk = (
    1.8 * (contract == "Month-to-month")
    + 1.2 * (internet_service == "Fiber optic")
    + 1.0 * (tech_support == "No")
    + 0.8 * (online_security == "No")
    + 0.5 * (paperless_billing == "Yes")
    + 0.4 * (payment_method == "Electronic check")
    + 0.35 * num_support_calls
    - 0.06 * tenure
    - 1.0 * (contract == "Two year")
)
prob = 1 / (1 + np.exp(-(risk - 2.0)))
churn = np.random.binomial(1, prob)

df = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "contract": contract,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "online_security": online_security,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "num_support_calls": num_support_calls,
    "churn": churn,
})

df.to_csv("data.csv", index=False)
print(f"Generated {len(df)} rows. Churn rate: {df['churn'].mean():.2%}")
print(df.head())
