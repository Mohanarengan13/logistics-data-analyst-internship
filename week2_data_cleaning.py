"""
Week 2 Task: Data Collection, Cleaning, and Preprocessing for
Logistics Analysis
-------------------------------------------------------------------
This script implements the cleaning and preprocessing pipeline
described in the Week 2 report: deduplication, category
standardization, missing-value imputation, outlier flagging,
and feature normalization.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1. Load raw logistics dataset
# ---------------------------------------------------------
df = pd.read_csv(
    "logistics_orders_raw.csv",
    parse_dates=["order_time", "delivery_time"]
)

# Delivery duration in hours (needed before imputation step below)
df["delivery_hours"] = (
    df["delivery_time"] - df["order_time"]
).dt.total_seconds() / 3600

# ---------------------------------------------------------
# 2. Remove duplicate orders, keep the most recently updated record
# ---------------------------------------------------------
df = df.sort_values("order_time").drop_duplicates(
    subset="order_id", keep="last"
)

# ---------------------------------------------------------
# 3. Standardize vehicle_type categories
# ---------------------------------------------------------
vehicle_map = {
    "van": "Van", "vn": "Van",
    "truck": "Truck",
    "motorbike": "Motorbike", "bike": "Motorbike",
}
df["vehicle_type"] = (
    df["vehicle_type"].str.strip().str.lower().map(vehicle_map)
)

# ---------------------------------------------------------
# 4. Impute missing delivery_time using median duration per
#    distance band and vehicle type
# ---------------------------------------------------------
df["distance_band"] = pd.cut(df["distance_km"], bins=[0, 5, 15, 30, 100])
median_duration = df.groupby(
    ["distance_band", "vehicle_type"]
)["delivery_hours"].transform("median")
df["delivery_hours"] = df["delivery_hours"].fillna(median_duration)

# ---------------------------------------------------------
# 5. Flag outliers in delivery_cost using the IQR method
# ---------------------------------------------------------
Q1, Q3 = df["delivery_cost"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df["cost_outlier"] = ~df["delivery_cost"].between(lower, upper)

# ---------------------------------------------------------
# 6. Normalize numeric features for downstream modeling
# ---------------------------------------------------------
scaler = MinMaxScaler()
df[["distance_km_norm", "delivery_cost_norm"]] = scaler.fit_transform(
    df[["distance_km", "delivery_cost"]]
)

# ---------------------------------------------------------
# 7. Save cleaned dataset
# ---------------------------------------------------------
df.to_csv("logistics_orders_clean.csv", index=False)
print("Cleaned dataset saved to logistics_orders_clean.csv")
print(f"Flagged outliers: {df['cost_outlier'].sum()} rows")
