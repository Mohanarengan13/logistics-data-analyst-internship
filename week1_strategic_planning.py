"""
Week 1 Task: Strategic Planning and Data Exploration in Logistics
-------------------------------------------------------------------
This script illustrates the KPI calculation and initial zone-clustering
approach proposed in the Week 1 strategic planning report for a
last-mile logistics delivery operator.
"""

import pandas as pd
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. Load simulated logistics dataset
# ---------------------------------------------------------
orders = pd.read_csv(
    "logistics_orders.csv",
    parse_dates=["order_time", "delivery_time"]
)

# ---------------------------------------------------------
# 2. Compute delivery duration in hours
# ---------------------------------------------------------
orders["delivery_hours"] = (
    orders["delivery_time"] - orders["order_time"]
).dt.total_seconds() / 3600

# ---------------------------------------------------------
# 3. KPI: On-Time Delivery Rate (OTD%)
# ---------------------------------------------------------
orders["on_time"] = orders["delivery_hours"] <= orders["promised_hours"]
otd_rate = orders["on_time"].mean() * 100
print(f"On-Time Delivery Rate: {otd_rate:.2f}%")

# ---------------------------------------------------------
# 4. KPI: Average Delivery Cost per Order
# ---------------------------------------------------------
avg_cost = orders["delivery_cost"].mean()
print(f"Average Delivery Cost per Order: {avg_cost:.2f}")

# ---------------------------------------------------------
# 5. Zone assignment via K-Means clustering (for route planning)
# ---------------------------------------------------------
coords = orders[["delivery_lat", "delivery_lon"]]
kmeans = KMeans(n_clusters=6, random_state=42)
orders["zone"] = kmeans.fit_predict(coords)

# Zones can then be used to assign vans and build optimized routes
zone_summary = orders.groupby("zone")["delivery_cost"].mean()
print("\nAverage delivery cost by zone:")
print(zone_summary)
