"""
Week 3 Task: Advanced Data Analysis and Visualization in Logistics
-------------------------------------------------------------------
This script performs exploratory data analysis and generates the
four visualizations described in the Week 3 report: a delivery-time
histogram, a distance-vs-cost scatter plot, a cost-by-vehicle-type
box plot, and a zone/day-of-week heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load cleaned dataset (output of week2_data_cleaning.py)
# ---------------------------------------------------------
df = pd.read_csv("logistics_orders_clean.csv")

# ---------------------------------------------------------
# 2. Summary statistics and correlation analysis
# ---------------------------------------------------------
print("Summary statistics:")
print(df[["delivery_hours", "delivery_cost", "distance_km", "shipment_volume"]].describe())

print("\nCorrelation matrix:")
print(df[["distance_km", "shipment_volume", "delivery_cost", "delivery_hours"]].corr())

# ---------------------------------------------------------
# 3. Histogram of delivery times
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["delivery_hours"], bins=30, kde=True)
plt.title("Distribution of Delivery Time (Hours)")
plt.xlabel("Delivery Hours")
plt.savefig("delivery_time_distribution.png")
plt.close()

# ---------------------------------------------------------
# 4. Scatter plot: distance vs cost, colored by vehicle type
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="distance_km", y="delivery_cost",
                 hue="vehicle_type", alpha=0.6)
plt.title("Delivery Cost vs Distance by Vehicle Type")
plt.savefig("cost_vs_distance.png")
plt.close()

# ---------------------------------------------------------
# 5. Box plot: cost distribution by vehicle type
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="vehicle_type", y="delivery_cost")
plt.title("Delivery Cost Distribution by Vehicle Type")
plt.savefig("cost_by_vehicle_type.png")
plt.close()

# ---------------------------------------------------------
# 6. Heatmap: average delivery time by zone and day of week
# ---------------------------------------------------------
df["day_of_week"] = pd.to_datetime(df["order_time"]).dt.day_name()
pivot = df.pivot_table(
    values="delivery_hours",
    index="delivery_zone",
    columns="day_of_week",
    aggfunc="mean"
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
plt.title("Average Delivery Time by Zone and Day of Week")
plt.savefig("delivery_time_heatmap.png")
plt.close()

print("\nAll visualizations saved as PNG files.")
