"""
Week 4 Task: Predictive Modeling and Optimization in Logistics Systems
-------------------------------------------------------------------
This script trains and tunes a Random Forest model to predict
delivery time, evaluates it, and applies the results to a simple
rule-based optimization for vehicle assignment and delay flagging.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. Load cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv("logistics_orders_clean.csv")

features = [
    "distance_km", "shipment_volume", "vehicle_type",
    "delivery_zone", "day_of_week", "hour_of_day"
]
target = "delivery_hours"

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 2. Build preprocessing + model pipeline
# ---------------------------------------------------------
categorical_cols = ["vehicle_type", "delivery_zone", "day_of_week"]

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
])

# ---------------------------------------------------------
# 3. Hyperparameter tuning with cross-validation
# ---------------------------------------------------------
param_grid = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [6, 10, 14],
}
grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_root_mean_squared_error")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ---------------------------------------------------------
# 4. Evaluate on held-out test set
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} hrs, MAE: {mae:.2f} hrs, R2: {r2:.3f}")
print(f"Best parameters: {grid.best_params_}")

# ---------------------------------------------------------
# 5. Optimization: flag high delay-risk orders using predictions
# ---------------------------------------------------------
df["predicted_hours"] = best_model.predict(df[features])
df["delay_risk"] = df["predicted_hours"] > df["promised_hours"]

high_risk_orders = df[df["delay_risk"]]
print(f"{len(high_risk_orders)} orders flagged as high delay-risk "
      f"for proactive rescheduling")
