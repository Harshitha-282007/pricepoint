import pandas as pd

from optimizer import optimize_dataframe, optimize_price
from simulate import simulate

df = pd.read_csv(r"C:\Users\Mithra Bijumon\pricepoint\data\pricing_output_new_new (1).csv")

# Optimize each SKU-week row
optimization_results = optimize_dataframe(df)

df = pd.concat([df, optimization_results], axis=1)

print(df.columns.tolist())

# Simulation
results = simulate(df)

print(results.columns.tolist())

results.to_csv("simulation_results.csv", index=False)

print(results.head())

dashboard = results[
    [
        "week_start_date",
        "sku_id",
        "category",
        "current_price",
        "recommended_price",
        "predicted_gross_margin",
        "price_change_pct",
        "current_quantity",
        "predicted_quantity",
        "volume_change_pct",
        "historical_revenue",
        "cost_price",
        "predicted_revenue",
        "revenue_change_pct",
        "historical_profit",
        "predicted_profit",
        "profit_change_pct",
        "elasticity",
        "elasticity_p_value",
    ]
].copy()

dashboard["elasticity_significant"] = (
    dashboard["elasticity_p_value"] < 0.05
)

dashboard.to_csv("dashboard_data.csv", index=False)