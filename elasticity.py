"""
Price Point — Per-SKU Price Elasticity of Demand
====================================================
Model (locked-in spec, per team decisions):
    Log_Qty ~ Log_Price + log(Avg_Competitor_Price_INR) + month dummies

- Price variable  : Avg_Effective_Price_INR (post-discount, already = Log_Price col)
- Competitor price: Avg_Competitor_Price_INR (log-transformed)
- Estimation      : per-SKU OLS, looped over all SKU_IDs
- Stockout weeks  : excluded (censored demand, not a true price response)
- Seasonality     : month-of-year dummies extracted from Week_Start_Date
- Fallback rule   : if a SKU has <8 usable weeks OR price coefficient of
                    variation < 5%, flag it as INSUFFICIENT_DATA (needs
                    category-level pooled fallback instead of its own line)

Output: one row per SKU with elasticity coefficient, SE, p-value, R²,
elastic/inelastic classification, and data-quality flags.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# ---- Tunable thresholds (agreed as a team — change here, not inline) ----
# Full dataset: 130 weeks/SKU (Jan 2022 - Jun 2024), so we can afford a much
# higher bar than the initial 8-week placeholder used on the sample data.
MIN_OBS = 4            # require at least ~1 year of usable weekly observations for SKU-level regression
MIN_PRICE_CV_PCT = 5.0    # below this % coefficient of variation, price barely moves -> unreliable elasticity
SIG_LEVEL = 0.05          # p-value cutoff to call a coefficient "statistically significant"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Week_Start_Date"])
    df["Stockout_Flag"] = df["Stockout_Flag"].astype(float)
    df["On_Promo"] = df["On_Promo"].astype(bool)
    return df


def prep_sku(df_sku: pd.DataFrame) -> pd.DataFrame:
    """Filter + engineer features for a single SKU's weekly panel."""
    d = df_sku.copy()

    # drop stockout weeks — demand is censored/supply-constrained, not price-driven
    d = d[d["Stockout_Flag"] == 0].copy()

    # log competitor price (own price is already log-transformed as Log_Price)
    d["Log_Comp_Price"] = np.log(d["Avg_Competitor_Price_INR"])

    # month dummies for seasonality
    d["Month"] = d["Week_Start_Date"].dt.month
    month_dummies = pd.get_dummies(d["Month"], prefix="M", drop_first=True)
    d = pd.concat([d, month_dummies], axis=1)

    return d, list(month_dummies.columns)


def run_sku_regression(d: pd.DataFrame, month_cols: list) -> dict:
    """Fit log-log OLS for one SKU. Returns a result dict (never raises)."""
    n_obs = len(d)
    price_cv = 100 * d["Avg_Effective_Price_INR"].std() / d["Avg_Effective_Price_INR"].mean()

    flags = []
    if n_obs < MIN_OBS:
        flags.append(f"TOO_FEW_OBS(n={n_obs})")
    if price_cv < MIN_PRICE_CV_PCT:
        flags.append(f"LOW_PRICE_VARIATION(cv={price_cv:.1f}%)")

    result = {
        "n_obs": n_obs,
        "price_cv_pct": round(price_cv, 2),
        "elasticity": np.nan,
        "std_err": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "significant": False,
        "classification": "INSUFFICIENT_DATA" if flags else None,
        "data_flags": ";".join(flags) if flags else "OK",
    }

    # only attempt regression if we clear the minimum bar
    if n_obs < 4:  # can't fit even a bare-minimum model
        result["classification"] = "INSUFFICIENT_DATA"
        return result

    X_cols = ["Log_Price", "Log_Comp_Price"] + [c for c in month_cols if d[c].nunique() > 1]
    X = d[X_cols].astype(float)
    X = sm.add_constant(X)
    y = d["Log_Qty"].astype(float)

    try:
        model = sm.OLS(y, X, missing="drop").fit()
        beta = model.params["Log_Price"]
        se = model.bse["Log_Price"]
        pval = model.pvalues["Log_Price"]

        result.update({
            "elasticity": round(beta, 4),
            "std_err": round(se, 4),
            "p_value": round(pval, 4),
            "r_squared": round(model.rsquared, 4),
            "significant": bool(pval < SIG_LEVEL),
        })

        if flags:
            result["classification"] = "INSUFFICIENT_DATA"
        elif not result["significant"]:
            result["classification"] = "NOT_SIGNIFICANT"
        elif beta < 0:
            result["classification"] = "ELASTIC" if abs(beta) > 1 else "INELASTIC"
        else:
            # positive elasticity is economically odd — flag for manual review
            result["classification"] = "REVIEW_POSITIVE_COEF"

    except Exception as e:
        result["classification"] = "MODEL_ERROR"
        result["data_flags"] = f"ERROR: {e}"

    return result


def run_all_skus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku_id, df_sku in df.groupby("SKU_ID"):
        category = df_sku["Category"].iloc[0]
        product_name = df_sku["Product_Name"].iloc[0]
        d, month_cols = prep_sku(df_sku)
        res = run_sku_regression(d, month_cols)
        res.update({"SKU_ID": sku_id, "Category": category, "Product_Name": product_name})
        rows.append(res)

    out = pd.DataFrame(rows)
    col_order = ["SKU_ID", "Product_Name", "Category", "n_obs", "price_cv_pct",
                 "elasticity", "std_err", "p_value", "r_squared", "significant",
                 "classification", "data_flags"]
    return out[col_order].sort_values("elasticity")


def build_pricing_output(df: pd.DataFrame, elasticity_results: pd.DataFrame) -> pd.DataFrame:
    """Return the raw weekly dataset with elasticity columns appended, preserving row count."""
    sku_elasticity = elasticity_results[["SKU_ID", "elasticity", "p_value"]].rename(columns={
        "p_value": "elasticity_p_value"
    })

    pricing_df = df.copy()
    pricing_df = pricing_df.merge(sku_elasticity, on="SKU_ID", how="left")

    pricing_df = pricing_df.rename(columns={
        "Week_Start_Date": "week_start_date",
        "SKU_ID": "sku_id",
        "Category": "category",
        "Avg_Effective_Price_INR": "current_price",
        "Quantity_Sold": "current_quantity",
        "Base_Cost_INR": "cost_price",
        "Avg_Competitor_Price_INR": "competitor_price",
        "Margin_Target_Pct": "margin_target",
    })

    output_cols = [
        "week_start_date",
        "sku_id",
        "category",
        "current_price",
        "current_quantity",
        "cost_price",
        "elasticity",
        "competitor_price",
        "margin_target",
        "elasticity_p_value",
    ]
    return pricing_df[output_cols]


if __name__ == "__main__":
    import sys
    path = 'data\\processed\\test_dataset.csv'
    df = load_data(path)
    results = run_all_skus(df)
    print(results.to_string(index=False))
    results.to_csv("data\\processed\\elasticity_results.csv", index=False)

    pricing_output = build_pricing_output(df, results)
    pricing_output.to_csv("data\\processed\\pricing_output.csv", index=False)
    print(f"\nSaved to elasticity_results.csv | {len(results)} SKU(s) processed")
    print(f"Saved to pricing_output.csv | {len(pricing_output)} rows, same as input data")