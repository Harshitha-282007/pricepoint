"""
Elasticity model with conditional, time-varying elasticity by promo state.

Each SKU has:
- normal elasticity = beta_price
- promo elasticity = beta_price + beta_price_promo

The optimizer still uses final_elasticity as the single scalar value while the
model output keeps separate promo-aware coefficients for analysis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

MIN_OBS = 20
MIN_PRICE_CV_PCT = 5.0
SIG_LEVEL = 0.05
VIF_THRESHOLD = 5.0


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Week_Start_Date"])
    df["Stockout_Flag"] = pd.to_numeric(df["Stockout_Flag"], errors="coerce").fillna(0)
    df["On_Promo"] = df["On_Promo"].astype(bool)
    df["Avg_Discount_Pct"] = pd.to_numeric(df["Avg_Discount_Pct"], errors="coerce").fillna(0)
    df["Avg_Competitor_Price_INR"] = pd.to_numeric(df["Avg_Competitor_Price_INR"], errors="coerce")
    df["Avg_Effective_Price_INR"] = pd.to_numeric(df["Avg_Effective_Price_INR"], errors="coerce")
    df["Quantity_Sold"] = pd.to_numeric(df["Quantity_Sold"], errors="coerce").fillna(0)
    df["Log_Price"] = np.log(df["Avg_Effective_Price_INR"].replace(0, np.nan))
    df["Log_Qty"] = np.log(df["Quantity_Sold"].where(df["Quantity_Sold"] > 0))
    return df


def calculate_vif(df: pd.DataFrame, features: list[str]) -> dict:
    X = df[features].astype(float)
    vif_values = {}
    for feature in features:
        idx = X.columns.get_loc(feature)
        vif_values[feature] = float(variance_inflation_factor(X.values, idx))
    return vif_values


def robust_lstsq_coef(X: pd.DataFrame, y: pd.Series) -> tuple[float, str]:
    """Numerical fallback only; not a replacement for HAC OLS inference."""
    X_clean = X.replace([np.inf, -np.inf], np.nan).dropna()
    y_clean = y.loc[X_clean.index].replace([np.inf, -np.inf], np.nan).dropna()
    X_clean = X_clean.loc[y_clean.index]

    if len(X_clean) < 3 or X_clean.shape[1] == 0:
        return np.nan, "insufficient_data"

    X_design = sm.add_constant(X_clean, has_constant="add")
    try:
        beta, *_ = np.linalg.lstsq(X_design.values, y_clean.values, rcond=None)
        log_price_idx = X_design.columns.get_loc("Log_Price") if "Log_Price" in X_design.columns else 1
        return float(beta[log_price_idx]), "lstsq"
    except Exception:
        return np.nan, "failed"


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d[d["Stockout_Flag"] == 0].copy()
    d = d[d["Quantity_Sold"] > 0].copy()
    d["Log_Comp_Price"] = np.log(d["Avg_Competitor_Price_INR"].replace(0, np.nan))
    d["Price_Promo_Interaction"] = d["Log_Price"] * d["On_Promo"].astype(float)
    d["Quarter"] = d["Week_Start_Date"].dt.quarter
    quarter_dummies = pd.get_dummies(d["Quarter"], prefix="Q", drop_first=True)
    d = pd.concat([d, quarter_dummies], axis=1)
    return d


def estimate_category_elasticity(df_category: pd.DataFrame) -> float:
    d = build_model_frame(df_category)
    if len(d) < 2:
        return np.nan

    feature_cols = [
        "Log_Price",
        "Log_Comp_Price",
        "Avg_Discount_Pct",
        "On_Promo",
        "Price_Promo_Interaction",
    ] + [c for c in d.columns if c.startswith("Q_") and d[c].nunique() > 1]

    X = d[feature_cols].astype(float)
    y = d["Log_Qty"].astype(float)

    try:
        X_design = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_design, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        return float(model.params["Log_Price"])
    except Exception:
        beta, _ = robust_lstsq_coef(X, y)
        return float(beta) if np.isfinite(beta) else np.nan


def run_sku_regression(d: pd.DataFrame) -> dict:
    n_obs = len(d)
    price_cv = 100 * d["Avg_Effective_Price_INR"].std(ddof=1) / d["Avg_Effective_Price_INR"].mean() if d["Avg_Effective_Price_INR"].mean() else 0

    flags = []
    if n_obs < MIN_OBS:
        flags.append(f"TOO_FEW_OBS(n={n_obs})")
    if price_cv < MIN_PRICE_CV_PCT:
        flags.append(f"LOW_PRICE_VARIATION(cv={price_cv:.1f}%)")

    result = {
        "n_obs": n_obs,
        "price_cv_pct": round(float(price_cv), 2),
        "sku_elasticity": np.nan,
        "elasticity": np.nan,
        "normal_elasticity": np.nan,
        "promo_elasticity": np.nan,
        "category_elasticity": np.nan,
        "final_elasticity": np.nan,
        "elasticity_source": "UNAVAILABLE",
        "std_err": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "significant": False,
        "vif_log_price": np.nan,
        "vif_log_comp_price": np.nan,
        "high_multicollinearity": False,
        "classification": "INSUFFICIENT_DATA" if flags else None,
        "data_flags": ";".join(flags) if flags else "OK",
    }

    if n_obs < MIN_OBS:
        result["classification"] = "INSUFFICIENT_DATA"
        return result

    feature_cols = [
        "Log_Price",
        "Log_Comp_Price",
        "Avg_Discount_Pct",
        "On_Promo",
        "Price_Promo_Interaction",
    ]
    quarter_cols = [c for c in d.columns if c.startswith("Q_") and d[c].nunique() > 1]
    feature_cols += quarter_cols

    X = d[feature_cols].astype(float)
    y = d["Log_Qty"].astype(float)

    try:
        X_design = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_design, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 2})

        beta_price = float(model.params["Log_Price"])
        beta_promo = float(model.params.get("Price_Promo_Interaction", 0.0))
        normal_elasticity = beta_price
        promo_elasticity = beta_price + beta_promo

        se = float(model.bse["Log_Price"])
        pval = float(model.pvalues["Log_Price"])

        vif_values = calculate_vif(X, ["Log_Price", "Log_Comp_Price"])
        high_vif = any(vif_value > VIF_THRESHOLD for vif_value in vif_values.values())

        result.update({
            "sku_elasticity": round(normal_elasticity, 4),
            "elasticity": round(normal_elasticity, 4),
            "normal_elasticity": round(normal_elasticity, 4),
            "promo_elasticity": round(promo_elasticity, 4),
            "final_elasticity": round(normal_elasticity, 4),
            "std_err": round(se, 4),
            "p_value": round(pval, 4),
            "r_squared": round(float(model.rsquared), 4),
            "significant": bool(pval < SIG_LEVEL),
            "vif_log_price": round(float(vif_values.get("Log_Price", np.nan)), 4),
            "vif_log_comp_price": round(float(vif_values.get("Log_Comp_Price", np.nan)), 4),
            "high_multicollinearity": bool(high_vif),
        })

        if high_vif:
            result["data_flags"] = f"{result['data_flags']};HIGH_VIF" if result["data_flags"] != 'OK' else 'HIGH_VIF'

        if flags:
            result["classification"] = "INSUFFICIENT_DATA"
        elif not result["significant"]:
            result["classification"] = "NOT_SIGNIFICANT"
        elif normal_elasticity < 0:
            result["classification"] = "ELASTIC" if abs(normal_elasticity) > 1 else "INELASTIC"
        else:
            result["classification"] = "REVIEW_POSITIVE_COEF"

    except Exception as e:
        beta_fallback, method = robust_lstsq_coef(X, y)
        if np.isfinite(beta_fallback):
            normal_elasticity = float(beta_fallback)
            # If the promo interaction is not estimable in a fallback case, keep the
            # most conservative assumption: promo elasticity remains a separate value
            # but defaults to the base elasticity rather than silently dropping it.
            promo_elasticity = float(beta_fallback)
            result.update({
                "sku_elasticity": round(normal_elasticity, 4),
                "elasticity": round(normal_elasticity, 4),
                "normal_elasticity": round(normal_elasticity, 4),
                "promo_elasticity": round(promo_elasticity, 4),
                "final_elasticity": round(normal_elasticity, 4),
                "std_err": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "significant": False,
                "classification": "FALLBACK_LSTSQ",
                "data_flags": f"{result['data_flags']};FALLBACK_LSTSQ({method})" if result['data_flags'] != 'OK' else f"FALLBACK_LSTSQ({method})",
            })
        else:
            result["classification"] = "MODEL_ERROR"
            result["data_flags"] = f"ERROR: {e}"

    return result


def run_all_skus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku_id, df_sku in df.groupby("SKU_ID"):
        category = df_sku["Category"].iloc[0]
        product_name = df_sku["Product_Name"].iloc[0]
        d = build_model_frame(df_sku)
        res = run_sku_regression(d)
        category_beta = estimate_category_elasticity(df[df["Category"] == category].copy())
        res.update({
            "SKU_ID": sku_id,
            "Category": category,
            "Product_Name": product_name,
            "category_elasticity": np.nan if pd.isna(category_beta) else round(float(category_beta), 4),
            "final_elasticity": np.nan,
            "elasticity_source": "UNAVAILABLE",
        })

        sku_is_reliable = (
            not bool(res["classification"] == "INSUFFICIENT_DATA")
            and bool(res["significant"])
            and not bool(res["high_multicollinearity"])
            and pd.notna(res["normal_elasticity"])
        )

        if sku_is_reliable:
            res["final_elasticity"] = res["normal_elasticity"]
            res["elasticity_source"] = "sku"
        elif not pd.isna(category_beta):
            res["final_elasticity"] = round(float(category_beta), 4)
            res["elasticity_source"] = "category"
            if pd.notna(res["normal_elasticity"]):
                res["final_elasticity"] = res["normal_elasticity"]

        rows.append(res)

    out = pd.DataFrame(rows)
    col_order = [
        "SKU_ID",
        "Product_Name",
        "Category",
        "n_obs",
        "price_cv_pct",
        "sku_elasticity",
        "normal_elasticity",
        "promo_elasticity",
        "category_elasticity",
        "final_elasticity",
        "elasticity_source",
        "elasticity",
        "std_err",
        "p_value",
        "r_squared",
        "significant",
        "vif_log_price",
        "vif_log_comp_price",
        "high_multicollinearity",
        "classification",
        "data_flags",
    ]
    return out[col_order].sort_values("final_elasticity", na_position="last")


def build_pricing_output(df: pd.DataFrame, elasticity_results: pd.DataFrame) -> pd.DataFrame:
    price_cols = [
        "SKU_ID",
        "normal_elasticity",
        "promo_elasticity",
        "final_elasticity",
        "p_value",
    ]
    sku_elasticity = elasticity_results[price_cols].rename(columns={
        "final_elasticity": "elasticity",
        "p_value": "elasticity_p_value",
    })

    pricing_df = df.copy().merge(sku_elasticity, on="SKU_ID", how="left")
    pricing_df["is_on_promo"] = pricing_df["On_Promo"].astype(bool)
    pricing_df["final_elasticity"] = np.where(
        pricing_df["is_on_promo"],
        pricing_df["promo_elasticity"],
        pricing_df["normal_elasticity"],
    )
    pricing_df["elasticity"] = pricing_df["final_elasticity"]

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
        "is_on_promo",
        "normal_elasticity",
        "promo_elasticity",
        "final_elasticity",
        "elasticity",
        "competitor_price",
        "margin_target",
        "elasticity_p_value",
    ]
    return pricing_df[output_cols]


if __name__ == "__main__":
    path = r"data\processed\test_dataset.csv"
    df = load_data(path)
    results = run_all_skus(df)
    print(results.to_string(index=False))
    results.to_csv(r"data\processed\elasticity_results_new_new.csv", index=False)

    pricing_output = build_pricing_output(df, results)
    pricing_output.to_csv(r"data\processed\pricing_output_new_new.csv", index=False)
    print(f"\nSaved to elasticity_results_new_new.csv | {len(results)} SKU(s) processed")
    print(f"Saved to pricing_output_new_new.csv | {len(pricing_output)} rows, same as input data")
