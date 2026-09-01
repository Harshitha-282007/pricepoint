"""Optimization utilities for the price-point project."""

import numpy as np
import pandas as pd
from demand_model import predict_quantity


def optimize_price(row):
    p0 = row["current_price"]
    q0 = row["current_quantity"]
    cost = row["cost_price"]
    elasticity = row["elasticity"]
    competitor = row["competitor_price"]
    margin_target = row["margin_target"]
    minimum_price = cost / (1 - margin_target)

    lower_price = max(p0 * 0.80, minimum_price)
    volume_max_price = p0 * (0.97 ** (1 / elasticity))
    upper_price = upper_price = min(p0 * 1.50, volume_max_price)

    if lower_price > upper_price:
        return None

    prices = np.linspace(lower_price, upper_price, 1000)

    candidates = []

    for price in prices:
        quantity = predict_quantity(
            p0,
            q0,
            price,
            elasticity
        )

        volume_change = quantity / q0 - 1
        revenue = price * quantity
        profit = (price - cost) * quantity
        gross_margin = (price - cost) / price

        candidates.append({
            "price": price,
            "quantity": quantity,
            "volume_change": volume_change,
            "revenue": revenue,
            "profit": profit,
            "gross_margin": gross_margin,
        })

    valid_candidates = [
        x for x in candidates
        if x["volume_change"] >= -0.03
        and x["price"] <= competitor * 1.10
    ]

    if not valid_candidates:
        return None

    best = max(
        valid_candidates,
        key=lambda x: x["profit"]
    )

    return {
        "recommended_price": best["price"],
        "predicted_quantity": best["quantity"],
        "predicted_revenue": best["revenue"],
        "predicted_gross_margin": best["gross_margin"] * 100,
        "predicted_profit": best["profit"],
        "volume_change_pct": best["volume_change"] * 100,
        "price_change_pct": (best["price"] / p0 - 1) * 100,
    }

def optimize_dataframe(df):
    results = []
    for _, row in df.iterrows():
        result = optimize_price(row)

        if result is None:
            result = {
                "recommended_price": np.nan,
                "predicted_quantity": np.nan,
                "predicted_revenue": np.nan,
                "predicted_profit": np.nan,
                "volume_change_pct": np.nan,
                "price_change_pct": np.nan,
                "predicted_gross_margin": np.nan,
            }

        results.append(result)

    return pd.DataFrame(results, index=df.index)