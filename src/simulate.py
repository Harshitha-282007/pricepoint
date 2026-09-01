"""Simulation utilities for the price-point project."""

def simulate(df):

    # -------------------------
    # Historical scenario
    # -------------------------

    df["historical_revenue"] = (
        df["current_price"]
        * df["current_quantity"]
    )

    df["historical_profit"] = (
        df["current_price"]
        - df["cost_price"]
    ) * df["current_quantity"]

    # -------------------------
    # Percentage changes
    # -------------------------

    df["revenue_change_pct"] = (
        df["predicted_revenue"]
        / df["historical_revenue"] - 1
    ) * 100

    df["profit_change_pct"] = (
        df["predicted_profit"]
        / df["historical_profit"] - 1
    ) * 100

    return df