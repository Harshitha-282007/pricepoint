"""
Loads the 5 raw CSVs, cleans each, and builds a single master dataset
at SKU-week grain (aligned to Monday week starts, matching competitor
and inventory data). Transactions are aggregated from daily -> weekly.

Usage:
    from data_cleaning import build_master_dataset
    master = build_master_dataset()
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# Loaders

def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "transactions.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    return df


def load_competitor_prices() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "competitor_prices.csv")
    df["Week_Start_Date"] = pd.to_datetime(df["Week_Start_Date"], format="%Y-%m-%d")
    return df


def load_inventory_levels() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "inventory_levels.csv")
    df["Week_Start_Date"] = pd.to_datetime(df["Week_Start_Date"], format="%Y-%m-%d")
    return df


def load_product_master() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "product_master.csv")


def load_promo_calendar(max_date: str = "2024-12-31") -> pd.DataFrame:
    """
    promo_calendar.csv has trailing junk columns (Unnamed: 4-9) from a
    stray delimiter in the source file, plus a few promo events that run
    past the transaction data's date range - both are dropped here.
    """
    df = pd.read_csv(RAW_DIR / "promo_calendar.csv")
    df = df[["Promo_Name", "Start_Date", "End_Date", "Discount_Pct"]].copy()
    df["Start_Date"] = pd.to_datetime(df["Start_Date"], format="%d-%m-%Y")
    df["End_Date"] = pd.to_datetime(df["End_Date"], format="%d-%m-%Y")
    df = df[df["Start_Date"] <= pd.to_datetime(max_date)].reset_index(drop=True)
    return df


# Cleaning / feature steps

def add_week_start(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add a Monday-aligned Week_Start_Date column, matching cp/inv grain."""
    df = df.copy()
    df["Week_Start_Date"] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit="D")
    return df


def flag_promo_days(tx: pd.DataFrame, promo: pd.DataFrame) -> pd.DataFrame:
    """Tag each transaction row with whether it falls within a promo window."""
    tx = tx.copy()
    tx["On_Promo"] = False
    tx["Promo_Name"] = pd.Series([pd.NA] * len(tx), dtype="object")
    for _, row in promo.iterrows():
        mask = (tx["Date"] >= row["Start_Date"]) & (tx["Date"] <= row["End_Date"])
        tx.loc[mask, "On_Promo"] = True
        tx.loc[mask, "Promo_Name"] = row["Promo_Name"]
    return tx


def aggregate_transactions_weekly(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily transactions to SKU-week grain to match competitor/
    inventory data. Price is quantity-weighted (a straight average would
    understate the price actually paid on high-volume days).
    """
    tx = add_week_start(tx, "Date")
    tx = tx.copy()
    tx["_rev"] = tx["Effective_Price_INR"] * tx["Quantity_Sold"]
    tx["_listed_rev"] = tx["Listed_Price_INR"] * tx["Quantity_Sold"]

    agg = tx.groupby(["SKU_ID", "Category", "Week_Start_Date"]).agg(
        Quantity_Sold=("Quantity_Sold", "sum"),
        Total_Revenue=("_rev", "sum"),
        Total_Listed_Revenue=("_listed_rev", "sum"),
        Avg_Discount_Pct=("Discount_Pct", "mean"),
        Max_Discount_Pct=("Discount_Pct", "max"),
        On_Promo=("On_Promo", "max"),
    ).reset_index()

    agg["Avg_Effective_Price_INR"] = agg["Total_Revenue"] / agg["Quantity_Sold"]
    agg["Avg_Listed_Price_INR"] = agg["Total_Listed_Revenue"] / agg["Quantity_Sold"]
    agg = agg.drop(columns=["Total_Listed_Revenue"])
    return agg


def aggregate_competitor_prices(cp: pd.DataFrame) -> pd.DataFrame:
    """
    Average across the 3 competitors per SKU-week, plus keep min (the
    toughest competitor) since that's usually what matters for positioning.
    """
    agg = cp.groupby(["SKU_ID", "Week_Start_Date"]).agg(
        Avg_Competitor_Price_INR=("Competitor_Price_INR", "mean"),
        Min_Competitor_Price_INR=("Competitor_Price_INR", "min"),
        Max_Competitor_Price_INR=("Competitor_Price_INR", "max"),
    ).reset_index()
    return agg


# Master build

def build_master_dataset(save: bool = True) -> pd.DataFrame:
    tx = load_transactions()
    cp = load_competitor_prices()
    inv = load_inventory_levels()
    pm = load_product_master()
    promo = load_promo_calendar()

    tx = flag_promo_days(tx, promo)
    tx_weekly = aggregate_transactions_weekly(tx)
    cp_weekly = aggregate_competitor_prices(cp)

    inv_slim = inv[["SKU_ID", "Week_Start_Date", "Beginning_Inventory_Units",
                     "Ending_Inventory_Units", "Stockout_Flag"]]

    master = tx_weekly.merge(cp_weekly, on=["SKU_ID", "Week_Start_Date"], how="left")
    master = master.merge(inv_slim, on=["SKU_ID", "Week_Start_Date"], how="left")
    master = master.merge(
        pm[["SKU_ID", "Product_Name", "Base_Cost_INR", "Margin_Target_Pct", "Shelf_Life_Days"]],
        on="SKU_ID", how="left"
    )

    # Derived fields useful for elasticity / margin work downstream
    master["Gross_Margin_Pct"] = (
        (master["Avg_Effective_Price_INR"] - master["Base_Cost_INR"]) / master["Avg_Effective_Price_INR"]
    )
    master["Price_vs_Competitor_Gap_Pct"] = (
        (master["Avg_Effective_Price_INR"] - master["Avg_Competitor_Price_INR"]) / master["Avg_Competitor_Price_INR"]
    )
    master["Log_Price"] = np.log(master["Avg_Effective_Price_INR"])
    master["Log_Qty"] = np.log(master["Quantity_Sold"].replace(0, np.nan))

    # Drop the partial first week: transactions start 2022-01-01 (a Saturday),
    # so its Monday-aligned week (2021-12-27) has only 2 days of transactions
    # and no matching competitor/inventory data (those start 2022-01-03).
    first_full_week = cp_weekly["Week_Start_Date"].min()
    master = master[master["Week_Start_Date"] >= first_full_week]

    master = master.sort_values(["SKU_ID", "Week_Start_Date"]).reset_index(drop=True)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        master.to_csv(PROCESSED_DIR / "master_dataset.csv", index=False)

    return master


def train_test_split_by_date(
    df: pd.DataFrame,
    cutoff_date: str = "2024-07-01",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split for the SKU-week panel.

    A random row split would leak information across the split (a week
    sitting in train right next to the same SKU's neighbouring week in
    test looks almost identical, so any model would appear more accurate
    than it really is). Instead, every SKU is split on the same cutoff
    date - all weeks before it go to train, all weeks on/after it go to
    test - so the test set represents a genuine future holdout period.

    Args:
        df: master dataset (or any subset with a Week_Start_Date column)
        cutoff_date: first week (YYYY-MM-DD) to include in the test set
        save: if True, writes train_dataset.csv / test_dataset.csv to
              data/processed/

    Returns:
        (train_df, test_df)
    """
    df = df.copy()
    df["Week_Start_Date"] = pd.to_datetime(df["Week_Start_Date"])
    cutoff = pd.to_datetime(cutoff_date)

    train = df[df["Week_Start_Date"] < cutoff].reset_index(drop=True)
    test = df[df["Week_Start_Date"] >= cutoff].reset_index(drop=True)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train.to_csv(PROCESSED_DIR / "train_dataset.csv", index=False)
        test.to_csv(PROCESSED_DIR / "test_dataset.csv", index=False)

    return train, test


if __name__ == "__main__":
    df = build_master_dataset()
    print(f"Master dataset built: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Saved to {PROCESSED_DIR / 'master_dataset.csv'}")
    print(f"\nNulls per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    train_df, test_df = train_test_split_by_date(df, cutoff_date="2024-07-01")
    print(f"\nTrain: {train_df.shape[0]} rows ({train_df['Week_Start_Date'].min().date()} to {train_df['Week_Start_Date'].max().date()})")
    print(f"Test:  {test_df.shape[0]} rows ({test_df['Week_Start_Date'].min().date()} to {test_df['Week_Start_Date'].max().date()})")
