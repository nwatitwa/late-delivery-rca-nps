from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

TIMESTAMP_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
]


def load_raw_tables(data_dir: str | Path = "data/raw") -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        "orders": pd.read_csv(data_dir / "olist_orders_dataset.csv", low_memory=False),
        "items": pd.read_csv(data_dir / "olist_order_items_dataset.csv", low_memory=False),
        "sellers": pd.read_csv(data_dir / "olist_sellers_dataset.csv", low_memory=False),
        "customers": pd.read_csv(data_dir / "olist_customers_dataset.csv", low_memory=False),
    }


def parse_timestamps(df: pd.DataFrame, columns=TIMESTAMP_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def build_order_delivery_legs(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    tables = load_raw_tables(data_dir)
    orders = parse_timestamps(tables["orders"])
    customers = tables["customers"]
    sellers = tables["sellers"]

    items = tables["items"].copy()
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")
    item_agg = (
        items.groupby(["order_id", "seller_id"], as_index=False)
        .agg(
            order_item_count=("order_item_id", "count"),
            leg_price=("price", "sum"),
            leg_freight_value=("freight_value", "sum"),
            shipping_limit_date=("shipping_limit_date", "max"),
        )
    )

    base = orders.merge(
        customers[["customer_id", "customer_state", "customer_city", "customer_zip_code_prefix"]],
        on="customer_id",
        how="left",
    )
    base = base.rename(columns={"customer_zip_code_prefix": "customer_zip"})

    legs = item_agg.merge(
        sellers[["seller_id", "seller_state", "seller_city", "seller_zip_code_prefix"]],
        on="seller_id",
        how="left",
    ).rename(columns={"seller_zip_code_prefix": "seller_zip"})

    out = base.merge(legs, on="order_id", how="left")
    out["route"] = out["seller_state"].fillna("UNK") + "->" + out["customer_state"].fillna("UNK")
    return out


def quality_summary(df: pd.DataFrame) -> dict:
    critical_cols = [
        "order_id",
        "seller_id",
        "customer_id",
        "seller_state",
        "customer_state",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "shipping_limit_date",
    ]

    return {
        "row_counts": {
            "rows": len(df),
            "orders": int(df["order_id"].nunique()),
            "order_seller_legs": int(df[["order_id", "seller_id"]].drop_duplicates().shape[0]),
        },
        "coverage": {
            "customer_state_coverage": float(1.0 - df["customer_state"].isna().mean()),
            "seller_state_coverage": float(1.0 - df["seller_state"].isna().mean()),
        },
        "missing_rates": df[critical_cols].isna().mean().sort_values(ascending=False),
        "duplicate_order_seller": int(df.duplicated(["order_id", "seller_id"]).sum()),
        "status_distribution": df["order_status"].value_counts(normalize=True),
    }


def derive_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    out = parse_timestamps(df)

    out["estimated_delivery_days"] = (
        out["order_estimated_delivery_date"] - out["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    out["actual_delivery_days"] = (
        out["order_delivered_customer_date"] - out["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    out["delay_days"] = out["actual_delivery_days"] - out["estimated_delivery_days"]

    out["seller_processing_days"] = (
        out["order_delivered_carrier_date"] - out["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    out["carrier_transit_days"] = (
        out["order_delivered_customer_date"] - out["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86400.0

    delivered = out["order_status"].eq("delivered") & out["order_delivered_customer_date"].notna()
    out["is_delivered"] = delivered
    out["is_late"] = pd.Series(np.where(delivered, (out["delay_days"] > 0).astype(int), pd.NA), dtype="Int64")
    return out


def rank_failure_routes(df: pd.DataFrame, min_samples: int = 100, top_n: int = 5) -> pd.DataFrame:
    d = df[df["is_delivered"]].copy()
    g = (
        d.groupby(["seller_state", "customer_state"], dropna=False)
        .agg(
            volume=("order_id", "count"),
            failure_rate=("is_late", "mean"),
            mean_delay_days=("delay_days", "mean"),
            mean_seller_processing_days=("seller_processing_days", "mean"),
            mean_carrier_transit_days=("carrier_transit_days", "mean"),
        )
        .reset_index()
    )
    g["route"] = g["seller_state"].fillna("UNK") + "->" + g["customer_state"].fillna("UNK")
    g = g[g["volume"] >= min_samples].copy()
    g = g.sort_values(["failure_rate", "volume", "seller_state", "customer_state"], ascending=[False, False, True, True])
    return g.head(top_n)


def temporal_train_test_split(df: pd.DataFrame, date_col: str, cutoff_quantile: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df[date_col].notna()].copy()
    cutoff = df[date_col].quantile(cutoff_quantile)
    return df[df[date_col] <= cutoff].copy(), df[df[date_col] > cutoff].copy()


def simulate_hybrid_edd(df: pd.DataFrame, risk_scores: pd.Series, quantile_level: float = 0.8):
    sim = df.copy()
    sim["risk_score"] = risk_scores.values
    d = sim[sim["is_delivered"]].copy()

    proc_q = d.groupby(["seller_id", "seller_state", "customer_state"])["seller_processing_days"].quantile(quantile_level)
    trans_q = d.groupby(["seller_state", "customer_state"])["carrier_transit_days"].quantile(quantile_level)

    d = d.merge(proc_q.rename("proc_q").reset_index(), on=["seller_id", "seller_state", "customer_state"], how="left")
    d = d.merge(trans_q.rename("transit_q").reset_index(), on=["seller_state", "customer_state"], how="left")

    d["proc_q"] = d["proc_q"].fillna(d["seller_processing_days"].quantile(quantile_level))
    d["transit_q"] = d["transit_q"].fillna(d["carrier_transit_days"].quantile(quantile_level))

    d["base_eta_days"] = d["proc_q"] + d["transit_q"]
    d["risk_buffer_days"] = pd.cut(d["risk_score"].fillna(0.5), bins=[-np.inf, 0.25, 0.5, 0.75, np.inf], labels=[0, 1, 2, 3]).astype(int)
    d["proposed_eta_days"] = np.ceil(d["base_eta_days"] + d["risk_buffer_days"])

    d["current_eta_days"] = np.ceil(d["estimated_delivery_days"])
    d["actual_days"] = np.ceil(d["actual_delivery_days"])
    d["current_late"] = (d["actual_days"] > d["current_eta_days"]).astype(int)
    d["proposed_late"] = (d["actual_days"] > d["proposed_eta_days"]).astype(int)

    metrics = pd.DataFrame(
        {
            "metric": [
                "current_late_rate",
                "proposed_late_rate",
                "late_rate_reduction_pp",
                "avg_current_promised_days",
                "avg_proposed_promised_days",
                "avg_promised_days_increase",
            ],
            "value": [
                d["current_late"].mean(),
                d["proposed_late"].mean(),
                (d["current_late"].mean() - d["proposed_late"].mean()) * 100,
                d["current_eta_days"].mean(),
                d["proposed_eta_days"].mean(),
                (d["proposed_eta_days"] - d["current_eta_days"]).mean(),
            ],
        }
    )
    return d, metrics
