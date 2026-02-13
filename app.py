from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from utils import (  # noqa: E402
    build_order_delivery_legs,
    derive_delivery_features,
    rank_failure_routes,
    simulate_hybrid_edd,
)

st.set_page_config(
    page_title="Late Delivery RCA and EDD Simulator",
    page_icon="📦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_feature_table() -> pd.DataFrame:
    processed_parquet = ROOT / "data" / "processed" / "order_delivery_legs.parquet"
    processed_csv = ROOT / "data" / "processed" / "order_delivery_legs.csv"

    if processed_parquet.exists():
        base = pd.read_parquet(processed_parquet)
    elif processed_csv.exists():
        base = pd.read_csv(processed_csv)
    else:
        base = build_order_delivery_legs(ROOT / "data" / "raw")

    feat = derive_delivery_features(base)
    delivered = feat[feat["is_delivered"]].copy()
    delivered["purchase_month"] = delivered["order_purchase_timestamp"].dt.to_period("M").astype(str)
    delivered["route"] = delivered["seller_state"].fillna("UNK") + "->" + delivered["customer_state"].fillna("UNK")
    return delivered


@st.cache_data(show_spinner=False)
def build_policy_table(delivered: pd.DataFrame, quantile_level: float, route_weight: float):
    route_risk = delivered.groupby(["seller_state", "customer_state"])["is_late"].mean().rename("route_late_rate")
    seller_risk = delivered.groupby("seller_id")["is_late"].mean().rename("seller_late_rate")

    sim_input = delivered.merge(route_risk.reset_index(), on=["seller_state", "customer_state"], how="left")
    sim_input = sim_input.merge(seller_risk.reset_index(), on="seller_id", how="left")

    seller_weight = 1.0 - route_weight
    sim_input["risk_score"] = (
        route_weight * sim_input["route_late_rate"].fillna(sim_input["is_late"].mean())
        + seller_weight * sim_input["seller_late_rate"].fillna(sim_input["is_late"].mean())
    )

    simulation, policy = simulate_hybrid_edd(
        sim_input,
        sim_input["risk_score"],
        quantile_level=quantile_level,
    )

    return simulation, policy


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def days(x: float) -> str:
    return f"{x:.2f} days"


delivered = load_feature_table()

st.title("Late Delivery Story and Solution")
st.caption("Simple view of the delivery problem: what is happening, why it happens, and which solution improves it.")

with st.sidebar:
    st.header("Scenario Controls")
    min_route_volume = st.slider("Route minimum volume", min_value=50, max_value=500, value=100, step=25)
    quantile_level = st.slider("EDD base quantile", min_value=0.60, max_value=0.95, value=0.80, step=0.05)
    route_weight = st.slider("Route risk weight", min_value=0.0, max_value=1.0, value=0.60, step=0.05)

late_rate = delivered["is_late"].mean()
avg_delay = delivered["delay_days"].mean()
avg_est = delivered["estimated_delivery_days"].mean()
avg_actual = delivered["actual_delivery_days"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Delivered Orders", f"{len(delivered):,}")
k2.metric("Late Delivery Rate", pct(late_rate))
k3.metric("Avg Estimated Days", days(avg_est))
k4.metric("Avg Actual Days", days(avg_actual))

st.markdown("### 1) Problem Framing: Delivery Promise Gap")
st.write(
    "A delivery is counted as a miss when it arrives later than promised. "
    "In this dashboard, miss rate means delivered orders where actual days are greater than estimated days."
)

c1, c2 = st.columns(2)
with c1:
    monthly = (
        delivered.groupby("purchase_month", as_index=False)
        .agg(late_rate=("is_late", "mean"), volume=("order_id", "count"))
        .sort_values("purchase_month")
    )
    monthly["late_rate_pct"] = monthly["late_rate"] * 100
    st.line_chart(monthly.set_index("purchase_month")["late_rate_pct"], height=280)
    st.caption("Monthly miss rate (%)")

with c2:
    dist = delivered[["estimated_delivery_days", "actual_delivery_days", "delay_days"]].describe(percentiles=[0.5, 0.8, 0.9, 0.95]).T
    st.dataframe(dist[["mean", "50%", "80%", "90%", "95%"]].round(2), use_container_width=True)
    st.caption("Delivery timing distribution snapshot")

st.markdown("### 2) Root Cause: Where the Delay Comes From")
cohort = (
    delivered.groupby("is_late", as_index=False)
    .agg(
        volume=("order_id", "count"),
        seller_processing_days=("seller_processing_days", "mean"),
        carrier_transit_days=("carrier_transit_days", "mean"),
        avg_delay_days=("delay_days", "mean"),
    )
)
cohort["cohort"] = cohort["is_late"].map({0: "On-time", 1: "Late"})
cohort["share_of_orders_pct"] = (cohort["volume"] / cohort["volume"].sum()) * 100

r1, r2 = st.columns([1, 1])
with r1:
    st.dataframe(
        cohort[
            [
                "cohort",
                "volume",
                "share_of_orders_pct",
                "seller_processing_days",
                "carrier_transit_days",
                "avg_delay_days",
            ]
        ]
        .rename(columns={"share_of_orders_pct": "share_of_orders (%)"})
        .round(1),
        use_container_width=True,
    )
with r2:
    comp = cohort.set_index("cohort")[["seller_processing_days", "carrier_transit_days"]]
    st.bar_chart(comp, height=300)

st.markdown("### 3) Hotspot Routes: Worst Performing Lanes")
routes = rank_failure_routes(delivered, min_samples=min_route_volume, top_n=5)
if routes.empty:
    st.warning("No routes meet the selected minimum volume threshold.")
else:
    routes_display = routes.copy()
    routes_display["failure_rate_pct"] = routes_display["failure_rate"] * 100
    st.dataframe(
        routes_display[
            [
                "route",
                "volume",
                "failure_rate_pct",
                "mean_delay_days",
                "mean_seller_processing_days",
                "mean_carrier_transit_days",
            ]
        ]
        .rename(columns={"failure_rate_pct": "failure_rate (%)"})
        .round(1),
        use_container_width=True,
    )
    route_chart = routes_display.set_index("route")[["failure_rate_pct"]]
    st.bar_chart(route_chart, height=280)

st.markdown("### 4) Solution: Hybrid EDD Policy Simulator")
st.write(
    "Proposed policy = baseline ETA from historical route/seller performance + a risk buffer. "
    "Adjust the sliders to see the trade-off between fewer late deliveries and longer promised delivery windows."
)

simulation, policy = build_policy_table(delivered, quantile_level, route_weight)
policy_map = policy.set_index("metric")["value"].to_dict()

s1, s2, s3 = st.columns(3)
s1.metric("Current Late Rate", pct(policy_map["current_late_rate"]))
s2.metric("Proposed Late Rate", pct(policy_map["proposed_late_rate"]))
s3.metric("Late-rate Improvement", f"{policy_map['late_rate_reduction_pp']:.1f} pp")

s4, s5 = st.columns(2)
s4.metric("Current Avg Promised Days", days(policy_map["avg_current_promised_days"]))
s5.metric("Proposed Avg Promised Days", days(policy_map["avg_proposed_promised_days"]))

band = simulation["base_eta_days"].quantile([0.5, 0.8, 0.9]).rename(index={0.5: "P50", 0.8: "P80", 0.9: "P90"})
st.dataframe(band.to_frame("days").round(2), use_container_width=False)
st.caption("Confidence-style ETA bands from baseline policy component")

st.markdown("### 5) Actionable Guidance")
st.markdown(
    "- **Operations**: Focus seller coaching on routes where seller processing time dominates cycle time.\n"
    "- **Carrier Management**: Prioritize contracts or interventions on top failing routes with high transit burden.\n"
    "- **Customer Experience**: Expose confidence-tier ETA messaging (P50/P80/P90) to set expectation quality."
)

with st.expander("Technical Notes"):
    st.write("Grain: order-seller leg. Failure: delivered late vs ETA. Route ranking tie-break: failure rate desc, volume desc.")
    st.write("Risk score in this dashboard is a weighted blend of route and seller historical late rates.")

st.markdown("### Key")
st.markdown(
    "- **ETA**: Estimated Time of Arrival (promised delivery date).\n"
    "- **EDD**: Estimated Delivery Date logic used to calculate the promise.\n"
    "- **RCA**: Root Cause Analysis.\n"
    "- **SLA**: Service Level Agreement.\n"
    "- **CX**: Customer Experience.\n"
    "- **pp**: Percentage points (difference between two percentages)."
)
