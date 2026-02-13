# Executive Summary

## Objective
Quantify late-delivery performance, isolate root causes, and propose a more reliable Estimated Delivery Date (EDD) policy.

## Analytical Scope
- Actual vs estimated delivery timing
- Seller processing vs carrier transit decomposition
- Top 5 highest-failure state-to-state routes
- Delay-risk predictive modeling and explainability
- Hybrid EDD redesign with scenario trade-off analysis

## Deliverables
- `notebooks/00_data_quality.ipynb`
- `notebooks/01_define_late_delivery.ipynb`
- `notebooks/02_root_cause_analysis.ipynb`
- `notebooks/03_nps_impact_and_solutions.ipynb`

## Strategy Translation
- Operations: reduce seller processing variance on high-risk lanes.
- Logistics: improve carrier performance on worst routes.
- Product: show confidence-tier ETA messaging (P50/P80/P90) for expectation management.
