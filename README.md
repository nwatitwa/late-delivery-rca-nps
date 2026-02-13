# Late Delivery RCA + EDD Redesign

This project turns late deliveries from a vague pain point into a measurable, explainable, and actionable system.

Instead of only asking **"How many orders were late?"**, this repo answers:
- **Where** delay is coming from (seller processing vs carrier transit)
- **Which** routes fail most often (state-to-state hotspots)
- **What** policy changes improve reliability (hybrid EDD simulation)
- **How** to explain trade-offs to non-technical stakeholders

---

## Why This Work Matters
Delivery promise quality is a customer trust issue. A missed promise hurts NPS, increases support load, and weakens repeat behavior.  
This repository is built to help teams move from reactive reporting to proactive decision-making.

Core objective:
- Diagnose the drivers of late delivery
- Prioritize interventions by impact
- Simulate policy changes before rollout

---

## What Is In This Repo

### 1) Analysis Notebooks (end-to-end story)
- `notebooks/00_data_quality.ipynb`  
  Data quality profiling + canonical analytic table creation at **order-seller leg** grain.

- `notebooks/01_define_late_delivery.ipynb`  
  Defines lateness (`actual > estimated`) and establishes baseline KPI behavior.

- `notebooks/02_root_cause_analysis.ipynb`  
  Splits delay into seller processing vs carrier transit and ranks top failing routes.

- `notebooks/03_nps_impact_and_solutions.ipynb`  
  Feature engineering, predictive risk framing, explainability, and hybrid EDD redesign logic.

### 2) Stakeholder Dashboard
- `app.py` (Streamlit)

The dashboard is designed for simple storytelling without losing technical credibility:
- Problem snapshot (how large the miss issue is)
- Root-cause decomposition
- Route hotspot visibility
- Interactive policy simulator for trade-off analysis

### 3) Reusable Pipeline Utilities
- `src/utils.py`

Shared functions power both notebooks and app:
- canonical table build
- delivery feature derivation
- route failure ranking
- hybrid EDD simulation

---

## Approach in One View
1. Build a clean, auditable delivery dataset.
2. Define failure consistently: delivered late vs promised ETA.
3. Decompose cycle time into operational components.
4. Identify high-risk lanes with volume-aware filtering.
5. Simulate a hybrid ETA/EDD policy:
   - historical quantile baseline
   - risk-based buffer
   - measurable service-vs-speed trade-off

---

## Run Locally
```bash
pip install -r requirements.txt
```

### Launch Dashboard
```bash
streamlit run app.py
```

### Open Notebooks
Start with notebook order `00 -> 03` for full analytical flow.

---

## Key Definitions
- **Late delivery**: delivered orders where `actual_delivery_days > estimated_delivery_days`
- **Route**: `seller_state -> customer_state`
- **Primary grain**: order-seller leg (supports multi-seller order diagnostics)
- **pp**: percentage points (difference between percentages)

---

## Outputs
- `reports/executive_summary.md` for business summary
- `reports/dashboard_walkthrough_and_results.pdf` for presentation/video walkthrough script
- `reports/figures/` for exported visuals

---

## Final Note
This repo is not just a dashboard project. It is a decision framework:  
**measure clearly, explain causally, and choose policy with explicit trade-offs.**
