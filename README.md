# Donor Lifecycle & Reactivation (Demo)

**Which lapsed donors should a charity call first?** This project turns raw donation history into a ranked call list for a follow-up team, using RFM segmentation and a machine-learning reactivation score.

> **Demo with synthetic data.** This repository recreates a project I led during a remote data analyst internship with a nonprofit. The client's data is confidential, so every donor, amount and date here is randomly generated (`generate_data.py`) with realistic patterns: zakat/infaq/sedekah programs, Ramadan spikes, 50–60 day giving cycles and a large dormant base.

**Live app:** [irsyad-donor-reactivation-demo.streamlit.app](https://irsyad-donor-reactivation-demo.streamlit.app/) · **Dashboard:** [Looker Studio](https://lookerstudio.google.com/reporting/d3139633-8c45-4394-98cb-cbda71031f90)

## The problem

Fundraising teams hold years of transactions but no clear answer to *"who do we contact this week?"* Calling everyone is expensive, and most of the donor base has gone quiet.

## Approach

1. **Features per donor:** recency, frequency, lifetime value, average gift, tenure, average days between gifts, how overdue the donor is versus their own rhythm, program variety and Ramadan share.
2. **Model:** a gradient-boosted classifier predicts whether a donor will give again in the next 180 days. Features are taken at a cut-off date and the label from the 180 days after it, so the model never sees the future. The original project used XGBoost; `pipeline.py` uses XGBoost when installed and falls back to scikit-learn's HistGradientBoosting. The scores committed in `data/` were produced with the scikit-learn model so the app opens instantly.
3. **Segmentation:** lifecycle segments (Active, At-Risk, Dormant, Former Loyal) and classic RFM segments (Champions, Cannot Lose Them, At Risk, Lost, ...).
4. **Priority score (0–100):** 50% predicted chance to give again + 30% lifetime value rank + 20% frequency rank, only for donors past their usual giving interval. The top 7% are marked **High** urgency.

## Results on the synthetic data

| Metric | Value |
| --- | --- |
| ROC-AUC (hold-out) | ~0.94 |
| Lift in top 10% of the ranking | ~8× better than random |
| Donors needing reactivation | ~97% |
| High-urgency call list | ~1,400 donors (~7%) |

The team can start with a short, ranked list instead of the full database, which cuts follow-up workload by over 90%. Numbers come from synthetic data and show the method, not real-world performance.

## Run it

```bash
pip install -r requirements.txt
python generate_data.py   # creates data/transactions.csv
python pipeline.py        # trains the model, writes donor_scores.csv etc.
streamlit run app.py
```

## Files

| File | What it does |
| --- | --- |
| `generate_data.py` | Builds the synthetic transaction dataset |
| `pipeline.py` | Features, model, segments, priority score, summary tables |
| `app.py` | Streamlit app: overview, priority queue, segments & RFM, trends |
| `looker/` | Lighter CSV tables used by the Looker Studio demo dashboard |

## Tools

Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly · Looker Studio

---
Muhammad Irsyad Dzulfaqor · [LinkedIn](https://www.linkedin.com/in/mirsyaddzulfaqor/)
