"""
Donor Lifecycle & Reactivation pipeline (demo, synthetic data).

Steps
1. Build per-donor features (recency, frequency, monetary, giving interval,
   favourite program) at a given cut-off date.
2. Train a classifier: "will this donor give again in the next 180 days?"
   Features are taken at CUTOFF, the label from the 180 days after it.
3. Score every donor at SNAPSHOT, segment them (lifecycle + RFM) and build a
   0-100 priority score and an urgency level for the follow-up team.

Outputs in data/:
  donor_scores.csv     one row per donor (the priority queue)
  monthly_trend.csv    donations per month and akad
  program_summary.csv  donations per program
  model_metrics.json   hold-out metrics of the model

Usage:  python pipeline.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier

    def make_model():
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric="logloss", random_state=42)
    MODEL_NAME = "XGBoost"
except ImportError:  # fallback so the pipeline still runs without xgboost
    from sklearn.ensemble import HistGradientBoostingClassifier

    def make_model():
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                              max_iter=300, random_state=42)
    MODEL_NAME = "HistGradientBoosting (xgboost not installed)"

DATA = Path(__file__).parent / "data"
SNAPSHOT = pd.Timestamp("2026-09-30")
HORIZON_DAYS = 180
CUTOFF = SNAPSHOT - pd.Timedelta(days=HORIZON_DAYS)
FEATURES = ["recency", "frequency", "monetary", "avg_amount", "tenure",
            "avg_interval", "overdue_ratio", "n_programs", "ramadan_share"]

RAMADAN = [("2020-04-24", "2020-05-23"), ("2021-04-13", "2021-05-12"),
           ("2022-04-02", "2022-05-01"), ("2023-03-23", "2023-04-21"),
           ("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"),
           ("2026-02-18", "2026-03-19")]


def load_transactions() -> pd.DataFrame:
    path = DATA / "transactions.csv"
    if not path.exists():
        import generate_data
        generate_data.main(str(DATA))
    tx = pd.read_csv(path, parse_dates=["tgl_donasi"])
    rm = np.zeros(len(tx), dtype=bool)
    for s, e in RAMADAN:
        rm |= tx.tgl_donasi.between(pd.Timestamp(s), pd.Timestamp(e))
    tx["is_ramadan"] = rm
    return tx


def build_features(tx: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """Per-donor features using only transactions on or before `at`."""
    t = tx[tx.tgl_donasi <= at].sort_values("tgl_donasi")
    g = t.groupby("id_donor")
    f = pd.DataFrame({
        "first_date": g.tgl_donasi.min(),
        "last_date": g.tgl_donasi.max(),
        "frequency": g.size(),
        "monetary": g.nominal.sum(),
        "avg_amount": g.nominal.mean(),
        "n_programs": g.program.nunique(),
        "ramadan_share": g.is_ramadan.mean(),
    })
    f["recency"] = (at - f.last_date).dt.days
    f["tenure"] = (at - f.first_date).dt.days
    gaps = t.groupby("id_donor").tgl_donasi.diff().dt.days
    f["avg_interval"] = gaps.groupby(t.id_donor).mean()
    # one-time donors have no interval: use a long default
    f["avg_interval"] = f["avg_interval"].fillna(365).clip(lower=7)
    f["overdue_ratio"] = f.recency / f.avg_interval
    f["favourite_program"] = t.groupby("id_donor").program.agg(lambda s: s.value_counts().index[0])
    f["favourite_akad"] = t.groupby("id_donor").akad.agg(lambda s: s.value_counts().index[0])
    return f


def train_model(tx: pd.DataFrame):
    feats = build_features(tx, CUTOFF)
    future = tx[(tx.tgl_donasi > CUTOFF) & (tx.tgl_donasi <= SNAPSHOT)].id_donor.unique()
    y = feats.index.isin(future).astype(int)
    X = feats[FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = make_model().fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    # how much better than random is the top 10% of the ranking?
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    metrics = {
        "model": MODEL_NAME,
        "cutoff": str(CUTOFF.date()), "horizon_days": HORIZON_DAYS,
        "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        "base_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, p)), 3),
        "pr_auc": round(float(average_precision_score(y_te, p)), 3),
        "top10_lift": round(float(y_te[top].mean() / y_te.mean()), 2),
        "note": "Synthetic data - metrics show the method works, not real-world performance.",
    }
    # refit on all rows for scoring
    model = make_model().fit(X, y)
    return model, metrics


def lifecycle_segment(r) -> str:
    high_value = r.monetary_pct >= 0.8
    frequent = r.frequency >= 6
    if r.tenure <= 90:
        return "New Donor"
    if r.overdue_ratio <= 1.5 and r.recency <= 180:
        if high_value:
            return "High-Value Active"
        return "Frequent Active" if frequent else "Regular Active"
    if r.recency <= 365:
        if high_value:
            return "High-Value At-Risk"
        if frequent:
            return "Frequent At-Risk"
    if frequent or high_value:
        return "Dormant (Former Loyal)"
    return "Dormant / Churn Risk"


def rfm_segment(r) -> str:
    R, F, M = r.R, r.F, r.M
    if R >= 4 and F >= 4 and M >= 4:
        return "Champions"
    if R <= 2 and F >= 4 and M >= 4:
        return "Cannot Lose Them"
    if F >= 4 and R >= 3:
        return "Loyal"
    if R >= 4 and F <= 1:
        return "New"
    if R >= 3 and F >= 2:
        return "Potential Loyalists"
    if R <= 2 and F >= 3:
        return "At Risk"
    if M >= 5 and F <= 2:
        return "Big Spenders"
    if R <= 1 and F <= 2:
        return "Lost"
    return "Others"


def urgency(row) -> str:
    if not row.needs_reactivation:
        return "Not needed"
    if row.priority_score >= HIGH_CUT:
        return "High"
    if row.priority_score >= MED_CUT:
        return "Medium"
    return "Low"


def score_donors(tx: pd.DataFrame, model) -> pd.DataFrame:
    d = build_features(tx, SNAPSHOT)
    d["p_reactivate"] = model.predict_proba(d[FEATURES])[:, 1]
    d["monetary_pct"] = d.monetary.rank(pct=True)
    d["frequency_pct"] = d.frequency.rank(pct=True)
    d["segment"] = d.apply(lifecycle_segment, axis=1)
    # RFM quintiles (1-5, 5 = best)
    d["R"] = pd.qcut(-d.recency.rank(method="first"), 5, labels=False) + 1
    d["F"] = pd.qcut(d.frequency.rank(method="first"), 5, labels=False) + 1
    d["M"] = pd.qcut(d.monetary.rank(method="first"), 5, labels=False) + 1
    d["rfm_segment"] = d.apply(rfm_segment, axis=1)
    d["needs_reactivation"] = (d.overdue_ratio > 1.5) | (d.recency > 180)
    # priority = chance they come back x how much they matter
    d["priority_score"] = (100 * (0.5 * d.p_reactivate + 0.3 * d.monetary_pct
                                  + 0.2 * d.frequency_pct)).round(1)
    d.loc[~d.needs_reactivation, "priority_score"] = 0.0
    need = d.loc[d.needs_reactivation, "priority_score"]
    global HIGH_CUT, MED_CUT
    HIGH_CUT, MED_CUT = need.quantile(0.93), need.quantile(0.70)
    d["urgency"] = d.apply(urgency, axis=1)
    d["days_overdue"] = (d.recency - d.avg_interval).clip(lower=0).round().astype(int)
    d["avg_interval"] = d.avg_interval.round(1)
    d["p_reactivate"] = d.p_reactivate.round(3)
    cols = ["segment", "rfm_segment", "urgency", "priority_score", "p_reactivate",
            "favourite_program", "favourite_akad", "last_date", "recency",
            "avg_interval", "days_overdue", "frequency", "monetary", "avg_amount",
            "R", "F", "M", "needs_reactivation"]
    return d[cols].sort_values("priority_score", ascending=False).reset_index()


def main():
    tx = load_transactions()
    model, metrics = train_model(tx)
    donors = score_donors(tx, model)
    donors.to_csv(DATA / "donor_scores.csv", index=False, date_format="%Y-%m-%d")

    m = tx.assign(bulan=tx.tgl_donasi.dt.to_period("M").dt.to_timestamp())
    m.groupby(["bulan", "akad"]).agg(total_nominal=("nominal", "sum"),
                                     jumlah_transaksi=("id_transaksi", "size"),
                                     jumlah_donor=("id_donor", "nunique")) \
     .reset_index().to_csv(DATA / "monthly_trend.csv", index=False, date_format="%Y-%m-%d")
    tx.groupby(["program", "akad"]).agg(total_nominal=("nominal", "sum"),
                                        jumlah_transaksi=("id_transaksi", "size"),
                                        jumlah_donor=("id_donor", "nunique")) \
      .reset_index().sort_values("total_nominal", ascending=False) \
      .to_csv(DATA / "program_summary.csv", index=False)
    (DATA / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(donors.segment.value_counts(normalize=True).round(3))
    print(donors.urgency.value_counts())


if __name__ == "__main__":
    main()
