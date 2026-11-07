"""Donor Lifecycle & Reactivation - Streamlit demo (synthetic data)."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).parent / "data"
URGENCY_ORDER = ["High", "Medium", "Low", "Not needed"]
URGENCY_COLORS = {"High": "#c0392b", "Medium": "#e67e22", "Low": "#95a5a6", "Not needed": "#27ae60"}
SEGMENT_ORDER = ["Dormant / Churn Risk", "Dormant (Former Loyal)", "High-Value At-Risk",
                 "Frequent At-Risk", "Regular Active", "High-Value Active",
                 "Frequent Active", "New Donor"]

st.set_page_config(page_title="Donor Reactivation Demo", page_icon="📊", layout="wide")


@st.cache_data
def load():
    if not (DATA / "donor_scores.csv").exists():
        import pipeline
        pipeline.main()
    donors = pd.read_csv(DATA / "donor_scores.csv", parse_dates=["last_date"])
    trend = pd.read_csv(DATA / "monthly_trend.csv", parse_dates=["bulan"])
    programs = pd.read_csv(DATA / "program_summary.csv")
    metrics = json.loads((DATA / "model_metrics.json").read_text())
    return donors, trend, programs, metrics


def rupiah(x: float) -> str:
    if x >= 1e9:
        return f"Rp{x / 1e9:,.2f} M".replace(",", "X").replace(".", ",").replace("X", ".")
    if x >= 1e6:
        return f"Rp{x / 1e6:,.1f} jt".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp{x:,.0f}".replace(",", ".")


donors, trend, programs, metrics = load()

st.title("Donor Lifecycle & Reactivation")
st.caption("Who should the follow-up team contact first? RFM segmentation + a machine-learning "
           "reactivation score turn donation history into a prioritized call list.")
st.info("**Demo with synthetic data.** All donors, amounts and dates are randomly generated "
        "and do not belong to any real organization.", icon="ℹ️")

# ---------------- sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    seg = st.multiselect("Lifecycle segment", SEGMENT_ORDER, default=[])
    urg = st.multiselect("Urgency", URGENCY_ORDER[:3], default=[])
    prog = st.multiselect("Favourite program", sorted(donors.favourite_program.unique()), default=[])
    st.divider()
    st.markdown(f"**Model:** {metrics['model']}  \n"
                f"ROC-AUC {metrics['roc_auc']} · PR-AUC {metrics['pr_auc']}  \n"
                f"Top-10% lift {metrics['top10_lift']}×  \n"
                f"Label: gave again within {metrics['horizon_days']} days")

f = donors.copy()
if seg:
    f = f[f.segment.isin(seg)]
if urg:
    f = f[f.urgency.isin(urg)]
if prog:
    f = f[f.favourite_program.isin(prog)]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Priority Queue", "Segments & RFM", "Trends & Programs"])

# ---------------- overview ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Donors", f"{len(f):,}".replace(",", "."))
    c2.metric("Need reactivation", f"{f.needs_reactivation.mean():.1%}" if len(f) else "-")
    c3.metric("High urgency", f"{(f.urgency == 'High').sum():,}".replace(",", "."))
    c4.metric("Lifetime donations", rupiah(f.monetary.sum()))

    left, right = st.columns(2)
    seg_counts = f.segment.value_counts().reindex(SEGMENT_ORDER).dropna().reset_index()
    seg_counts.columns = ["segment", "donors"]
    left.plotly_chart(px.bar(seg_counts, x="donors", y="segment", orientation="h",
                             title="Donors by lifecycle segment")
                      .update_layout(yaxis={"categoryorder": "total ascending"}),
                      use_container_width=True)
    urg_counts = f.urgency.value_counts().reindex(URGENCY_ORDER).dropna().reset_index()
    urg_counts.columns = ["urgency", "donors"]
    right.plotly_chart(px.bar(urg_counts, x="urgency", y="donors", color="urgency",
                              color_discrete_map=URGENCY_COLORS, title="Donors by urgency")
                       .update_layout(showlegend=False), use_container_width=True)

    high = f[f.urgency == "High"]
    if len(high):
        st.success(f"**Recommendation:** start with the {len(high):,} high-urgency donors "
                   f"({len(high) / max(len(f), 1):.1%} of this view). Their most common favourite "
                   f"program is **{high.favourite_program.mode()[0]}**, so lead with it in the message.")

# ---------------- priority queue ----------------
with tab2:
    st.subheader("Reactivation priority queue")
    st.caption("Priority score (0-100) = 50% predicted chance to give again + 30% lifetime value "
               "rank + 20% frequency rank. Only donors past their usual giving interval are scored.")
    queue = f[f.needs_reactivation].sort_values("priority_score", ascending=False)
    show = queue[["id_donor", "urgency", "priority_score", "p_reactivate", "segment",
                  "favourite_program", "last_date", "days_overdue", "frequency", "monetary"]] \
        .rename(columns={"id_donor": "Donor ID", "urgency": "Urgency",
                         "priority_score": "Priority score", "p_reactivate": "P(give again)",
                         "segment": "Segment", "favourite_program": "Favourite program",
                         "last_date": "Last gift", "days_overdue": "Days overdue",
                         "frequency": "Gifts", "monetary": "Lifetime (Rp)"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=520,
                 column_config={
                     "Priority score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                     "P(give again)": st.column_config.NumberColumn(format="%.2f"),
                     "Last gift": st.column_config.DateColumn(format="YYYY-MM-DD"),
                     "Lifetime (Rp)": st.column_config.NumberColumn(format="%,.0f"),
                 })
    st.download_button("Download queue (CSV)", show.to_csv(index=False).encode(),
                       "priority_queue.csv", "text/csv")

# ---------------- segments & RFM ----------------
with tab3:
    left, right = st.columns(2)
    avg = f[f.needs_reactivation].groupby("segment").priority_score.mean().reset_index()
    left.plotly_chart(px.bar(avg, x="priority_score", y="segment", orientation="h",
                             title="Average priority score by segment",
                             labels={"priority_score": "avg priority score"})
                      .update_layout(yaxis={"categoryorder": "total ascending"}),
                      use_container_width=True)
    rfm = f.rfm_segment.value_counts().reset_index()
    rfm.columns = ["rfm_segment", "donors"]
    right.plotly_chart(px.bar(rfm, x="donors", y="rfm_segment", orientation="h",
                              title="RFM segments")
                       .update_layout(yaxis={"categoryorder": "total ascending"}),
                       use_container_width=True)
    active = f[f.frequency > 1]
    st.plotly_chart(px.histogram(active, x="avg_interval", nbins=60, range_x=[0, 400],
                                 title="Typical days between gifts (repeat donors)",
                                 labels={"avg_interval": "average interval (days)"}),
                    use_container_width=True)

# ---------------- trends ----------------
with tab4:
    st.plotly_chart(px.bar(trend, x="bulan", y="total_nominal", color="akad",
                           title="Monthly donations by akad (spikes = Ramadan)",
                           labels={"bulan": "", "total_nominal": "Rp"}),
                    use_container_width=True)
    top = programs.groupby("program", as_index=False).total_nominal.sum() \
        .sort_values("total_nominal", ascending=False).head(10)
    st.plotly_chart(px.bar(top, x="total_nominal", y="program", orientation="h",
                           title="Top programs by total donations", labels={"total_nominal": "Rp"})
                    .update_layout(yaxis={"categoryorder": "total ascending"}),
                    use_container_width=True)

st.caption("Built by Muhammad Irsyad Dzulfaqor · Python, scikit-learn/XGBoost, Streamlit · "
           "Code: see the GitHub repository.")
