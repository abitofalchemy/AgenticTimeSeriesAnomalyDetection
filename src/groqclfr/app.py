"""Streamlit dashboard. Run with `groqclfr-dashboard` or `streamlit run src/groqclfr/app.py`."""

import altair as alt
import pandas as pd
import streamlit as st

from groqclfr import agent, config
from groqclfr.data import DataSourceError, load_covid_data
from groqclfr.pipeline import analyze

SEVERITY_ORDER = ["MINOR", "WARNING", "CRITICAL"]
SEVERITY_COLORS = ["#fab219", "#ec835a", "#d03b3b"]  # status palette: warning / serious / critical
SEVERITY_SHAPES = ["circle", "square", "triangle-up"]
LINE_COLOR = "#2a78d6"


@st.cache_data(ttl=3600, show_spinner="Fetching data from disease.sh…")
def cached_load(country: str, days: int) -> pd.DataFrame:
    return load_covid_data(country, days)


def build_chart(df: pd.DataFrame) -> alt.LayerChart:
    base = alt.Chart(df).encode(x=alt.X("Date:T", title=None))
    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover", empty=False)

    line = base.mark_line(strokeWidth=2, color=LINE_COLOR).encode(
        y=alt.Y("Cases:Q", title="Daily new cases (after corrections)")
    )
    rule = (
        base.mark_rule(color="gray", strokeWidth=1)
        .encode(
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m-%d"),
                alt.Tooltip("Cases:Q", format=",", title="Cases (corrected)"),
                alt.Tooltip("Original Cases:Q", format=",", title="Cases (reported)"),
            ],
        )
        .add_params(hover)
    )

    anomalies = df[df["Anomaly"]]
    points = alt.Chart(anomalies).encode(x="Date:T")
    moved = points.mark_rule(strokeDash=[3, 3], color="gray").encode(y="Original Cases:Q", y2="Cases:Q")
    markers = points.mark_point(size=90, filled=True, stroke="white", strokeWidth=2, opacity=1).encode(
        y="Original Cases:Q",
        color=alt.Color("Severity:N", scale=alt.Scale(domain=SEVERITY_ORDER, range=SEVERITY_COLORS)),
        shape=alt.Shape("Severity:N", scale=alt.Scale(domain=SEVERITY_ORDER, range=SEVERITY_SHAPES)),
        tooltip=[
            alt.Tooltip("Date:T", format="%Y-%m-%d"),
            alt.Tooltip("Original Cases:Q", format=",", title="Reported"),
            alt.Tooltip("Cases:Q", format=",", title="After action"),
            alt.Tooltip("Baseline:Q", format=",.0f"),
            "Severity:N",
            "Decision:N",
            "Decided By:N",
            "Reason:N",
        ],
    )
    return alt.layer(line, rule, moved, markers).properties(height=380)


def main() -> None:
    st.set_page_config(page_title="COVID-19 Anomaly Agent", layout="wide")
    st.title("COVID-19 Anomaly Detection Agent")
    st.caption(
        "Daily new cases from disease.sh (Johns Hopkins CSSE data, which ended on 2023-03-09). "
        "MINOR anomalies are auto-corrected by rule; WARNING/CRITICAL are decided by a GroqCloud agent."
    )

    with st.sidebar:
        country = st.text_input("Country (name, ISO code, or 'all')", config.DEFAULT_COUNTRY)
        days = st.slider("Days of history", 14, 365, config.DEFAULT_DAYS)
        use_agent = st.toggle("Use AI agent", value=True)
        run = st.button("Run pipeline", type="primary", width="stretch")

    if run:
        try:
            raw = cached_load(country.strip(), days)
        except DataSourceError as exc:
            st.error(f"Data source error: {exc}")
            return
        client = agent.get_client() if use_agent else None
        if use_agent and client is None:
            st.warning("GROQ_API_KEY not set; WARNING/CRITICAL anomalies will be flagged for review.")
        progress = st.progress(0.0, text="Handling anomalies…")
        st.session_state.result = analyze(
            raw, client=client, use_agent=use_agent, on_progress=lambda n, t: progress.progress(n / t)
        )
        st.session_state.country = country
        progress.empty()

    df = st.session_state.get("result")
    if df is None:
        st.info("Choose a country and click **Run pipeline**.")
        return

    anomalies = df[df["Anomaly"]]
    cols = st.columns(5)
    cols[0].metric("Days analyzed", len(df))
    cols[1].metric("Anomalies", len(anomalies))
    cols[2].metric("Auto-corrected", int((anomalies["Decision"] == config.FIX_ANOMALY).sum()))
    cols[3].metric("Kept as signal", int((anomalies["Decision"] == config.KEEP_ANOMALY).sum()))
    cols[4].metric("Flagged for review", int((anomalies["Decision"] == config.FLAG_FOR_REVIEW).sum()))

    st.subheader(f"Daily new cases: {st.session_state.country}")
    st.altair_chart(build_chart(df), width="stretch")
    st.caption("Markers show the reported value of each anomaly; a dashed line connects it to the corrected value.")

    tab_anom, tab_all = st.tabs(["Anomalies", "All data"])
    shown = ["Date", "Original Cases", "Cases", "Baseline", "Growth", "ZScore", "Severity",
             "Decision", "Decided By", "Action", "Reason"]  # fmt: skip
    with tab_anom:
        st.dataframe(anomalies[shown], hide_index=True, width="stretch")
    with tab_all:
        st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), "covid_anomalies.csv", "text/csv")


main()
