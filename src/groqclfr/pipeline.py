"""End-to-end pipeline: load -> detect -> classify -> decide -> act."""

from collections.abc import Callable

import pandas as pd

from groqclfr import agent, config
from groqclfr.actions import apply_decision
from groqclfr.data import load_covid_data
from groqclfr.detection import MINOR, compute_severity, detect_anomalies

ProgressFn = Callable[[int, int], None]


def handle_anomalies(
    df: pd.DataFrame, client=None, use_agent: bool = True, on_progress: ProgressFn | None = None
) -> pd.DataFrame:
    """Decide and act on each anomaly (hybrid policy). Returns a new frame.

    MINOR anomalies are auto-corrected by rule; WARNING/CRITICAL go to the LLM agent
    (or are flagged for review when the agent is disabled or unavailable).
    """
    df = df.copy()
    df["Original Cases"] = df["Cases"]
    for col in ("Decision", "Decided By", "Reason", "Action"):
        df[col] = ""

    anomalies = df.index[df["Anomaly"]].tolist()
    for n, idx in enumerate(anomalies, start=1):
        if df.loc[idx, "Severity"] == MINOR:
            decision = agent.Decision(config.FIX_ANOMALY, "Minor deviation from baseline")
            df.loc[idx, "Decided By"] = "rule"
        elif use_agent:
            decision = agent.decide(agent.build_observation(df, idx), client)
            df.loc[idx, "Decided By"] = "agent"
        else:
            decision = agent.Decision(config.FLAG_FOR_REVIEW, "Agent disabled")
            df.loc[idx, "Decided By"] = "rule"

        df.loc[idx, "Reason"] = decision.reason
        apply_decision(df, idx, decision.action)
        if on_progress:
            on_progress(n, len(anomalies))
    return df


def analyze(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run detection, severity and handling on an already-loaded daily-cases frame."""
    return handle_anomalies(compute_severity(detect_anomalies(df)), **kwargs)


def run_pipeline(
    country: str = config.DEFAULT_COUNTRY,
    days: int = config.DEFAULT_DAYS,
    use_agent: bool = True,
    on_progress: ProgressFn | None = None,
) -> pd.DataFrame:
    client = agent.get_client() if use_agent else None
    return analyze(load_covid_data(country, days), client=client, use_agent=use_agent, on_progress=on_progress)
