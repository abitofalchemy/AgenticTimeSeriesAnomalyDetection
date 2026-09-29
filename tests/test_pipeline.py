import json
from types import SimpleNamespace

import pandas as pd
import pytest

from groqclfr import config
from groqclfr.agent import build_observation, decide, parse_decision
from groqclfr.data import to_daily_cases
from groqclfr.detection import compute_severity, detect_anomalies
from groqclfr.pipeline import analyze


def make_df(cases: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=len(cases)), "Cases": cases})


def flat_with(overrides: dict[int, int], n: int = 30, level: int = 10_000) -> pd.DataFrame:
    cases = [level] * n
    for i, v in overrides.items():
        cases[i] = v
    return make_df(cases)


class FakeClient:
    """Mimics groq.Groq().chat.completions.create and records calls."""

    def __init__(self, reply: str):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.reply = reply

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


# --- data -------------------------------------------------------------------

def test_to_daily_cases_diffs_cumulative_and_sorts():
    df = to_daily_cases({"1/3/23": 130, "1/1/23": 100, "1/2/23": 110})
    assert df["Cases"].tolist() == [10, 20]
    assert df["Date"].dt.day.tolist() == [2, 3]


# --- detection --------------------------------------------------------------

def test_flat_series_has_no_anomalies():
    assert not detect_anomalies(flat_with({}))["Anomaly"].any()


def test_spike_is_detected_and_critical():
    df = compute_severity(detect_anomalies(flat_with({20: 50_000})))
    assert df.index[df["Anomaly"]].tolist() == [20]
    assert df.loc[20, "Severity"] == "CRITICAL"


def test_negative_revision_is_detected():
    df = detect_anomalies(flat_with({15: -3_000}))
    assert df.loc[15, "Anomaly"]


@pytest.mark.parametrize(
    "value, expected",
    [(16_000, "WARNING"), (25_000, "CRITICAL")],
)
def test_severity_levels_from_growth(value, expected):
    df = compute_severity(detect_anomalies(flat_with({20: value})))
    assert df.loc[20, "Severity"] == expected


def test_small_absolute_increase_is_minor():
    # +60% growth but only +300 cases: surge rule doesn't fire, z-score does -> MINOR
    df = compute_severity(detect_anomalies(flat_with({20: 800}, level=500)))
    assert df.loc[20, "Anomaly"] and df.loc[20, "Severity"] == "MINOR"


# --- agent ------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, action",
    [
        ('{"decision": "KEEP_ANOMALY", "reason": "sustained"}', "KEEP_ANOMALY"),
        ("fix_anomaly", "FIX_ANOMALY"),
        ("Decision: FLAG_FOR_REVIEW", "FLAG_FOR_REVIEW"),
        ("FIX_ANOMALY or KEEP_ANOMALY?", "FLAG_FOR_REVIEW"),  # ambiguous
        ("", "FLAG_FOR_REVIEW"),
    ],
)
def test_parse_decision(text, action):
    assert parse_decision(text).action == action


def test_decide_without_client_flags():
    assert decide({}, None).action == config.FLAG_FOR_REVIEW


def test_decide_on_client_error_flags():
    class Broken:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: 1 / 0))

    assert decide({}, Broken()).action == config.FLAG_FOR_REVIEW


def test_observation_includes_context():
    df = compute_severity(detect_anomalies(flat_with({20: 50_000})))
    obs = build_observation(df, 20)
    assert obs["cases"] == 50_000 and obs["baseline_same_weekday_median"] == 10_000
    assert len(obs["previous_days"]) == config.CONTEXT_DAYS and len(obs["following_days"]) == 3


# --- pipeline ---------------------------------------------------------------

def test_minor_is_fixed_by_rule_without_llm():
    client = FakeClient(json.dumps({"decision": "KEEP_ANOMALY", "reason": "x"}))
    df = analyze(flat_with({20: 800}, level=500), client=client)
    assert client.calls == []
    assert df.loc[20, "Decision"] == "FIX_ANOMALY" and df.loc[20, "Decided By"] == "rule"
    assert df.loc[20, "Cases"] == 500 and df.loc[20, "Original Cases"] == 800


def test_critical_goes_to_agent():
    client = FakeClient(json.dumps({"decision": "KEEP_ANOMALY", "reason": "real surge"}))
    df = analyze(flat_with({20: 50_000}), client=client)
    assert len(client.calls) == 1
    assert df.loc[20, "Decision"] == "KEEP_ANOMALY" and df.loc[20, "Reason"] == "real surge"
    assert df.loc[20, "Cases"] == 50_000


def test_agent_fix_smooths_value():
    df = analyze(flat_with({20: 50_000}), client=FakeClient('{"decision": "FIX_ANOMALY"}'))
    assert df.loc[20, "Cases"] == 10_000


def test_no_agent_flags_severe():
    df = analyze(flat_with({20: 50_000}), use_agent=False)
    assert df.loc[20, "Decision"] == "FLAG_FOR_REVIEW"


def test_anomaly_on_first_day_cannot_be_fixed():
    df = analyze(flat_with({0: 50_000}), client=FakeClient('{"decision": "FIX_ANOMALY"}'))
    assert df.loc[0, "Decision"] == "FLAG_FOR_REVIEW"


def test_weekly_reporting_cycle_is_not_anomalous():
    # Reports only on Mondays/Wednesdays, zero otherwise: a steady weekly pattern.
    week = [30_000, 0, 20_000, 0, 0, 0, 0]
    df = compute_severity(detect_anomalies(make_df(week * 8)))
    assert not df.loc[7:, "Anomaly"].any()
