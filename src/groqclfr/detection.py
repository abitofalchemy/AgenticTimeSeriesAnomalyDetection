"""Statistical anomaly detection and severity classification on daily new cases."""

import numpy as np
import pandas as pd

from groqclfr import config

MINOR, WARNING, CRITICAL = "MINOR", "WARNING", "CRITICAL"


def weekday_baseline(cases: pd.Series, weeks: int = config.BASELINE_WEEKS) -> pd.Series:
    """Median of the same weekday over the prior `weeks` weeks (NaN during the first week).

    Many jurisdictions report on fixed weekdays, so a trailing daily mean mistakes the
    weekly reporting cycle for anomalies; comparing like-for-like weekdays does not.
    """
    lagged = pd.concat([cases.shift(7 * w) for w in range(1, weeks + 1)], axis=1)
    return lagged.median(axis=1, skipna=True)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add Baseline, Growth, ZScore and Anomaly columns. Returns a new frame.

    A day is anomalous if any of:
      - Z-score spike: |x - mean| > ZSCORE_THRESHOLD * std over the whole window
      - growth surge: growth vs. same-weekday baseline > GROWTH_THRESHOLD and
        the absolute increase is at least MIN_ABS_INCREASE
      - impossible value: negative daily cases (a downward revision)
    """
    df = df.copy()
    cases = df["Cases"].astype(float)

    std = cases.std(ddof=0)
    df["ZScore"] = (cases - cases.mean()) / std if std > 0 else 0.0

    df["Baseline"] = weekday_baseline(cases)
    deviation = cases - df["Baseline"]
    df["Growth"] = deviation / np.maximum(df["Baseline"].abs(), 1)

    spike = df["ZScore"].abs() > config.ZSCORE_THRESHOLD
    surge = (df["Growth"] > config.GROWTH_THRESHOLD) & (deviation >= config.MIN_ABS_INCREASE)
    negative = cases < 0
    df["Anomaly"] = spike | surge | negative
    return df


def classify_severity(cases: float, baseline: float, growth: float) -> str:
    """Severity for a single anomalous day, based on deviation from its baseline."""
    if pd.isna(baseline):
        return WARNING  # no history to compare against: needs judgement
    if abs(cases - baseline) < config.MIN_ABS_INCREASE:
        return MINOR
    if abs(growth) >= config.CRITICAL_GROWTH:
        return CRITICAL
    if abs(growth) >= config.WARNING_GROWTH:
        return WARNING
    return MINOR


def compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Severity column (empty string for non-anomalous rows). Returns a new frame."""
    df = df.copy()
    df["Severity"] = [
        classify_severity(row.Cases, row.Baseline, row.Growth) if row.Anomaly else ""
        for row in df.itertuples()
    ]
    return df
