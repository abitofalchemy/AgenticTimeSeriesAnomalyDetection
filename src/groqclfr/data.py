"""Data ingestion from the disease.sh historical API."""

import pandas as pd
import requests

from groqclfr import config


class DataSourceError(RuntimeError):
    """Raised when the API returns no usable timeline."""


def fetch_cumulative_cases(country: str, days: int) -> dict[str, int]:
    """Return the raw {"m/d/yy": cumulative_cases} mapping for a country (or "all")."""
    url = f"{config.API_BASE_URL}/{country}"
    response = requests.get(url, params={"lastdays": days}, timeout=config.REQUEST_TIMEOUT_S)
    payload = response.json() if response.content else {}
    if not response.ok:
        raise DataSourceError(payload.get("message") or f"HTTP {response.status_code} for {country!r}")

    # /historical/all has no "timeline" wrapper; per-country responses do.
    timeline = payload.get("timeline", payload)
    cases = timeline.get("cases")
    if not cases:
        raise DataSourceError(f"No case timeline returned for {country!r}")
    return cases


def to_daily_cases(cumulative: dict[str, int]) -> pd.DataFrame:
    """Convert a cumulative timeline into daily new cases.

    Negative values (downward revisions by the reporting agency) are kept on purpose:
    they are data-quality anomalies the detector should see.
    """
    df = (
        pd.DataFrame(list(cumulative.items()), columns=["Date", "Cumulative"])
        .assign(Date=lambda d: pd.to_datetime(d["Date"], format="%m/%d/%y"))
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df["Cases"] = df["Cumulative"].diff()
    return df.dropna(subset=["Cases"]).astype({"Cases": "int64"}).reset_index(drop=True)


def load_covid_data(country: str = config.DEFAULT_COUNTRY, days: int = config.DEFAULT_DAYS) -> pd.DataFrame:
    """Load `days` rows of daily new cases (columns: Date, Cumulative, Cases)."""
    # One extra day so the first diff isn't lost.
    return to_daily_cases(fetch_cumulative_cases(country, days + 1))
