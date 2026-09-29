"""Apply a decision (FIX / KEEP / FLAG) to an anomalous row, in place."""

import pandas as pd

from groqclfr import config
from groqclfr.detection import weekday_baseline


def fix_anomaly(df: pd.DataFrame, idx: int) -> bool:
    """Replace Cases with the same-weekday baseline. Returns False if there is no history.

    The baseline is recomputed from current values so earlier corrections are respected.
    """
    baseline = weekday_baseline(df["Cases"].astype(float)).loc[idx]
    if pd.isna(baseline):
        return False
    df.loc[idx, "Cases"] = round(baseline)
    return True


def apply_decision(df: pd.DataFrame, idx: int, action: str) -> None:
    df.loc[idx, "Decision"] = action
    if action == config.FIX_ANOMALY:
        if fix_anomaly(df, idx):
            df.loc[idx, "Action"] = f"Auto-corrected (same-weekday median, {config.BASELINE_WEEKS} weeks)"
        else:
            df.loc[idx, "Decision"] = config.FLAG_FOR_REVIEW
            df.loc[idx, "Action"] = "Flagged for human review (no history to correct from)"
    elif action == config.KEEP_ANOMALY:
        df.loc[idx, "Action"] = "Accepted as a real outbreak signal"
    else:
        df.loc[idx, "Action"] = "Flagged for human review"
