"""Tunable thresholds and settings for the pipeline."""

# Data source (disease.sh wraps JHU CSSE, which stopped updating on 2023-03-09)
API_BASE_URL = "https://disease.sh/v3/covid-19/historical"
REQUEST_TIMEOUT_S = 15
DEFAULT_COUNTRY = "usa"
DEFAULT_DAYS = 90

# Detection
ZSCORE_THRESHOLD = 3.0          # |x - mean| > k * std  -> spike
GROWTH_THRESHOLD = 0.4          # growth vs. baseline -> surge
BASELINE_WEEKS = 3              # baseline = median of the same weekday over the prior N weeks
CONTEXT_DAYS = 7                # preceding days shown to the agent
MIN_ABS_INCREASE = 500          # deviations smaller than this are MINOR at most

# Severity (applied to |growth| vs. baseline)
CRITICAL_GROWTH = 1.0
WARNING_GROWTH = 0.4

# Agent
GROQ_MODEL = "openai/gpt-oss-120b"
FIX_ANOMALY = "FIX_ANOMALY"
KEEP_ANOMALY = "KEEP_ANOMALY"
FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
VALID_ACTIONS = (FIX_ANOMALY, KEEP_ANOMALY, FLAG_FOR_REVIEW)
