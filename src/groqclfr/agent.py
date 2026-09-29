"""GroqCloud-backed agent that decides how to handle a significant anomaly."""

import json
import os
import re
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

from groqclfr import config

SYSTEM_PROMPT = f"""\
You are an AI agent monitoring COVID-19 daily new-case time-series data for data quality.
Many jurisdictions report on fixed weekdays, so compare against the same weekday in prior weeks.
For each anomaly you are shown, decide exactly one action:
- {config.FIX_ANOMALY}: reporting noise or artifact (e.g. batch dump after missing days,
  weekend catch-up, isolated spike that immediately reverts, negative revision).
- {config.KEEP_ANOMALY}: a plausible real epidemiological signal (e.g. sustained rise).
- {config.FLAG_FOR_REVIEW}: severe or ambiguous; a human should look at it.
Respond with a JSON object: {{"decision": "<one of the three>", "reason": "<one sentence>"}}"""


@dataclass
class Decision:
    action: str
    reason: str


def get_client():
    """Return a Groq client, or None if GROQ_API_KEY isn't configured."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        return None
    from groq import Groq

    return Groq()


def build_observation(df: pd.DataFrame, idx: int, context_after: int = 3) -> dict:
    """Summarize an anomalous row plus its surrounding values for the prompt."""
    row = df.loc[idx]
    before = df.loc[max(0, idx - config.CONTEXT_DAYS) : idx - 1, "Cases"]
    after = df.loc[idx + 1 : idx + context_after, "Cases"]
    same_weekday = [int(df.loc[idx - 7 * w, "Cases"]) for w in range(1, config.BASELINE_WEEKS + 1) if idx - 7 * w >= 0]
    return {
        "date": row["Date"].strftime("%Y-%m-%d"),
        "weekday": row["Date"].day_name(),
        "cases": int(row["Cases"]),
        "baseline_same_weekday_median": None if pd.isna(row["Baseline"]) else round(float(row["Baseline"])),
        "same_weekday_prior_weeks": same_weekday,
        "growth_vs_baseline_pct": None if pd.isna(row["Growth"]) else round(float(row["Growth"]) * 100, 1),
        "zscore": round(float(row["ZScore"]), 2),
        "severity": row["Severity"],
        "previous_days": [int(v) for v in before],
        "following_days": [int(v) for v in after],
    }


def build_agent_prompt(obs: dict) -> str:
    return "Observed anomaly:\n" + json.dumps(obs, indent=2)


def parse_decision(text: str) -> Decision:
    """Extract a decision from model output; anything unclear becomes FLAG_FOR_REVIEW."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
        action = str(data.get("decision", "")).strip().upper()
        if action in config.VALID_ACTIONS:
            return Decision(action, str(data.get("reason", "")).strip())
    except (json.JSONDecodeError, AttributeError):
        pass

    found = set(re.findall("|".join(config.VALID_ACTIONS), text.upper()))
    if len(found) == 1:
        return Decision(found.pop(), "")
    return Decision(config.FLAG_FOR_REVIEW, f"Unparseable agent response: {text[:80]!r}")


def decide(obs: dict, client) -> Decision:
    """Ask the LLM for a decision. Failures fall back to FLAG_FOR_REVIEW."""
    if client is None:
        return Decision(config.FLAG_FOR_REVIEW, "Agent unavailable (GROQ_API_KEY not set)")
    try:
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_agent_prompt(obs)},
            ],
            response_format={"type": "json_object"},
        )
        return parse_decision(completion.choices[0].message.content)
    except Exception as exc:  # network/auth/rate-limit: never let one call kill the run
        return Decision(config.FLAG_FOR_REVIEW, f"Agent error: {type(exc).__name__}: {exc}")
