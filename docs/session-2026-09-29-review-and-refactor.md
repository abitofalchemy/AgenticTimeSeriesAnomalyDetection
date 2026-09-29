# Session log: codebase review and refactor (2026-09-29)

Working session with Claude Code: reviewing `src/` and turning it into a working, tested pipeline.

## 1. Request

> Review this codebase … see the code under the `src/` folder and help me organize and make it actionable. Consult with me if there are questions.

Project context given: COVID-19 anomaly detection on disease.sh data, with Z-score and growth-rate detection,
MINOR/WARNING/CRITICAL severity, a GroqCloud agent choosing FIX/KEEP/FLAG, rolling-mean
correction of minor anomalies, and a Streamlit dashboard.

## 2. Initial review

`src/groqclfr/` held 7 files (138 lines) that looked like notebook cells copied out one by one. **None of them could run.**

| Problem | Where |
|---|---|
| No imports anywhere (`requests`, `pd`, `np`, `Agent`, `Groq`) | every file |
| Undefined `build_observation`, `VALID_ACTIONS`, `df` | `create_agent_groq_cloud.py` |
| Agent loop at module level (would run on import) | `create_agent_groq_cloud.py` |
| `pyproject.toml` had no dependencies; `requirements.txt` was missing pandas/numpy/requests | config |
| No Streamlit dashboard existed | — |
| `main()` only printed "Hello" | `__init__.py` |
| `.env` (with `GROQ_API_KEY`) not in `.gitignore` | `.gitignore` |

### Logic bugs
1. **Detection ran on cumulative counts.** A test call to the API showed the data is cumulative and **ends 2023-03-09**, because disease.sh uses Johns Hopkins CSSE data, which stopped then. A Z-score over a series that only goes up is meaningless.
2. **`compute_severity` had fall-through bugs.** The branch for rows without enough history had no `continue`, the minimum-increase check was `if` where `elif` was needed, drops always became MINOR, and a `NaN` baseline silently became MINOR.
3. **The fix didn't match the spec.** `fix_anomaly` used a hard-coded 3-day mean and ignored `ROLLING_WINDOW = 7`.
4. **Reading the agent's response was fragile.** It used `response.messages[-1].content` with an exact-match check against a reasoning model's output.
5. **The agent had too little context.** It got only date, cases and severity.
6. **Ingestion had no error handling.** There was no timeout and no status check. `country="all"` returns a different JSON shape, and an unknown country crashed with a `KeyError`.

## 3. Decisions (asked and answered)

| Question | Choice |
|---|---|
| Data treatment | Daily new cases (`diff()`), keep disease.sh, label it as historical |
| Who decides actions | **Hybrid:** MINOR is auto-fixed by rule; WARNING/CRITICAL go to the LLM agent |
| Agent library | Groq Python SDK directly (phidata is superseded by agno; this is a single classification call) |
| Scope | Full working pipeline: restructure, fixes, CLI, dashboard, tests, README |
| Baseline (asked after the first real-data run) | **Median of the same weekday over the prior 3 weeks** |
| Live Groq test using the key in `.env` | Approved |

### Baseline experiment
With a trailing 7-day mean, weekday reporting cycles made most days look anomalous. Anomalies found over 90 days:

| Country | Trailing 7-day mean | Same weekday last week | Median of 3 same weekdays |
|---|---|---|---|
| usa | 31 | 10 | 7 |
| germany | 30 | 5 | 5 |
| india | 2 | 3 | 2 |
| all | 14 | 2 | 0 |

## 4. Resulting layout

```
src/groqclfr/
  config.py     thresholds, model id (openai/gpt-oss-120b), API settings
  data.py       disease.sh ingestion → daily new cases; DataSourceError
  detection.py  weekday_baseline, detect_anomalies, classify/compute_severity
  agent.py      get_client, build_observation, prompt, parse_decision, decide
  actions.py    fix_anomaly (weekday-median), apply_decision
  pipeline.py   handle_anomalies (hybrid policy), analyze, run_pipeline
  cli.py        `groqclfr` (tabulate report, --no-agent, --csv) and `groqclfr-dashboard`
  app.py        Streamlit dashboard (KPIs, Altair chart w/ severity markers, tables, CSV download)
tests/test_pipeline.py   21 tests, fake Groq client, no network
```

**Detection rules.** A day is anomalous if any of these holds:
- |z| > 3
- growth over the baseline > 40% **and** at least 500 cases above it
- negative daily cases

**Severity** comes from |growth| over the baseline: ≥ 100% is CRITICAL, ≥ 40% is WARNING. A deviation under 500 cases is MINOR, and a day with no baseline is WARNING.

**Agent.** The agent receives the weekday, the baseline, the same weekday in prior weeks, the previous 7 days and the next 3 days. It replies in JSON mode. Any failure (unreadable reply, API error, missing key) becomes `FLAG_FOR_REVIEW`.

## 5. Verification

- `uv run pytest`: **21 passed**.
- CLI without the agent: USA 90 days, 7 anomalies. An unknown country gives a clean error, and `all` works.
- **Live agent run (USA, 90 days):** 7 anomalies. The rule fixed 1 MINOR day and the agent handled 6 in about 8 s. It classified all 6 as reporting artifacts (`FIX_ANOMALY`), with reasons such as "Negative case count is impossible and indicates a reporting artifact error." No `KEEP_ANOMALY` decision appeared live; that path is covered only by tests.
- **Streamlit (headless test harness):** no exceptions. Metrics, chart, tables and download render, and a bad country shows an error. The deprecated `use_container_width` was replaced with `width="stretch"`.
- **Chart check:** the chart was exported to PNG and inspected. The weekly cycle is visible, and the flagged days break the pattern.

## 6. Git

- `.env` and `*.csv` were added to `.gitignore`.
- A search for secrets turned up nothing.
- **Staged (not committed):** `.gitignore`, `.python-version`, `README.md`, `pyproject.toml`, `uv.lock`, the 9 new modules and `tests/test_pipeline.py`.
- **Left unstaged on purpose:** `requirements.txt` and the old fragment files. Deleting them from the session was blocked because they are untracked.

## 7. Follow-ups

- [ ] Delete the obsolete files:
  `rm requirements.txt src/groqclfr/{anom_detection,severity_classification,create_agent_groq_cloud,prompt_ai_agent,decision_ai,data_ingestion}.py`
- [ ] Make the first commit.
- [ ] Optional: switch to a data source that still updates, if "live" data matters.
- [ ] Optional: tune the thresholds in `config.py` per country, since reporting patterns differ.
