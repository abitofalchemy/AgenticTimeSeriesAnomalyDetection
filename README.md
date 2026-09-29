# COVID-19 Anomaly Detection with an AI Agent

This work is a replication of https://github.com/rautmadhura4/anomaly_detection_agent/tree/main. The original work was presented by Madhura Raut (https://towardsdatascience.com/building-an-ai-agent-to-detect-and-handle-anomalies-in-time-series-data/)


End-to-end agentic anomaly detection and handling on COVID-19 time-series data:

1. **Ingest**: daily new cases from the public [disease.sh](https://disease.sh) API
   (cumulative totals are differenced into daily counts).
2. **Detect**: a day is anomalous if it is a Z-score spike (|z| > 3), a growth surge
   (> 40% and ≥ 500 cases above baseline), or negative (a downward revision).
3. **Classify**: severity `MINOR` / `WARNING` / `CRITICAL` from growth vs. baseline.
4. **Decide** (hybrid policy):
   - `MINOR` → auto-corrected by rule, no LLM call.
   - `WARNING` / `CRITICAL` → a GroqCloud LLM agent (`openai/gpt-oss-120b`) chooses
     `FIX_ANOMALY`, `KEEP_ANOMALY` or `FLAG_FOR_REVIEW` and gives a one-sentence reason.
     Unparseable replies, API errors or a missing key fall back to `FLAG_FOR_REVIEW`.
5. **Act**: fixes replace the value with the baseline; the original is kept in `Original Cases`.
6. **Visualize**: a Streamlit dashboard.

**Baseline** = median of the same weekday over the prior 3 weeks. Many jurisdictions report
on fixed weekdays, so a trailing daily mean flags the weekly reporting cycle itself
(USA, 90 days: 31 anomalies with a 7-day mean vs. 7 with the weekday median).

> **Data caveat:** disease.sh's historical endpoint is backed by Johns Hopkins CSSE,
> which stopped updating on **2023-03-09**. "Last N days" means the N days ending then.

## Setup

```bash
uv sync
echo "GROQ_API_KEY=gsk_..." > .env   # optional; without it severe anomalies are flagged
```

## Usage

```bash
uv run groqclfr --country usa --days 90            # CLI report
uv run groqclfr --country germany --no-agent        # rules only, no LLM calls
uv run groqclfr --country all --csv out.csv         # global totals, save full result
uv run groqclfr-dashboard                           # Streamlit dashboard
uv run pytest                                       # tests (no network needed)
```

## Layout

```
src/groqclfr/
  config.py     thresholds, model id, API settings
  data.py       disease.sh ingestion → daily new cases
  detection.py  anomaly detection + severity classification
  agent.py      observation/prompt building, Groq call, decision parsing
  actions.py    apply FIX / KEEP / FLAG to a row
  pipeline.py   run_pipeline(): load → detect → classify → decide → act
  cli.py        `groqclfr` and `groqclfr-dashboard` entry points
  app.py        Streamlit dashboard
tests/          unit tests with a fake Groq client
```

All thresholds live in `config.py`.
