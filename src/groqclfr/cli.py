"""Command-line entry points."""

import argparse
import subprocess
import sys
from pathlib import Path

from tabulate import tabulate

from groqclfr import config
from groqclfr.data import DataSourceError
from groqclfr.pipeline import run_pipeline

REPORT_COLUMNS = ["Date", "Original Cases", "Cases", "Severity", "Decision", "Decided By", "Action", "Reason"]


def main() -> None:
    parser = argparse.ArgumentParser(description="COVID-19 anomaly detection with a GroqCloud agent")
    parser.add_argument("--country", default=config.DEFAULT_COUNTRY, help='country name/ISO code, or "all"')
    parser.add_argument("--days", type=int, default=config.DEFAULT_DAYS, help="days of history (ends 2023-03-09)")
    parser.add_argument("--no-agent", action="store_true", help="skip LLM calls; WARNING/CRITICAL get flagged")
    parser.add_argument("--csv", type=Path, help="also write the full result to this CSV file")
    args = parser.parse_args()

    try:
        df = run_pipeline(args.country, args.days, use_agent=not args.no_agent)
    except DataSourceError as exc:
        sys.exit(f"Data source error: {exc}")

    anomalies = df[df["Anomaly"]]
    print(f"{args.country}: {len(df)} days, {len(anomalies)} anomalies\n")
    if not anomalies.empty:
        report = anomalies[REPORT_COLUMNS].assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"))
        print(tabulate(report, headers="keys", tablefmt="github", showindex=False, maxcolwidths=[None] * 7 + [50]))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    app = Path(__file__).with_name("app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))
