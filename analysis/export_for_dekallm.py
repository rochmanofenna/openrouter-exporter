"""Dump all collected DekaLLM data to CSV for sharing.

Pulls from the running Prometheus, writes three CSVs into analysis/out/:
- dekallm_daily_totals.csv : raw running-total samples per (model, date)
- dekallm_hourly_rate.csv  : 1-hour token delta (smoothed demand signal)
- dekallm_30min_rate.csv   : 30-min token delta (forecasting target)

Run from repo root after the exporter+Prometheus stack is up:
    python analysis/export_for_dekallm.py

Env knobs:
    PROM_URL    default http://localhost:9090
    DAYS        default 14 (capture the full retention window we have)
    PROVIDER    default dekallm
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
DAYS = int(os.environ.get("DAYS", "14"))
PROVIDER = os.environ.get("PROVIDER", "dekallm")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

END = datetime.now(timezone.utc).replace(microsecond=0)
START = END - timedelta(days=DAYS)

QUERIES = {
    "daily_totals": (
        f'openrouter_provider_tokens_daily{{provider="{PROVIDER}"}}',
        1800,  # sample every 30 min — running totals don't move faster
    ),
    "hourly_rate": (
        f'sum by (model_id) (clamp_min(delta(openrouter_provider_tokens_daily{{provider="{PROVIDER}"}}[1h]), 0))',
        3600,
    ),
    "30min_rate": (
        f'sum by (model_id) (clamp_min(delta(openrouter_provider_tokens_daily{{provider="{PROVIDER}"}}[30m]), 0))',
        1800,
    ),
}


def prom_range(query: str, step: int) -> list[dict]:
    resp = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": START.timestamp(),
            "end": END.timestamp(),
            "step": step,
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")
    return payload["data"]["result"]


def main() -> None:
    print(f"Exporting {DAYS}d of DekaLLM data: {START.isoformat()} -> {END.isoformat()}")

    for name, (query, step) in QUERIES.items():
        series = prom_range(query, step)
        rows = []
        for s in series:
            labels = s["metric"]
            for ts, val in s["values"]:
                rows.append({
                    "timestamp_utc": datetime.fromtimestamp(int(float(ts)), tz=timezone.utc).isoformat(),
                    "model_id": labels.get("model_id", ""),
                    "date": labels.get("date", ""),
                    "value": float(val),
                })

        path = OUT / f"dekallm_{name}.csv"
        if not rows:
            print(f"  {path.name}: no data")
            continue

        rows.sort(key=lambda r: (r["model_id"], r["timestamp_utc"]))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "model_id", "date", "value"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {path.name}: {len(rows)} rows, {len(series)} series")

    readme = OUT / "README.txt"
    readme.write_text(
        f"""DekaLLM token volume data exported from OpenRouter provider chart.

Window:   {START.date()} -> {END.date()} (UTC, {DAYS} days)
Provider: {PROVIDER}
Source:   https://openrouter.ai/provider/{PROVIDER} chart payload, scraped via
          custom Prometheus exporter (github.com/rochmanofenna/openrouter-exporter)

Files:
- dekallm_daily_totals.csv : per (model, date) daily running-total samples
                             at 30-min cadence. Today's bucket grows intraday.
- dekallm_hourly_rate.csv  : 1-hour token delta per model (PromQL clamp_min so
                             upstream re-tally corrections don't go negative).
                             Smoothest representation of demand.
- dekallm_30min_rate.csv   : 30-min token delta per model. Used as the target
                             for forecasting; noisier due to upstream batching.

Schema (all CSVs):
    timestamp_utc : ISO-8601 UTC
    model_id      : OpenRouter model identifier
    date          : YYYY-MM-DD UTC, only set for daily_totals
    value         : tokens (running total or delta depending on file)

Notes:
- z-ai/glm-4.7 traffic dropped to 0 on 2026-05-07 and never recovered.
- Tokens are aggregate across all DekaLLM customers — provider-level slice
  from OpenRouter's public chart, not per-customer breakdown.
""")
    print(f"  {readme.name}: written")
    print(f"\nDone. Files in {OUT}")


if __name__ == "__main__":
    main()
