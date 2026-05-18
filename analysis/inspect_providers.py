"""Audit cross-provider data we've been collecting in Prometheus.

We added deepinfra, fireworks, together to OPENROUTER_ACTIVITY_PROVIDERS on
May 12. This script checks what's actually been captured:

1. Models per provider — how many distinct models has each been serving?
2. Date coverage per provider — when did we start, are there gaps?
3. Daily totals last 7 days — order-of-magnitude comparison vs DekaLLM
4. Cross-provider model overlap — which models are served by both DekaLLM and
   competitors? Those are candidate features for the DekaLLM forecaster
   (e.g., "DekaLLM's mistral volume relative to deepinfra's mistral volume").

Usage:
    python analysis/inspect_providers.py
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")


def prom_query(query: str) -> list[dict]:
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")
    return payload["data"]["result"]


def prom_range(query: str, start: datetime, end: datetime, step: int) -> list[dict]:
    r = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={"query": query, "start": start.timestamp(), "end": end.timestamp(), "step": step},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")
    return payload["data"]["result"]


def main() -> None:
    print("=" * 80)
    print("Provider summary — what's in Prometheus right now")
    print("=" * 80)

    # 1. List distinct providers we've been scraping
    res = prom_query("count by (provider) (openrouter_provider_tokens_daily)")
    providers = sorted([r["metric"]["provider"] for r in res])
    print(f"\nProviders captured ({len(providers)}):")
    for p in providers:
        print(f"  {p}")

    # 2. Per-provider summary: distinct models, distinct dates, total series
    print(f"\n{'='*80}")
    print(f"Per-provider: distinct models + dates seen")
    print(f"{'='*80}\n")
    summary = {}
    for p in providers:
        res = prom_query(f'count by (model_id) (openrouter_provider_tokens_daily{{provider="{p}"}})')
        models = sorted([(r["metric"]["model_id"], int(float(r["value"][1]))) for r in res], key=lambda x: -x[1])
        res2 = prom_query(f'count by (date) (openrouter_provider_tokens_daily{{provider="{p}"}})')
        dates = sorted([r["metric"]["date"] for r in res2])
        summary[p] = {"models": models, "dates": dates}
        print(f"  {p}: {len(models)} models, {len(dates)} distinct dates "
              f"({dates[0] if dates else 'n/a'} -> {dates[-1] if dates else 'n/a'})")

    # 3. Today's totals per provider — order-of-magnitude check
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n{'='*80}")
    print(f"Today's running totals per provider ({today}, intraday)")
    print(f"{'='*80}\n")
    for p in providers:
        res = prom_query(
            f'sum(openrouter_provider_tokens_daily{{provider="{p}",date="{today}"}})'
        )
        if res:
            total = float(res[0]["value"][1])
            print(f"  {p:18s}  {total/1e9:6.2f}B tokens (running, intraday)")
        else:
            print(f"  {p:18s}  no data for today")

    # 4. Last 7 days of daily totals per provider
    print(f"\n{'='*80}")
    print(f"Last 7 days — daily tokens per provider (sum of all models)")
    print(f"{'='*80}\n")
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=8)
    rows = []
    for p in providers:
        series = prom_range(
            f'sum by (date) (openrouter_provider_tokens_daily{{provider="{p}"}})',
            start, end, 3600,
        )
        for s in series:
            d = s["metric"].get("date", "")
            if not d:
                continue
            # Take the max value seen for this date (running totals are monotonic within a day)
            values = [float(v[1]) for v in s["values"]]
            if values:
                rows.append({"date": d, "provider": p, "max_total": max(values)})

    if rows:
        df = pd.DataFrame(rows).pivot_table(
            index="date", columns="provider", values="max_total", aggfunc="max"
        ).fillna(0) / 1e9
        # only show last 8 days
        df = df.tail(8)
        print(df.round(2).to_string())
        print(f"\n  Values in billions of tokens (running daily total seen across hour samples).")
    else:
        print("  No daily history available.")

    # 5. Cross-provider model overlap — which models does DekaLLM share with competitors?
    print(f"\n{'='*80}")
    print(f"Cross-provider model overlap")
    print(f"{'='*80}\n")
    if "dekallm" not in summary:
        print("  No DekaLLM data — can't compute overlap.")
        return
    dekallm_models = {m for m, _ in summary["dekallm"]["models"]}
    for p in providers:
        if p == "dekallm":
            continue
        other_models = {m for m, _ in summary[p]["models"]}
        shared = dekallm_models & other_models
        only_other = other_models - dekallm_models
        print(f"\n  dekallm ∩ {p}:  {len(shared)} shared model(s)")
        for m in sorted(shared):
            print(f"    [shared]  {m}")
        if len(only_other) > 0:
            print(f"  {p} only:  {len(only_other)} models (showing top 10 by date count)")
            top = sorted(summary[p]["models"], key=lambda x: -x[1])[:10]
            for m, n in top:
                if m in only_other:
                    print(f"    [{p}]  {m}  ({n} dates)")

    print(f"\n{'='*80}")
    print("Notes")
    print("=" * 80)
    print("""
- "Shared models" are candidates for a market-context feature in the DekaLLM
  forecaster: e.g., "DekaLLM's mistral volume today as a share of all-provider
  mistral volume today." That would capture whether DekaLLM is gaining or
  losing share of the market.

- Per-day totals are computed as the max running-counter value seen across
  hourly samples for each day — accurate for completed days, approximates the
  intraday peak for today's still-in-progress day.

- If a provider has 0 models or no dates: check the OPENROUTER_ACTIVITY_PROVIDERS
  env var on the VM is correctly listing it, and check `docker compose logs
  openrouter-exporter | grep provider_errors` for scrape failures.
""")


if __name__ == "__main__":
    main()
