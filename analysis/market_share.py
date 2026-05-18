"""Compute DekaLLM's market share over time for cross-provider models.

For each model that DekaLLM and at least one competitor serve, compute the
ratio of DekaLLM's daily token volume to the total across all providers.

Usage:
    python analysis/market_share.py

Default: gpt-oss-120b (served by all 4 providers). Override with:
    MODEL_ID=mistralai/mistral-nemo python analysis/market_share.py

Outputs:
    analysis/out/market_share_<model_sanitized>.csv
    analysis/out/market_share_<model_sanitized>.png
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-120b")
DAYS = int(os.environ.get("DAYS", "30"))
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


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
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=DAYS)
    print(f"Computing market share for: {MODEL_ID}")
    print(f"Window: {start.date()} -> {end.date()} ({DAYS} days)\n")

    # Per (provider, date), get the running daily total for this model.
    # Query at 1-hour step; for each (provider, date) we keep the max value seen
    # (which is the end-of-day total for completed UTC days).
    query = f'openrouter_provider_tokens_daily{{model_id="{MODEL_ID}"}}'
    series = prom_range(query, start, end, 3600)
    if not series:
        print(f"No data for {MODEL_ID}. Check the model_id label matches.")
        return

    rows = []
    for s in series:
        provider = s["metric"].get("provider", "unknown")
        date = s["metric"].get("date", "")
        if not date:
            continue
        # max over all hourly samples for this (provider, date) = end-of-day total
        max_val = max(float(v[1]) for v in s["values"])
        rows.append({"date": date, "provider": provider, "daily_tokens": max_val})

    if not rows:
        print(f"No (date, provider) data points extracted for {MODEL_ID}.")
        return

    df = pd.DataFrame(rows)
    # Pivot: rows = date, columns = provider, values = daily tokens
    wide = df.pivot_table(index="date", columns="provider", values="daily_tokens", aggfunc="max").fillna(0)
    wide = wide.sort_index()

    # Compute total and DekaLLM share
    if "dekallm" not in wide.columns:
        print(f"No DekaLLM data for {MODEL_ID}. They may not serve this model.")
        return
    wide["total"] = wide.sum(axis=1)
    wide["dekallm_share_pct"] = 100 * wide["dekallm"] / wide["total"]

    print(f"=== Daily market share for {MODEL_ID} (last {DAYS} days) ===\n")
    # Print just the recent days clearly
    recent = wide.tail(min(20, len(wide))).copy()
    for c in recent.columns:
        if c == "dekallm_share_pct":
            continue
        recent[c] = (recent[c] / 1e9).round(2)
    recent["dekallm_share_pct"] = recent["dekallm_share_pct"].round(2)
    print(recent.to_string())
    print(f"\n  (Provider columns shown in billions of tokens; share column in %)")

    out_csv = OUT / f"market_share_{MODEL_ID.replace('/', '_')}.csv"
    wide.to_csv(out_csv)
    print(f"\nSaved CSV -> {out_csv}")

    # Plot: stacked area of provider volumes + share line on secondary axis
    try:
        import matplotlib.pyplot as plt

        # Drop derived columns for the stack plot
        providers_only = [c for c in wide.columns if c not in ("total", "dekallm_share_pct")]
        # Order so dekallm is on bottom for visibility
        cols = ["dekallm"] + [c for c in providers_only if c != "dekallm"]
        plot_data = wide[cols].copy()

        fig, ax = plt.subplots(figsize=(12, 5.5))
        plot_data.plot.area(ax=ax, alpha=0.7, stacked=True, colormap="tab10")
        ax.set_title(f"{MODEL_ID} — daily token volume by provider", fontsize=11)
        ax.set_ylabel("daily tokens")
        ax.set_xlabel("date (UTC)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.15)

        # Secondary axis: DekaLLM share as a percentage
        ax2 = ax.twinx()
        ax2.plot(wide.index, wide["dekallm_share_pct"], color="red", linewidth=2, linestyle="--",
                 marker="o", markersize=4, label="DekaLLM share (%)")
        ax2.set_ylabel("DekaLLM share (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(fontsize=8, loc="upper right")

        # Rotate x labels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")

        fig.tight_layout()
        out_png = OUT / f"market_share_{MODEL_ID.replace('/', '_')}.png"
        fig.savefig(out_png, dpi=120)
        print(f"Saved plot -> {out_png}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
