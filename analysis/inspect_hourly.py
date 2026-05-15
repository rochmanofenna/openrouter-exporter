"""Inspect raw hourly token-rate data from Prometheus before building peak-hour forecaster.

Pulls the last N days of hourly deltas per (provider, model_id) and surfaces:
- Per-model count of valid (non-NaN) hours, missing hours, negative-delta hours
- Magnitude of negative deltas — are they noise (correction blips) or huge (mid-day re-tally)?
- When negatives occur — at UTC midnight (expected, counter reset) or random hours (real corrections)?
- Compares raw delta vs clamp_min(0) vs (deltas using sum-by-model-then-clamp)
- Per-model 7-day hourly profile saved as PNG

The point of this script is to decide whether clamping negatives is safe before the
peak-hour forecaster is built. We want to see the data, not assume.

Usage:
    python analysis/inspect_hourly.py

Env:
    PROM_URL    default http://localhost:9090
    DAYS        default 7
    PROVIDER    default dekallm
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
DAYS = int(os.environ.get("DAYS", "7"))
PROVIDER = os.environ.get("PROVIDER", "dekallm")
STEP_SECONDS = 3600  # 1h
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def prom_range(query: str, start: datetime, end: datetime, step: int) -> pd.DataFrame:
    resp = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={"query": query, "start": start.timestamp(), "end": end.timestamp(), "step": step},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")
    rows = []
    for series in payload["data"]["result"]:
        labels = series["metric"]
        for ts, val in series["values"]:
            rows.append({
                "ts": pd.Timestamp(int(float(ts)), unit="s", tz="UTC"),
                "value": float(val),
                **labels,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=DAYS)
    print(f"Inspecting {DAYS} days of hourly data: {start.isoformat()} -> {end.isoformat()}\n")

    # --- Query A: raw delta() summed across date label per model ---
    q_raw = (
        f'sum by (model_id) (delta(openrouter_provider_tokens_daily{{provider="{PROVIDER}"}}[1h]))'
    )
    df_raw = prom_range(q_raw, start, end, STEP_SECONDS)
    df_raw = df_raw.rename(columns={"value": "delta_raw"})

    # --- Query B: same but clamped at zero ---
    q_clamp = f'clamp_min({q_raw}, 0)'
    df_clamp = prom_range(q_clamp, start, end, STEP_SECONDS)
    df_clamp = df_clamp.rename(columns={"value": "delta_clamped"})

    if df_raw.empty:
        print("No data returned. Is the exporter running and Prometheus reachable?")
        return

    df = df_raw.merge(df_clamp[["ts", "model_id", "delta_clamped"]], on=["ts", "model_id"], how="left")
    df = df.sort_values(["model_id", "ts"]).reset_index(drop=True)
    df["hour_utc"] = df["ts"].dt.hour
    df["date_utc"] = df["ts"].dt.date
    df["dow_utc"] = df["ts"].dt.dayofweek
    df["is_negative"] = df["delta_raw"] < 0
    df["is_zero"] = df["delta_raw"] == 0

    # --- Per-model diagnostics ---
    print("=" * 80)
    print("Per-model hourly data quality")
    print("=" * 80)
    summary_rows = []
    for m, grp in df.groupby("model_id"):
        n = len(grp)
        n_pos = (grp["delta_raw"] > 0).sum()
        n_neg = (grp["delta_raw"] < 0).sum()
        n_zero = (grp["delta_raw"] == 0).sum()
        n_nan = grp["delta_raw"].isna().sum()
        neg_mag = grp.loc[grp["delta_raw"] < 0, "delta_raw"]
        neg_summary = (
            f"min={neg_mag.min():.2e} median={neg_mag.median():.2e} max={neg_mag.max():.2e}"
            if len(neg_mag) else "n/a"
        )
        pos_mag = grp.loc[grp["delta_raw"] > 0, "delta_raw"]
        pos_summary = (
            f"median={pos_mag.median():.2e} p95={pos_mag.quantile(0.95):.2e} max={pos_mag.max():.2e}"
            if len(pos_mag) else "n/a"
        )
        print(f"\n  {m}")
        print(f"    total hours: {n}   pos: {n_pos}   neg: {n_neg}   zero: {n_zero}   nan: {n_nan}")
        print(f"    positive delta:  {pos_summary}")
        print(f"    negative delta:  {neg_summary}")
        summary_rows.append({
            "model": m, "n_hours": n, "n_pos": n_pos, "n_neg": n_neg,
            "n_zero": n_zero, "n_nan": n_nan,
            "median_pos": pos_mag.median() if len(pos_mag) else np.nan,
            "max_neg_abs": neg_mag.abs().max() if len(neg_mag) else 0,
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "inspect_hourly_summary.csv", index=False)

    # --- When do negative deltas occur? ---
    print("\n" + "=" * 80)
    print("Distribution of negative deltas by UTC hour of day")
    print("=" * 80)
    neg = df[df["is_negative"]].copy()
    if len(neg) == 0:
        print("  No negative deltas in the window. Clamp_min is a no-op.")
    else:
        # Heuristic: if most negatives cluster at hour 0 (UTC midnight), they're
        # reset artifacts. If they're scattered through the day, they're real
        # mid-day corrections.
        by_hour = neg.groupby("hour_utc").size().to_dict()
        print(f"  Total negative-delta hours: {len(neg)}")
        print(f"  Negatives at UTC hour 0 (midnight reset suspect): {by_hour.get(0, 0)}")
        print(f"  Negatives elsewhere (real corrections): {len(neg) - by_hour.get(0, 0)}")
        print()
        print("  hour_utc | count | pct of negatives")
        for h in range(24):
            c = by_hour.get(h, 0)
            if c > 0:
                bar = "#" * int(c / max(by_hour.values()) * 30)
                print(f"  {h:>2}       | {c:>5} | {bar}")

    # --- Verdict suggestion based on data ---
    print("\n" + "=" * 80)
    print("Clamp-handling recommendation")
    print("=" * 80)
    total_hours = len(df)
    total_neg = int(df["is_negative"].sum())
    pct_neg = 100 * total_neg / total_hours if total_hours else 0
    midnight_neg = int(((df["is_negative"]) & (df["hour_utc"] == 0)).sum())
    midnight_pct_of_neg = 100 * midnight_neg / total_neg if total_neg else 0

    print(f"  Negative hours: {total_neg} of {total_hours} ({pct_neg:.1f}%)")
    print(f"  Of those, at UTC midnight: {midnight_neg} ({midnight_pct_of_neg:.0f}%)")
    if pct_neg < 1:
        print("  -> Almost no negatives. clamp_min(0) is essentially free; use it for cleanliness.")
    elif midnight_pct_of_neg > 70:
        print("  -> Negatives concentrate at midnight. Reset artifact, safe to clamp.")
    else:
        print("  -> Negatives spread throughout the day suggest real upstream re-tally events.")
        print("     Clamping discards information about these corrections; consider keeping raw")
        print("     and handling negatives explicitly in the forecaster (e.g. residual analysis).")

    # --- Save raw hourly profile per model for visualization ---
    long_path = OUT / "inspect_hourly_long.csv"
    df.to_csv(long_path, index=False)
    print(f"\n  Saved long-format hourly data -> {long_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not installed; skipping plots)")
        return

    # One subplot per model — 7-day hourly profile
    models = sorted(df["model_id"].unique())
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        grp = df[df["model_id"] == m].sort_values("ts")
        ax.plot(grp["ts"], grp["delta_raw"] / 1e6, linewidth=0.9, label="raw", alpha=0.8)
        ax.plot(grp["ts"], grp["delta_clamped"] / 1e6, linewidth=0.9, label="clamped", alpha=0.6)
        ax.set_title(m, fontsize=9)
        ax.axhline(0, color="white", linewidth=0.4, alpha=0.5)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("UTC")
    fig.suptitle(f"Hourly delta per model — last {DAYS} days (tokens/M per hour, raw vs clamp_min)", y=1.001)
    fig.tight_layout()
    out_path = OUT / "inspect_hourly_per_model.png"
    fig.savefig(out_path, dpi=120)
    print(f"  Saved per-model plot -> {out_path}")

    # Hour-of-day profile averaged across days (sanity-check the diurnal shape)
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    for m in models:
        grp = df[df["model_id"] == m]
        hour_avg = grp.groupby("hour_utc")["delta_clamped"].mean() / 1e6
        ax2.plot(hour_avg.index, hour_avg.values, marker="o", markersize=3, linewidth=1.2, label=m.split("/")[-1][:24])
    ax2.set_xticks(range(0, 24))
    ax2.set_xlabel("hour of day (UTC)  —  add 7 for GMT+7")
    ax2.set_ylabel("avg tokens/hour (millions, clamped)")
    ax2.set_title(f"Average hourly profile per model — last {DAYS} days")
    ax2.grid(alpha=0.15)
    ax2.legend(fontsize=8, loc="upper left")
    fig2.tight_layout()
    profile_path = OUT / "inspect_hourly_diurnal_avg.png"
    fig2.savefig(profile_path, dpi=120)
    print(f"  Saved diurnal-average plot -> {profile_path}")


if __name__ == "__main__":
    main()
