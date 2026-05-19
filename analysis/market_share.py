"""Compute DekaLLM's market share over time for one model, against ALL providers.

Pulls per-day, per-provider token counts from the local analytics API (which
mirrors OpenRouter's data) and computes DekaLLM's share of total volume.

Data source upgrade vs prior version: previously we scraped 4 hand-picked
providers (deepinfra/fireworks/together) into Prometheus and joined them
locally. The API exposes all providers serving each model with up to 91
days of history, so we no longer need to maintain a provider allowlist
and the historical window is ~3× deeper.

Usage:
    python analysis/market_share.py

Default model: mistralai/mistral-nemo (DekaLLM's biggest growth story).
Override:
    MODEL_ID=openai/gpt-oss-120b python analysis/market_share.py
    DAYS=30 python analysis/market_share.py    # limit window

Outputs:
    analysis/out/market_share_<model_sanitized>.csv
    analysis/out/market_share_<model_sanitized>.png
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from analysis.api_client import default_client, require_alive

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/mistral-nemo")
DAYS = int(os.environ.get("DAYS", "0"))  # 0 = full history available
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def fetch_share_frame(client, model_id: str, days: int = 0) -> pd.DataFrame:
    """Return a wide DataFrame indexed by date, with one column per provider plus derived totals.

    days=0 means "use everything the API returns" — usually up to 91 days.
    """
    resolved = client.resolve(model_id)
    rows = client.db_model_provider_tokens(resolved)
    if not rows and resolved != model_id:
        rows = client.db_model_provider_tokens(model_id)
    if not rows:
        return pd.DataFrame()

    flat = []
    for row in rows:
        date = row.get("date")
        if not date:
            continue
        for provider, tokens in row.get("providers", {}).items():
            flat.append({"date": date, "provider": provider, "tokens": int(tokens)})

    df = pd.DataFrame(flat)
    if df.empty:
        return df

    wide = (
        df.pivot_table(index="date", columns="provider", values="tokens", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )

    if days > 0:
        wide = wide.tail(days)

    if "dekallm" not in wide.columns:
        # add empty column so downstream math works
        wide["dekallm"] = 0

    wide["total"] = wide.sum(axis=1)
    wide["dekallm_share_pct"] = 100 * wide["dekallm"] / wide["total"]
    return wide


def print_recent(wide: pd.DataFrame, n: int = 21) -> None:
    if wide.empty:
        print("No data.")
        return
    providers = [c for c in wide.columns if c not in ("total", "dekallm_share_pct")]
    recent = wide.tail(min(n, len(wide))).copy()
    for c in providers + ["total"]:
        recent[c] = (recent[c] / 1e9).round(2)
    recent["dekallm_share_pct"] = recent["dekallm_share_pct"].round(2)
    print(recent.to_string())
    print("\n  (Provider columns shown in billions of tokens; share column in %)")


def plot(wide: pd.DataFrame, model_id: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    providers = [c for c in wide.columns if c not in ("total", "dekallm_share_pct")]
    # Put dekallm at the bottom of the stack so its growth is visually obvious.
    cols = ["dekallm"] + sorted(p for p in providers if p != "dekallm")
    plot_data = wide[cols].copy()

    fig, ax = plt.subplots(figsize=(13, 6))
    plot_data.plot.area(ax=ax, alpha=0.75, stacked=True, colormap="tab20")
    ax.set_title(f"{model_id} — daily tokens by provider ({len(wide)} days)", fontsize=12)
    ax.set_ylabel("daily tokens")
    ax.set_xlabel("date")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.15)

    ax2 = ax.twinx()
    ax2.plot(
        wide.index, wide["dekallm_share_pct"],
        color="red", linewidth=2, linestyle="--", marker="o", markersize=4,
        label="DekaLLM share (%)",
    )
    ax2.set_ylabel("DekaLLM share (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(fontsize=8, loc="upper right")

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot -> {out_path}")


def main() -> None:
    client = require_alive()
    resolved = client.resolve(MODEL_ID)
    suffix = f" (resolved as {resolved})" if resolved != MODEL_ID else ""
    print(f"Computing market share for: {MODEL_ID}{suffix}")
    print(f"Window: {DAYS or 'all available'} days\n")

    wide = fetch_share_frame(client, MODEL_ID, days=DAYS)
    if wide.empty:
        print(f"No data for {MODEL_ID}.")
        return

    print(f"=== Daily market share for {MODEL_ID} ({len(wide)} days) ===\n")
    print_recent(wide, n=min(21, len(wide)))

    # Headline numbers
    dek_appeared = wide[wide["dekallm"] > 0]
    if not dek_appeared.empty:
        entry_date = dek_appeared.index[0]
        entry_share = float(dek_appeared.iloc[0]["dekallm_share_pct"])
        peak_share = float(dek_appeared["dekallm_share_pct"].max())
        current_share = float(wide.iloc[-1]["dekallm_share_pct"])
        print(
            f"\nDekaLLM: entered on {entry_date} at {entry_share:.2f}% share, "
            f"peaked at {peak_share:.2f}%, currently {current_share:.2f}% "
            f"({len(dek_appeared)} days in market)."
        )

    sanitized = MODEL_ID.replace("/", "_")
    out_csv = OUT / f"market_share_{sanitized}.csv"
    wide.to_csv(out_csv)
    print(f"\nSaved CSV  -> {out_csv}")

    out_png = OUT / f"market_share_{sanitized}.png"
    plot(wide, MODEL_ID, out_png)


if __name__ == "__main__":
    main()
