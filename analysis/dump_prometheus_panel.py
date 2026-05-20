"""Dump the full (model, provider, date) panel from Prometheus to a single CSV.

The Go exporter on dekavm has been scraping OpenRouter for ~90 days across
4 providers (dekallm, deepinfra, fireworks, together). All the per-day token
volumes plus per-(model, provider) lever values (price, throughput, latency,
uptime) live in Prometheus right now — but never got persisted anywhere we
can re-read after a restart.

This script joins everything into one wide CSV with rows of:
    (model_id, provider, date,
     daily_tokens,
     input_price, output_price,
     throughput_p50, latency_p50_ms,
     uptime_1d_pct)

That CSV is the panel needed to fit the share formula (Experiment A onward).

Usage:
    python analysis/dump_prometheus_panel.py
    # produces analysis/out/share_panel.csv

Env:
    PROM_URL      default http://localhost:9090
    OUT_PATH      default analysis/out/share_panel.csv
    LOOKBACK_DAYS default 100
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Token metric model_ids carry an OpenRouter permaslug date suffix
# (e.g. google/gemma-4-26b-a4b-it-20260403), but endpoint feature metrics
# use the base slug (no suffix). Strip the suffix when joining the two.
_DATE_SUFFIX = re.compile(r"-\d{8}$")


def base_slug(model_id: str) -> str:
    return _DATE_SUFFIX.sub("", model_id)

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
OUT_PATH = Path(os.environ.get("OUT_PATH", OUT / "share_panel.csv"))
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "100"))

# Endpoint metrics use provider_name (display name); token metric uses
# provider (slug). Map between them.
PROVIDER_NAME_HINTS = {
    "dekallm":   ["DekaLLM", "Dekallm", "dekallm"],
    "deepinfra": ["DeepInfra", "Deepinfra", "deepinfra"],
    "fireworks": ["Fireworks", "Fireworks AI", "fireworks"],
    "together":  ["Together", "Together AI", "together"],
}


def prom(query: str) -> list[dict]:
    url = f"{PROM_URL}/api/v1/query?query={urllib.parse.quote(query)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        body = json.loads(r.read())
    if body.get("status") != "success":
        raise RuntimeError(f"prom error: {body}")
    return body["data"]["result"]


def prom_range(query: str, start: datetime, end: datetime, step: int) -> list[dict]:
    url = (f"{PROM_URL}/api/v1/query_range?"
           f"query={urllib.parse.quote(query)}"
           f"&start={start.timestamp()}&end={end.timestamp()}&step={step}")
    with urllib.request.urlopen(url, timeout=60) as r:
        body = json.loads(r.read())
    if body.get("status") != "success":
        raise RuntimeError(f"prom error: {body}")
    return body["data"]["result"]


def resolve_provider_names() -> dict[str, str]:
    """Map every token-side slug -> display name used in endpoint metrics.

    Auto-discovers all (slug, display_name) pairs by querying both metrics
    and matching by lowercase. Falls back to PROVIDER_NAME_HINTS for tricky
    cases (compound names like "Google AI Studio").
    """
    feat_res = prom("group by (provider_name) (openrouter_model_input_price_dollars_per_million_tokens)")
    display_names = [r["metric"]["provider_name"] for r in feat_res]

    tok_res = prom("group by (provider) (openrouter_provider_tokens_daily)")
    slugs = [r["metric"]["provider"] for r in tok_res]

    mapping: dict[str, str] = {}
    for slug in slugs:
        # 1) Try hardcoded hints first (handles "google-ai-studio" -> "Google AI Studio")
        hints = PROVIDER_NAME_HINTS.get(slug, [slug])
        for hint in hints:
            for d in display_names:
                if d.lower() == hint.lower():
                    mapping[slug] = d
                    break
            if slug in mapping:
                break

        # 2) Try exact case-insensitive match on the slug itself
        if slug not in mapping:
            for d in display_names:
                if d.lower() == slug.lower():
                    mapping[slug] = d
                    break

        # 3) Try collapsed comparison: strip non-alphanumerics, lowercase
        if slug not in mapping:
            slug_norm = "".join(c for c in slug.lower() if c.isalnum())
            for d in display_names:
                d_norm = "".join(c for c in d.lower() if c.isalnum())
                if d_norm == slug_norm:
                    mapping[slug] = d
                    break

        # 4) Substring fallback
        if slug not in mapping:
            for d in display_names:
                if slug.lower() in d.lower().replace(" ", "") or \
                   d.lower().replace(" ", "") in slug.lower():
                    mapping[slug] = d
                    break

    return mapping


def pull_daily_tokens(start: datetime, end: datetime) -> pd.DataFrame:
    series = prom_range("openrouter_provider_tokens_daily", start, end, 3600)
    rows = []
    for s in series:
        provider = s["metric"].get("provider")
        date = s["metric"].get("date")
        model_id = s["metric"].get("model_id")
        if not (provider and date and model_id):
            continue
        # The metric is the running daily total; end-of-day value = max over hourly samples
        max_val = max(float(v[1]) for v in s["values"])
        rows.append({"model_id": model_id, "provider": provider,
                     "date": date, "daily_tokens": max_val})
    return pd.DataFrame(rows)


def pull_model_activity(start: datetime, end: datetime) -> pd.DataFrame:
    """Per-(model, date) request count and token totals from the /activity scrape.

    This is the *aggregate* across all providers for a model on a given date —
    used as the denominator for tokens-per-request[M], which the decomposition
    treats as a model-level property to avoid provider-routing endogeneity.

    Only models in OPENROUTER_ACTIVITY_MODELS have data here (currently
    DekaLLM's portfolio).
    """
    queries = {
        "model_requests":          "openrouter_model_activity_requests",
        "model_prompt_tokens":     "openrouter_model_activity_prompt_tokens",
        "model_completion_tokens": "openrouter_model_activity_completion_tokens",
    }
    per_metric: dict[str, pd.DataFrame] = {}
    for col, q in queries.items():
        series = prom_range(q, start, end, 86400)
        rows = []
        for s in series:
            model_id = s["metric"].get("model_id")
            date_label = s["metric"].get("date")  # the activity metric carries 'date' as a label
            if not model_id:
                continue
            for ts, val in s["values"]:
                # Prefer the metric's own date label (UTC date the activity
                # references). Fall back to sample timestamp if label missing.
                if date_label:
                    date = date_label
                else:
                    date = datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()
                rows.append({"model_id": model_id, "date": date, col: float(val)})
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        # Multiple samples per (model, date) → take max (running total)
        df = df.groupby(["model_id", "date"], as_index=False)[col].max()
        per_metric[col] = df

    out = None
    for col, df in per_metric.items():
        if out is None:
            out = df
        else:
            out = out.merge(df, on=["model_id", "date"], how="outer")
    if out is None or out.empty:
        return pd.DataFrame()
    # Derive total + tpr
    out["model_total_tokens_activity"] = (
        out["model_prompt_tokens"].fillna(0) + out["model_completion_tokens"].fillna(0)
    )
    return out


def pull_endpoint_features(start: datetime, end: datetime,
                           name_to_slug: dict[str, str]) -> pd.DataFrame:
    """Per-(model, provider, date) snapshot of price / throughput / latency / uptime."""
    queries = {
        "input_price":    'openrouter_model_input_price_dollars_per_million_tokens',
        "output_price":   'openrouter_model_output_price_dollars_per_million_tokens',
        "throughput_p50": 'openrouter_endpoint_throughput_tokens_per_second{quantile="p50"}',
        "latency_p50_ms": 'openrouter_endpoint_latency_milliseconds{quantile="p50"}',
        "uptime_1d_pct":  'openrouter_endpoint_uptime_percentage_last_1d',
    }
    per_metric: dict[str, pd.DataFrame] = {}
    for col, q in queries.items():
        series = prom_range(q, start, end, 86400)  # daily step
        rows = []
        for s in series:
            model_id = s["metric"].get("model_id")
            pname = s["metric"].get("provider_name")
            if not (model_id and pname):
                continue
            slug = name_to_slug.get(pname)
            if slug is None:
                continue
            for ts, val in s["values"]:
                date = datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()
                rows.append({"model_id": model_id, "provider": slug,
                             "date": date, col: float(val)})
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df = df.groupby(["model_id", "provider", "date"], as_index=False)[col].mean()
        per_metric[col] = df

    out = None
    for col, df in per_metric.items():
        if out is None:
            out = df
        else:
            out = out.merge(df, on=["model_id", "provider", "date"], how="outer")
    return out if out is not None else pd.DataFrame()


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the share formula actually wants to fit on:
       share targets, days-since-added, normalized ratios, uptime tier."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["model_id", "provider", "date"]).reset_index(drop=True)

    # Token share is the target (within-model, within-date)
    daily_total = df.groupby(["model_id", "date"])["daily_tokens"].transform("sum")
    df["model_daily_total"] = daily_total
    df["token_share"] = df["daily_tokens"] / daily_total.replace(0, pd.NA)

    # (M, P) ramp: days since this provider first appeared serving this model
    first = (df[df["daily_tokens"] > 0]
             .groupby(["model_id", "provider"])["date"].min()
             .rename("provider_added_date"))
    df = df.merge(first, on=["model_id", "provider"], how="left")
    df["days_since_provider_added"] = (df["date"] - df["provider_added_date"]).dt.days

    # M-level newness (rough proxy — actual launch date may predate our window)
    model_first = df.groupby("model_id")["date"].min().rename("model_first_date")
    df = df.merge(model_first, on="model_id", how="left")
    df["days_since_model_seen"] = (df["date"] - df["model_first_date"]).dt.days

    # Throughput relative to per-(M, date) median
    med_tp = df.groupby(["model_id", "date"])["throughput_p50"].transform("median")
    df["throughput_ratio_to_median"] = df["throughput_p50"] / med_tp.replace(0, pd.NA)

    # Price relative to per-(M, date) cheapest (1.0 = cheapest)
    min_in = df.groupby(["model_id", "date"])["input_price"].transform("min")
    df["input_price_ratio_to_min"] = df["input_price"] / min_in.replace(0, pd.NA)
    min_out = df.groupby(["model_id", "date"])["output_price"].transform("min")
    df["output_price_ratio_to_min"] = df["output_price"] / min_out.replace(0, pd.NA)

    # Uptime tier matches OpenRouter's actual production buckets
    def tier(p):
        if pd.isna(p):
            return None
        if p >= 95:
            return "full"
        if p >= 80:
            return "degraded"
        return "fallback"
    df["uptime_tier"] = df["uptime_1d_pct"].apply(tier)

    # Model-level tokens-per-request, derived from /activity totals.
    # NaN for models we don't scrape activity on (most non-DekaLLM models).
    if "model_total_tokens_activity" in df.columns and "model_requests" in df.columns:
        req = df["model_requests"].astype(float)
        toks = df["model_total_tokens_activity"].astype(float)
        df["model_tpr"] = toks / req.where(req > 0)
    else:
        df["model_tpr"] = float("nan")

    return df


def main() -> None:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"Pulling panel from Prometheus at {PROM_URL}")
    print(f"Window: {start.date()} → {end.date()} ({LOOKBACK_DAYS} days)\n")

    print("1/4 Resolving provider name ↔ slug map...")
    name_map = resolve_provider_names()
    name_to_slug = {v: k for k, v in name_map.items()}
    for slug, name in name_map.items():
        print(f"    {slug:10s} → {name}")

    print("\n2/4 Pulling daily token volumes...")
    tokens = pull_daily_tokens(start, end)
    print(f"    {len(tokens):,} (model, provider, date) rows")

    print("\n3/4 Pulling endpoint features (price, throughput, latency, uptime)...")
    features = pull_endpoint_features(start, end, name_to_slug)
    print(f"    {len(features):,} feature rows")

    print("\n4/5 Pulling model-level activity (requests, prompt/completion tokens)...")
    activity = pull_model_activity(start, end)
    print(f"    {len(activity):,} model-activity rows")

    print("\n5/5 Joining and deriving columns...")
    # Join via base_slug (strip date suffix) so permaslug-tagged token rows
    # match base-slug-tagged feature/activity rows.
    tokens["base_slug"] = tokens["model_id"].apply(base_slug)
    features_renamed = features.rename(columns={"model_id": "base_slug"})
    panel = tokens.merge(
        features_renamed,
        on=["base_slug", "provider", "date"],
        how="left",
    )
    if not activity.empty:
        activity_renamed = activity.rename(columns={"model_id": "base_slug"})
        panel = panel.merge(activity_renamed, on=["base_slug", "date"], how="left")
    else:
        for col in ("model_requests", "model_prompt_tokens",
                    "model_completion_tokens", "model_total_tokens_activity"):
            panel[col] = pd.NA
    panel = add_derived(panel)

    print(f"\nFinal panel: {len(panel):,} rows")
    print(f"  models:    {panel['model_id'].nunique()}")
    print(f"  providers: {panel['provider'].nunique()}")
    print(f"  dates:     {panel['date'].nunique()}  "
          f"({panel['date'].min().date()} → {panel['date'].max().date()})")

    print(f"\nCoverage by column:")
    for col in ["token_share", "input_price", "output_price",
                "throughput_p50", "latency_p50_ms", "uptime_1d_pct"]:
        n = panel[col].notna().sum()
        pct = 100.0 * n / len(panel) if len(panel) else 0
        print(f"  {col:24s}  {n:>5,} non-null  ({pct:.1f}%)")

    feat_cols = ["token_share", "input_price", "output_price",
                 "throughput_p50", "latency_p50_ms", "uptime_1d_pct"]
    complete = panel.dropna(subset=feat_cols)
    print(f"\n  ROWS COMPLETE FOR FITTING (all features non-null): {len(complete):,}")
    if len(complete) > 0:
        print(f"    across {complete['model_id'].nunique()} models, "
              f"{complete['provider'].nunique()} providers, "
              f"{complete['date'].nunique()} dates")

    panel.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    # Also write a forward-filled variant for the fit scripts. Features change
    # slowly so carrying a recent reading forward across the (M, P) timeline
    # is a safe approximation that lets the fit use 90 days of token data.
    filled = panel.copy()
    filled = filled.sort_values(["base_slug", "provider", "date"])
    feat_cols = ["input_price", "output_price", "throughput_p50",
                 "latency_p50_ms", "uptime_1d_pct"]
    filled[feat_cols] = filled.groupby(["base_slug", "provider"])[feat_cols].ffill()
    if "model_tpr" in filled.columns:
        filled["model_tpr"] = filled.groupby("base_slug")["model_tpr"].ffill()
    filled_path = OUT_PATH.with_name("share_panel_filled.csv")
    filled.to_csv(filled_path, index=False)
    print(f"Saved: {filled_path}")
    complete_filled = filled.dropna(subset=feat_cols + ["token_share"])
    print(f"  Forward-filled complete-for-fitting rows: {len(complete_filled):,} "
          f"({complete_filled['model_id'].nunique()} models, "
          f"{complete_filled['provider'].nunique()} providers, "
          f"{complete_filled['date'].nunique()} dates)")


if __name__ == "__main__":
    main()
