#!/usr/bin/env python3
"""Per-model share growth for DekaLLM's portfolio.

Uses the local analytics API at http://localhost:8000 to pull per-day,
per-provider token counts for each model DekaLLM serves. Computes
DekaLLM's share over time and reports entry date, peak share, current
share, and growth multiple.
"""

import json
import urllib.parse
import urllib.request

API = "http://localhost:8000"

# DekaLLM's current 5-model portfolio as of 2026-05-18.
DEKALLM_MODELS = [
    "mistralai/mistral-nemo",
    "openai/gpt-oss-120b",
    "nvidia/nemotron-3-super-120b-a12b-20230311",
    "google/gemma-4-26b-a4b-it-20260403",
    "qwen/qwen3.5-35b-a3b-20260224",
]


def fetch(path: str) -> dict:
    url = f"{API}{path}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def resolve_slug(slug: str) -> str:
    try:
        body = fetch(f"/scrape/resolve/{urllib.parse.quote(slug, safe='')}")
        return body.get("data", {}).get("permaslug") or slug
    except Exception:
        return slug


def provider_tokens(slug: str) -> list[dict]:
    encoded = urllib.parse.quote(slug, safe="")
    body = fetch(f"/db/models/{encoded}/provider-tokens")
    return body.get("data", {}).get("data", [])


def compute_share_series(rows: list[dict], provider: str = "dekallm"):
    series = []
    for row in rows:
        date = row["date"]
        providers = row.get("providers", {})
        total = sum(providers.values())
        prov = providers.get(provider, 0)
        if total > 0:
            series.append((date, 100.0 * prov / total, prov, total))
    series.sort(key=lambda x: x[0])
    return series


def summarize(slug: str, series):
    if not series:
        return {"model": slug, "error": "no data"}

    appeared = [s for s in series if s[1] > 0]
    if not appeared:
        return {"model": slug, "error": "DekaLLM never appeared"}

    entry_date, entry_share, _, _ = appeared[0]
    peak_date, peak_share, _, _ = max(appeared, key=lambda x: x[1])
    current_date, current_share, _, _ = series[-1]

    growth = current_share / entry_share if entry_share > 0 else None

    return {
        "model": slug,
        "history_days": len(series),
        "entry_date": entry_date,
        "entry_share": entry_share,
        "peak_date": peak_date,
        "peak_share": peak_share,
        "current_date": current_date,
        "current_share": current_share,
        "growth_mult": growth,
        "days_in_market": len(appeared),
    }


def fmt_pct(x):
    return f"{x:5.2f}%" if x is not None else "  —  "


def fmt_growth(x):
    if x is None:
        return "   —  "
    if x >= 10:
        return f"{x:5.1f}x"
    return f"{x:5.2f}x"


def main():
    print()
    print(f"{'Model':<52} {'Hist':>5} {'Entry':<12} {'Entry%':>7} {'Peak%':>7} {'Now%':>7} {'Growth':>7} {'Days':>5}")
    print("-" * 120)

    summaries = []
    for slug in DEKALLM_MODELS:
        resolved = resolve_slug(slug)
        try:
            rows = provider_tokens(resolved)
        except Exception:
            try:
                rows = provider_tokens(slug)
                resolved = slug
            except Exception as e:
                print(f"{slug:<52}  ERROR: {e}")
                continue

        series = compute_share_series(rows)
        s = summarize(resolved, series)
        s["resolved"] = resolved
        s["series"] = series
        summaries.append(s)

        if "error" in s:
            print(f"{slug:<52}  {s['error']}")
            continue

        print(
            f"{slug:<52} {s['history_days']:>5} {s['entry_date']:<12} "
            f"{fmt_pct(s['entry_share'])} {fmt_pct(s['peak_share'])} "
            f"{fmt_pct(s['current_share'])} {fmt_growth(s['growth_mult'])} "
            f"{s['days_in_market']:>5}"
        )

    print()
    for s in summaries:
        if "error" in s:
            continue
        print(f"\n{s['model']} — last 21 days of DekaLLM share")
        for date, share, _, total in s["series"][-21:]:
            bar = "█" * int(share / 2)
            tot_b = total / 1e9
            print(f"  {date}  {share:>5.1f}%  total={tot_b:5.1f}B  {bar}")


if __name__ == "__main__":
    main()
