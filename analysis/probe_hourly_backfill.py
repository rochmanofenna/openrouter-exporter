"""Probe the DekaLLM operator dashboard for hourly drill-down endpoints.

We have ~9 days of hourly data from the Prometheus scraper but ~28 days of daily
data from the CSV exports. If the operator dashboard exposes hourly granularity
for past days, we can backfill April and save 3 weeks of waiting.

This script:
1. Fetches the dashboard HTML with the session cookie
2. Greps the body for hints of hourly endpoints (any URL or symbol containing
   "hour", "granularity", "drill", "interval", "csv", etc.)
3. Tries a list of plausible URL variations and reports status + body shape
4. Reports whether anything obviously hourly is exposed

Usage:
    python analysis/probe_hourly_backfill.py

Env:
    OPENROUTER_ACTIVITY_SESSION_COOKIE  required
    PROVIDER                            default dekallm
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import requests

PROVIDER = os.environ.get("PROVIDER", "dekallm")
BASE = "https://openrouter.ai"
COOKIE = os.environ.get("OPENROUTER_ACTIVITY_SESSION_COOKIE", "")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def fetch(url: str, rsc: bool = False) -> tuple[int, str, dict]:
    """Return (status, body, headers)."""
    headers = {
        "User-Agent": "openrouter-exporter/1.0",
        "Cookie": f"__session={COOKIE}",
    }
    if rsc:
        headers["RSC"] = "1"
    r = requests.get(url, headers=headers, timeout=30)
    return r.status_code, r.text, dict(r.headers)


def grep_terms(body: str, terms: list[str]) -> dict[str, list[str]]:
    """For each term, return up to 5 surrounding-context snippets."""
    out = {}
    for t in terms:
        rx = re.compile(rf".{{,60}}{re.escape(t)}.{{,60}}", re.IGNORECASE)
        matches = rx.findall(body)
        if matches:
            out[t] = matches[:5]
    return out


def find_url_patterns(body: str) -> list[str]:
    """Find URL-ish strings (paths starting with / or full https URLs)."""
    rx = re.compile(r'(?:https?:\/\/[^\s"\'<>]+|\/[a-zA-Z0-9_\-\/\?\&\=\.]{6,})')
    found = set(rx.findall(body))
    # Filter to interesting candidates
    keep = []
    for u in found:
        l = u.lower()
        if any(k in l for k in ["hour", "drill", "granul", "interval", "usage", "stats", "activity", "report", "csv", "download", "api/v1/provider"]):
            keep.append(u)
    return sorted(set(keep))[:50]


def main() -> None:
    if not COOKIE:
        print("ERROR: OPENROUTER_ACTIVITY_SESSION_COOKIE not set.")
        print("Run with: source .env && python analysis/probe_hourly_backfill.py")
        return

    print("=" * 70)
    print("Probe 1: dashboard HTML at /provider/<slug>/dashboard")
    print("=" * 70)
    url = f"{BASE}/provider/{PROVIDER}/dashboard"
    status, body, _ = fetch(url)
    print(f"  GET {url}  -> {status}, body {len(body)} bytes")
    if status == 200:
        terms = ["hour", "hourly", "by-hour", "by_hour", "granularity", "interval",
                 "drilldown", "drill-down", "csv", "download", "timeseries", "time_series"]
        hits = grep_terms(body, terms)
        if hits:
            print(f"\n  Term hits in dashboard HTML:")
            for t, samples in hits.items():
                print(f"\n    '{t}' ({len(samples)} samples):")
                for s in samples[:3]:
                    print(f"       ...{s}...")
        else:
            print("  No hourly-suggesting terms found in dashboard HTML.")

        urls = find_url_patterns(body)
        if urls:
            print(f"\n  Candidate URL patterns ({len(urls)}):")
            for u in urls:
                print(f"    {u}")

    print("\n" + "=" * 70)
    print("Probe 2: dashboard RSC fetch (Next.js Server Component payload)")
    print("=" * 70)
    status, body, _ = fetch(url, rsc=True)
    print(f"  GET {url} [RSC]  -> {status}, body {len(body)} bytes")
    if status == 200:
        terms = ["hour", "hourly", "granularity", "interval", "drilldown", "timeseries"]
        hits = grep_terms(body, terms)
        if hits:
            print(f"\n  Term hits in RSC payload:")
            for t, samples in hits.items():
                print(f"\n    '{t}' ({len(samples)} samples):")
                for s in samples[:3]:
                    print(f"       ...{s}...")
        urls = find_url_patterns(body)
        if urls:
            print(f"\n  Candidate URL patterns in RSC ({len(urls)}):")
            for u in urls[:30]:
                print(f"    {u}")

    print("\n" + "=" * 70)
    print("Probe 3: try plausible hourly endpoint variations")
    print("=" * 70)
    candidates = [
        f"/provider/{PROVIDER}/dashboard?range=hourly",
        f"/provider/{PROVIDER}/dashboard?granularity=hour",
        f"/provider/{PROVIDER}/dashboard?interval=1h",
        f"/provider/{PROVIDER}/usage",
        f"/provider/{PROVIDER}/usage?range=hourly",
        f"/provider/{PROVIDER}/usage?period=hour",
        f"/provider/{PROVIDER}/stats",
        f"/provider/{PROVIDER}/stats?hourly=true",
        f"/provider/{PROVIDER}/activity",
        f"/provider/{PROVIDER}/activity?hourly=true",
        f"/provider/{PROVIDER}/report",
        f"/provider/{PROVIDER}/report?period=hourly",
        f"/api/v1/provider/{PROVIDER}",
        f"/api/v1/provider/{PROVIDER}/usage",
        f"/api/v1/provider/{PROVIDER}/usage?period=hour",
        f"/api/v1/provider/{PROVIDER}/stats",
        f"/api/v1/provider/{PROVIDER}/hourly",
        f"/api/v1/providers/{PROVIDER}/usage",
    ]
    for path in candidates:
        u = f"{BASE}{path}"
        try:
            status, body, headers = fetch(u)
            ct = headers.get("Content-Type", "").split(";")[0]
            size = len(body)
            preview = body[:120].replace("\n", "\\n").replace("\r", "")
            has_hour = "hour" in body.lower()[:5000]
            has_x = '"x":"' in body[:5000]
            has_date = '"date":"' in body[:5000]
            note = []
            if has_hour: note.append("'hour' found")
            if has_x: note.append("'x:' found (chart data?)")
            if has_date: note.append("'date:' found")
            note_str = "  | " + ", ".join(note) if note else ""
            print(f"  {status}  {ct:30s}  {size:>7} B  {path}{note_str}")
            if status == 200 and ("hour" in body.lower()[:50000] and size > 1000):
                # Save promising responses for inspection
                safe = path.replace("/", "_").replace("?", "_").replace("=", "_").replace("&", "_")
                outpath = OUT / f"probe_response{safe}.txt"
                with open(outpath, "w") as f:
                    f.write(body[:200_000])
                print(f"     -> saved to {outpath.name}")
        except Exception as e:
            print(f"  ERR   ...                            {path}  ({e})")

    print("\n" + "=" * 70)
    print("Probe 4: model-level activity page (we already use this; check schema)")
    print("=" * 70)
    candidates = [
        f"/openai/gpt-oss-120b/activity",
        f"/openai/gpt-oss-120b/activity?range=hourly",
        f"/openai/gpt-oss-120b/stats",
        f"/openai/gpt-oss-120b/usage",
    ]
    for path in candidates:
        u = f"{BASE}{path}"
        try:
            for use_rsc in [False, True]:
                status, body, headers = fetch(u, rsc=use_rsc)
                ct = headers.get("Content-Type", "").split(";")[0]
                has_hour = "hour" in body.lower()[:30000]
                has_x = '"x":"' in body[:30000]
                tag = "RSC " if use_rsc else "HTML"
                note = []
                if has_hour: note.append("'hour' found")
                if has_x: note.append("'x:' found")
                note_str = "  | " + ", ".join(note) if note else ""
                print(f"  [{tag}] {status}  {ct:30s}  {len(body):>7} B  {path}{note_str}")
        except Exception as e:
            print(f"  ERR ... {path} ({e})")

    print("\n" + "=" * 70)
    print("Recommendation")
    print("=" * 70)
    print("""
Look at the output above:

* If any endpoint returned a body containing 'hour' and an 'x:' / 'date:'
  array structure → that's a candidate for hourly backfill. Inspect the
  saved response file in analysis/out/.

* If everything returned the same daily-aggregated chart (looks like x = date
  strings 'YYYY-MM-DD 00:00:00'), no hourly drill-down is exposed at the URL
  level. The dashboard might have a UI drill-down that calls a different
  endpoint client-side — open Firefox DevTools Network tab, click around the
  dashboard, and look for any XHR/fetch calls with hourly data.

* If nothing showed up at all, we don't have a backfill path. Wait for the
  scraper to accumulate (timeline: peak-hour usable ~June 5).
""")


if __name__ == "__main__":
    main()
