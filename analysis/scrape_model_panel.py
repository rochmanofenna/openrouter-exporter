"""Scrape per-(model, provider, date) token volumes directly from OpenRouter.

Built because:
  - The Go exporter only scrapes provider-chart endpoints (one HTTP request
    per configured provider). We only configured 4 providers, so we only
    have token data for those 4. For most DekaLLM models, fewer than 3 of
    those 4 actually serve the model, so the panel is too thin to fit on.
  - The local analytics API has per-model provider-tokens data with 18+
    providers per model, but is currently unreachable.
  - We need cross-model + cross-provider variance to identify the share
    formula's coefficients, and the Path 2b decision was to scrape it
    ourselves so we're not blocked when the API is down.

Mechanism:
  - Hits the OpenRouter model page using the same RSC-flavored request the
    Go exporter uses for provider chart pages (see client/activity.go).
    The page contains an embedded `"data":[{"x":"...","ys":{...}}, ...]`
    chart array whose `ys` maps PROVIDER → daily-running-total. This is
    the inverse of the provider chart (where `ys` maps MODEL → tokens).
  - Tries a few candidate URLs per model since we don't know the exact
    path; logs which one worked so we can lock it in later.

Outputs (per the mentor's defensive design):
  - analysis/raw/{date}/{model_slug_safe}.json
      Raw response, one file per (date, model). If OpenRouter changes its
      JSON schema we can re-parse history rather than lose it.
  - analysis/out/model_panel_history.csv
      Parsed long-form table: (date, model_id, provider, tokens, scraped_at).
      Idempotent: replaces (date, model, provider) rows on each run rather
      than appending duplicates.

Auth:
  OPENROUTER_SESSION_COOKIE — same env var the Go exporter uses. Required.
  Get it from a logged-in OpenRouter session (DevTools → Application →
  Cookies → __session) or from whatever secret store the VM is using.

Usage:
  # Smoke-test against one model (writes nothing, prints anchor diagnostics):
  python analysis/scrape_model_panel.py --smoke mistralai/mistral-nemo

  # Full scrape (DekaLLM portfolio + a popularity sample):
  python analysis/scrape_model_panel.py

  # Custom model list:
  MODELS_FILE=analysis/target_models.txt python analysis/scrape_model_panel.py

Suggested cron (daily, after 23:55 UTC for finalized day-totals):
  55 23 * * *  cd /path/to/repo && python analysis/scrape_model_panel.py >> /var/log/scrape_model_panel.log 2>&1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

HISTORY_CSV = OUT / "model_panel_history.csv"

OPENROUTER_BASE = "https://openrouter.ai"
USER_AGENT = "openrouter-exporter/1.0 (python panel scraper)"
SESSION_COOKIE = os.environ.get("OPENROUTER_SESSION_COOKIE", "")
RSC_HEADER = "1"
REQUEST_TIMEOUT = 30
INTER_REQUEST_SLEEP = float(os.environ.get("INTER_REQUEST_SLEEP", "1.0"))

# Anchor mirrors client/activity.go providerChartAnchor — the page contains
# several `"data":[` arrays but only the per-day chart starts with `{"x":`.
CHART_ANCHOR = '"data":[{"x":"'

# Try these URL templates in order until one returns a parseable chart.
# Locked-in pattern (whichever works first) gets logged so future runs
# can skip the others. The Go exporter's provider chart lives at
# /provider/{slug}; the inverse for per-model provider tokens is one of:
CANDIDATE_URL_TEMPLATES = [
    "{base}/{slug}",
    "{base}/{slug}/providers",
    "{base}/{slug}/analytics",
]

# DekaLLM's current portfolio (we know these from the share_growth.py output).
DEKALLM_MODELS = [
    "mistralai/mistral-nemo",
    "openai/gpt-oss-120b",
    "nvidia/nemotron-3-super-120b-a12b",
    "google/gemma-4-26b-a4b-it",
    "qwen/qwen3.5-35b-a3b",
]

# Popularity sample for cross-model variance (so the regression isn't
# conditioned on DekaLLM's presence). Plain guesses based on what's
# typically high-volume on OpenRouter; tune via MODELS_FILE.
POPULAR_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-1.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-large",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.1-405b-instruct",
]


def safe_slug(slug: str) -> str:
    """File-safe form of a model slug (slashes → __)."""
    return slug.replace("/", "__")


def load_model_list() -> list[str]:
    """Return the target model list, from MODELS_FILE if set, else the default."""
    path_env = os.environ.get("MODELS_FILE")
    if path_env:
        p = Path(path_env)
        if not p.exists():
            sys.exit(f"MODELS_FILE not found: {p}")
        return [line.strip() for line in p.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    return DEKALLM_MODELS + POPULAR_MODELS


def fetch_url(url: str) -> tuple[int, str]:
    """GET with RSC + session cookie. Returns (status_code, body) or raises."""
    if not SESSION_COOKIE:
        sys.exit(
            "OPENROUTER_SESSION_COOKIE is required. Set it before running.\n"
            "Same value the Go exporter uses — grab from /etc/default/openrouter-exporter or wherever you keep it."
        )
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "RSC": RSC_HEADER,
        "Cookie": "__session=" + SESSION_COOKIE,
        "Accept": "*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""


def find_matching_bracket(s: str, start: int) -> int:
    """Mirror of client/activity.go findMatchingBracket — string-aware."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        if escaped:
            escaped = False
            continue
        if in_string:
            if s[i] == "\\":
                escaped = True
            elif s[i] == '"':
                in_string = False
            continue
        c = s[i]
        if c == '"':
            in_string = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_chart(body: str) -> list[dict] | None:
    """Extract the [{x, ys}] chart array from the RSC response body. Returns
    None if the anchor isn't present (this URL doesn't expose the chart)."""
    idx = body.find(CHART_ANCHOR)
    if idx == -1:
        return None
    array_start = idx + len(CHART_ANCHOR) - len('{"x":"') - 1
    array_end = find_matching_bracket(body, array_start)
    if array_end == -1:
        return None
    try:
        return json.loads(body[array_start:array_end + 1])
    except json.JSONDecodeError:
        return None


def chart_to_rows(model_id: str, chart: list[dict], scraped_at: str) -> list[dict]:
    """Flatten one chart into long-form (date, provider, tokens) rows."""
    rows = []
    for point in chart:
        x = point.get("x", "")
        # x is "YYYY-MM-DD HH:MM:SS" UTC; the date prefix is what we want.
        date = x.split(" ", 1)[0] if x else ""
        ys = point.get("ys") or {}
        if not date or not isinstance(ys, dict):
            continue
        for provider, tokens in ys.items():
            if not isinstance(tokens, (int, float)):
                continue
            rows.append({
                "date": date,
                "model_id": model_id,
                "provider": str(provider),
                "tokens": int(tokens),
                "scraped_at": scraped_at,
            })
    return rows


def scrape_model(model_id: str, scraped_at: str, run_date: str,
                 url_lock: dict[str, str]) -> tuple[list[dict], str | None]:
    """Try candidate URLs; return (rows, successful_url) or ([], None) on failure.

    Persists the raw body of the first successful response to RAW_DIR/{date}/{slug}.json.
    """
    slug = safe_slug(model_id)
    # If a URL template already worked this run, prefer it (skips the
    # 404-fishing for subsequent models).
    templates = ([url_lock["template"]] if url_lock.get("template")
                 else CANDIDATE_URL_TEMPLATES)

    for template in templates:
        url = template.format(base=OPENROUTER_BASE, slug=model_id)
        status, body = fetch_url(url)
        if status == 404:
            continue
        if status != 200:
            print(f"    [{model_id}] {url} → HTTP {status}")
            continue
        chart = parse_chart(body)
        if chart is None:
            print(f"    [{model_id}] {url} → no chart anchor in response")
            continue

        # Persist raw response
        day_dir = RAW_DIR / run_date
        day_dir.mkdir(parents=True, exist_ok=True)
        raw_path = day_dir / f"{slug}.json"
        # Save the chart slice (small), not the whole RSC body (potentially huge)
        with open(raw_path, "w") as f:
            json.dump({
                "scraped_at": scraped_at,
                "model_id": model_id,
                "url": url,
                "chart": chart,
            }, f, indent=2)

        url_lock["template"] = template  # lock the working template
        return chart_to_rows(model_id, chart, scraped_at), url

    return [], None


# ---------------------------------------------------------------------------
# Idempotent CSV merge
# ---------------------------------------------------------------------------
HISTORY_FIELDS = ["date", "model_id", "provider", "tokens", "scraped_at"]


def load_existing_history() -> dict[tuple[str, str, str], dict]:
    """Read the history CSV into a dict keyed by (date, model_id, provider)."""
    if not HISTORY_CSV.exists():
        return {}
    out: dict[tuple[str, str, str], dict] = {}
    with open(HISTORY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["date"], row["model_id"], row["provider"])
            out[key] = row
    return out


def write_history(rows_by_key: dict[tuple[str, str, str], dict]) -> None:
    """Replace HISTORY_CSV with the merged set, sorted for stable diffs."""
    sorted_rows = sorted(
        rows_by_key.values(),
        key=lambda r: (r["date"], r["model_id"], r["provider"])
    )
    tmp = HISTORY_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_rows)
    tmp.replace(HISTORY_CSV)


def merge_rows(existing: dict, new_rows: list[dict]) -> tuple[int, int]:
    """Merge new_rows into existing (in-place). Returns (added, replaced)."""
    added = replaced = 0
    for r in new_rows:
        key = (r["date"], r["model_id"], r["provider"])
        if key in existing:
            replaced += 1
        else:
            added += 1
        existing[key] = r
    return added, replaced


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test(model_id: str) -> None:
    """Hit each candidate URL once and report what each returned. Writes nothing."""
    if not SESSION_COOKIE:
        sys.exit("OPENROUTER_SESSION_COOKIE required for smoke test")
    print(f"Smoke test for {model_id}\n")
    for template in CANDIDATE_URL_TEMPLATES:
        url = template.format(base=OPENROUTER_BASE, slug=model_id)
        print(f"--- {url}")
        status, body = fetch_url(url)
        print(f"  HTTP {status}, body len={len(body):,}")
        if status != 200:
            continue
        chart = parse_chart(body)
        if chart is None:
            anchor_idx = body.find(CHART_ANCHOR)
            print(f"  chart anchor `{CHART_ANCHOR}` found at: {anchor_idx}")
            # Show what arrays the page DOES expose
            print(f"  body excerpt around offset 0..500: {body[:500]!r}")
            continue
        ys_keys = set()
        for p in chart:
            if isinstance(p, dict) and isinstance(p.get("ys"), dict):
                ys_keys.update(p["ys"].keys())
        print(f"  chart points: {len(chart)}")
        print(f"  unique ys keys (these become 'provider'): {sorted(ys_keys)}")
        if chart:
            sample = chart[-1]  # most recent point
            print(f"  most-recent point: x={sample.get('x')!r}, ys keys={list((sample.get('ys') or {}).keys())[:6]}...")
        print("  ^ if ys keys look like provider names, this URL is correct.")
        return  # stop on first success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", metavar="MODEL_ID",
                    help="Probe URLs for one model and exit (writes nothing)")
    args = ap.parse_args()

    if args.smoke:
        smoke_test(args.smoke)
        return

    models = load_model_list()
    print(f"Target model list: {len(models)} models")
    for m in models:
        print(f"  {m}")

    scraped_at = datetime.now(timezone.utc).isoformat()
    run_date = datetime.now(timezone.utc).date().isoformat()

    existing = load_existing_history()
    print(f"\nExisting history: {len(existing):,} rows")

    url_lock: dict[str, str] = {}
    total_new = total_replaced = 0
    failures = []

    for m in models:
        print(f"\n[{m}]")
        new_rows, url = scrape_model(m, scraped_at, run_date, url_lock)
        if not new_rows:
            print(f"  FAILED — no chart from any candidate URL")
            failures.append(m)
            time.sleep(INTER_REQUEST_SLEEP)
            continue
        added, replaced = merge_rows(existing, new_rows)
        total_new += added
        total_replaced += replaced
        latest_date = max(r["date"] for r in new_rows)
        latest_providers = sorted({r["provider"] for r in new_rows if r["date"] == latest_date})
        print(f"  {url}")
        print(f"  +{added:,} new rows, {replaced:,} updated. "
              f"Latest day {latest_date}: {len(latest_providers)} providers")
        time.sleep(INTER_REQUEST_SLEEP)

    write_history(existing)

    print(f"\n{'='*78}")
    print(f"Done. Saved {len(existing):,} total rows to {HISTORY_CSV}")
    print(f"  This run: +{total_new:,} new, {total_replaced:,} updated, "
          f"{len(failures)} failures")
    if failures:
        print(f"  Failed models (likely don't exist or wrong slug):")
        for m in failures:
            print(f"    {m}")
    if url_lock.get("template"):
        print(f"\nWorking URL template: {url_lock['template']}")
        print("  Reorder CANDIDATE_URL_TEMPLATES so this is first; saves probing.")

    # Sanity check: row count vs prior runs (mentor's defensive ask)
    today_rows = sum(1 for r in existing.values() if r["date"] == run_date)
    yest = (datetime.now(timezone.utc).date().toordinal() - 1)
    yest_date = datetime.fromordinal(yest).date().isoformat()
    yest_rows = sum(1 for r in existing.values() if r["date"] == yest_date)
    print(f"\nSanity check: today={today_rows} rows, yesterday={yest_rows} rows")
    if yest_rows > 0 and today_rows < 0.5 * yest_rows:
        print("  WARNING: today's row count is <50% of yesterday's. "
              "Possible silent breakage; investigate before trusting fits.")


if __name__ == "__main__":
    main()
