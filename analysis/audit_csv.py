"""Audit the DekaLLM daily CSV exports — sanity-check coverage and consistency.

For each `dekallm_daily_report_2026_*.csv` in the repo root, prints:
- Total rows
- Per-date row count, total tokens, total cost, and which models appeared
- Per-model row count across all dates

Then cross-checks any pair of CSVs for the same (date, model) and flags
discrepancies — useful when you have both a stale and a fresh CSV on disk.

Usage:
    python analysis/audit_csv.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith('"20'):
                continue
            parts = list(csv.reader([line]))[0]
            if len(parts) < 10:
                continue
            try:
                rows.append({
                    "date": parts[0],
                    "model": parts[1],
                    "requests": int(parts[2]),
                    "input_tokens": int(parts[3]),
                    "output_tokens": int(parts[4]),
                    "total_tokens": int(parts[6]),
                    "total_cost": float(parts[9]),
                })
            except (ValueError, IndexError):
                # Skip TOTAL summary row or malformed lines
                continue
    return rows


def short(model: str) -> str:
    """Compact model name for printing."""
    s = model.split("/")[-1]
    return s[:24]


def audit_one(path: Path) -> dict[tuple[str, str], dict]:
    rows = load_csv(path)
    print(f"\n=== {path.name} ({path.stat().st_size} bytes, {len(rows)} data rows) ===")

    by_date: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, int] = defaultdict(int)
    by_key: dict[tuple[str, str], dict] = {}

    for r in rows:
        by_date[r["date"]].append(r)
        by_model[r["model"]] += 1
        by_key[(r["date"], r["model"])] = r

    # Detect duplicate (date, model) within one CSV
    seen = set()
    duplicates = []
    for r in rows:
        k = (r["date"], r["model"])
        if k in seen:
            duplicates.append(k)
        seen.add(k)
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate (date, model) pairs within this CSV")
        for k in duplicates[:5]:
            print(f"    {k}")

    print(f"\n  rows per date:")
    for d in sorted(by_date):
        day_rows = by_date[d]
        tokens = sum(x["total_tokens"] for x in day_rows)
        cost = sum(x["total_cost"] for x in day_rows)
        models = sorted(short(x["model"]) for x in day_rows)
        print(f"    {d}: {len(day_rows)} models, {tokens/1e9:6.2f}B tokens, ${cost:8.2f}")
        print(f"            {models}")

    print(f"\n  rows per model (across all dates in this CSV):")
    for m in sorted(by_model):
        print(f"    {by_model[m]:3d}  {m}")

    return by_key


def cross_check(audits: dict[str, dict[tuple[str, str], dict]]) -> None:
    """Compare overlapping (date, model) rows across CSVs."""
    if len(audits) < 2:
        return

    names = sorted(audits.keys())
    print(f"\n=== Cross-CSV consistency check ===")
    print(f"Comparing: {names}")

    all_keys: set[tuple[str, str]] = set()
    for keys in audits.values():
        all_keys.update(keys.keys())

    mismatches = 0
    for key in sorted(all_keys):
        values_per_csv = {}
        for name, keys in audits.items():
            if key in keys:
                values_per_csv[name] = keys[key]
        if len(values_per_csv) < 2:
            continue  # only in one CSV, nothing to compare

        # Compare total_tokens and total_cost
        token_values = set(round(v["total_tokens"], 0) for v in values_per_csv.values())
        cost_values = set(round(v["total_cost"], 2) for v in values_per_csv.values())
        if len(token_values) > 1 or len(cost_values) > 1:
            mismatches += 1
            d, m = key
            print(f"\n  MISMATCH for {d} / {short(m)}:")
            for name, v in values_per_csv.items():
                print(f"    {name}: tokens={v['total_tokens']:,}  cost=${v['total_cost']:.2f}")

    if mismatches == 0:
        overlap = sum(
            1 for k in all_keys
            if sum(1 for keys in audits.values() if k in keys) >= 2
        )
        print(f"  OK — {overlap} overlapping (date, model) pairs, all values match")
    else:
        print(f"\n  {mismatches} mismatch(es) found. The newer CSV is usually authoritative.")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    paths = sorted(root.glob("dekallm_daily_report_2026_*.csv"))
    if not paths:
        print("No dekallm_daily_report_2026_*.csv files found.")
        return

    print(f"Auditing {len(paths)} CSV file(s):")
    for p in paths:
        print(f"  {p.name}")

    audits = {p.name: audit_one(p) for p in paths}
    cross_check(audits)


if __name__ == "__main__":
    main()
