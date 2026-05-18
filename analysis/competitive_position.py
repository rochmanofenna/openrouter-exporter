"""Competitive position analysis — where DekaLLM leads/lags on the levers that
drive OpenRouter routing share.

For each model DekaLLM serves, this script pulls the current values of every
routing-relevant lever from Prometheus:

- Input price ($/M tokens)
- Output price ($/M tokens)
- Throughput (p50 tokens/sec)
- Latency (p50 ms)
- Uptime (last 1d %)

It ranks DekaLLM against the other providers we scrape (deepinfra, fireworks,
together), shows the gap to leader on each lever, and identifies the
highest-leverage move per model.

OpenRouter routing modes that prioritize each lever:
- :floor mode + default mode  -> price (lower wins)
- :nitro mode                 -> throughput (higher wins)
- default mode                -> latency (lower wins) + uptime (higher wins)
- user-specified              -> quantization, context length

So if DekaLLM is the 3rd-cheapest on output price but the leader on throughput,
they'll naturally win nitro-mode users but lose floor-mode users. Knowing which
levers they trail on tells the GPU and pricing teams exactly what to fix to
gain share.

Outputs:
- analysis/out/competitive_position.md   (meeting-ready report)
- analysis/out/competitive_position.csv  (per-model lever table)

Usage:
    python analysis/competitive_position.py

Env:
    PROM_URL     default http://localhost:9090
    PROVIDERS    comma-separated; default dekallm,deepinfra,fireworks,together
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
PROVIDERS_SLUGS = [s.strip() for s in os.environ.get(
    "PROVIDERS", "dekallm,deepinfra,fireworks,together"
).split(",")]
DEKALLM_SLUG = "dekallm"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Endpoint metrics use the display name (e.g. "DekaLLM"), the provider chart
# metric uses the slug (e.g. "dekallm"). Map between them so we can join.
PROVIDER_NAME_HINTS = {
    "dekallm":    ["DekaLLM", "Dekallm", "dekallm"],
    "deepinfra":  ["DeepInfra", "Deepinfra", "deepinfra"],
    "fireworks":  ["Fireworks", "Fireworks AI", "fireworks"],
    "together":   ["Together", "Together AI", "together"],
}


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


# ----------------------------------------------------------------------------
# Resolve provider_name (endpoint metrics) <-> provider slug (chart metric)
# ----------------------------------------------------------------------------
def resolve_provider_names() -> dict[str, str]:
    """Map each tracked slug to the display name actually used in endpoint metrics."""
    res = prom_query("group by (provider_name) (openrouter_model_input_price_dollars_per_million_tokens)")
    discovered = [r["metric"]["provider_name"] for r in res]
    mapping = {}
    for slug in PROVIDERS_SLUGS:
        hints = PROVIDER_NAME_HINTS.get(slug, [slug])
        # Try exact hint match first, then case-insensitive, then substring
        found = None
        for hint in hints:
            for d in discovered:
                if d == hint:
                    found = d
                    break
            if found:
                break
        if not found:
            for hint in hints:
                for d in discovered:
                    if d.lower() == hint.lower():
                        found = d
                        break
                if found:
                    break
        if not found:
            for hint in hints:
                for d in discovered:
                    if hint.lower() in d.lower() or d.lower() in hint.lower():
                        found = d
                        break
                if found:
                    break
        mapping[slug] = found  # may be None if not found
    return mapping


# ----------------------------------------------------------------------------
# Identify DekaLLM's models
# ----------------------------------------------------------------------------
def get_dekallm_models() -> list[str]:
    res = prom_query(
        f'count by (model_id) (openrouter_provider_tokens_daily{{provider="{DEKALLM_SLUG}"}})'
    )
    models = sorted({r["metric"]["model_id"] for r in res})
    return models


# ----------------------------------------------------------------------------
# Pull lever values per (provider, model_id)
# ----------------------------------------------------------------------------
def get_levers_for_model(model_id: str, name_map: dict[str, str]) -> pd.DataFrame:
    """Return a DataFrame indexed by provider_slug with columns for each lever."""
    queries = {
        "input_price":  ("min", "openrouter_model_input_price_dollars_per_million_tokens", ""),
        "output_price": ("min", "openrouter_model_output_price_dollars_per_million_tokens", ""),
        "throughput":   ("max", "openrouter_endpoint_throughput_tokens_per_second", ',quantile="p50"'),
        "latency_ms":   ("min", "openrouter_endpoint_latency_milliseconds", ',quantile="p50"'),
        "uptime_1d":    ("max", "openrouter_endpoint_uptime_percentage_last_1d", ""),
    }

    # Build reverse map name->slug
    name_to_slug = {v: k for k, v in name_map.items() if v}

    metrics: dict[str, dict[str, float]] = {}
    for col, (agg, base, qfilter) in queries.items():
        q = f'{agg} by (provider_name) ({base}{{model_id="{model_id}"{qfilter}}})'
        try:
            res = prom_query(q)
        except Exception:
            res = []
        for r in res:
            pn = r["metric"].get("provider_name", "")
            slug = name_to_slug.get(pn)
            if slug is None:
                continue
            v = float(r["value"][1])
            metrics.setdefault(slug, {})[col] = v

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.index.name = "provider"
    # Ensure all expected columns exist even if no data
    for col in queries:
        if col not in df.columns:
            df[col] = float("nan")
    return df[list(queries.keys())]


# ----------------------------------------------------------------------------
# Pull recent market share
# ----------------------------------------------------------------------------
def get_market_share(model_id: str, days: int = 7) -> dict[str, float]:
    """Return per-slug average share over the last `days` days."""
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    q = f'openrouter_provider_tokens_daily{{model_id="{model_id}"}}'
    series = prom_range(q, start, end, 3600)

    per_provider_date: dict[tuple[str, str], float] = {}
    for s in series:
        provider = s["metric"].get("provider", "")
        date = s["metric"].get("date", "")
        if not date or not provider:
            continue
        max_val = max(float(v[1]) for v in s["values"])
        per_provider_date[(provider, date)] = max(max_val, per_provider_date.get((provider, date), 0))

    # Sum per (provider, date) is already in per_provider_date
    # Compute per-day total then per-day share, then average shares
    dates = sorted({d for _, d in per_provider_date})
    if not dates:
        return {p: 0.0 for p in PROVIDERS_SLUGS}

    per_day_total: dict[str, float] = {}
    for (p, d), v in per_provider_date.items():
        per_day_total[d] = per_day_total.get(d, 0) + v

    shares: dict[str, list[float]] = {}
    for (p, d), v in per_provider_date.items():
        total = per_day_total.get(d, 0)
        if total > 0:
            shares.setdefault(p, []).append(v / total)

    return {p: (sum(vs) / len(vs)) if vs else 0.0 for p, vs in shares.items()}


# ----------------------------------------------------------------------------
# Ranking + leverage analysis per model
# ----------------------------------------------------------------------------
HIGHER_IS_BETTER = {"throughput": True, "uptime_1d": True,
                    "input_price": False, "output_price": False, "latency_ms": False}


def rank_and_gap(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add `<col>_rank` and `<col>_gap_pct_vs_leader` columns to df."""
    if col not in df.columns or df[col].isna().all():
        df[f"{col}_rank"] = float("nan")
        df[f"{col}_gap_pct"] = float("nan")
        return df

    higher_better = HIGHER_IS_BETTER[col]
    series = df[col].dropna()
    if len(series) == 0:
        df[f"{col}_rank"] = float("nan")
        df[f"{col}_gap_pct"] = float("nan")
        return df

    leader_value = series.max() if higher_better else series.min()
    ranks = series.rank(ascending=not higher_better, method="min").astype(int)
    if higher_better:
        # gap_pct: how much smaller than leader, negative = behind (e.g. -25% slower)
        gaps = (series - leader_value) / leader_value * 100
    else:
        # gap_pct: how much larger than leader (positive = more expensive/slower than leader)
        gaps = (series - leader_value) / leader_value * 100

    df[f"{col}_rank"] = ranks.reindex(df.index)
    df[f"{col}_gap_pct"] = gaps.reindex(df.index)
    return df


LEVER_LABELS = {
    "input_price":  ("Input price ($/M)",   "lower-better", "cut input price"),
    "output_price": ("Output price ($/M)",  "lower-better", "cut output price"),
    "throughput":   ("Throughput (t/s p50)", "higher-better", "improve throughput"),
    "latency_ms":   ("Latency (ms p50)",    "lower-better", "reduce latency"),
    "uptime_1d":    ("Uptime (% last 1d)",  "higher-better", "improve uptime"),
}


def fmt_value(col: str, v: float) -> str:
    if pd.isna(v):
        return "—"
    if col in ("input_price", "output_price"):
        return f"${v:.3f}"
    if col == "throughput":
        return f"{v:,.0f} t/s"
    if col == "latency_ms":
        return f"{v:,.0f} ms"
    if col == "uptime_1d":
        return f"{v:.2f}%"
    return f"{v:.3f}"


def per_model_report(model_id: str, df: pd.DataFrame, share: dict[str, float]) -> str:
    """Return a markdown chunk for one model."""
    out = []
    out.append(f"## {model_id}\n")

    # Show DekaLLM's share + total
    dek_share = share.get(DEKALLM_SLUG, 0.0) * 100
    out.append(f"**DekaLLM market share (7-day avg)**: {dek_share:.2f}%\n")
    out.append(f"**Providers serving this model**: {len(df)}\n")

    # Rank each lever
    for col in LEVER_LABELS:
        df = rank_and_gap(df, col)

    # Build a leaderboard table
    out.append("\n### Leaderboard\n")
    out.append("| Provider | Input $/M | Output $/M | Throughput | Latency | Uptime | 7d share |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for slug in df.index:
        row = df.loc[slug]
        share_pct = share.get(slug, 0.0) * 100
        out.append(
            f"| **{slug}** | {fmt_value('input_price', row.get('input_price'))} | "
            f"{fmt_value('output_price', row.get('output_price'))} | "
            f"{fmt_value('throughput', row.get('throughput'))} | "
            f"{fmt_value('latency_ms', row.get('latency_ms'))} | "
            f"{fmt_value('uptime_1d', row.get('uptime_1d'))} | "
            f"{share_pct:.2f}% |"
        )

    # DekaLLM's per-lever gap
    if DEKALLM_SLUG not in df.index:
        out.append(f"\n_No lever data for DekaLLM on this model._\n")
        return "\n".join(out)

    dek = df.loc[DEKALLM_SLUG]
    out.append("\n### DekaLLM lever position\n")
    out.append("| Lever | DekaLLM value | Rank | Gap vs leader | Action if pursued |")
    out.append("|---|---:|---:|---:|---|")
    actions = []
    for col, (label, _dir, action_phrase) in LEVER_LABELS.items():
        v = dek.get(col)
        rank = dek.get(f"{col}_rank")
        gap = dek.get(f"{col}_gap_pct")
        if pd.isna(v) or pd.isna(rank):
            out.append(f"| {label} | — | — | — | (no data) |")
            continue
        n_providers = df[col].notna().sum()
        rank_str = f"{int(rank)} / {int(n_providers)}"
        gap_str = f"{gap:+.1f}%" if not pd.isna(gap) else "—"
        # Action only meaningful if not leader
        action = action_phrase if int(rank) > 1 else "(already leader)"
        out.append(f"| {label} | {fmt_value(col, v)} | {rank_str} | {gap_str} | {action} |")
        if int(rank) > 1 and not pd.isna(gap):
            # Score: weighted by |gap|. Larger absolute gap = bigger improvement potential.
            actions.append((col, abs(float(gap)), action_phrase, int(rank), float(gap)))

    # Highest-leverage move
    if actions:
        actions.sort(key=lambda x: -x[1])
        col, abs_gap, action, rank, gap = actions[0]
        out.append(
            f"\n**Highest-leverage move**: **{action}** "
            f"(currently rank {rank}, {gap:+.1f}% vs leader). "
            f"Closing this gap likely produces the largest share gain per unit effort, "
            f"though absolute leverage depends on whether OpenRouter users for this model "
            f"sort by this lever (price-sensitive vs speed-sensitive)."
        )
    else:
        out.append("\n**DekaLLM is the leader on every measured lever** for this model.")

    return "\n".join(out)


# ----------------------------------------------------------------------------
# Aggregated summary
# ----------------------------------------------------------------------------
def overall_summary(model_data: dict[str, dict]) -> str:
    """Aggregate across models — where DekaLLM consistently leads/lags."""
    out = ["## Aggregated summary across all DekaLLM models\n"]
    rows = []
    for model_id, payload in model_data.items():
        dek_row = payload["levers"].loc[DEKALLM_SLUG] if DEKALLM_SLUG in payload["levers"].index else None
        if dek_row is None:
            continue
        for col in LEVER_LABELS:
            rank = dek_row.get(f"{col}_rank")
            gap = dek_row.get(f"{col}_gap_pct")
            if pd.isna(rank):
                continue
            rows.append({
                "model_id": model_id,
                "lever": col,
                "rank": int(rank),
                "gap_pct": float(gap) if not pd.isna(gap) else 0.0,
                "is_leader": int(rank) == 1,
            })
    if not rows:
        out.append("_No data._")
        return "\n".join(out)

    agg = pd.DataFrame(rows)
    summary = agg.groupby("lever").agg(
        avg_rank=("rank", "mean"),
        avg_gap_pct=("gap_pct", "mean"),
        leader_count=("is_leader", "sum"),
        model_count=("rank", "count"),
    ).round(2)
    summary["leader_count"] = summary["leader_count"].astype(int)
    summary["model_count"] = summary["model_count"].astype(int)

    out.append("Average rank and gap across models DekaLLM serves "
               "(rank 1 = leader; gap_pct = signed % from leader):\n")
    out.append("| Lever | Avg rank | Avg gap vs leader | Leader on # models | Of total |")
    out.append("|---|---:|---:|---:|---:|")
    for lever, row in summary.iterrows():
        label = LEVER_LABELS[lever][0]
        out.append(
            f"| {label} | {row['avg_rank']:.1f} | {row['avg_gap_pct']:+.1f}% | "
            f"{row['leader_count']} | {row['model_count']} |"
        )

    # Pick the lever with the worst average rank as the "strategic priority"
    summary_sorted = summary.sort_values("avg_rank", ascending=False)
    worst_lever = summary_sorted.index[0]
    worst_row = summary_sorted.iloc[0]
    out.append(
        f"\n**Strategic priority**: the lever where DekaLLM consistently lags most is "
        f"**{LEVER_LABELS[worst_lever][0]}** (avg rank {worst_row['avg_rank']:.1f}, "
        f"avg gap {worst_row['avg_gap_pct']:+.1f}% vs leader). "
        f"Improving this universally would lift share across the portfolio."
    )

    out.append(
        f"\n**Sustainable advantage candidates**: levers where DekaLLM already leads on "
        f"some models — worth investing to maintain and replicate the recipe."
    )
    leader_levers = summary[summary["leader_count"] > 0].sort_values("leader_count", ascending=False)
    for lever, row in leader_levers.iterrows():
        out.append(
            f"- **{LEVER_LABELS[lever][0]}**: leader on {int(row['leader_count'])} / "
            f"{int(row['model_count'])} models"
        )
    if leader_levers.empty:
        out.append("- _(none yet)_")

    return "\n".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    print(f"Pulling lever data from {PROM_URL}\n")

    name_map = resolve_provider_names()
    print(f"Provider slug -> display name resolution:")
    for slug, name in name_map.items():
        print(f"  {slug:12s} -> {name or '(NOT FOUND — check endpoint scrape config)'}")

    dek_models = get_dekallm_models()
    print(f"\nDekaLLM models with token-daily data: {len(dek_models)}")
    for m in dek_models:
        print(f"  {m}")

    print("\nGenerating per-model competitive position reports...")
    model_data = {}
    sections = []
    for model_id in dek_models:
        levers = get_levers_for_model(model_id, name_map)
        if levers.empty or levers.isna().all().all():
            print(f"  {model_id}: no lever data (likely DekaLLM-only or unscraped)")
            continue
        for col in LEVER_LABELS:
            levers = rank_and_gap(levers, col)
        share = get_market_share(model_id)
        model_data[model_id] = {"levers": levers, "share": share}
        sections.append(per_model_report(model_id, levers, share))
        print(f"  {model_id}: {len(levers)} providers compared")

    summary = overall_summary(model_data)

    # Write markdown report
    md_path = OUT / "competitive_position.md"
    with open(md_path, "w") as f:
        f.write(f"# DekaLLM Competitive Position on OpenRouter\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("Routing levers compared across providers: price (input + output), "
                "throughput (p50 t/s), latency (p50 ms), uptime (last 1d %). "
                "Each lever maps to a specific OpenRouter routing mode that prioritizes it.\n\n")
        f.write("---\n\n")
        f.write(summary)
        f.write("\n\n---\n\n# Per-model breakdown\n\n")
        f.write("\n\n---\n\n".join(sections))
    print(f"\nMarkdown report -> {md_path}")

    # Write flat CSV with all model + lever data
    csv_rows = []
    for model_id, payload in model_data.items():
        df = payload["levers"]
        share = payload["share"]
        for slug in df.index:
            row = {"model_id": model_id, "provider": slug, "share_7d": share.get(slug, 0.0)}
            for col in LEVER_LABELS:
                row[col] = df.loc[slug].get(col)
                row[f"{col}_rank"] = df.loc[slug].get(f"{col}_rank")
                row[f"{col}_gap_pct"] = df.loc[slug].get(f"{col}_gap_pct")
            csv_rows.append(row)
    csv_path = OUT / "competitive_position.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Flat CSV          -> {csv_path}")

    # Console summary
    print("\n" + "=" * 78)
    print("Summary printed below — full markdown report at:")
    print(f"  {md_path}")
    print("=" * 78 + "\n")
    print(summary)


if __name__ == "__main__":
    main()
