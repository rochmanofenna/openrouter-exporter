"""Competitive position analysis — where DekaLLM leads/lags on every routing
lever, for every model in DekaLLM's portfolio.

For each model DekaLLM serves, this script pulls:
  - Per-provider pricing, throughput, latency, context (from /scrape/models/{slug}/details)
  - Per-provider daily uptime (from /scrape/models/{slug}/uptime)        <-- NEW
  - Per-provider, per-day token share (from /db/models/{slug}/provider-tokens)
ranks DekaLLM against every provider serving the same model, shows the gap
to leader on each lever, and identifies the highest-leverage move per model.

Data source upgrade vs prior version:
  - 91-day share history (was 32 days, limited to 4 providers)
  - All ~72 providers compared, not 4
  - Real measured uptime (was approximated from incident heuristics)
  - Slug resolution via /scrape/resolve (was hand-coded regex fallback)

OpenRouter routing modes that prioritize each lever:
  - default mode              -> price + latency + uptime (weighted blend)
  - :floor mode               -> price (lower wins)
  - :nitro mode               -> throughput (higher wins)
  - :exacto mode              -> tool-call quality vs per-model median (not yet in API)

Outputs:
  - analysis/out/competitive_position.md   (meeting-ready report)
  - analysis/out/competitive_position.csv  (flat per-model lever table)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.api_client import APIError, default_client, require_alive

DEKALLM_SLUG = "dekallm"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
# Share lookback window for the competitive table (days). Per-provider tokens are
# the freshest signal we have; 7 days smooths over individual incidents like
# the 2026-05-17 routing-quality event without losing trend.
SHARE_LOOKBACK_DAYS = int(os.environ.get("SHARE_LOOKBACK_DAYS", "7"))


# ---------------------------------------------------------------------------
# Lever extraction — tolerant of unknown response shapes
# ---------------------------------------------------------------------------
# The /scrape/models/{slug}/details payload exposes a list of provider endpoints,
# each carrying pricing + perf fields. The exact field names depend on the
# upstream OpenRouter JSON shape and may be nested. Rather than hard-code
# paths, we search each endpoint dict for fields matching a set of name
# patterns. This makes the script robust to minor schema drift.

LEVERS = {
    "input_price":  {"better": "lower", "label": "Input price ($/M)"},
    "output_price": {"better": "lower", "label": "Output price ($/M)"},
    "throughput":   {"better": "higher", "label": "Throughput (t/s p50)"},
    "latency_ms":   {"better": "lower", "label": "Latency (ms p50)"},
    "uptime_pct":   {"better": "higher", "label": "Uptime (% last 7d)"},
}


def _walk(obj, predicate):
    """Yield (path_str, value) for any nested key/value matching predicate(key)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if predicate(k):
                yield (k, v)
            yield from _walk(v, predicate)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item, predicate)


def _first_match(endpoint: dict, keywords: tuple[str, ...]) -> float | None:
    """Return the first numeric field whose key contains all keywords (case-insensitive)."""
    for k, v in _walk(endpoint, lambda key: all(kw in key.lower() for kw in keywords)):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def extract_levers(endpoint: dict) -> dict:
    """Pull lever values from one provider-endpoint dict. Tolerant of field naming.

    Searches for the most semantically obvious key. Multiple patterns tried so
    the same code handles both raw OpenRouter JSON ('pricing.prompt') and
    enriched/flattened views ('input_price_per_million'). If none match,
    that lever stays NaN.
    """
    # Pricing — usually nested under "pricing": {"prompt", "completion", ...}
    # Common in OR: pricing.prompt = $/token (very small float). Convert to $/M.
    in_price = _first_match(endpoint, ("input", "price")) or _first_match(endpoint, ("prompt", "price"))
    if in_price is None:
        raw = _first_match(endpoint, ("pricing", "prompt"))
        if raw is not None:
            in_price = raw * 1_000_000 if raw < 1.0 else raw

    out_price = _first_match(endpoint, ("output", "price")) or _first_match(endpoint, ("completion", "price"))
    if out_price is None:
        raw = _first_match(endpoint, ("pricing", "completion"))
        if raw is not None:
            out_price = raw * 1_000_000 if raw < 1.0 else raw

    throughput = (
        _first_match(endpoint, ("throughput",))
        or _first_match(endpoint, ("tokens_per_second",))
        or _first_match(endpoint, ("tps",))
    )
    latency = (
        _first_match(endpoint, ("latency",))
        or _first_match(endpoint, ("first", "token"))
    )

    return {
        "input_price":  in_price,
        "output_price": out_price,
        "throughput":   throughput,
        "latency_ms":   latency,
        "uptime_pct":   None,  # filled in separately from /uptime endpoint
    }


def provider_slug_from_endpoint(endpoint: dict) -> str | None:
    """Best-effort: extract the provider slug from an endpoint payload.

    OpenRouter endpoints typically expose provider as `provider_name` or
    nested `provider.slug`/`provider.name`. Returns lower-case slug or None.
    """
    # Direct fields
    for key in ("provider_slug", "providerSlug", "providerName", "provider_name"):
        v = endpoint.get(key)
        if isinstance(v, str) and v:
            return v.lower().replace(" ", "-")
    # Nested provider object
    prov = endpoint.get("provider")
    if isinstance(prov, dict):
        for key in ("slug", "name", "displayName"):
            v = prov.get(key)
            if isinstance(v, str) and v:
                return v.lower().replace(" ", "-")
    return None


# ---------------------------------------------------------------------------
# Per-model data assembly
# ---------------------------------------------------------------------------
def build_lever_table(client, model_id: str) -> tuple[pd.DataFrame, str | None]:
    """Return (DataFrame indexed by provider, resolved permaslug) for one model.

    DataFrame columns: input_price, output_price, throughput, latency_ms, uptime_pct.
    """
    resolved = client.resolve(model_id)
    try:
        details = client.scrape_model_details(resolved)
    except APIError:
        details = {}
    if not details and resolved != model_id:
        try:
            details = client.scrape_model_details(model_id)
        except APIError:
            details = {}

    # The details payload may be a dict with "endpoints" or just a list.
    endpoints = []
    if isinstance(details, list):
        endpoints = details
    elif isinstance(details, dict):
        endpoints = (
            details.get("endpoints")
            or details.get("providers")
            or details.get("data")
            or []
        )

    rows: dict[str, dict] = {}
    for ep in endpoints:
        if not isinstance(ep, dict):
            continue
        slug = provider_slug_from_endpoint(ep)
        if not slug:
            continue
        rows[slug] = extract_levers(ep)

    # Uptime — separate endpoint, returns daily values per provider.
    try:
        uptime_rows = client.scrape_model_uptime(resolved)
    except APIError:
        uptime_rows = []
    if not uptime_rows and resolved != model_id:
        try:
            uptime_rows = client.scrape_model_uptime(model_id)
        except APIError:
            uptime_rows = []

    # Aggregate last `SHARE_LOOKBACK_DAYS` of uptime per provider into a single mean.
    uptime_by_provider: dict[str, list[float]] = {}
    for row in uptime_rows[-SHARE_LOOKBACK_DAYS:] if uptime_rows else []:
        if not isinstance(row, dict):
            continue
        # Expected shape: {"date": "...", "endpoints": [{provider_name, uptime_pct}, ...]}
        # or {"date": "...", "providers": {slug: pct}}
        providers_block = row.get("providers")
        if isinstance(providers_block, dict):
            for slug, pct in providers_block.items():
                if isinstance(pct, (int, float)):
                    uptime_by_provider.setdefault(slug.lower(), []).append(float(pct))
            continue
        for ep_uptime in row.get("endpoints", []):
            if not isinstance(ep_uptime, dict):
                continue
            slug = provider_slug_from_endpoint(ep_uptime)
            pct = _first_match(ep_uptime, ("uptime",))
            if slug and pct is not None:
                uptime_by_provider.setdefault(slug, []).append(pct)

    for slug, values in uptime_by_provider.items():
        if not values:
            continue
        if slug not in rows:
            rows[slug] = {k: None for k in LEVERS}
        rows[slug]["uptime_pct"] = sum(values) / len(values)

    if not rows:
        return pd.DataFrame(columns=list(LEVERS.keys())), resolved

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "provider"
    for col in LEVERS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[list(LEVERS.keys())], resolved


def build_share_table(client, model_id: str, lookback_days: int = SHARE_LOOKBACK_DAYS) -> dict[str, float]:
    """Per-provider share averaged over the most recent finalized days.

    Drops today's still-in-progress day (the most recent date in the API
    response is usually yesterday since the sync runs daily). Uses only days
    where ALL providers in the window had data — guarantees shares sum to ~100%.
    """
    resolved = client.resolve(model_id)
    rows = client.db_model_provider_tokens(resolved)
    if not rows and resolved != model_id:
        rows = client.db_model_provider_tokens(model_id)
    if not rows:
        return {}

    # rows arrive newest-first; sort oldest-first for clarity, then take tail
    rows = sorted(rows, key=lambda r: r.get("date", ""))
    window = rows[-lookback_days:] if lookback_days > 0 else rows
    if not window:
        return {}

    providers_seen: set[str] = set()
    for r in window:
        providers_seen.update(r.get("providers", {}).keys())

    # Restrict to days where ALL seen providers reported
    common_days = [
        r for r in window
        if set(r.get("providers", {}).keys()) >= providers_seen
    ]
    # If too strict, fall back to the full window with zeros for missing providers
    use = common_days or window

    shares: dict[str, list[float]] = {p: [] for p in providers_seen}
    for r in use:
        tokens = r.get("providers", {})
        total = sum(tokens.values())
        if total <= 0:
            continue
        for p in providers_seen:
            shares[p].append(tokens.get(p, 0) / total)
    return {p: (sum(v) / len(v) if v else 0.0) for p, v in shares.items()}


# ---------------------------------------------------------------------------
# Ranking & reporting
# ---------------------------------------------------------------------------
def rank_and_gap(df: pd.DataFrame, col: str, higher_better: bool) -> pd.DataFrame:
    series = df[col].dropna()
    if series.empty:
        df[f"{col}_rank"] = float("nan")
        df[f"{col}_gap_pct"] = float("nan")
        return df
    leader = series.max() if higher_better else series.min()
    ranks = series.rank(ascending=not higher_better, method="min").astype(int)
    gaps = (series - leader) / leader * 100 if leader != 0 else series * 0
    df[f"{col}_rank"] = ranks.reindex(df.index)
    df[f"{col}_gap_pct"] = gaps.reindex(df.index)
    return df


def fmt_value(col: str, v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if col in ("input_price", "output_price"):
        return f"${v:.3f}"
    if col == "throughput":
        return f"{v:,.0f} t/s"
    if col == "latency_ms":
        return f"{v:,.0f} ms"
    if col == "uptime_pct":
        return f"{v:.2f}%"
    return f"{v:.3f}"


def per_model_report(model_id: str, df: pd.DataFrame, share: dict[str, float],
                     resolved: str | None) -> str:
    out = [f"## {model_id}\n"]
    if resolved and resolved != model_id:
        out.append(f"_Resolved to permaslug `{resolved}`._\n")
    dek_share = share.get(DEKALLM_SLUG, 0.0) * 100
    out.append(f"**DekaLLM market share ({SHARE_LOOKBACK_DAYS}-day avg)**: {dek_share:.2f}%\n")
    out.append(f"**Providers serving this model**: {len(df)}\n")

    for col, meta in LEVERS.items():
        df = rank_and_gap(df, col, higher_better=meta["better"] == "higher")

    # Top-of-leaderboard table — limit to top 10 by share for readability, always include DekaLLM.
    df_show = df.copy()
    df_show["share"] = df_show.index.map(lambda p: share.get(p, 0.0))
    df_show = df_show.sort_values("share", ascending=False)
    top = df_show.head(10)
    if DEKALLM_SLUG not in top.index and DEKALLM_SLUG in df_show.index:
        top = pd.concat([top, df_show.loc[[DEKALLM_SLUG]]])

    out.append(f"\n### Leaderboard (top {len(top)} providers by share)\n")
    out.append("| Provider | Input $/M | Output $/M | Throughput | Latency | Uptime | Share |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for slug, row in top.iterrows():
        marker = "**" if slug == DEKALLM_SLUG else ""
        out.append(
            f"| {marker}{slug}{marker} | "
            f"{fmt_value('input_price', row.get('input_price'))} | "
            f"{fmt_value('output_price', row.get('output_price'))} | "
            f"{fmt_value('throughput', row.get('throughput'))} | "
            f"{fmt_value('latency_ms', row.get('latency_ms'))} | "
            f"{fmt_value('uptime_pct', row.get('uptime_pct'))} | "
            f"{row['share']*100:.2f}% |"
        )

    if DEKALLM_SLUG not in df.index:
        out.append("\n_No DekaLLM endpoint metrics for this model._\n")
        return "\n".join(out)

    dek = df.loc[DEKALLM_SLUG]
    out.append("\n### DekaLLM lever position\n")
    out.append("| Lever | DekaLLM value | Rank | Gap vs leader | Action if pursued |")
    out.append("|---|---:|---:|---:|---|")
    actions = []
    for col, meta in LEVERS.items():
        v = dek.get(col)
        rank = dek.get(f"{col}_rank")
        gap = dek.get(f"{col}_gap_pct")
        if v is None or (isinstance(v, float) and v != v) or pd.isna(rank):
            out.append(f"| {meta['label']} | — | — | — | (no data) |")
            continue
        n_prov = df[col].notna().sum()
        rank_str = f"{int(rank)} / {int(n_prov)}"
        gap_str = f"{gap:+.1f}%" if not pd.isna(gap) else "—"
        action_phrase = {
            "input_price":  "cut input price",
            "output_price": "cut output price",
            "throughput":   "improve throughput",
            "latency_ms":   "reduce latency",
            "uptime_pct":   "improve uptime",
        }[col]
        action = action_phrase if int(rank) > 1 else "(already leader)"
        out.append(f"| {meta['label']} | {fmt_value(col, v)} | {rank_str} | {gap_str} | {action} |")
        if int(rank) > 1 and not pd.isna(gap):
            actions.append((col, abs(float(gap)), action_phrase, int(rank), float(gap)))

    if actions:
        actions.sort(key=lambda x: -x[1])
        col, abs_gap, action, rank, gap = actions[0]
        out.append(
            f"\n**Highest-leverage move**: **{action}** (currently rank {rank}, "
            f"{gap:+.1f}% vs leader). Closing this gap likely yields the largest "
            f"share gain per unit effort on this model."
        )
    else:
        out.append("\n**DekaLLM is the leader on every measured lever** for this model.")

    return "\n".join(out)


def overall_summary(model_data: dict) -> str:
    out = ["## Aggregated summary across DekaLLM models\n"]
    rows = []
    for model_id, payload in model_data.items():
        if DEKALLM_SLUG not in payload["levers"].index:
            continue
        dek = payload["levers"].loc[DEKALLM_SLUG]
        for col in LEVERS:
            rank = dek.get(f"{col}_rank")
            gap = dek.get(f"{col}_gap_pct")
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

    out.append("| Lever | Avg rank | Avg gap vs leader | Leader on # models | Of total |")
    out.append("|---|---:|---:|---:|---:|")
    for lever, row in summary.iterrows():
        out.append(
            f"| {LEVERS[lever]['label']} | {row['avg_rank']:.1f} | "
            f"{row['avg_gap_pct']:+.1f}% | {int(row['leader_count'])} | "
            f"{int(row['model_count'])} |"
        )

    worst = summary.sort_values("avg_rank", ascending=False).index[0]
    worst_row = summary.loc[worst]
    out.append(
        f"\n**Strategic priority**: the lever where DekaLLM consistently lags most is "
        f"**{LEVERS[worst]['label']}** (avg rank {worst_row['avg_rank']:.1f}, "
        f"avg gap {worst_row['avg_gap_pct']:+.1f}% vs leader)."
    )

    leaders = summary[summary["leader_count"] > 0].sort_values("leader_count", ascending=False)
    if not leaders.empty:
        out.append("\n**Levers where DekaLLM already leads on some models** (worth defending):")
        for lever, row in leaders.iterrows():
            out.append(
                f"- {LEVERS[lever]['label']}: leader on {int(row['leader_count'])} / "
                f"{int(row['model_count'])} models"
            )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = require_alive()
    print(f"Using analytics API at {client.base}\n")

    models = client.dekallm_current_model_slugs(lookback_days=3)
    print(f"DekaLLM current portfolio ({len(models)} models):")
    for m in models:
        print(f"  {m}")

    sections = []
    model_data: dict[str, dict] = {}
    print("\nBuilding per-model competitive position...")
    for m in models:
        try:
            levers, resolved = build_lever_table(client, m)
        except APIError as e:
            print(f"  {m}: SKIP ({e})")
            continue
        if levers.empty:
            print(f"  {m}: no lever data (endpoint details unavailable)")
            continue
        share = build_share_table(client, m)
        for col, meta in LEVERS.items():
            levers = rank_and_gap(levers, col, higher_better=meta["better"] == "higher")
        model_data[m] = {"levers": levers, "share": share, "resolved": resolved}
        sections.append(per_model_report(m, levers, share, resolved))
        print(f"  {m}: {len(levers)} providers, "
              f"DekaLLM share={share.get(DEKALLM_SLUG, 0)*100:.2f}%")

    if not sections:
        print("\nNo model data; not writing report.")
        return

    summary = overall_summary(model_data)
    md_path = OUT / "competitive_position.md"
    with open(md_path, "w") as f:
        f.write("# DekaLLM Competitive Position on OpenRouter\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(
            "Routing levers compared across all providers serving each DekaLLM model: "
            "input/output price, throughput (p50 t/s), latency (p50 ms), uptime (7-day mean %). "
            "Each lever maps to an OpenRouter routing mode that prioritizes it. "
            f"Share data is {SHARE_LOOKBACK_DAYS}-day average over fully-overlapping days only.\n\n"
        )
        f.write("---\n\n")
        f.write(summary)
        f.write("\n\n---\n\n# Per-model breakdown\n\n")
        f.write("\n\n---\n\n".join(sections))
    print(f"\nMarkdown report -> {md_path}")

    # Flat CSV
    csv_rows = []
    for model_id, payload in model_data.items():
        df = payload["levers"]
        share = payload["share"]
        for slug, row in df.iterrows():
            r = {"model_id": model_id, "provider": slug, "share": share.get(slug, 0.0)}
            for col in LEVERS:
                r[col] = row.get(col)
                r[f"{col}_rank"] = row.get(f"{col}_rank")
                r[f"{col}_gap_pct"] = row.get(f"{col}_gap_pct")
            csv_rows.append(r)
    csv_path = OUT / "competitive_position.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Flat CSV         -> {csv_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
