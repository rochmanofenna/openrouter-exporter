"""Demand-wave landscape: competitive map for the OSS models DekaLLM should consider.

Background. Per the OpenRouter market research:
- ~45% of OSS token volume on OpenRouter is now Chinese OSS (Kimi K2, MiniMax,
  DeepSeek, Qwen, GLM)
- Reasoning + agent + coding workloads drive demand
- Tool-call accuracy varies by ~32 points on the same model weights between
  providers (Fireworks 100% vs Chutes 68% on K2)
- The cheap-and-bad-tool-call cluster (Chutes, Novita) is a known dead-end
- DekaLLM currently sits in that cluster: cheapest, but bottom-quartile quality

This script pulls every provider's lever metrics for a list of demand-wave
models and produces a competitive landscape per model:

- Provider count
- Price/throughput/uptime distribution (quartiles)
- DekaLLM's current position (if they serve it)
- Where DekaLLM would land if they entered at their typical price point
- "Difficulty" score: combination of competition density and quality bar

Outputs:
- analysis/out/demand_wave_landscape.md   (meeting report)
- analysis/out/demand_wave_landscape.csv  (flat per-(model,provider) data)
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
DEKALLM_NAME = "DekaLLM"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Candidate demand-wave models. Try each with and without the date suffix;
# whichever returns data, we use. List based on inspect_providers.py output
# + OpenRouter research findings (top OSS models by volume Q1 2026).
DEMAND_WAVE_MODELS = [
    # Moonshot Kimi family — coding/agent
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.5-0127",
    "moonshotai/kimi-k2-0905",
    # DeepSeek family — reasoning + coding
    "deepseek/deepseek-v3.2-20251201",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1",
    # Z.AI / GLM family
    "z-ai/glm-5-20260211",
    "z-ai/glm-5",
    "z-ai/glm-5.1-20260406",
    "z-ai/glm-5.1",
    "z-ai/glm-4.7-flash-20260119",
    "z-ai/glm-4.7-flash",
    # MiniMax family
    "minimax/minimax-m2.5-20260211",
    "minimax/minimax-m2.5",
    "minimax/minimax-m2.1",
    # Qwen big-context / coder
    "qwen/qwen3.5-397b-a17b-20260216",
    "qwen/qwen3.5-9b-20260310",
    "qwen/qwen3-235b-a22b-07-25",
    "qwen/qwen3-coder",
    # OpenAI OSS - smaller siblings of what DekaLLM already serves
    "openai/gpt-oss-20b",
]


_DATE_SUFFIX = re.compile(r"-\d{8}$")
_DATE_DASH_SUFFIX = re.compile(r"-\d{4,8}$")  # also strips -MMDD suffixes like -0127


def prom_query(query: str) -> list[dict]:
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")
    return payload["data"]["result"]


def find_served_model_id(candidate: str) -> str | None:
    """Try the candidate and its date-stripped variants. Return the form that
    returns endpoint data, or None."""
    forms = [candidate]
    stripped = _DATE_SUFFIX.sub("", candidate)
    if stripped != candidate:
        forms.append(stripped)
    stripped2 = _DATE_DASH_SUFFIX.sub("", candidate)
    if stripped2 not in forms:
        forms.append(stripped2)

    for f in forms:
        probe = prom_query(
            f'openrouter_model_input_price_dollars_per_million_tokens{{model_id="{f}"}}'
        )
        if probe:
            return f
    return None


def pull_levers(model_id: str) -> pd.DataFrame:
    """For a single model, pull every provider's lever metrics. Returns
    DataFrame indexed by provider_name with columns for each lever."""
    queries = {
        "input_price":  ("min", "openrouter_model_input_price_dollars_per_million_tokens", ""),
        "output_price": ("min", "openrouter_model_output_price_dollars_per_million_tokens", ""),
        "throughput":   ("max", "openrouter_endpoint_throughput_tokens_per_second", ',quantile="p50"'),
        "latency_ms":   ("min", "openrouter_endpoint_latency_milliseconds", ',quantile="p50"'),
        "uptime_1d":    ("max", "openrouter_endpoint_uptime_percentage_last_1d", ""),
        "context_len":  ("max", "openrouter_model_info", ''),  # used for filter, drop later
    }

    metrics: dict[str, dict[str, float]] = {}
    for col, (agg, base, qfilter) in queries.items():
        if base == "openrouter_model_info":
            # context length comes from a different label set; skip for now
            continue
        q = f'{agg} by (provider_name) ({base}{{model_id="{model_id}"{qfilter}}})'
        try:
            res = prom_query(q)
        except Exception:
            res = []
        for r in res:
            pn = r["metric"].get("provider_name", "")
            if not pn:
                continue
            metrics.setdefault(pn, {})[col] = float(r["value"][1])

    if not metrics:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.index.name = "provider_name"
    for col in ("input_price", "output_price", "throughput", "latency_ms", "uptime_1d"):
        if col not in df.columns:
            df[col] = float("nan")
    return df[["input_price", "output_price", "throughput", "latency_ms", "uptime_1d"]]


# ---------------------------------------------------------------------------
# Quartile / clustering analysis
# ---------------------------------------------------------------------------
def quartile_label(series: pd.Series, value: float, higher_better: bool) -> str:
    """Return Q1/Q2/Q3/Q4 label for value within series. Q1 = best."""
    if pd.isna(value) or len(series.dropna()) < 4:
        return "—"
    quartiles = series.dropna().quantile([0.25, 0.50, 0.75]).values
    if higher_better:
        if value >= quartiles[2]:
            return "Q1 (top)"
        elif value >= quartiles[1]:
            return "Q2"
        elif value >= quartiles[0]:
            return "Q3"
        else:
            return "Q4 (bottom)"
    else:
        if value <= quartiles[0]:
            return "Q1 (top)"
        elif value <= quartiles[1]:
            return "Q2"
        elif value <= quartiles[2]:
            return "Q3"
        else:
            return "Q4 (bottom)"


def fmt_money(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"${v:.3f}"


def fmt_num(v, decimals=0) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.{decimals}f}"


# ---------------------------------------------------------------------------
# Report per model
# ---------------------------------------------------------------------------
def per_model_section(model_id: str, lookup_form: str, df: pd.DataFrame,
                      dekallm_reference: dict | None) -> str:
    out = []
    out.append(f"\n## {model_id}\n")
    if lookup_form != model_id:
        out.append(f"_(Endpoint metrics found under `{lookup_form}`)_\n")

    n_providers = len(df)
    out.append(f"**Providers serving:** {n_providers}\n")
    if n_providers < 2:
        out.append("_Only one provider — no real competitive map._\n")
        return "\n".join(out)

    # Leader by each lever
    leaders = {
        "Cheapest input": (df["input_price"].idxmin() if df["input_price"].notna().any() else None,
                           df["input_price"].min()),
        "Cheapest output": (df["output_price"].idxmin() if df["output_price"].notna().any() else None,
                            df["output_price"].min()),
        "Fastest throughput": (df["throughput"].idxmax() if df["throughput"].notna().any() else None,
                               df["throughput"].max()),
        "Lowest latency": (df["latency_ms"].idxmin() if df["latency_ms"].notna().any() else None,
                           df["latency_ms"].min()),
        "Highest uptime": (df["uptime_1d"].idxmax() if df["uptime_1d"].notna().any() else None,
                           df["uptime_1d"].max()),
    }

    out.append("### Per-lever leaders\n")
    out.append("| Lever | Leader | Value |")
    out.append("|---|---|---:|")
    out.append(f"| Cheapest input | {leaders['Cheapest input'][0] or '—'} | {fmt_money(leaders['Cheapest input'][1])}/M |")
    out.append(f"| Cheapest output | {leaders['Cheapest output'][0] or '—'} | {fmt_money(leaders['Cheapest output'][1])}/M |")
    out.append(f"| Fastest throughput | {leaders['Fastest throughput'][0] or '—'} | {fmt_num(leaders['Fastest throughput'][1])} t/s |")
    out.append(f"| Lowest latency | {leaders['Lowest latency'][0] or '—'} | {fmt_num(leaders['Lowest latency'][1])} ms |")
    out.append(f"| Highest uptime | {leaders['Highest uptime'][0] or '—'} | {fmt_num(leaders['Highest uptime'][1], 2)}% |")

    # Full leaderboard sorted by output price
    sorted_df = df.sort_values("output_price", na_position="last")
    out.append("\n### Full leaderboard (sorted by output price, cheapest first)\n")
    out.append("| Provider | Input $/M | Output $/M | Throughput | Latency | Uptime |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for provider, row in sorted_df.iterrows():
        is_dekallm = provider == DEKALLM_NAME
        marker = "**" if is_dekallm else ""
        out.append(
            f"| {marker}{provider}{marker} | "
            f"{fmt_money(row['input_price'])} | {fmt_money(row['output_price'])} | "
            f"{fmt_num(row['throughput'])} t/s | "
            f"{fmt_num(row['latency_ms'])} ms | "
            f"{fmt_num(row['uptime_1d'], 2)}% |"
        )

    # Quartile clusters (if enough providers)
    if n_providers >= 4:
        out.append("\n### Quartile clusters\n")
        q_table = []
        for provider, row in df.iterrows():
            q_table.append({
                "Provider": provider,
                "Output $/M tier": quartile_label(df["output_price"], row["output_price"], higher_better=False),
                "Throughput tier": quartile_label(df["throughput"], row["throughput"], higher_better=True),
                "Uptime tier": quartile_label(df["uptime_1d"], row["uptime_1d"], higher_better=True),
            })
        qdf = pd.DataFrame(q_table).set_index("Provider")
        out.append("| Provider | Output price tier | Throughput tier | Uptime tier |")
        out.append("|---|---|---|---|")
        for p, row in qdf.iterrows():
            marker = "**" if p == DEKALLM_NAME else ""
            out.append(
                f"| {marker}{p}{marker} | {row['Output $/M tier']} | "
                f"{row['Throughput tier']} | {row['Uptime tier']} |"
            )

    # If DekaLLM doesn't currently serve, project where they'd land
    if DEKALLM_NAME not in df.index and dekallm_reference is not None:
        out.append("\n### DekaLLM entry projection\n")
        dek_typical_in = dekallm_reference["typical_input_price"]
        dek_typical_out = dekallm_reference["typical_output_price"]
        dek_typical_tps = dekallm_reference["typical_throughput"]
        dek_typical_uptime = dekallm_reference["typical_uptime"]

        out.append(
            f"DekaLLM does **not** currently serve this model. Projecting their "
            f"position if they entered at the price/throughput/uptime they show "
            f"on the models they do serve:\n\n"
            f"- DekaLLM's typical input price: **{fmt_money(dek_typical_in)}/M**\n"
            f"- DekaLLM's typical output price: **{fmt_money(dek_typical_out)}/M**\n"
            f"- DekaLLM's typical throughput: **{fmt_num(dek_typical_tps)} t/s**\n"
            f"- DekaLLM's typical uptime: **{fmt_num(dek_typical_uptime, 2)}%**\n"
        )

        # Where would DekaLLM rank
        if n_providers >= 4:
            price_q = quartile_label(df["output_price"], dek_typical_out, higher_better=False)
            tput_q = quartile_label(df["throughput"], dek_typical_tps, higher_better=True)
            uptime_q = quartile_label(df["uptime_1d"], dek_typical_uptime, higher_better=True)
            out.append(f"\n**Projected entry tier:** output price **{price_q}**, "
                       f"throughput **{tput_q}**, uptime **{uptime_q}**.\n")

            # Heuristic verdict
            uptime_threshold_ok = dek_typical_uptime >= 95.0
            advantages = []
            disadvantages = []
            if "Q1" in price_q:
                advantages.append("would be a top-quartile price option")
            elif "Q4" in price_q:
                disadvantages.append("price would not be competitive — entry pointless without a price edge")
            if "Q1" in tput_q:
                advantages.append("throughput is top-tier (rare for cheap providers)")
            elif "Q4" in tput_q:
                disadvantages.append("throughput would be bottom-tier — Auto Exacto risk for tool traffic")
            if not uptime_threshold_ok:
                disadvantages.append("uptime <95% means degraded routing tier on day one")

            if advantages and not disadvantages:
                out.append("**Verdict: attractive entry.** " + " | ".join(advantages))
            elif advantages and disadvantages:
                out.append(f"**Verdict: mixed.** Strengths: {'; '.join(advantages)}. "
                           f"Risks: {'; '.join(disadvantages)}.")
            elif disadvantages:
                out.append(f"**Verdict: skip unless quality improves.** {' | '.join(disadvantages)}.")
            else:
                out.append("**Verdict: middle of pack.** No clear wedge, but no clear blocker either.")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# DekaLLM reference: typical values across the models they currently serve
# ---------------------------------------------------------------------------
def dekallm_reference_profile() -> dict | None:
    """Compute DekaLLM's typical price/throughput/uptime across the models
    they currently serve, for projection onto new candidate models."""
    # Get list of models DekaLLM serves (from chart data)
    res = prom_query(
        'count by (model_id) (openrouter_provider_tokens_daily{provider="dekallm"})'
    )
    if not res:
        return None
    dek_models = [r["metric"]["model_id"] for r in res]

    input_prices = []
    output_prices = []
    throughputs = []
    uptimes = []
    for m in dek_models:
        served = find_served_model_id(m)
        if served is None:
            continue
        for col, agg, base, qfilter in [
            ("input_price", "min", "openrouter_model_input_price_dollars_per_million_tokens", ""),
            ("output_price", "min", "openrouter_model_output_price_dollars_per_million_tokens", ""),
            ("throughput", "max", "openrouter_endpoint_throughput_tokens_per_second", ',quantile="p50"'),
            ("uptime_1d", "max", "openrouter_endpoint_uptime_percentage_last_1d", ""),
        ]:
            q = f'{agg}({base}{{model_id="{served}",provider_name="{DEKALLM_NAME}"{qfilter}}})'
            try:
                res = prom_query(q)
            except Exception:
                res = []
            if res:
                v = float(res[0]["value"][1])
                if col == "input_price":
                    input_prices.append(v)
                elif col == "output_price":
                    output_prices.append(v)
                elif col == "throughput":
                    throughputs.append(v)
                elif col == "uptime_1d":
                    uptimes.append(v)

    if not input_prices:
        return None
    return {
        "typical_input_price": sum(input_prices) / len(input_prices),
        "typical_output_price": sum(output_prices) / len(output_prices) if output_prices else None,
        "typical_throughput": sum(throughputs) / len(throughputs) if throughputs else None,
        "typical_uptime": sum(uptimes) / len(uptimes) if uptimes else None,
    }


# ---------------------------------------------------------------------------
# Aggregated summary across candidate models
# ---------------------------------------------------------------------------
def opportunity_score(df: pd.DataFrame, dek_ref: dict) -> dict:
    """Return a heuristic score for whether DekaLLM should enter this model.
    Higher = more attractive."""
    if dek_ref is None:
        return {"score": 0, "reasons": ["no DekaLLM reference available"]}

    n_providers = len(df)
    if n_providers < 2:
        return {"score": 0, "reasons": ["only one or no competitor"]}

    dek_out = dek_ref["typical_output_price"]
    dek_tps = dek_ref["typical_throughput"]

    score = 0
    reasons = []

    # Lower competition = easier to break in
    if n_providers <= 3:
        score += 3
        reasons.append(f"low competition ({n_providers} providers)")
    elif n_providers <= 6:
        score += 1
        reasons.append(f"moderate competition ({n_providers} providers)")
    else:
        score -= 1
        reasons.append(f"high competition ({n_providers} providers — Auto Exacto stricter)")

    # Where DekaLLM's typical output price lands
    out_prices = df["output_price"].dropna()
    if len(out_prices) >= 2 and dek_out is not None:
        if dek_out <= out_prices.min():
            score += 3
            reasons.append("DekaLLM price would be cheapest")
        elif dek_out <= out_prices.median():
            score += 1
            reasons.append("DekaLLM price would be below median")
        else:
            score -= 2
            reasons.append("DekaLLM price would not be competitive")

    # Throughput floor risk (if Auto Exacto evaluates this)
    if dek_tps is not None and len(df["throughput"].dropna()) >= 4:
        tput_25th = df["throughput"].quantile(0.25)
        if dek_tps < tput_25th:
            score -= 2
            reasons.append("DekaLLM throughput would be bottom quartile (Auto Exacto risk)")

    return {"score": score, "reasons": reasons}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Querying Prometheus at {PROM_URL}\n")

    # Compute DekaLLM reference profile first
    dek_ref = dekallm_reference_profile()
    if dek_ref:
        print("DekaLLM reference profile (typical values across models they serve):")
        for k, v in dek_ref.items():
            print(f"  {k}: {v}")
    else:
        print("Warning: could not compute DekaLLM reference profile.")
    print()

    # Try each candidate model
    print("Probing candidate demand-wave models...")
    served_models = []
    for m in DEMAND_WAVE_MODELS:
        served = find_served_model_id(m)
        if served:
            print(f"  ✓ {m} -> data found at '{served}'")
            served_models.append((m, served))
        else:
            print(f"  ✗ {m} -> no endpoint data on OpenRouter")
    print(f"\nFound {len(served_models)} of {len(DEMAND_WAVE_MODELS)} candidate models on OpenRouter.")

    # Pull lever data + assemble report
    sections = []
    csv_rows = []
    summary_rows = []
    for original_id, served_id in served_models:
        df = pull_levers(served_id)
        if df.empty:
            continue
        sections.append(per_model_section(original_id, served_id, df, dek_ref))

        opp = opportunity_score(df, dek_ref) if dek_ref else {"score": 0, "reasons": []}
        summary_rows.append({
            "model_id": original_id,
            "n_providers": len(df),
            "median_output_price": df["output_price"].median(),
            "min_output_price": df["output_price"].min(),
            "median_throughput": df["throughput"].median(),
            "max_throughput": df["throughput"].max(),
            "median_uptime": df["uptime_1d"].median(),
            "dekallm_serves": DEKALLM_NAME in df.index,
            "opportunity_score": opp["score"],
            "opportunity_reasons": "; ".join(opp["reasons"]),
        })
        for provider, row in df.iterrows():
            csv_rows.append({
                "model_id": original_id,
                "provider_name": provider,
                **row.to_dict(),
            })

    # Write CSV
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(OUT / "demand_wave_landscape.csv", index=False)
        print(f"\nFlat CSV -> {OUT / 'demand_wave_landscape.csv'}")

    # Write markdown report
    md = []
    md.append("# Demand-Wave Model Landscape — Where Should DekaLLM Compete?\n")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    md.append(
        "\nBased on OpenRouter market research: ~45% of OSS token volume is now "
        "Chinese OSS coding/agent models (Kimi K2, MiniMax M2, DeepSeek, Qwen Coder, "
        "GLM). DekaLLM currently serves none of these except qwen3.5-35b. This "
        "report maps each demand-wave model's competitive landscape and projects "
        "DekaLLM's entry tier.\n"
    )

    if dek_ref:
        md.append("\n## DekaLLM reference profile\n")
        md.append("Averaged across the models DekaLLM currently serves on OpenRouter:\n")
        md.append("| Metric | Value |")
        md.append("|---|---:|")
        md.append(f"| Typical input price | {fmt_money(dek_ref['typical_input_price'])}/M |")
        md.append(f"| Typical output price | {fmt_money(dek_ref['typical_output_price'])}/M |")
        md.append(f"| Typical throughput | {fmt_num(dek_ref['typical_throughput'])} t/s |")
        md.append(f"| Typical uptime | {fmt_num(dek_ref['typical_uptime'], 2)}% |")
        md.append("\nThis is the entry profile we project onto each candidate model below.\n")

    # Opportunity ranking
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("opportunity_score", ascending=False)
        md.append("\n## Opportunity ranking (highest to lowest)\n")
        md.append("Heuristic score combining competition density, price competitiveness, "
                  "and quality floor risk. Positive = attractive to enter; negative = skip.\n")
        md.append("\n| Score | Model | # Providers | Median output $/M | Median t/s | Why |")
        md.append("|---:|---|---:|---:|---:|---|")
        for _, r in summary_df.iterrows():
            already = " *(already serving)*" if r["dekallm_serves"] else ""
            md.append(
                f"| {int(r['opportunity_score']):+d} | {r['model_id']}{already} | "
                f"{int(r['n_providers'])} | {fmt_money(r['median_output_price'])} | "
                f"{fmt_num(r['median_throughput'])} t/s | {r['opportunity_reasons']} |"
            )

    # Per-model detail
    md.append("\n---\n\n# Per-model breakdown\n")
    md.extend(sections)

    md.append("\n---\n\n## Caveats\n")
    md.append(
        "- **Opportunity score is heuristic, not financial.** It captures whether "
        "DekaLLM would be a competitive entrant, not whether they'd be profitable.\n"
        "- **Tool-call accuracy not measured here.** Critical missing dimension. "
        "DekaLLM's typical TauBench score of ~0.58 suggests they'd land in the "
        "bottom-quartile tool-call cluster (Chutes, Novita, Together) unless "
        "the serving stack is upgraded. Auto Exacto would demote them for tool-using "
        "traffic on any model with ≥4 providers.\n"
        "- **Throughput projection assumes current GPU stack.** A GPU upgrade "
        "(see margin_analysis.py L40S → H100) would shift the projection upward.\n"
        "- **Uptime <95% is a step-function disadvantage.** DekaLLM averages "
        "~93% on the models they serve, putting them in OpenRouter's 'degraded' "
        "routing tier on entry.\n"
    )

    md_path = OUT / "demand_wave_landscape.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Markdown report -> {md_path}\n")

    # Console summary
    print("=" * 78)
    print("Opportunity ranking (top 5):")
    print("=" * 78)
    if summary_rows:
        for r in summary_df.head(5).itertuples():
            already = " (already serving)" if r.dekallm_serves else ""
            print(f"  score={r.opportunity_score:+d}  {r.model_id}{already}")
            print(f"      {r.n_providers} providers, median output ${r.median_output_price:.3f}/M, "
                  f"median {r.median_throughput:.0f} t/s")
            print(f"      reasons: {r.opportunity_reasons}")
            print()


if __name__ == "__main__":
    main()
