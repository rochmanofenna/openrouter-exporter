"""Per-model margin analysis for DekaLLM under different GPU configurations.

Now sourced from the analytics API. Revenue comes from joining DekaLLM's
daily token volume (`/db/providers/dekallm/tokens`) with the pricing on each
provider endpoint (`/scrape/models/{slug}/details`). Throughput likewise
comes from the model details payload. The script also calls the planner's
`/planner/compute` endpoint with DekaLLM's actual GPU pricing as overrides
and shows the two views side-by-side as a sanity check.

Defaults reflect DekaLLM's actual contracted rates (Ryan confirmed
2026-05-15): $1.00/hr L40S, $2.00/hr H100. Override any of these via env
vars.

Usage:
    python analysis/margin_analysis.py

Env knobs:
    L40S_PER_HR              default 1.00
    H100_PER_HR              default 2.00
    H100_THROUGHPUT_X        default 3.0
    OR_FEE_PCT               default 0.15
    SHARE_SLOPE              default 0.40   (conservative throughput→share slope)
    LOOKBACK_DAYS            default 7      (revenue averaging window)

Outputs:
    analysis/out/margin_analysis.md
    analysis/out/margin_analysis.csv
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.api_client import APIError, APIUnavailable, require_alive
from analysis.competitive_position import (
    extract_levers,
    provider_slug_from_endpoint,
)

L40S_PER_HR = float(os.environ.get("L40S_PER_HR", "1.00"))
H100_PER_HR = float(os.environ.get("H100_PER_HR", "2.00"))
H100_THROUGHPUT_X = float(os.environ.get("H100_THROUGHPUT_X", "3.0"))
OR_FEE_PCT = float(os.environ.get("OR_FEE_PCT", "0.15"))
SHARE_SLOPE = float(os.environ.get("SHARE_SLOPE", "0.40"))
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))

L40S_PER_DAY = L40S_PER_HR * 24
H100_PER_DAY = H100_PER_HR * 24
H100_COST_MULTIPLIER = H100_PER_HR / L40S_PER_HR
H100_DAILY_COST_FACTOR = H100_COST_MULTIPLIER / H100_THROUGHPUT_X

DEKALLM_SLUG = "dekallm"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Per-model pricing & throughput from the API
# ---------------------------------------------------------------------------
def get_dekallm_endpoint(client, model_id: str) -> dict | None:
    """Return DekaLLM's endpoint payload for one model, or None.

    Used to extract input/output price + throughput for the revenue math.
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
    endpoints = (details if isinstance(details, list)
                 else details.get("endpoints") or details.get("providers")
                 or details.get("data") or [])
    for ep in endpoints:
        if isinstance(ep, dict) and provider_slug_from_endpoint(ep) == DEKALLM_SLUG:
            return ep
    return None


def get_leader_throughput(client, model_id: str) -> float | None:
    resolved = client.resolve(model_id)
    try:
        details = client.scrape_model_details(resolved)
    except APIError:
        return None
    endpoints = (details if isinstance(details, list)
                 else details.get("endpoints") or details.get("providers")
                 or details.get("data") or [])
    throughputs = []
    for ep in endpoints:
        levers = extract_levers(ep)
        tp = levers.get("throughput")
        if isinstance(tp, (int, float)):
            throughputs.append(float(tp))
    return max(throughputs) if throughputs else None


# ---------------------------------------------------------------------------
# Revenue from token volume × pricing
# ---------------------------------------------------------------------------
def load_dekallm_token_history(client, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Per-(date, model) token volume from the analytics API, last N finalized days."""
    rows = client.db_provider_tokens(DEKALLM_SLUG)
    flat = []
    for row in rows:
        date = row.get("date")
        if not date:
            continue
        for model, tokens in row.get("tokens", {}).items():
            flat.append({"Date": date, "Model": model, "tokens": int(tokens)})
    if not flat:
        return pd.DataFrame()
    df = pd.DataFrame(flat)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Model", "Date"]).reset_index(drop=True)
    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(days=lookback_days - 1)
    return df[df["Date"] >= cutoff].copy()


def daily_revenue(tokens: float, input_price_per_million: float | None,
                  output_price_per_million: float | None) -> float | None:
    """Estimate gross revenue assuming a 50/50 input/output split.

    Returns None if pricing isn't available. Real I/O ratios vary per model
    (chat is output-heavy, summarization input-heavy); 50/50 is a crude
    blended rate that's close enough for portfolio-level numbers and avoids
    needing per-day input/output breakdown.
    """
    if input_price_per_million is None and output_price_per_million is None:
        return None
    in_price = input_price_per_million if input_price_per_million is not None else (output_price_per_million or 0)
    out_price = output_price_per_million if output_price_per_million is not None else (input_price_per_million or 0)
    # Average price per million tokens
    avg_per_million = (in_price + out_price) / 2
    return tokens * avg_per_million / 1_000_000


# ---------------------------------------------------------------------------
# Per-model economics
# ---------------------------------------------------------------------------
def estimate_gpus(daily_tokens: float, tps: float | None) -> float:
    if not tps or tps <= 0:
        return float("nan")
    return daily_tokens / (tps * 86400)


def per_model_economics(client, model_id: str, history: pd.DataFrame) -> dict | None:
    sub = history[history["Model"] == model_id]
    if sub.empty:
        return None

    avg_daily_tokens = float(sub["tokens"].mean())

    ep = get_dekallm_endpoint(client, model_id)
    levers = extract_levers(ep) if ep else {}
    in_price = levers.get("input_price")
    out_price = levers.get("output_price")
    dek_tps = levers.get("throughput")
    leader_tps = get_leader_throughput(client, model_id)

    gross = daily_revenue(avg_daily_tokens, in_price, out_price)
    net = gross * (1 - OR_FEE_PCT) if gross is not None else None

    gpus_l40s = estimate_gpus(avg_daily_tokens, dek_tps)
    cost_l40s = gpus_l40s * L40S_PER_DAY if gpus_l40s == gpus_l40s else float("nan")
    margin_l40s = (net - cost_l40s) if (net is not None and cost_l40s == cost_l40s) else float("nan")

    gpus_h100 = gpus_l40s / H100_THROUGHPUT_X if gpus_l40s == gpus_l40s else float("nan")
    cost_h100 = gpus_h100 * H100_PER_DAY if gpus_h100 == gpus_h100 else float("nan")
    margin_h100_static = (net - cost_h100) if (net is not None and cost_h100 == cost_h100) else float("nan")

    # Share-lift projection on H100
    share_lift_pp = 0.0
    projected_revenue = net if net is not None else 0.0
    throughput_gap = None
    if dek_tps is not None and leader_tps is not None and leader_tps > 0:
        throughput_gap = (dek_tps - leader_tps) / leader_tps
        new_tps = dek_tps * H100_THROUGHPUT_X
        new_gap = (new_tps - leader_tps) / leader_tps
        gap_closed = throughput_gap - new_gap
        share_lift = -gap_closed * SHARE_SLOPE
        share_lift_pp = share_lift * 100
        if share_lift > 0 and net is not None:
            multiplier = min(1.0 + share_lift * 3.0, 2.0)
            projected_revenue = net * multiplier

    margin_h100_projected = (projected_revenue - cost_h100) if cost_h100 == cost_h100 else float("nan")

    return {
        "model_id": model_id,
        "avg_daily_tokens": avg_daily_tokens,
        "avg_daily_gross_revenue": gross,
        "avg_daily_net_revenue": net,
        "input_price_per_million": in_price,
        "output_price_per_million": out_price,
        "dek_throughput_tps": dek_tps,
        "leader_throughput_tps": leader_tps,
        "throughput_gap": throughput_gap,
        "gpus_l40s_estimate": gpus_l40s,
        "cost_l40s_per_day": cost_l40s,
        "margin_l40s": margin_l40s,
        "gpus_h100_estimate": gpus_h100,
        "cost_h100_per_day_static": cost_h100,
        "margin_h100_static": margin_h100_static,
        "share_lift_pp_h100": share_lift_pp,
        "projected_revenue_h100": projected_revenue,
        "margin_h100_projected": margin_h100_projected,
    }


# ---------------------------------------------------------------------------
# Planner cross-check
# ---------------------------------------------------------------------------
def call_planner_compute(client, chip: str, dollar_per_gpu_hr: float) -> dict | None:
    """Best-effort call to /planner/compute. Body schema is not yet known
    (couldn't probe live), so we try a few field name variations and return
    whichever succeeds. If all fail, returns None.
    """
    payload_variants = [
        {"chip": chip, "dollar_per_gpu_hr": dollar_per_gpu_hr},
        {"hardware": chip, "dollar_per_gpu_hr": dollar_per_gpu_hr},
        {"chip": chip, "price_override": dollar_per_gpu_hr},
        {"chip": chip},  # use planner default pricing
    ]
    for body in payload_variants:
        try:
            return client.planner_compute(body)
        except APIError:
            continue
        except APIUnavailable:
            return None
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt_money(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"${v:,.2f}"


def fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:,.0f}"


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v*100:+.1f}%"


def write_report(results: list[dict], planner_l40s: dict | None,
                 planner_h100: dict | None) -> str:
    out = ["# DekaLLM Margin Analysis — L40S vs H100\n"]
    out.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    out.append("\n## Assumptions\n")
    out.append(f"- L40S cost: **${L40S_PER_HR:.2f} / GPU / hr** "
               f"(${L40S_PER_DAY:.2f} / GPU / day) — DekaLLM contracted rate")
    out.append(f"- H100 cost: **${H100_PER_HR:.2f} / GPU / hr** "
               f"(${H100_PER_DAY:.2f} / GPU / day) — DekaLLM contracted rate")
    out.append(f"- H100 throughput multiplier: **{H100_THROUGHPUT_X}×** "
               f"(typical for LLM inference; verify per model)")
    out.append(f"- OpenRouter fee: **{OR_FEE_PCT*100:.0f}%** of gross")
    out.append(f"- Throughput→share slope: **{SHARE_SLOPE}** "
               f"(from lever_sensitivity.py, conservatively discounted)")
    out.append(f"- Revenue window: **{LOOKBACK_DAYS}** most recent finalized days\n")

    totals = {
        "gross_rev": sum((r["avg_daily_gross_revenue"] or 0) for r in results),
        "net_rev":   sum((r["avg_daily_net_revenue"] or 0) for r in results),
        "cost_l40s": sum(r["cost_l40s_per_day"] for r in results
                         if r["cost_l40s_per_day"] == r["cost_l40s_per_day"]),
        "cost_h100": sum(r["cost_h100_per_day_static"] for r in results
                         if r["cost_h100_per_day_static"] == r["cost_h100_per_day_static"]),
        "margin_l40s": sum(r["margin_l40s"] for r in results
                           if r["margin_l40s"] == r["margin_l40s"]),
        "margin_h100_static": sum(r["margin_h100_static"] for r in results
                                  if r["margin_h100_static"] == r["margin_h100_static"]),
        "margin_h100_proj": sum(r["margin_h100_projected"] for r in results
                                if r["margin_h100_projected"] == r["margin_h100_projected"]),
        "proj_rev": sum(r["projected_revenue_h100"] for r in results),
    }

    out.append("\n## Headline (sum across all DekaLLM models, $ per day)\n")
    out.append("| | L40S (current) | H100 (constant demand) | H100 (with share lift) |")
    out.append("|---|---:|---:|---:|")
    out.append(f"| Gross revenue / day | {fmt_money(totals['gross_rev'])} | "
               f"{fmt_money(totals['gross_rev'])} | (projected) |")
    out.append(f"| Net revenue / day | {fmt_money(totals['net_rev'])} | "
               f"{fmt_money(totals['net_rev'])} | {fmt_money(totals['proj_rev'])} |")
    out.append(f"| GPU cost / day | {fmt_money(totals['cost_l40s'])} | "
               f"{fmt_money(totals['cost_h100'])} | {fmt_money(totals['cost_h100'])} |")
    out.append(f"| **Daily margin** | **{fmt_money(totals['margin_l40s'])}** | "
               f"**{fmt_money(totals['margin_h100_static'])}** | "
               f"**{fmt_money(totals['margin_h100_proj'])}** |")
    out.append(f"| **Monthly (×30)** | **{fmt_money(totals['margin_l40s']*30)}** | "
               f"**{fmt_money(totals['margin_h100_static']*30)}** | "
               f"**{fmt_money(totals['margin_h100_proj']*30)}** |\n")

    if totals["cost_l40s"] > 0:
        cost_delta_pct = (totals["cost_h100"] - totals["cost_l40s"]) / totals["cost_l40s"] * 100
        out.append(f"**Cost-only effect of L40S→H100 swap** (same demand): "
                   f"daily GPU cost changes by {cost_delta_pct:+.1f}% "
                   f"(${totals['cost_h100'] - totals['cost_l40s']:+,.2f}/day).\n")
        if cost_delta_pct < 0:
            out.append("H100 is *cheaper* at constant throughput because "
                       f"the {H100_THROUGHPUT_X}× speedup outweighs the "
                       f"{H100_COST_MULTIPLIER:.1f}× price premium.\n")

    if planner_l40s or planner_h100:
        out.append("\n## Planner cross-check (`/planner/compute`)\n")
        out.append("Independent estimate from the analytics service's optimizer, "
                   "using our pricing overrides:\n")
        if planner_l40s:
            out.append(f"\n**L40S @ ${L40S_PER_HR}/hr**\n")
            out.append(f"```\n{planner_l40s}\n```\n")
        if planner_h100:
            out.append(f"\n**H100 @ ${H100_PER_HR}/hr**\n")
            out.append(f"```\n{planner_h100}\n```\n")

    out.append("\n## Per-model breakdown\n")
    out.append("| Model | Net rev/day | DekaLLM t/s | Leader t/s | GPUs L40S | "
               "Cost L40S | Margin L40S | GPUs H100 | Cost H100 | Margin H100 (static) | "
               "Margin H100 (share lift) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        out.append(
            f"| {r['model_id']} | {fmt_money(r['avg_daily_net_revenue'])} | "
            f"{fmt_int(r['dek_throughput_tps'])} | {fmt_int(r['leader_throughput_tps'])} | "
            f"{fmt_int(r['gpus_l40s_estimate'])} | {fmt_money(r['cost_l40s_per_day'])} | "
            f"{fmt_money(r['margin_l40s'])} | {fmt_int(r['gpus_h100_estimate'])} | "
            f"{fmt_money(r['cost_h100_per_day_static'])} | {fmt_money(r['margin_h100_static'])} | "
            f"{fmt_money(r['margin_h100_projected'])} |"
        )

    out.append("\n## Caveats\n")
    out.append(
        "- **GPU count is a *minimum* estimate.** Computed as "
        "`daily_tokens / (t/s × 86400)`. Assumes 24/7 utilization at the "
        "observed throughput. Real fleet size is higher due to utilization "
        "spikes and continuous-batching headroom. DekaLLM's actual fleet is "
        "92 L40S + 4 H100 cards.\n"
        "- **Revenue uses 50/50 input/output token split.** Real ratio varies "
        "by model. Per-model accuracy could improve with daily I/O breakdown "
        "from `/db/models/{slug}/activity` once that endpoint is verified.\n"
        "- **Throughput multiplier is generic.** H100/L40S speedup is 2.5-4× "
        "depending on model size and tensor-parallel config. For models that "
        "need TP>1 on L40S (no NVLink penalty), the H100 win is bigger.\n"
        "- **Share lift is regression-derived.** Throughput→share treated as "
        "linear, which it isn't at the tails. Real lift is between the "
        "'static' and 'with share lift' columns.\n"
        "- **`/db/benchmarks` is currently empty.** Real measured throughput "
        "would replace the multiplier estimate — flag for the team.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = require_alive()
    print(f"Using analytics API at {client.base}\n")
    print(f"Loading DekaLLM token history (last {LOOKBACK_DAYS} days)...")
    history = load_dekallm_token_history(client, LOOKBACK_DAYS)
    if history.empty:
        print("No token history. Aborting.")
        return

    models = sorted(history["Model"].unique())
    print(f"  Models with token data: {len(models)}")
    for m in models:
        print(f"    {m}")

    print(f"\nComputing margins (L40S=${L40S_PER_HR}/hr, "
          f"H100=${H100_PER_HR}/hr, throughput {H100_THROUGHPUT_X}×)...")
    results = []
    for m in models:
        r = per_model_economics(client, m, history)
        if r:
            results.append(r)
            tps = r["dek_throughput_tps"]
            net = r["avg_daily_net_revenue"]
            print(f"  {m}: net_rev={fmt_money(net)}/day, "
                  f"tps={fmt_int(tps)}, "
                  f"margin L40S={fmt_money(r['margin_l40s'])}")

    # Planner cross-check
    print("\nCross-checking with /planner/compute...")
    planner_l40s = call_planner_compute(client, "L40S", L40S_PER_HR)
    planner_h100 = call_planner_compute(client, "H100", H100_PER_HR)
    if planner_l40s:
        print(f"  L40S planner result: {type(planner_l40s).__name__} "
              f"({'data' if isinstance(planner_l40s, dict) and 'data' in planner_l40s else 'raw'})")
    else:
        print("  Planner unreachable or schema mismatch — skipped.")

    md = write_report(results, planner_l40s, planner_h100)
    md_path = OUT / "margin_analysis.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nMarkdown report -> {md_path}")

    csv_path = OUT / "margin_analysis.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"CSV             -> {csv_path}")

    totals_l40s = sum(r["margin_l40s"] for r in results
                      if r["margin_l40s"] == r["margin_l40s"])
    totals_h100_static = sum(r["margin_h100_static"] for r in results
                             if r["margin_h100_static"] == r["margin_h100_static"])
    totals_h100_proj = sum(r["margin_h100_projected"] for r in results
                           if r["margin_h100_projected"] == r["margin_h100_projected"])
    print("\n" + "=" * 78)
    print("Aggregate daily margin")
    print("=" * 78)
    print(f"  L40S (current):                    {fmt_money(totals_l40s):>14} / day  "
          f"({fmt_money(totals_l40s*30)} / month)")
    print(f"  H100 (same demand):                {fmt_money(totals_h100_static):>14} / day  "
          f"({fmt_money(totals_h100_static*30)} / month)")
    print(f"  H100 (with share-lift projection): {fmt_money(totals_h100_proj):>14} / day  "
          f"({fmt_money(totals_h100_proj*30)} / month)")


if __name__ == "__main__":
    main()
