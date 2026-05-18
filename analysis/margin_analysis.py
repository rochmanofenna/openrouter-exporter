"""Per-model margin analysis for DekaLLM under different GPU configurations.

Compares the current L40S setup against a hypothetical H100 swap. For each
model DekaLLM serves, computes:

- Revenue/day: from CSV historical actuals (gross, then net after OR's 15% fee)
- GPU cost/day: assumed $/GPU/hr × estimated GPUs in service
- Current margin: net revenue − GPU cost
- Projected H100 margin: using a configurable throughput multiplier and
  the lever-sensitivity regression's throughput slope to estimate share lift

The analysis is sensitive to several assumptions you should set explicitly
(see the CONFIG block). Defaults are 2026-era industry rentals; replace with
DekaLLM's actual contracted prices for the real number.

Usage:
    python analysis/margin_analysis.py

Env knobs (all optional):
    L40S_PER_HR              default 1.40   (USD per GPU per hour)
    H100_PER_HR              default 3.00   (USD per GPU per hour)
    H100_THROUGHPUT_X        default 3.0    (H100 throughput multiplier vs L40S)
    OR_FEE_PCT               default 0.15   (OpenRouter platform fee fraction)
    SHARE_SLOPE              default 0.758  (from lever_sensitivity.py throughput slope)

Outputs:
    analysis/out/margin_analysis.md       markdown report for the meeting
    analysis/out/margin_analysis.csv      per-model table
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Config — change these to match DekaLLM's actual contracts
# ---------------------------------------------------------------------------
L40S_PER_HR = float(os.environ.get("L40S_PER_HR", "1.40"))     # ≈ baseline cloud L40S rental
H100_PER_HR = float(os.environ.get("H100_PER_HR", "3.00"))     # ≈ H100 80GB rental
H100_THROUGHPUT_X = float(os.environ.get("H100_THROUGHPUT_X", "3.0"))   # 2.5-3.5x typical for LLM inference
OR_FEE_PCT = float(os.environ.get("OR_FEE_PCT", "0.15"))       # OpenRouter takes 15%
# Throughput slope from lever_sensitivity.py (share change per unit gap closed).
# Conservative: regression said +0.758 but the projection is widely held to be
# overstated due to linear extrapolation. Use a discounted slope as default.
SHARE_SLOPE = float(os.environ.get("SHARE_SLOPE", "0.40"))     # ~half of regression slope, conservative

L40S_PER_DAY = L40S_PER_HR * 24
H100_PER_DAY = H100_PER_HR * 24
H100_COST_MULTIPLIER = H100_PER_HR / L40S_PER_HR
# Net cost ratio: if you go faster by X but each GPU costs more by Y, GPUs needed
# drops by X but each costs Y more. Net daily cost change factor:
H100_DAILY_COST_FACTOR = H100_COST_MULTIPLIER / H100_THROUGHPUT_X

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
DEKALLM_SLUG = "dekallm"
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load DekaLLM revenue history from CSVs
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    with open(path) as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    df = pd.read_csv(path, skiprows=header_idx)
    df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Requests", "Input Tokens", "Output Tokens", "Cached Tokens",
              "Total Tokens", "Input Cost", "Output Cost", "Total Cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_dekallm_history(lookback_days: int = 7) -> pd.DataFrame:
    """Return DekaLLM daily revenue table for the last `lookback_days` complete days."""
    paths = sorted(glob.glob(str(ROOT / "dekallm_daily_report_2026_*.csv")))
    if not paths:
        raise FileNotFoundError("No dekallm_daily_report_*.csv found in repo root")
    frames = [load_csv(Path(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date", "Model"])
    df = df.sort_values(["Model", "Date"]).reset_index(drop=True)

    # Drop today's partial day
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df["Date"] < today]

    # Keep last lookback_days complete days
    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(days=lookback_days - 1)
    df = df[df["Date"] >= cutoff]
    return df


# ---------------------------------------------------------------------------
# Throughput per model from Prometheus
# ---------------------------------------------------------------------------
def prom_query(query: str) -> list[dict]:
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")
    return payload["data"]["result"]


_DATE_SUFFIX = re.compile(r"-\d{8}$")


def get_dekallm_throughput(model_permaslug: str) -> float | None:
    """Get DekaLLM's p50 throughput (t/s) for a model, trying base slug if permaslug fails."""
    for candidate in (model_permaslug, _DATE_SUFFIX.sub("", model_permaslug)):
        q = (f'max(openrouter_endpoint_throughput_tokens_per_second'
             f'{{model_id="{candidate}",provider_name="DekaLLM",quantile="p50"}})')
        res = prom_query(q)
        if res:
            return float(res[0]["value"][1])
    return None


def get_leader_throughput(model_permaslug: str) -> float | None:
    """Get the highest p50 throughput across any provider for this model."""
    for candidate in (model_permaslug, _DATE_SUFFIX.sub("", model_permaslug)):
        q = (f'max(openrouter_endpoint_throughput_tokens_per_second'
             f'{{model_id="{candidate}",quantile="p50"}})')
        res = prom_query(q)
        if res:
            return float(res[0]["value"][1])
    return None


# ---------------------------------------------------------------------------
# Per-model unit economics
# ---------------------------------------------------------------------------
def estimate_gpus_in_service(daily_tokens: float, t_per_sec: float) -> float:
    """How many GPUs (on average) it takes to serve this volume at this throughput.
    Assumes 24/7 utilization, single-replica per GPU. Real life has utilization
    fluctuations — the number is a lower-bound estimate of fleet size."""
    if t_per_sec <= 0:
        return float("nan")
    tokens_per_day_per_gpu = t_per_sec * 86400
    return daily_tokens / tokens_per_day_per_gpu


def per_model_economics(model_id: str, history: pd.DataFrame) -> dict | None:
    sub = history[history["Model"] == model_id]
    if sub.empty:
        return None
    avg_daily_tokens = float(sub["Total Tokens"].mean())
    avg_daily_input_tokens = float(sub["Input Tokens"].mean())
    avg_daily_output_tokens = float(sub["Output Tokens"].mean())
    avg_daily_gross_revenue = float(sub["Total Cost"].mean())  # this is what OR charges customers
    avg_daily_requests = float(sub["Requests"].mean())

    # DekaLLM's net revenue after OpenRouter fee
    avg_daily_net_revenue = avg_daily_gross_revenue * (1 - OR_FEE_PCT)

    # DekaLLM observed throughput
    dek_tps = get_dekallm_throughput(model_id)
    leader_tps = get_leader_throughput(model_id)
    throughput_gap = None
    if dek_tps is not None and leader_tps is not None and leader_tps > 0:
        throughput_gap = (dek_tps - leader_tps) / leader_tps  # negative if behind

    # Current L40S economics
    gpus_l40s = (estimate_gpus_in_service(avg_daily_tokens, dek_tps)
                 if dek_tps else float("nan"))
    cost_l40s_per_day = gpus_l40s * L40S_PER_DAY if gpus_l40s == gpus_l40s else float("nan")
    margin_l40s = avg_daily_net_revenue - cost_l40s_per_day if cost_l40s_per_day == cost_l40s_per_day else float("nan")

    # H100 swap: throughput rises by H100_THROUGHPUT_X. GPUs needed at same volume falls.
    # If demand stays the same, the cost change is the H100_DAILY_COST_FACTOR.
    gpus_h100 = gpus_l40s / H100_THROUGHPUT_X if gpus_l40s == gpus_l40s else float("nan")
    cost_h100_per_day_static = gpus_h100 * H100_PER_DAY if gpus_h100 == gpus_h100 else float("nan")
    margin_h100_static = avg_daily_net_revenue - cost_h100_per_day_static if cost_h100_per_day_static == cost_h100_per_day_static else float("nan")

    # Project share lift if H100 closes part of the throughput gap.
    # New throughput = current × H100_THROUGHPUT_X. New gap = (new_tput - leader)/leader.
    projected_revenue_h100 = avg_daily_net_revenue  # baseline
    share_lift_pp = 0.0
    if (dek_tps is not None and leader_tps is not None and throughput_gap is not None):
        new_dek_tps = dek_tps * H100_THROUGHPUT_X
        new_gap = (new_dek_tps - leader_tps) / leader_tps
        gap_closed = throughput_gap - new_gap  # how much of the (negative) gap was reduced
        # share lift per regression: slope * (-gap_closed)
        share_lift = -gap_closed * SHARE_SLOPE  # positive expected if dek was behind
        share_lift_pp = share_lift * 100
        # if share lifts by X pp, revenue scales by (current_share + share_lift) / current_share
        # but we don't have share directly here — approximate with revenue scaling.
        # The simpler model: revenue scales proportionally to share. So:
        # if share_lift is +5pp and current share is 10pp, revenue scales by 1.5x.
        # We don't have current share here, so we just estimate the additional revenue from
        # share_lift relative to the model's TOTAL market revenue (= demand × price).
        # Approximation: assume current dekallm revenue corresponds to current share fraction;
        # share lift of share_lift_pp gives additional revenue ≈
        #   avg_daily_gross_revenue * (share_lift / current_share)
        # but again share isn't here. Punt: just flag the share lift in pp.
        # Conservative: project up to a 2x revenue multiplier capped.
        if share_lift > 0:
            multiplier = min(1.0 + share_lift * 3.0, 2.0)  # rough scaling; capped at 2x
            projected_revenue_h100 = avg_daily_net_revenue * multiplier

    margin_h100_projected = projected_revenue_h100 - cost_h100_per_day_static if cost_h100_per_day_static == cost_h100_per_day_static else float("nan")

    return {
        "model_id": model_id,
        "avg_daily_tokens": avg_daily_tokens,
        "avg_daily_input_tokens": avg_daily_input_tokens,
        "avg_daily_output_tokens": avg_daily_output_tokens,
        "avg_daily_requests": avg_daily_requests,
        "avg_daily_gross_revenue": avg_daily_gross_revenue,
        "avg_daily_net_revenue": avg_daily_net_revenue,
        "dek_throughput_tps": dek_tps,
        "leader_throughput_tps": leader_tps,
        "throughput_gap": throughput_gap,
        "gpus_l40s_estimate": gpus_l40s,
        "cost_l40s_per_day": cost_l40s_per_day,
        "margin_l40s": margin_l40s,
        "gpus_h100_estimate": gpus_h100,
        "cost_h100_per_day_static": cost_h100_per_day_static,
        "margin_h100_static": margin_h100_static,
        "share_lift_pp_h100": share_lift_pp,
        "projected_revenue_h100": projected_revenue_h100,
        "margin_h100_projected": margin_h100_projected,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt_money(v) -> str:
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
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


def write_report(results: list[dict]) -> str:
    out = []
    out.append("# DekaLLM Margin Analysis — L40S vs H100\n")
    out.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")

    out.append("\n## Assumptions used\n")
    out.append(f"- L40S cost: **${L40S_PER_HR:.2f} / GPU / hr** (${L40S_PER_DAY:.2f} / GPU / day)")
    out.append(f"- H100 cost: **${H100_PER_HR:.2f} / GPU / hr** (${H100_PER_DAY:.2f} / GPU / day)")
    out.append(f"- H100 throughput multiplier: **{H100_THROUGHPUT_X}×** (typical for LLM inference)")
    out.append(f"- OpenRouter fee: **{OR_FEE_PCT*100:.0f}%** of gross revenue")
    out.append(f"- Throughput share-slope: **{SHARE_SLOPE}** "
               f"(from lever_sensitivity.py, conservatively discounted)\n")

    out.append("Override any of these with env vars before re-running. "
               "Defaults are 2026-era industry rentals; replace with DekaLLM's "
               "actual contracted prices for the real number.\n")

    # Aggregate totals
    totals = {
        "gross_rev": sum(r["avg_daily_gross_revenue"] for r in results),
        "net_rev":   sum(r["avg_daily_net_revenue"] for r in results),
        "cost_l40s": sum(r["cost_l40s_per_day"] for r in results if r["cost_l40s_per_day"] == r["cost_l40s_per_day"]),
        "cost_h100": sum(r["cost_h100_per_day_static"] for r in results if r["cost_h100_per_day_static"] == r["cost_h100_per_day_static"]),
        "margin_l40s": sum(r["margin_l40s"] for r in results if r["margin_l40s"] == r["margin_l40s"]),
        "margin_h100_static": sum(r["margin_h100_static"] for r in results if r["margin_h100_static"] == r["margin_h100_static"]),
        "margin_h100_proj": sum(r["margin_h100_projected"] for r in results if r["margin_h100_projected"] == r["margin_h100_projected"]),
    }

    out.append("\n## Headline — daily numbers, summed across all DekaLLM models\n")
    out.append(f"| | L40S (current) | H100 (constant demand) | H100 (with share lift) |")
    out.append(f"|---|---:|---:|---:|")
    out.append(f"| Gross revenue / day | {fmt_money(totals['gross_rev'])} | {fmt_money(totals['gross_rev'])} | (model-projected, see below) |")
    out.append(f"| Net revenue / day (after OR fee) | {fmt_money(totals['net_rev'])} | {fmt_money(totals['net_rev'])} | {fmt_money(sum(r['projected_revenue_h100'] for r in results))} |")
    out.append(f"| GPU cost / day | {fmt_money(totals['cost_l40s'])} | {fmt_money(totals['cost_h100'])} | {fmt_money(totals['cost_h100'])} |")
    out.append(f"| **Daily margin** | **{fmt_money(totals['margin_l40s'])}** | **{fmt_money(totals['margin_h100_static'])}** | **{fmt_money(totals['margin_h100_proj'])}** |")
    out.append(f"| **Monthly margin (×30)** | **{fmt_money(totals['margin_l40s']*30)}** | **{fmt_money(totals['margin_h100_static']*30)}** | **{fmt_money(totals['margin_h100_proj']*30)}** |\n")

    # Net cost-only effect of swap (constant demand)
    if totals["cost_l40s"] > 0:
        cost_change_pct = (totals["cost_h100"] - totals["cost_l40s"]) / totals["cost_l40s"] * 100
        out.append(f"**Cost-only effect of swap** (same demand, faster GPUs): "
                   f"daily GPU cost changes by {cost_change_pct:+.1f}% "
                   f"(${totals['cost_h100'] - totals['cost_l40s']:+.2f}/day).\n")
        if cost_change_pct < 0:
            out.append("H100 is *cheaper* to operate at the same throughput because "
                       "the throughput multiplier outweighs the per-GPU price premium. "
                       "Even before any share-growth benefit.\n")
        else:
            out.append("H100 is *more expensive* to operate at the same throughput. "
                       "Only justifiable if the share lift from faster routing pays back "
                       "the cost premium.\n")

    out.append("\n## Per-model breakdown\n")
    out.append("| Model | Net rev/day | DekaLLM t/s | Leader t/s | GPUs (L40S) | Cost L40S | Margin L40S | GPUs (H100) | Cost H100 | Margin H100 (static) | Margin H100 (with share lift) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        out.append(
            f"| {r['model_id']} | "
            f"{fmt_money(r['avg_daily_net_revenue'])} | "
            f"{fmt_int(r['dek_throughput_tps'])} | "
            f"{fmt_int(r['leader_throughput_tps'])} | "
            f"{fmt_int(r['gpus_l40s_estimate'])} | "
            f"{fmt_money(r['cost_l40s_per_day'])} | "
            f"{fmt_money(r['margin_l40s'])} | "
            f"{fmt_int(r['gpus_h100_estimate'])} | "
            f"{fmt_money(r['cost_h100_per_day_static'])} | "
            f"{fmt_money(r['margin_h100_static'])} | "
            f"{fmt_money(r['margin_h100_projected'])} |"
        )

    out.append("\n## Caveats\n")
    out.append("- **GPU count is an estimate.** Computed as `daily_tokens / (t/s × 86400)`. "
               "Assumes 24/7 utilization; real fleet size is higher because of utilization spikes.\n")
    out.append("- **Throughput multiplier is generic.** H100-vs-L40S speedup varies by model size, "
               "quantization, and inference engine. 2.5-3.5× is typical for LLM inference; for very large "
               "models (>70B), the speedup can be 4-5× because H100's larger VRAM removes tensor-parallel "
               "overhead.\n")
    out.append("- **Share lift is regression-derived.** Treats throughput→share as linear, which it isn't at "
               "the tails. Real lift is between the 'static' and 'projected' columns.\n")
    out.append("- **Doesn't include amortized capex / data-center costs / engineering time** to migrate. "
               "Add those before committing.\n")
    out.append("- **Numbers come from OpenRouter dashboard data only.** Any private/enterprise customers "
               "not routed through OpenRouter are not in this analysis.\n")
    return "\n".join(out)


def main() -> None:
    print(f"Loading DekaLLM CSV history (last 7 complete days)...")
    history = load_dekallm_history(lookback_days=7)
    if history.empty:
        print("No CSV history found — make sure dekallm_daily_report_*.csv files exist.")
        return

    models = sorted(history["Model"].unique())
    print(f"Models with revenue data: {len(models)}")
    for m in models:
        print(f"  {m}")

    print(f"\nComputing margins under L40S (${L40S_PER_HR:.2f}/hr) vs H100 (${H100_PER_HR:.2f}/hr, {H100_THROUGHPUT_X}x faster)...")
    results = []
    for m in models:
        r = per_model_economics(m, history)
        if r:
            results.append(r)
            tps = r["dek_throughput_tps"]
            print(f"  {m}: net_rev=${r['avg_daily_net_revenue']:.2f}/day, "
                  f"tps={tps if tps else 'unknown'}, "
                  f"L40S margin=${r['margin_l40s']:.2f}, "
                  f"H100 margin (static)=${r['margin_h100_static']:.2f}")

    # Markdown report
    md = write_report(results)
    md_path = OUT / "margin_analysis.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nMarkdown report -> {md_path}")

    # CSV
    csv_path = OUT / "margin_analysis.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"CSV              -> {csv_path}")

    # Console summary
    totals_l40s = sum(r["margin_l40s"] for r in results if r["margin_l40s"] == r["margin_l40s"])
    totals_h100_static = sum(r["margin_h100_static"] for r in results if r["margin_h100_static"] == r["margin_h100_static"])
    totals_h100_proj = sum(r["margin_h100_projected"] for r in results if r["margin_h100_projected"] == r["margin_h100_projected"])

    print("\n" + "=" * 78)
    print("Aggregate daily margin (across all DekaLLM models)")
    print("=" * 78)
    print(f"  L40S (current):                  ${totals_l40s:>10,.2f}/day  (${totals_l40s*30:,.2f}/month)")
    print(f"  H100 (same demand, faster GPUs): ${totals_h100_static:>10,.2f}/day  (${totals_h100_static*30:,.2f}/month)")
    print(f"  H100 (with share-lift projection): ${totals_h100_proj:>10,.2f}/day  (${totals_h100_proj*30:,.2f}/month)")
    if totals_l40s != 0:
        delta_static_pct = (totals_h100_static - totals_l40s) / abs(totals_l40s) * 100
        delta_proj_pct = (totals_h100_proj - totals_l40s) / abs(totals_l40s) * 100
        print(f"\n  H100 vs L40S (static):           {delta_static_pct:+.1f}% margin change")
        print(f"  H100 vs L40S (with share lift):  {delta_proj_pct:+.1f}% margin change")


if __name__ == "__main__":
    main()
