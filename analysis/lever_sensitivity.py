"""Lever sensitivity — empirically estimate how much each routing lever moves
OpenRouter market share, so DekaLLM can prioritize where to invest.

For each shared model we observe (provider, share, price, throughput, latency,
uptime). Pooling all (model, provider) observations, we regress share onto
each lever and report:

- Marginal slope per lever (the simplest, most communicable number)
- R^2 per lever (how much variation in share that lever explains)
- A joint regression with all levers (partial effects)
- For DekaLLM specifically: "if you closed your gap on lever X to the
  cross-model average leader, predicted share lift is Y percentage points"

Caveats baked into the report:
- Cross-sectional only — we can't separate "DekaLLM is small because it's
  slow" from "DekaLLM is small for other reasons and also happens to be slow."
- Limited sample size (a handful of models, 2-4 providers each).
- Lever values are mostly static over our observation window, so we use a
  pooled cross-sectional snapshot rather than time-series.

Outputs:
- analysis/out/lever_sensitivity.md      — markdown report for the meeting
- analysis/out/lever_sensitivity.csv     — raw (model, provider, lever, share) table
- analysis/out/lever_sensitivity_*.png   — scatter plot per lever
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
PROVIDERS_SLUGS = [s.strip() for s in os.environ.get(
    "PROVIDERS", "dekallm,deepinfra,fireworks,together"
).split(",")]
DEKALLM_SLUG = "dekallm"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Endpoint metrics use display names; chart metric uses slugs.
PROVIDER_NAME_HINTS = {
    "dekallm":    ["DekaLLM", "Dekallm", "dekallm"],
    "deepinfra":  ["DeepInfra", "Deepinfra", "deepinfra"],
    "fireworks":  ["Fireworks", "Fireworks AI", "fireworks"],
    "together":   ["Together", "Together AI", "together"],
}

_DATE_SUFFIX = re.compile(r"-\d{8}$")


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
# Provider name resolution (same as competitive_position.py)
# ----------------------------------------------------------------------------
def resolve_provider_names() -> dict[str, str]:
    res = prom_query("group by (provider_name) (openrouter_model_input_price_dollars_per_million_tokens)")
    discovered = [r["metric"]["provider_name"] for r in res]
    mapping = {}
    for slug in PROVIDERS_SLUGS:
        hints = PROVIDER_NAME_HINTS.get(slug, [slug])
        found = None
        for hint in hints:
            for d in discovered:
                if d == hint or d.lower() == hint.lower() or hint.lower() in d.lower():
                    found = d
                    break
            if found:
                break
        mapping[slug] = found
    return mapping


def candidate_endpoint_model_ids(permaslug: str) -> list[str]:
    candidates = [permaslug]
    stripped = _DATE_SUFFIX.sub("", permaslug)
    if stripped != permaslug:
        candidates.append(stripped)
    return candidates


# ----------------------------------------------------------------------------
# Build the pooled (model, provider) snapshot
# ----------------------------------------------------------------------------
def get_dekallm_models() -> list[str]:
    res = prom_query(
        f'count by (model_id) (openrouter_provider_tokens_daily{{provider="{DEKALLM_SLUG}"}})'
    )
    return sorted({r["metric"]["model_id"] for r in res})


def get_levers_for_model(model_id: str, name_map: dict[str, str]) -> dict[str, dict[str, float]]:
    """Return {provider_slug: {lever_name: value}}. Uses the slug-stripping
    fallback for endpoint metrics."""
    queries = {
        "input_price":  ("min", "openrouter_model_input_price_dollars_per_million_tokens", ""),
        "output_price": ("min", "openrouter_model_output_price_dollars_per_million_tokens", ""),
        "throughput":   ("max", "openrouter_endpoint_throughput_tokens_per_second", ',quantile="p50"'),
        "latency_ms":   ("min", "openrouter_endpoint_latency_milliseconds", ',quantile="p50"'),
        "uptime_1d":    ("max", "openrouter_endpoint_uptime_percentage_last_1d", ""),
    }
    name_to_slug = {v: k for k, v in name_map.items() if v}

    resolved = None
    for candidate in candidate_endpoint_model_ids(model_id):
        probe = prom_query(
            f'openrouter_model_input_price_dollars_per_million_tokens{{model_id="{candidate}"}}'
        )
        if probe:
            resolved = candidate
            break
    if resolved is None:
        return {}

    out: dict[str, dict[str, float]] = {}
    for col, (agg, base, qfilter) in queries.items():
        q = f'{agg} by (provider_name) ({base}{{model_id="{resolved}"{qfilter}}})'
        try:
            res = prom_query(q)
        except Exception:
            res = []
        for r in res:
            slug = name_to_slug.get(r["metric"].get("provider_name", ""))
            if slug is None:
                continue
            out.setdefault(slug, {})[col] = float(r["value"][1])
    return out


def get_clean_share(model_id: str, lookback_days: int = 7) -> dict[str, float]:
    """Compute share per provider using only days where ALL providers serving
    this model had data. Returns provider_slug -> share (0.0-1.0), averaged
    across the common days."""
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=lookback_days + 1)
    q = f'openrouter_provider_tokens_daily{{model_id="{model_id}"}}'
    series = prom_range(q, start, end, 3600)

    # (provider, date) -> max(running total) observed during the window
    per_pd: dict[tuple[str, str], float] = {}
    for s in series:
        provider = s["metric"].get("provider", "")
        date = s["metric"].get("date", "")
        if not date or not provider:
            continue
        v = max(float(v[1]) for v in s["values"])
        per_pd[(provider, date)] = max(v, per_pd.get((provider, date), 0))

    if not per_pd:
        return {}

    # Find providers that have any data in window
    providers_in_window = {p for (p, _) in per_pd.keys()}
    # Find dates where ALL of those providers had data
    dates_per_provider: dict[str, set[str]] = {p: set() for p in providers_in_window}
    for (p, d), _v in per_pd.items():
        dates_per_provider[p].add(d)
    if not dates_per_provider:
        return {}
    common_dates = set.intersection(*dates_per_provider.values())
    today = datetime.now(timezone.utc).date().isoformat()
    common_dates = sorted([d for d in common_dates if d < today])  # drop in-progress today
    if not common_dates:
        return {}

    # For each common date, compute per-provider share of total
    shares_by_provider: dict[str, list[float]] = {p: [] for p in providers_in_window}
    for d in common_dates:
        total = sum(per_pd.get((p, d), 0.0) for p in providers_in_window)
        if total <= 0:
            continue
        for p in providers_in_window:
            shares_by_provider[p].append(per_pd.get((p, d), 0.0) / total)
    return {p: float(np.mean(vs)) if vs else 0.0 for p, vs in shares_by_provider.items()}


# ----------------------------------------------------------------------------
# Regression helpers (no scipy/statsmodels needed)
# ----------------------------------------------------------------------------
def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple ordinary least squares y = a + b*x. Returns slope, intercept,
    r_squared, n, and standard error of slope."""
    n = len(x)
    if n < 3:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "se_slope": float("nan")}
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = ((x - x_mean) ** 2).sum()
    Sxy = ((x - x_mean) * (y - y_mean)).sum()
    if Sxx == 0:
        return {"n": n, "slope": 0.0, "intercept": y_mean, "r2": 0.0,
                "se_slope": float("nan")}
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse = ss_res / (n - 2) if n > 2 else float("nan")
    se_slope = (mse / Sxx) ** 0.5 if mse == mse and Sxx > 0 else float("nan")
    return {"n": n, "slope": slope, "intercept": intercept, "r2": r2, "se_slope": se_slope}


# ----------------------------------------------------------------------------
# Build the snapshot DataFrame
# ----------------------------------------------------------------------------
def build_snapshot(name_map: dict[str, str]) -> pd.DataFrame:
    models = get_dekallm_models()
    rows = []
    for model_id in models:
        levers = get_levers_for_model(model_id, name_map)
        share = get_clean_share(model_id)
        if not levers and not share:
            continue
        providers = sorted(set(levers.keys()) | set(share.keys()))
        for p in providers:
            row = {
                "model_id": model_id,
                "provider": p,
                "share": share.get(p, np.nan),
            }
            for lever in ("input_price", "output_price", "throughput", "latency_ms", "uptime_1d"):
                row[lever] = levers.get(p, {}).get(lever, np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)

    # Per-model gap vs leader on each lever
    HIGHER_IS_BETTER = {"throughput": True, "uptime_1d": True,
                        "input_price": False, "output_price": False, "latency_ms": False}
    for lever, higher_better in HIGHER_IS_BETTER.items():
        gaps = []
        for model_id, grp in df.groupby("model_id"):
            vals = grp[lever]
            valid = vals.dropna()
            if len(valid) == 0:
                continue
            leader = valid.max() if higher_better else valid.min()
            if leader == 0 or pd.isna(leader):
                continue
            for idx in grp.index:
                v = df.loc[idx, lever]
                if pd.isna(v):
                    continue
                if higher_better:
                    # negative gap = behind leader
                    gap = (v - leader) / leader  # 0 for leader, negative for laggards
                else:
                    # positive gap = more expensive/slower than leader
                    gap = (v - leader) / leader  # 0 for leader, positive for laggards
                df.loc[idx, f"{lever}_gap"] = gap
    return df


# ----------------------------------------------------------------------------
# Marginal regressions per lever
# ----------------------------------------------------------------------------
LEVER_LABELS = {
    "input_price":  ("Input price", "lower-better"),
    "output_price": ("Output price", "lower-better"),
    "throughput":   ("Throughput", "higher-better"),
    "latency_ms":   ("Latency", "lower-better"),
    "uptime_1d":    ("Uptime", "higher-better"),
}


def marginal_regressions(df: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for lever in LEVER_LABELS:
        gap_col = f"{lever}_gap"
        sub = df.dropna(subset=[gap_col, "share"])
        if len(sub) < 3:
            out[lever] = {"n": len(sub), "slope": np.nan, "r2": np.nan, "se_slope": np.nan}
            continue
        x = sub[gap_col].to_numpy()
        y = sub["share"].to_numpy()
        out[lever] = ols(x, y)
    return out


def joint_regression(df: pd.DataFrame) -> dict | None:
    """Multivariate OLS: share ~ all lever gaps simultaneously. Returns dict
    with coefs per lever, intercept, R^2, n. Uses numpy pinv for stability."""
    cols = [f"{l}_gap" for l in LEVER_LABELS]
    sub = df.dropna(subset=cols + ["share"])
    if len(sub) < len(cols) + 2:
        return None
    X = sub[cols].to_numpy()
    y = sub["share"].to_numpy()
    X1 = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X1, y, rcond=None)
    intercept = float(coefs[0])
    betas = {col: float(c) for col, c in zip(cols, coefs[1:])}
    y_pred = X1 @ coefs
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"n": len(sub), "intercept": intercept, "betas": betas, "r2": r2}


# ----------------------------------------------------------------------------
# DekaLLM-specific projection
# ----------------------------------------------------------------------------
def dekallm_share_projection(df: pd.DataFrame, marginals: dict) -> list[dict]:
    """For each lever: estimate DekaLLM's potential share lift if it closes the
    average gap to leader, holding others constant. Uses the marginal slope
    (not the joint coefficient) for clarity."""
    dek = df[df["provider"] == DEKALLM_SLUG]
    out = []
    for lever, label in LEVER_LABELS.items():
        gap_col = f"{lever}_gap"
        avg_gap = dek[gap_col].dropna().mean()
        if pd.isna(avg_gap):
            continue
        slope = marginals[lever]["slope"]
        if pd.isna(slope):
            continue
        # If DekaLLM closes the gap, new gap = 0. Share lift = -avg_gap * slope.
        # (slope is share change per 1 unit of gap; gap going from avg_gap to 0
        # is a change of -avg_gap; share change is slope * (-avg_gap).)
        projected_lift = -avg_gap * slope  # in absolute share units (0.0-1.0)
        out.append({
            "lever": lever,
            "label": label[0],
            "current_dekallm_avg_gap_pct": avg_gap * 100,
            "slope_share_per_unit_gap": slope,
            "projected_share_lift_pp": projected_lift * 100,  # percentage points
        })
    out.sort(key=lambda r: -r["projected_share_lift_pp"])
    return out


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(df: pd.DataFrame, marginals: dict, joint: dict | None,
           projection: list[dict]) -> str:
    out = []
    out.append("# DekaLLM Lever Sensitivity Analysis\n")
    out.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    out.append("\nQuestion: which routing lever, if improved, would yield the "
               "biggest market-share lift for DekaLLM?\n")

    out.append("\n## Method\n")
    out.append(
        "- For each (model, provider) in the OpenRouter sample, observe: share "
        "(7-day average over fully-overlapping days only), input/output price, "
        "throughput (p50), latency (p50), uptime (1d).\n"
        "- For each lever, compute the provider's gap to the leader on that "
        "model: signed normalized gap = (value - leader_value) / leader_value.\n"
        "- Regress share onto each lever's gap (marginal OLS, all models pooled).\n"
        "- Also fit a joint OLS with all four lever gaps to estimate partial "
        "effects.\n"
    )

    # Marginal table
    out.append("\n## Marginal sensitivity (one lever at a time)\n")
    out.append("Each row: how much share *changes* per 1-unit change in the "
               "lever's gap-vs-leader, pooled across all (model, provider) "
               "observations.\n\n")
    out.append("| Lever | n | Slope (share per unit gap) | R² | SE of slope | Interpretation |")
    out.append("|---|---:|---:|---:|---:|---|")
    for lever, label in LEVER_LABELS.items():
        m = marginals[lever]
        slope = m["slope"]
        if pd.isna(slope):
            out.append(f"| {label[0]} | {m['n']} | — | — | — | (no fit) |")
            continue
        # Convert slope into "% share change per 10% lever change" for intuition
        per_10pct = slope * 0.10 * 100  # share is 0-1; gap is fractional; result in pp
        sign = "+" if per_10pct > 0 else ""
        interp = (
            f"A 10% improvement on this lever (closing 10% of the gap) is "
            f"associated with {sign}{per_10pct:.2f} pp share change."
        )
        out.append(f"| {label[0]} | {m['n']} | {slope:+.3f} | {m['r2']:.3f} | "
                   f"{m['se_slope']:.3f} | {interp} |")

    out.append("\nNote: slope sign depends on lever direction. For "
               "*lower-better* levers (price, latency), the gap is positive "
               "when worse than leader, so a positive slope means worse "
               "providers have more share (unlikely) and a negative slope "
               "means being more expensive/slower costs share. For "
               "*higher-better* levers (throughput, uptime), the gap is "
               "negative when worse than leader, so a positive slope means "
               "being faster/more reliable gains share.\n")

    # Joint regression
    out.append("\n## Joint regression (all levers simultaneously)\n")
    if joint is None:
        out.append("_Insufficient data for joint fit (need at least 7 complete "
                   "observations across all levers)._\n")
    else:
        out.append(f"n = {joint['n']}, R² = {joint['r2']:.3f}\n\n")
        out.append("Each coefficient is the partial effect of that lever, "
                   "holding the others constant.\n\n")
        out.append("| Lever | Partial effect (share per unit gap) |")
        out.append("|---|---:|")
        for lever, label in LEVER_LABELS.items():
            beta = joint["betas"].get(f"{lever}_gap", float("nan"))
            out.append(f"| {label[0]} | {beta:+.3f} |")
        out.append(
            f"\nIntercept ≈ {joint['intercept']:.3f} (predicted share when a "
            f"provider is tied with the leader on every lever — i.e. average "
            f"share for a tied provider)."
        )

    # DekaLLM projection
    out.append("\n## DekaLLM share-lift projection per lever\n")
    out.append(
        "If DekaLLM closes its **average current gap** to the leader on this "
        "lever (across all shared models), the marginal-regression-implied "
        "share lift in percentage points is:\n\n"
    )
    out.append("| Lever | DekaLLM avg current gap | Projected share lift (pp) | Action |")
    out.append("|---|---:|---:|---|")
    for r in projection:
        if r["projected_share_lift_pp"] > 0:
            action = f"close gap → +{r['projected_share_lift_pp']:.2f} pp share"
        elif r["projected_share_lift_pp"] < 0:
            action = f"already favorable — moving further would lose share"
        else:
            action = "neutral"
        out.append(f"| {r['label']} | {r['current_dekallm_avg_gap_pct']:+.1f}% "
                   f"| {r['projected_share_lift_pp']:+.2f} pp | {action} |")

    # Caveats
    out.append("\n## Caveats\n")
    out.append(
        "- **Cross-sectional, not causal.** A provider being slow and small "
        "doesn't prove slowness causes smallness. Could be confounded by "
        "customer mix, marketing, or other unobservables.\n"
        "- **Limited sample.** Most models in the sample have only 2-4 "
        "providers. Estimates have wide confidence intervals.\n"
        "- **Levers are mostly static over our window.** We can't validate "
        "with within-provider over-time variation yet. A V2 with 3+ months "
        "of data could do panel regression with provider fixed effects.\n"
        "- **OpenRouter's actual routing algorithm is private.** We infer "
        "lever effects from behavior; the true causal model is opaque.\n"
        "- **'Closing the gap' assumes everything else stays equal.** In "
        "reality, lowering price hurts margin even if it gains share.\n"
    )

    out.append("\n## Snapshot used for fitting\n")
    out.append("Pooled (model, provider) table:\n\n")
    snap = df[["model_id", "provider", "share", "input_price", "output_price",
               "throughput", "latency_ms", "uptime_1d"]].copy()
    snap = snap.dropna(subset=["share"], how="all")
    # Manual markdown table (avoids tabulate dependency)
    out.append("| model_id | provider | share | input_price | output_price | throughput | latency_ms | uptime_1d |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in snap.iterrows():
        def fmt(v: float) -> str:
            if pd.isna(v):
                return "—"
            return f"{v:.3f}"
        out.append(
            f"| {r['model_id']} | {r['provider']} | {fmt(r['share'])} | "
            f"{fmt(r['input_price'])} | {fmt(r['output_price'])} | "
            f"{fmt(r['throughput'])} | {fmt(r['latency_ms'])} | {fmt(r['uptime_1d'])} |"
        )

    return "\n".join(out)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_scatter(df: pd.DataFrame, marginals: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(LEVER_LABELS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (lever, (label, direction)) in zip(axes, LEVER_LABELS.items()):
        gap_col = f"{lever}_gap"
        sub = df.dropna(subset=[gap_col, "share"])
        if len(sub) == 0:
            ax.set_title(f"{label}\n(no data)")
            continue
        for _, row in sub.iterrows():
            color = "red" if row["provider"] == DEKALLM_SLUG else "tab:blue"
            ax.scatter(row[gap_col] * 100, row["share"] * 100, color=color, s=70, alpha=0.8)
        # Draw fit line
        m = marginals[lever]
        if not pd.isna(m["slope"]):
            xs = np.array([sub[gap_col].min(), sub[gap_col].max()])
            ys = m["intercept"] + m["slope"] * xs
            ax.plot(xs * 100, ys * 100, color="black", linewidth=1, linestyle="--",
                    label=f"slope={m['slope']:+.2f}, R²={m['r2']:.2f}")
            ax.legend(fontsize=7, loc="best")
        ax.set_title(f"{label}\n({direction})")
        ax.set_xlabel("gap vs leader (%)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("market share (%)")
    fig.suptitle("Cross-sectional share vs. lever gap — red = DekaLLM observations", y=1.02)
    fig.tight_layout()
    out_path = OUT / "lever_sensitivity_scatter.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved scatter plot -> {out_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    print(f"Querying Prometheus at {PROM_URL}\n")
    name_map = resolve_provider_names()
    print(f"Provider name resolution: {name_map}\n")

    df = build_snapshot(name_map)
    if df.empty:
        print("No (model, provider) observations found.")
        return

    print(f"Built pooled snapshot: {len(df)} (model, provider) rows across "
          f"{df['model_id'].nunique()} models")

    marginals = marginal_regressions(df)
    joint = joint_regression(df)
    projection = dekallm_share_projection(df, marginals)

    # Save raw snapshot
    csv_path = OUT / "lever_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved snapshot     -> {csv_path}")

    # Save markdown report
    md = report(df, marginals, joint, projection)
    md_path = OUT / "lever_sensitivity.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved markdown     -> {md_path}")

    # Plot
    plot_scatter(df, marginals)

    # Console summary
    print("\n" + "=" * 78)
    print("Marginal sensitivity (slope = share change per unit gap-vs-leader)")
    print("=" * 78)
    for lever, m in marginals.items():
        if pd.isna(m["slope"]):
            print(f"  {lever:14s}  (insufficient data, n={m['n']})")
        else:
            print(f"  {lever:14s}  slope={m['slope']:+.4f}  R²={m['r2']:.3f}  "
                  f"n={m['n']}  SE={m['se_slope']:.4f}")

    print("\n" + "=" * 78)
    print("DekaLLM projected share lift per lever (closing avg gap to leader)")
    print("=" * 78)
    for r in projection:
        sign = "+" if r["projected_share_lift_pp"] >= 0 else ""
        print(f"  {r['label']:18s}  avg gap {r['current_dekallm_avg_gap_pct']:+6.1f}%  "
              f"-> projected lift {sign}{r['projected_share_lift_pp']:+.2f} pp")


if __name__ == "__main__":
    main()
