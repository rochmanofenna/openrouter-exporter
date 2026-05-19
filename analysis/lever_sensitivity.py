"""Lever sensitivity — empirically estimate how much each routing lever moves
OpenRouter market share, pooled across all (model, provider) observations.

Pulls the per-model competitive snapshot from the analytics API, fits OLS
regressions of share on lever gap, and projects DekaLLM's share lift if it
closes its average gap to the leader on each lever.

Differences vs the prior Prometheus-based version:
  - All providers serving each model (was: 4 hand-picked competitors)
  - 91-day per-model share window vs 32 days — denser sample for fitting
  - Real measured uptime is now a lever (was: omitted as unmeasurable)
  - Slug resolution via API (no regex fallback needed)

Outputs:
  - analysis/out/lever_sensitivity.md
  - analysis/out/lever_sensitivity.csv  — raw pooled snapshot
  - analysis/out/lever_sensitivity_scatter.png
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.api_client import default_client, require_alive
from analysis.competitive_position import (
    LEVERS,
    build_lever_table,
    build_share_table,
)

DEKALLM_SLUG = "dekallm"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
SHARE_LOOKBACK_DAYS = int(os.environ.get("SHARE_LOOKBACK_DAYS", "14"))


HIGHER_IS_BETTER = {name: meta["better"] == "higher" for name, meta in LEVERS.items()}


# ---------------------------------------------------------------------------
# OLS (no scipy dependency — small data, hand-rolled is fine)
# ---------------------------------------------------------------------------
def ols(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3:
        return {"n": n, "slope": np.nan, "intercept": np.nan, "r2": np.nan, "se_slope": np.nan}
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = ((x - x_mean) ** 2).sum()
    Sxy = ((x - x_mean) * (y - y_mean)).sum()
    if Sxx == 0:
        return {"n": n, "slope": 0.0, "intercept": y_mean, "r2": 0.0, "se_slope": np.nan}
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse = ss_res / (n - 2) if n > 2 else np.nan
    se_slope = (mse / Sxx) ** 0.5 if mse == mse and Sxx > 0 else np.nan
    return {"n": n, "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2), "se_slope": float(se_slope)}


# ---------------------------------------------------------------------------
# Build pooled (model, provider) snapshot
# ---------------------------------------------------------------------------
def build_snapshot(client) -> pd.DataFrame:
    models = client.dekallm_current_model_slugs(lookback_days=3)
    print(f"Building snapshot across {len(models)} DekaLLM models...")
    rows = []
    for m in models:
        try:
            levers, _ = build_lever_table(client, m)
        except Exception as e:
            print(f"  {m}: skip ({e})")
            continue
        share = build_share_table(client, m, lookback_days=SHARE_LOOKBACK_DAYS)
        if levers.empty and not share:
            continue
        providers = sorted(set(levers.index) | set(share.keys()))
        for p in providers:
            row = {"model_id": m, "provider": p, "share": share.get(p, np.nan)}
            for lever in LEVERS:
                row[lever] = levers.loc[p, lever] if p in levers.index else np.nan
            rows.append(row)
        print(f"  {m}: {len(providers)} providers")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Compute signed gap vs leader per lever, normalized.
    # Convention: gap = (value - leader) / leader.
    #   Higher-better levers (throughput, uptime): leader has max value, others negative.
    #   Lower-better levers (price, latency):       leader has min value, others positive.
    # This means a positive slope on a higher-better gap means "closer to leader = more share"
    # and a negative slope on a lower-better gap means "further from leader = less share".
    for lever, higher_better in HIGHER_IS_BETTER.items():
        gaps = []
        for model_id, grp in df.groupby("model_id"):
            valid = grp[lever].dropna()
            if valid.empty:
                continue
            leader = valid.max() if higher_better else valid.min()
            if leader == 0 or pd.isna(leader):
                continue
            for idx in grp.index:
                v = df.loc[idx, lever]
                if pd.isna(v):
                    continue
                df.loc[idx, f"{lever}_gap"] = (v - leader) / leader
    return df


def marginal_regressions(df: pd.DataFrame) -> dict:
    out = {}
    for lever in LEVERS:
        gap_col = f"{lever}_gap"
        if gap_col not in df.columns:
            out[lever] = {"n": 0, "slope": np.nan, "r2": np.nan, "se_slope": np.nan,
                          "intercept": np.nan}
            continue
        sub = df.dropna(subset=[gap_col, "share"])
        if len(sub) < 3:
            out[lever] = {"n": len(sub), "slope": np.nan, "r2": np.nan, "se_slope": np.nan,
                          "intercept": np.nan}
            continue
        out[lever] = ols(sub[gap_col].to_numpy(), sub["share"].to_numpy())
    return out


def joint_regression(df: pd.DataFrame) -> dict | None:
    cols = [f"{l}_gap" for l in LEVERS if f"{l}_gap" in df.columns]
    if not cols:
        return None
    sub = df.dropna(subset=cols + ["share"])
    if len(sub) < len(cols) + 2:
        return None
    X = sub[cols].to_numpy()
    y = sub["share"].to_numpy()
    X1 = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X1, y, rcond=None)
    intercept = float(coefs[0])
    betas = {c: float(b) for c, b in zip(cols, coefs[1:])}
    y_pred = X1 @ coefs
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"n": len(sub), "intercept": intercept, "betas": betas, "r2": r2}


def dekallm_projection(df: pd.DataFrame, marginals: dict) -> list[dict]:
    dek = df[df["provider"] == DEKALLM_SLUG]
    out = []
    for lever, meta in LEVERS.items():
        gap_col = f"{lever}_gap"
        if gap_col not in dek.columns:
            continue
        avg_gap = dek[gap_col].dropna().mean()
        if pd.isna(avg_gap):
            continue
        slope = marginals[lever]["slope"]
        if pd.isna(slope):
            continue
        # Closing the gap = moving gap to 0; share change = slope * (-avg_gap)
        projected_lift = -avg_gap * slope
        out.append({
            "lever": lever,
            "label": meta["label"],
            "current_dekallm_avg_gap_pct": avg_gap * 100,
            "slope_share_per_unit_gap": slope,
            "projected_share_lift_pp": projected_lift * 100,
        })
    out.sort(key=lambda r: -r["projected_share_lift_pp"])
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(df: pd.DataFrame, marginals: dict, joint: dict | None,
           projection: list[dict]) -> str:
    out = ["# DekaLLM Lever Sensitivity Analysis\n"]
    out.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    out.append("\nQuestion: which routing lever, if improved, would yield the "
               "biggest market-share lift for DekaLLM?\n")

    out.append("\n## Data source\n")
    out.append(
        "- Pooled (model, provider) snapshot from the local analytics API.\n"
        "- Models: every model in DekaLLM's current 3-day portfolio.\n"
        f"- Share: {SHARE_LOOKBACK_DAYS}-day average from /db/models/{{slug}}/provider-tokens.\n"
        "- Levers: input/output price, throughput (p50 t/s), latency (p50 ms), "
        "uptime (mean of daily % over the same window).\n"
        "- Gap definition: (value − leader_value) / leader_value, signed.\n"
    )

    out.append("\n## Marginal sensitivity (one lever at a time)\n")
    out.append("| Lever | n | Slope (share per unit gap) | R² | SE of slope | Interpretation |")
    out.append("|---|---:|---:|---:|---:|---|")
    for lever, meta in LEVERS.items():
        m = marginals.get(lever, {})
        slope = m.get("slope", np.nan)
        if pd.isna(slope):
            out.append(f"| {meta['label']} | {m.get('n', 0)} | — | — | — | (insufficient data) |")
            continue
        per_10pct = slope * 0.10 * 100  # share is 0-1, gap is fractional, output in pp
        sign = "+" if per_10pct > 0 else ""
        out.append(
            f"| {meta['label']} | {m['n']} | {slope:+.3f} | {m['r2']:.3f} | "
            f"{m['se_slope']:.3f} | "
            f"10% gap closure → {sign}{per_10pct:.2f} pp share |"
        )

    out.append("\nNote: slope sign depends on lever direction. For *lower-better* "
               "levers (price, latency), positive gap = worse than leader. For "
               "*higher-better* levers (throughput, uptime), negative gap = worse.\n")

    out.append("\n## Joint regression (all levers simultaneously)\n")
    if joint is None:
        out.append("_Insufficient data for joint fit._\n")
    else:
        out.append(f"n = {joint['n']}, R² = {joint['r2']:.3f}\n")
        out.append("Each coefficient is the partial effect, holding others constant.\n\n")
        out.append("| Lever | Partial effect (share per unit gap) |")
        out.append("|---|---:|")
        for lever, meta in LEVERS.items():
            beta = joint["betas"].get(f"{lever}_gap", np.nan)
            out.append(f"| {meta['label']} | {beta:+.3f} |")
        out.append(f"\nIntercept ≈ {joint['intercept']:.3f} (predicted share for a "
                   f"provider tied with the leader on every lever).")

    out.append("\n## DekaLLM share-lift projection per lever\n")
    out.append(
        "If DekaLLM closes its **average current gap** to the leader on each "
        "lever (across its models), the marginal-regression-implied share lift is:\n\n"
    )
    out.append("| Lever | DekaLLM avg current gap | Projected share lift (pp) | Action |")
    out.append("|---|---:|---:|---|")
    for r in projection:
        if r["projected_share_lift_pp"] > 0:
            action = f"close gap → +{r['projected_share_lift_pp']:.2f} pp share"
        elif r["projected_share_lift_pp"] < 0:
            action = "already favorable — moving further loses share"
        else:
            action = "neutral"
        out.append(
            f"| {r['label']} | {r['current_dekallm_avg_gap_pct']:+.1f}% | "
            f"{r['projected_share_lift_pp']:+.2f} pp | {action} |"
        )

    out.append("\n## Caveats\n")
    out.append(
        "- **Cross-sectional, not causal.** A small slow provider doesn't prove "
        "slowness causes smallness.\n"
        "- **Sample size still limited.** Even with 91-day history, snapshot "
        "uses static lever values × ~5 DekaLLM models × N providers each. "
        "Confidence intervals are wide.\n"
        "- **Levers correlate.** Cheap providers often run lower precision = "
        "worse tool-call quality. Multi-lever causality is unidentified.\n"
        "- **Tool-call accuracy is not in this regression.** That lever isn't "
        "in the API schema; getting it requires running K2-Vendor-Verifier-style "
        "tests against actual provider endpoints.\n"
    )
    return "\n".join(out)


def plot_scatter(df: pd.DataFrame, marginals: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(LEVERS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (lever, meta) in zip(axes, LEVERS.items()):
        gap_col = f"{lever}_gap"
        if gap_col not in df.columns:
            ax.set_title(f"{meta['label']}\n(no gap data)")
            continue
        sub = df.dropna(subset=[gap_col, "share"])
        if sub.empty:
            ax.set_title(f"{meta['label']}\n(no data)")
            continue
        for _, row in sub.iterrows():
            color = "red" if row["provider"] == DEKALLM_SLUG else "tab:blue"
            ax.scatter(row[gap_col] * 100, row["share"] * 100,
                       color=color, s=70, alpha=0.8)
        m = marginals[lever]
        if not pd.isna(m["slope"]):
            xs = np.array([sub[gap_col].min(), sub[gap_col].max()])
            ys = m["intercept"] + m["slope"] * xs
            ax.plot(xs * 100, ys * 100, color="black", linewidth=1, linestyle="--",
                    label=f"slope={m['slope']:+.2f}, R²={m['r2']:.2f}")
            ax.legend(fontsize=7, loc="best")
        ax.set_title(f"{meta['label']}\n({meta['better']}-better)")
        ax.set_xlabel("gap vs leader (%)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("market share (%)")
    fig.suptitle("Cross-sectional share vs lever gap — red = DekaLLM observations", y=1.02)
    fig.tight_layout()
    out_path = OUT / "lever_sensitivity_scatter.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved scatter -> {out_path}")


def main() -> None:
    client = require_alive()
    print(f"Using analytics API at {client.base}\n")

    df = build_snapshot(client)
    if df.empty:
        print("No (model, provider) observations.")
        return

    print(f"\nBuilt pooled snapshot: {len(df)} rows across "
          f"{df['model_id'].nunique()} models")

    marginals = marginal_regressions(df)
    joint = joint_regression(df)
    projection = dekallm_projection(df, marginals)

    csv_path = OUT / "lever_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved snapshot -> {csv_path}")

    md = report(df, marginals, joint, projection)
    md_path = OUT / "lever_sensitivity.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved markdown -> {md_path}")

    plot_scatter(df, marginals)

    print("\n" + "=" * 78)
    print("Marginal sensitivity (slope = share per unit gap)")
    print("=" * 78)
    for lever, m in marginals.items():
        if pd.isna(m["slope"]):
            print(f"  {lever:14s}  (insufficient data, n={m.get('n', 0)})")
        else:
            print(f"  {lever:14s}  slope={m['slope']:+.4f}  R²={m['r2']:.3f}  "
                  f"n={m['n']}  SE={m['se_slope']:.4f}")

    print("\n" + "=" * 78)
    print("DekaLLM projected share lift per lever (closing avg gap to leader)")
    print("=" * 78)
    for r in projection:
        sign = "+" if r["projected_share_lift_pp"] >= 0 else ""
        print(f"  {r['label']:24s}  avg gap {r['current_dekallm_avg_gap_pct']:+6.1f}%  "
              f"→ projected lift {sign}{r['projected_share_lift_pp']:+.2f} pp")


if __name__ == "__main__":
    main()
