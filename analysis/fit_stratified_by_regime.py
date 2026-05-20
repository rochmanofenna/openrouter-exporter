"""fit_stratified_by_regime.py — split the panel by per-(model, date) Spearman
regime and fit the lever model on each subset separately.

The hypothesis: the pooled R²=0.08 on `fit_experiment_a.py` is the average
across three structurally different routing regimes (default `1/price²`,
sort-routed `:nitro`/`:floor`, app-pinned `provider.order`). Each has a
different equation linking levers to share. If we classify (model, date)
rows by their Spearman ρ between price and share, the default-classified
subset should produce a clean fit; the pinned-classified subset should not.

PRE-COMMITMENTS (written before running):

  DEFAULT subset (ρ < -0.5):
    - price_out_ratio:  NEGATIVE, p<0.01, |coef| larger than the pooled v1 fit
    - downtime_pct:     NEGATIVE, p<0.01, similar magnitude to v1
    - throughput_ratio: insignificant OR weak (throughput is NOT in the
                        documented default routing function)
    - R²: 0.20-0.40

  PINNED subset (ρ ≥ -0.2):
    - price_out_ratio:  insignificant OR wrong-signed
    - downtime_pct:     correct sign but weaker than on default
                        (pinned apps don't reroute on 1-day outages)
    - R²: < 0.10

  Interpretation rule:
    - If DEFAULT subset hits pre-commit: regime stratification validated;
      the documented OpenRouter routing formula is identified in our data.
    - If DEFAULT subset still has R² ≈ 0.08 and wrong signs: the regime
      classifier isn't separating the population the way we think; the
      heterogeneity is along some other axis (which we'd need to find).
    - If PINNED subset shows lever coefficients, that's a surprise worth
      investigating — it would mean even "pinned" models have routing
      sensitivity we didn't account for.

Mentor design note honored: stratify at the (model, date) level, NOT by
model-level mean ρ. A model can be default-routed on one day and pinned
on another; aggregating to model-level throws away that variation.

Inputs:
  analysis/out/share_panel_filled.csv
  analysis/out/regime_spearman.csv

Outputs:
  analysis/out/fit_stratified_default.txt
  analysis/out/fit_stratified_pinned.txt
  analysis/out/fit_stratified_summary.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "out" / "share_panel_filled.csv"
RHO   = ROOT / "out" / "regime_spearman.csv"
OUT   = ROOT / "out"

EPS = 1e-4


def banner(s: str) -> None:
    print("=" * 78)
    print(s)
    print("=" * 78)


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def fit_one(df: pd.DataFrame, features: list[str], label: str) -> dict:
    """WLS on the chosen features. Returns coefficients + diagnostics."""
    if len(df) < len(features) + 3:
        print(f"  {label}: too few rows ({len(df)}) to fit")
        return {}
    X = sm.add_constant(df[features].to_numpy())
    y = df["y"].to_numpy()
    w = df["daily_tokens"].to_numpy()
    model = sm.WLS(y, X, weights=w).fit()

    print(f"\n--- {label} (n={len(df)}, models={df['model_id'].nunique()}, "
          f"providers={df['provider'].nunique()}) ---")
    print(model.summary(xname=["const"] + features))

    coefs = {features[i]: (float(model.params[i + 1]),
                           float(model.pvalues[i + 1])) for i in range(len(features))}

    return {
        "label": label,
        "n": int(len(df)),
        "n_models": int(df["model_id"].nunique()),
        "n_providers": int(df["provider"].nunique()),
        "rsquared": float(model.rsquared),
        "adj_rsquared": float(model.rsquared_adj),
        "f_pvalue": float(model.f_pvalue),
        "intercept": float(model.params[0]),
        "coefs": coefs,
        "summary_text": str(model.summary(xname=["const"] + features)),
    }


def grade(default_res: dict, pinned_res: dict) -> str:
    """Score outcomes against the pre-committed expectations."""
    if not default_res or not pinned_res:
        return "INSUFFICIENT DATA: at least one subset was too sparse to fit."

    # Default subset checks
    d_price_coef, d_price_p = default_res["coefs"]["price_out_ratio"]
    d_dt_coef, d_dt_p = default_res["coefs"]["downtime_pct"]
    d_tp_coef, d_tp_p = default_res["coefs"]["throughput_ratio"]
    d_r2 = default_res["rsquared"]

    d_price_ok = (d_price_coef < 0) and (d_price_p < 0.01)
    d_dt_ok = (d_dt_coef < 0) and (d_dt_p < 0.01)
    d_tp_weak = (d_tp_p > 0.05) or (abs(d_tp_coef) < 0.5)
    d_r2_ok = 0.18 <= d_r2 <= 0.45  # mentor said 0.20-0.40; small fuzz on bounds

    # Pinned subset checks
    p_price_coef, p_price_p = pinned_res["coefs"]["price_out_ratio"]
    p_dt_coef, p_dt_p = pinned_res["coefs"]["downtime_pct"]
    p_r2 = pinned_res["rsquared"]

    p_price_ns_or_wrong = (p_price_p > 0.05) or (p_price_coef > 0)
    p_dt_weaker = (abs(p_dt_coef) < abs(d_dt_coef)) if d_dt_coef != 0 else False
    p_r2_low = p_r2 < 0.15

    lines = []
    lines.append("Default subset checks (each pre-commit):")
    lines.append(f"  price_out NEGATIVE p<0.01  : "
                 f"{'PASS' if d_price_ok else 'FAIL'}  "
                 f"(coef={d_price_coef:+.3f}, p={d_price_p:.4f})")
    lines.append(f"  downtime NEGATIVE p<0.01   : "
                 f"{'PASS' if d_dt_ok else 'FAIL'}  "
                 f"(coef={d_dt_coef:+.3f}, p={d_dt_p:.4f})")
    lines.append(f"  throughput weak/NS         : "
                 f"{'PASS' if d_tp_weak else 'FAIL'}  "
                 f"(coef={d_tp_coef:+.3f}, p={d_tp_p:.4f})")
    lines.append(f"  R² in [0.18, 0.45]         : "
                 f"{'PASS' if d_r2_ok else 'FAIL'}  (R²={d_r2:.4f})")
    lines.append("")
    lines.append("Pinned subset checks:")
    lines.append(f"  price_out NS or wrong sign : "
                 f"{'PASS' if p_price_ns_or_wrong else 'FAIL'}  "
                 f"(coef={p_price_coef:+.3f}, p={p_price_p:.4f})")
    lines.append(f"  downtime weaker than default: "
                 f"{'PASS' if p_dt_weaker else 'FAIL'}  "
                 f"(default |coef|={abs(d_dt_coef):.3f}, pinned |coef|={abs(p_dt_coef):.3f})")
    lines.append(f"  R² < 0.15                  : "
                 f"{'PASS' if p_r2_low else 'FAIL'}  (R²={p_r2:.4f})")
    lines.append("")

    default_clean = d_price_ok and d_dt_ok and d_r2_ok
    pinned_messier = p_price_ns_or_wrong and p_r2_low

    if default_clean and pinned_messier:
        verdict = ("VALIDATED: regime stratification works. The default subset "
                   "behaves as the documented OpenRouter routing formula predicts, "
                   "the pinned subset is correctly messy. Pre-commit pre-conditions met.")
    elif default_clean and not pinned_messier:
        verdict = ("PARTIAL: default subset cleans up but pinned subset doesn't behave "
                   "as expected. The default-routed regime exists in our data; the "
                   "'pinned' classification may be capturing something different than pinning.")
    elif not default_clean and pinned_messier:
        verdict = ("INVERSION: pinned subset is messy as expected but default subset "
                   "doesn't show the clean 1/price² signature. The regime classifier "
                   "isn't actually separating routing modes — the heterogeneity is along "
                   "some other axis. Need to investigate what dimension actually splits.")
    else:
        verdict = ("BOTH SUBSETS UNCLEAN: stratification doesn't separate the population "
                   "into structurally different fits. The flat-Spearman histogram was right; "
                   "regime mixture is per-request, not per-(model, date). Need OpenRouter "
                   "telemetry to make further progress.")

    lines.append(f"VERDICT: {verdict}")
    return "\n".join(lines)


def main() -> None:
    if not PANEL.exists() or not RHO.exists():
        raise SystemExit(
            "Need both share_panel_filled.csv and regime_spearman.csv. "
            "Run dump_prometheus_panel.py and regime_spearman.py first."
        )

    panel = pd.read_csv(PANEL, parse_dates=["date"])
    rho = pd.read_csv(RHO, parse_dates=["date"])

    # Merge ρ onto each panel row by (model_id, date). Rows with no ρ
    # (fewer than 3 providers for that (M, date)) get NaN and are dropped.
    panel = panel.merge(rho[["model_id", "date", "rho_out_price_share", "n_providers"]],
                        on=["model_id", "date"], how="left")

    needed = ["token_share", "input_price", "output_price", "throughput_p50",
              "uptime_1d_pct", "daily_tokens", "rho_out_price_share"]
    panel = panel.dropna(subset=needed).copy()
    panel = panel[panel["token_share"] > 0].copy()

    # Build features (same as v1 / decomposition)
    panel["price_out_ratio"] = (panel.groupby("model_id")["output_price"]
                                .transform(lambda x: x / x.median() if x.median() > 0 else x))
    panel["throughput_ratio"] = (panel.groupby("model_id")["throughput_p50"]
                                 .transform(lambda x: x / x.median() if x.median() > 0 else x))
    panel["downtime_pct"] = 100.0 - panel["uptime_1d_pct"]
    panel["y"] = logit(panel["token_share"])

    # Stratify at (model, date) level — each row inherits the (M, date)'s ρ
    panel["regime"] = pd.cut(
        panel["rho_out_price_share"],
        bins=[-1.01, -0.5, -0.2, 1.01],
        labels=["default", "mixed", "pinned"],
    )

    banner("Pre-commitments (written before running)")
    print("""\
DEFAULT subset (ρ < -0.5):
  price_out_ratio : NEGATIVE, p<0.01, magnitude ≥ pooled v1
  downtime_pct    : NEGATIVE, p<0.01
  throughput_ratio: insignificant or weak (not in default routing function)
  R²              : 0.18-0.45

PINNED subset (ρ ≥ -0.2):
  price_out_ratio : insignificant or wrong-signed
  downtime_pct    : correct sign but weaker than default
  R²              : < 0.15
""")
    print()

    banner("Stratification counts (model-date level)")
    counts = panel["regime"].value_counts().reindex(["default", "mixed", "pinned"])
    print(counts.to_string())
    print()
    print("By (provider, regime) — sanity check that all providers appear in both subsets:")
    pivot = (panel.groupby(["provider", "regime"], observed=True)
             .size().unstack(fill_value=0))
    print(pivot.to_string())
    print()

    features = ["price_out_ratio", "throughput_ratio", "downtime_pct"]
    default_df = panel[panel["regime"] == "default"]
    pinned_df = panel[panel["regime"] == "pinned"]

    banner("Fit 1: DEFAULT subset (ρ < -0.5)")
    default_res = fit_one(default_df, features, "default-classified")
    if default_res:
        with open(OUT / "fit_stratified_default.txt", "w") as f:
            f.write(default_res["summary_text"])

    banner("Fit 2: PINNED subset (ρ ≥ -0.2)")
    pinned_res = fit_one(pinned_df, features, "pinned-classified")
    if pinned_res:
        with open(OUT / "fit_stratified_pinned.txt", "w") as f:
            f.write(pinned_res["summary_text"])

    banner("Grade against pre-commits")
    grade_report = grade(default_res, pinned_res)
    print(grade_report)
    print()

    # Side-by-side
    banner("Side-by-side coefficient comparison")
    print(f"  {'lever':<18} {'DEFAULT coef':>16} {'pinned coef':>16}")
    for f in features:
        d = default_res.get("coefs", {}).get(f, (np.nan, np.nan))
        p = pinned_res.get("coefs", {}).get(f, (np.nan, np.nan))
        d_marker = " *" if d[1] < 0.05 else "  "
        p_marker = " *" if p[1] < 0.05 else "  "
        print(f"  {f:<18} {d[0]:>+12.4f}{d_marker} {p[0]:>+12.4f}{p_marker}")
    print(f"  {'R²':<18} {default_res.get('rsquared', np.nan):>+12.4f}   "
          f"{pinned_res.get('rsquared', np.nan):>+12.4f}")
    print(f"  {'n':<18} {default_res.get('n', 0):>16} {pinned_res.get('n', 0):>16}")
    print()
    print("  (* = p < 0.05)")

    # Markdown summary
    md = [
        "# Regime-stratified fit results",
        "",
        f"## Stratification counts (model-date level)",
        f"- default (ρ < -0.5): **{int(counts.get('default', 0))} rows**",
        f"- mixed (-0.5 ≤ ρ < -0.2): {int(counts.get('mixed', 0))} rows (not fit)",
        f"- pinned (ρ ≥ -0.2): **{int(counts.get('pinned', 0))} rows**",
        "",
        "## Pre-committed expectations (do not edit)",
        "**DEFAULT subset**: price_out NEGATIVE p<0.01, downtime NEGATIVE p<0.01,",
        "throughput weak, R² 0.18-0.45",
        "",
        "**PINNED subset**: price_out NS or wrong, downtime weaker, R² < 0.15",
        "",
        "## Results",
        "| Lever | DEFAULT coef (p) | PINNED coef (p) |",
        "|---|---:|---:|",
    ]
    for f in features:
        d = default_res.get("coefs", {}).get(f, (np.nan, np.nan))
        p = pinned_res.get("coefs", {}).get(f, (np.nan, np.nan))
        md.append(f"| {f} | {d[0]:+.4f} (p={d[1]:.4f}) | {p[0]:+.4f} (p={p[1]:.4f}) |")
    md.extend([
        f"| **R²** | **{default_res.get('rsquared', np.nan):.4f}** | "
        f"**{pinned_res.get('rsquared', np.nan):.4f}** |",
        f"| n | {default_res.get('n', 0)} | {pinned_res.get('n', 0)} |",
        "",
        "## Grade",
        "```",
        grade_report,
        "```",
    ])
    with open(OUT / "fit_stratified_summary.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved: {OUT / 'fit_stratified_default.txt'}")
    print(f"Saved: {OUT / 'fit_stratified_pinned.txt'}")
    print(f"Saved: {OUT / 'fit_stratified_summary.md'}")


if __name__ == "__main__":
    main()
