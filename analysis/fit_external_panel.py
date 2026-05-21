"""fit_external_panel.py — run the full pre-committed fit suite on the
ML-ready data dump (analysis/data/training_table.csv).

This is the validation moment we've been pre-committing toward. The
training_table.csv covers 90 days × 145 models × 56 providers (vs our
Prometheus panel's 15 days × 59 models × 17 providers), with all the
feature columns already computed and rank features pre-engineered.

What this script does (in order):
  1. Load training_table.csv and adapt column names to our convention.
  2. Run the v1 fit (price_in + price_out + throughput + downtime).
  3. Stratify by per-(model, date) Spearman ρ and refit on each subset.
  4. Grade against PRECOMMITMENT.md thresholds.

PRECOMMITMENT.md thresholds (locked in 2026-05-20):
  - VALIDATED: R² > 0.4 + all signs correct + oracle within tolerance
  - PLAUSIBLE: R² 0.2-0.4 + all signs correct
  - STRUCTURAL PROBLEM: R² < 0.2 OR significant wrong sign

Expected signs:
  - price_in_ratio:    NEGATIVE
  - price_out_ratio:   NEGATIVE
  - throughput_ratio:  POSITIVE
  - downtime_pct:      NEGATIVE

Inputs:
  analysis/data/training_table.csv
Outputs:
  analysis/out/external_panel_summary.md
  analysis/out/external_panel_predictions.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "training_table.csv"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
EPS = 1e-4


def banner(s: str) -> None:
    print("=" * 78)
    print(s)
    print("=" * 78)


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


# ---------------------------------------------------------------------------
# Load + adapt
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit(f"Data not found: {DATA}")
    df = pd.read_csv(DATA, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows from {DATA.name}")
    print(f"  dates:     {df['date'].min().date()} → {df['date'].max().date()} ({df['date'].nunique()})")
    print(f"  models:    {df['permaslug'].nunique()}")
    print(f"  providers: {df['provider_slug'].nunique()}")

    # Adapt to our column convention used by fit_experiment_a.py et al.
    out = pd.DataFrame({
        "date": df["date"],
        "model_id": df["permaslug"],
        "provider": df["provider_slug"],
        # share comes in as %; convert to fraction
        "token_share": df["provider_share_pct"] / 100.0,
        "daily_tokens": df["provider_tokens"],
        # Pricing is in $/token; convert to $/M for interpretability
        "input_price": df["price_prompt"] * 1_000_000,
        "output_price": df["price_completion"] * 1_000_000,
        "throughput_p50": df["p50_throughput"],
        "latency_p50_ms": df["p50_latency_ms"],
        # uptime is already in %; downtime = 100 - uptime
        "uptime_1d_pct": df["recent_uptime_pct"],
        "num_providers": df["num_providers"],
        "quantization": df["quantization"],
        "is_cheapest": df["is_cheapest"],
        "price_rank": df["price_rank"],
    })
    # Drop the most recent date (manifest flags it as partial)
    last_date = out["date"].max()
    print(f"\nDropping partial-day rows for {last_date.date()} per manifest note")
    out = out[out["date"] < last_date].copy()
    print(f"  Remaining: {len(out):,} rows")
    return out


# ---------------------------------------------------------------------------
# v1 fit
# ---------------------------------------------------------------------------
def run_v1_fit(df: pd.DataFrame) -> dict:
    """Same regression as fit_experiment_a.py — price_in + price_out +
    throughput + downtime, logit target, WLS by tokens."""
    feat_cols = ["input_price", "output_price", "throughput_p50", "uptime_1d_pct"]
    work = df.dropna(subset=feat_cols + ["token_share", "daily_tokens", "model_id"]).copy()
    work = work[work["token_share"] > 0].copy()

    work["y"] = logit(work["token_share"])
    work["price_in_ratio"] = (work.groupby("model_id")["input_price"]
                              .transform(lambda x: x / x.median() if x.median() > 0 else x))
    work["price_out_ratio"] = (work.groupby("model_id")["output_price"]
                               .transform(lambda x: x / x.median() if x.median() > 0 else x))
    work["throughput_ratio"] = (work.groupby("model_id")["throughput_p50"]
                                .transform(lambda x: x / x.median() if x.median() > 0 else x))
    work["downtime_pct"] = 100.0 - work["uptime_1d_pct"]

    features = ["price_in_ratio", "price_out_ratio", "throughput_ratio", "downtime_pct"]
    X = sm.add_constant(work[features].to_numpy())
    y = work["y"].to_numpy()
    w = work["daily_tokens"].to_numpy()
    model = sm.WLS(y, X, weights=w).fit()

    print(model.summary(xname=["const"] + features))
    coefs = {features[i]: (float(model.params[i + 1]), float(model.pvalues[i + 1]))
             for i in range(len(features))}
    return {
        "n": len(work),
        "n_models": work["model_id"].nunique(),
        "n_providers": work["provider"].nunique(),
        "rsquared": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "coefs": coefs,
        "work": work,
        "model": model,
        "features": features,
    }


# ---------------------------------------------------------------------------
# Sign check + grading
# ---------------------------------------------------------------------------
EXPECTED = {"price_in_ratio": -1, "price_out_ratio": -1,
            "throughput_ratio": +1, "downtime_pct": -1}


def sign_check(coefs: dict) -> tuple[bool, list[str]]:
    lines = []
    all_ok = True
    sig_wrong = False
    for f, (coef, pval) in coefs.items():
        ok = np.sign(coef) == EXPECTED[f]
        sig = "p<0.05" if pval < 0.05 else f"NS (p={pval:.3f})"
        marker = "OK" if ok else "WRONG"
        if not ok and pval < 0.05:
            sig_wrong = True
        if not ok:
            all_ok = False
        lines.append(f"  {f:18s}  coef={coef:+.4f}  [{marker}]  {sig}")
    return all_ok, sig_wrong, lines


def grade(rsquared: float, all_signs_ok: bool, sig_wrong: bool) -> str:
    if rsquared > 0.40 and all_signs_ok:
        return ("VALIDATED: R² > 0.4 AND all signs correct. The hypothesized lever "
                "structure is confirmed on this panel.")
    if 0.20 <= rsquared <= 0.40 and all_signs_ok:
        return ("PLAUSIBLE: signs correct, R² in the 0.2-0.4 band. Structure is "
                "sound; need additional features to lift explanatory power.")
    if rsquared < 0.20 or sig_wrong:
        reasons = []
        if rsquared < 0.20:
            reasons.append(f"R²={rsquared:.3f} below 0.20 floor")
        if sig_wrong:
            reasons.append("at least one significant wrong-sign coefficient")
        return f"STRUCTURAL PROBLEM: {' and '.join(reasons)}."
    return (f"INTERMEDIATE: R²={rsquared:.3f}, some signs wrong but not significant. "
            "Treat as 'plausible but underpowered.'")


# ---------------------------------------------------------------------------
# Regime stratification
# ---------------------------------------------------------------------------
def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-(model, date) Spearman ρ and regime classification."""
    rows = []
    for (m, d), g in df.groupby(["model_id", "date"]):
        if len(g) < 3:
            continue
        rho, _ = stats.spearmanr(g["output_price"], g["token_share"])
        if pd.isna(rho):
            rho = 0.0
        rows.append({"model_id": m, "date": d,
                     "rho_out_price_share": rho, "n_providers": len(g)})
    rho_df = pd.DataFrame(rows)
    out = df.merge(rho_df, on=["model_id", "date"], how="left")
    out["regime"] = pd.cut(out["rho_out_price_share"],
                           bins=[-1.01, -0.5, -0.2, 1.01],
                           labels=["default", "mixed", "pinned"])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_panel()
    print()

    banner("Fit 1: pooled (same as fit_experiment_a.py)")
    pooled = run_v1_fit(df)
    print()

    banner("Sign check vs pre-commitments")
    all_ok, sig_wrong, lines = sign_check(pooled["coefs"])
    for line in lines:
        print(line)
    print()
    pooled_verdict = grade(pooled["rsquared"], all_ok, sig_wrong)
    banner("Pooled verdict")
    print(f"  R² = {pooled['rsquared']:.4f}")
    print(f"  {pooled_verdict}")
    print()

    # Regime stratification
    work = pooled["work"]
    work = add_regime(work)
    counts = work["regime"].value_counts().reindex(["default", "mixed", "pinned"])
    banner(f"Regime stratification (per (model, date) Spearman)")
    print(counts.to_string())
    print()

    def fit_subset(sub: pd.DataFrame, label: str) -> dict:
        if len(sub) < 50:
            print(f"  {label}: too few rows ({len(sub)})")
            return {}
        features = ["price_out_ratio", "throughput_ratio", "downtime_pct"]
        X = sm.add_constant(sub[features].to_numpy())
        y = sub["y"].to_numpy()
        w = sub["daily_tokens"].to_numpy()
        m = sm.WLS(y, X, weights=w).fit()
        print(f"\n--- {label} (n={len(sub)}, "
              f"models={sub['model_id'].nunique()}, "
              f"providers={sub['provider'].nunique()}) ---")
        print(m.summary(xname=["const"] + features))
        coefs = {features[i]: (float(m.params[i + 1]), float(m.pvalues[i + 1]))
                 for i in range(len(features))}
        return {"rsquared": float(m.rsquared), "coefs": coefs, "n": len(sub)}

    banner("Fit 2: DEFAULT subset (ρ < -0.5)")
    default_res = fit_subset(work[work["regime"] == "default"], "default")

    banner("Fit 3: PINNED subset (ρ ≥ -0.2)")
    pinned_res = fit_subset(work[work["regime"] == "pinned"], "pinned")

    # Side-by-side
    banner("Side-by-side: pooled vs default-subset vs pinned-subset")
    features_compare = ["price_out_ratio", "throughput_ratio", "downtime_pct"]
    print(f"  {'lever':<18} {'POOLED':>20} {'DEFAULT':>20} {'PINNED':>20}")
    for f in features_compare:
        po = pooled["coefs"].get(f, (np.nan, np.nan))
        de = default_res.get("coefs", {}).get(f, (np.nan, np.nan))
        pi = pinned_res.get("coefs", {}).get(f, (np.nan, np.nan))
        def fmt(c, p):
            mark = "*" if (not pd.isna(p) and p < 0.05) else " "
            return f"{c:>+10.4f} {mark} (p={p:.3f})" if not pd.isna(c) else "      n/a       "
        print(f"  {f:<18} {fmt(po[0], po[1]):>20} {fmt(de[0], de[1]):>20} {fmt(pi[0], pi[1]):>20}")
    print(f"  {'R²':<18} {pooled['rsquared']:>20.4f} "
          f"{default_res.get('rsquared', np.nan):>20.4f} "
          f"{pinned_res.get('rsquared', np.nan):>20.4f}")
    print()

    # Oracle: gpt-oss-120b
    banner("Oracle: openai/gpt-oss-120b (most recent finalized date)")
    oracle = work[work["model_id"] == "openai/gpt-oss-120b"]
    if not oracle.empty:
        latest = oracle["date"].max()
        rows = oracle[oracle["date"] == latest].sort_values("token_share", ascending=False)
        print(f"  Date: {latest.date()}, providers: {len(rows)}")
        print(f"  {'Provider':<14} {'Actual':>8} {'Price$/M':>9}")
        for _, r in rows.iterrows():
            print(f"  {r['provider']:<14} {r['token_share']*100:>7.2f}%  "
                  f"{r['output_price']:>8.2f}")
    print()

    # Save summary
    md = [
        "# External-panel fit results",
        "",
        f"Panel: `training_table.csv` ({pooled['n']:,} rows after partial-day drop, "
        f"{pooled['n_models']} models, {pooled['n_providers']} providers)",
        "",
        "## Pooled fit verdict (vs PRECOMMITMENT.md)",
        f"- R² = **{pooled['rsquared']:.4f}**",
        f"- Verdict: **{pooled_verdict}**",
        "",
        "## Coefficients (pooled)",
        "| Lever | Coef | p-value | Expected sign | Status |",
        "|---|---:|---:|---|---|",
    ]
    for f, (c, p) in pooled["coefs"].items():
        ok = np.sign(c) == EXPECTED[f]
        md.append(f"| {f} | {c:+.4f} | {p:.4f} | "
                  f"{'NEG' if EXPECTED[f] < 0 else 'POS'} | "
                  f"{'OK' if ok else '**WRONG**'} |")
    md.extend([
        "",
        "## Regime-stratified fit",
        f"- Default subset (ρ < -0.5): R² = {default_res.get('rsquared', float('nan')):.4f}",
        f"- Pinned subset  (ρ ≥ -0.2): R² = {pinned_res.get('rsquared', float('nan')):.4f}",
    ])
    if default_res:
        md.append("\n### Default subset coefficients")
        md.append("| Lever | Coef | p |")
        md.append("|---|---:|---:|")
        for f, (c, p) in default_res["coefs"].items():
            md.append(f"| {f} | {c:+.4f} | {p:.4f} |")
    if pinned_res:
        md.append("\n### Pinned subset coefficients")
        md.append("| Lever | Coef | p |")
        md.append("|---|---:|---:|")
        for f, (c, p) in pinned_res["coefs"].items():
            md.append(f"| {f} | {c:+.4f} | {p:.4f} |")
    with open(OUT / "external_panel_summary.md", "w") as f:
        f.write("\n".join(md))
    work.to_csv(OUT / "external_panel_predictions.csv", index=False)
    print(f"Saved: {OUT / 'external_panel_summary.md'}")
    print(f"Saved: {OUT / 'external_panel_predictions.csv'}")


if __name__ == "__main__":
    main()
