"""Experiment A — linear OLS baseline for the token-share formula.

Hypothesis under test (your handwritten formula):
    token_share[M, P] ≈ routing_weight[M, P] × multi_homing × pool_eligible
    routing_weight[M, P] ~ f(price competitiveness, uptime, throughput vs median,
                            tool-call accuracy, days since added)

Method (per mentor walkthrough):
  - Logit-transform token_share so it's unbounded
  - Normalize price/throughput per-model (raw $/M isn't comparable across models)
  - Fit WLS weighted by daily_tokens (more volume = more reliable share estimate)
  - Report coefficient signs vs pre-committed expectations
  - Check VIF for collinearity
  - Oracle row: predict openai/gpt-oss-120b across providers; compare to truth

Window constraint: feature history is only 15 days, so this fit is dominated
by *cross-sectional* variance (across providers within model), not temporal
variance. Don't split by date — split by model when you want a real holdout.

Inputs:
    analysis/out/share_panel_filled.csv   (from dump_prometheus_panel.py)

Outputs:
    analysis/out/experiment_a_predictions.csv
    analysis/out/experiment_a_summary.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError as e:
    raise SystemExit(f"statsmodels required: pip install statsmodels  ({e})")

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "out" / "share_panel_filled.csv"
OUT = ROOT / "out"


def banner(s: str) -> None:
    print("=" * 78)
    print(s)
    print("=" * 78)


def main() -> None:
    # ---- 0. Pre-commitment ---------------------------------------------------
    # Written down BEFORE any fit output. If predictions disagree with these
    # expectations, debug the fit; if they agree, the formula is doing what
    # we hoped. Either way, no rationalizing after the fact.
    banner("PRE-COMMITMENT (do not edit after running)")
    print("Oracle expectation — openai/gpt-oss-120b, recent days:")
    print("  deepinfra : 85-95%  (high throughput, mid price, dominant)")
    print("  dekallm   :  5-10%  (cheapest by ~3x but slowest by ~6-10x)")
    print("  together  :  1-3%   (priciest, mid throughput, marginal)")
    print()
    print("Expected coefficient signs:")
    print("  price_in_ratio    : NEGATIVE  (cheaper → more share)")
    print("  price_out_ratio   : NEGATIVE")
    print("  throughput_ratio  : POSITIVE  (faster → more share)")
    print("  downtime_pct      : NEGATIVE  (downtime → less share)")
    print()

    # ---- 1. Load and clean ---------------------------------------------------
    if not PANEL.exists():
        raise SystemExit(
            f"Panel not found: {PANEL}\n"
            "Run `python analysis/dump_prometheus_panel.py` first."
        )
    df = pd.read_csv(PANEL, parse_dates=["date"])
    banner(f"Loaded panel: {len(df):,} rows")
    print(f"  models     : {df['model_id'].nunique()}")
    print(f"  providers  : {sorted(df['provider'].unique())}")
    print(f"  date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print()

    needed_features = ["input_price", "output_price", "throughput_p50", "uptime_1d_pct"]
    needed = needed_features + ["token_share", "daily_tokens", "model_id"]
    before = len(df)
    df = df.dropna(subset=needed).copy()
    dropped_missing = before - len(df)

    zero_share = int((df["token_share"] == 0).sum())
    df = df[df["token_share"] > 0].copy()

    print(f"  Dropped {dropped_missing:,} rows missing features/share")
    if zero_share:
        print(f"  Dropped {zero_share:,} zero-share rows "
              "(cannot logit; flag for hurdle-model follow-up)")
    print(f"  Fitting on {len(df):,} rows / "
          f"{df['model_id'].nunique()} models / "
          f"{df['provider'].nunique()} providers / "
          f"{df['date'].nunique()} dates")
    print()

    # ---- 2. Feature construction --------------------------------------------
    # Logit target so it's unbounded
    eps = 1e-4
    s = df["token_share"].clip(eps, 1 - eps)
    df["y"] = np.log(s / (1 - s))

    # Within-model normalization. Cheap-model providers all sub-$1/M; pricey-
    # model providers all $10+/M. Ratio-to-median makes them comparable.
    df["price_in_ratio"] = (df.groupby("model_id")["input_price"]
                            .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["price_out_ratio"] = (df.groupby("model_id")["output_price"]
                             .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["throughput_ratio"] = (df.groupby("model_id")["throughput_p50"]
                              .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["downtime_pct"] = 100.0 - df["uptime_1d_pct"]

    features = ["price_in_ratio", "price_out_ratio", "throughput_ratio", "downtime_pct"]

    banner("Feature variance (post-normalization)")
    print(df[features].describe().round(3).to_string())
    # If any std is ~0, that feature is dead — usually means only one provider
    # in that subset, so per-model normalization collapsed everything to 1.0.
    near_dead = [f for f in features if df[f].std() < 0.01]
    if near_dead:
        print(f"\n  WARNING: near-zero variance on {near_dead} — feature won't fit")
    print()

    # ---- 3. WLS fit ----------------------------------------------------------
    X = df[features].values
    X_const = sm.add_constant(X)
    y = df["y"].values
    weights = df["daily_tokens"].values  # token-share → weight by token volume

    model = sm.WLS(y, X_const, weights=weights).fit()

    banner("WLS fit summary (weights = daily_tokens)")
    print(model.summary())
    print()

    # ---- 4. Sign check vs pre-commitment ------------------------------------
    banner("Sign check vs pre-committed expectations")
    expected_sign = {"price_in_ratio": -1, "price_out_ratio": -1,
                     "throughput_ratio": 1, "downtime_pct": -1}
    all_ok = True
    for i, feat in enumerate(features):
        coef = model.params[i + 1]  # +1 for const at index 0
        pval = model.pvalues[i + 1]
        ok = np.sign(coef) == expected_sign[feat]
        all_ok &= ok
        marker = "OK " if ok else "WRONG SIGN"
        sig = "p<0.05" if pval < 0.05 else f"NS (p={pval:.3f})"
        print(f"  {feat:18s}  coef={coef:+.4f}  [{marker}]  {sig}")
    if all_ok:
        print("\n  All signs match expectation.")
    else:
        print("\n  AT LEAST ONE WRONG SIGN — debug the fit, not the formula.")
    print()

    # ---- 5. VIF (collinearity) ----------------------------------------------
    banner("VIF (>5 = collinearity warning; coefficients individually unreliable)")
    for i, feat in enumerate(features):
        vif = variance_inflation_factor(X_const, i + 1)
        flag = "  <-- HIGH" if vif > 5 else ""
        print(f"  {feat:18s}  VIF={vif:.2f}{flag}")
    print()

    # ---- 6. Predictions + oracle row ----------------------------------------
    df["y_pred"] = model.predict(X_const)
    # Raw sigmoid of y_pred; then normalize within (model, date) so shares sum to 1.
    df["share_pred_raw"] = 1.0 / (1.0 + np.exp(-df["y_pred"]))
    df["share_pred"] = (df.groupby(["model_id", "date"])["share_pred_raw"]
                        .transform(lambda x: x / x.sum() if x.sum() > 0 else x))
    df["resid"] = df["token_share"] - df["share_pred"]

    banner("Oracle: openai/gpt-oss-120b — predicted vs actual")
    oracle = df[df["model_id"] == "openai/gpt-oss-120b"].copy()
    if oracle.empty:
        print("  NO DATA — gpt-oss-120b not in fitted set.")
    else:
        latest = oracle["date"].max()
        latest_rows = oracle[oracle["date"] == latest].sort_values("token_share", ascending=False)
        print(f"  Most recent date: {latest.date()}\n")
        print(f"  {'Provider':<12} {'Actual':>10} {'Predicted':>11} {'Δ':>10}")
        for _, r in latest_rows.iterrows():
            print(f"  {r['provider']:<12} "
                  f"{r['token_share']*100:>9.2f}%  "
                  f"{r['share_pred']*100:>10.2f}%  "
                  f"{r['resid']*100:>+9.2f}pp")
        print()
        # Mean over last 5 days per provider
        print(f"  Mean over last 5 days:")
        print(f"  {'Provider':<12} {'Actual':>10} {'Predicted':>11}")
        for p in sorted(oracle["provider"].unique()):
            sub = oracle[oracle["provider"] == p].sort_values("date").tail(5)
            print(f"  {p:<12} "
                  f"{sub['token_share'].mean()*100:>9.2f}%  "
                  f"{sub['share_pred'].mean()*100:>10.2f}%")
    print()

    # ---- 7. Aggregate fit quality -------------------------------------------
    banner("Aggregate fit quality")
    print(f"  R² (on logit target, WLS) : {model.rsquared:.4f}")
    print(f"  Adj R²                    : {model.rsquared_adj:.4f}")
    print(f"  F-statistic p-value       : {model.f_pvalue:.4g}")
    # On the actual share scale (not logit)
    valid = df.dropna(subset=["share_pred"])
    if len(valid) > 1:
        mae_share = np.average(np.abs(valid["token_share"] - valid["share_pred"]),
                               weights=valid["daily_tokens"])
        # Pearson on logit predictions
        corr_logit = np.corrcoef(df["y"], df["y_pred"])[0, 1]
        # Pearson on share predictions (after normalization)
        corr_share = np.corrcoef(valid["token_share"], valid["share_pred"])[0, 1]
        print(f"  Volume-weighted MAE (share scale): {mae_share*100:.2f} pp")
        print(f"  Pearson r (logit)         : {corr_logit:.4f}")
        print(f"  Pearson r (share)         : {corr_share:.4f}")
    print()

    # ---- 8. Per-provider residual bias (DekaLLM is what we care about) ------
    banner("Per-provider residual bias (Δ = actual − predicted, in pp)")
    print(f"  {'Provider':<12} {'mean Δ':>10} {'median Δ':>10} {'n':>6}")
    for p in sorted(df["provider"].unique()):
        sub = df[df["provider"] == p]
        mean_d = sub["resid"].mean() * 100
        med_d = sub["resid"].median() * 100
        print(f"  {p:<12} {mean_d:>+9.2f}pp  {med_d:>+9.2f}pp  {len(sub):>6}")
    print()

    # ---- 9. Persist ----------------------------------------------------------
    pred_csv = OUT / "experiment_a_predictions.csv"
    df.to_csv(pred_csv, index=False)
    with open(OUT / "experiment_a_summary.txt", "w") as f:
        f.write(str(model.summary()))
        f.write("\n\nSign check:\n")
        for i, feat in enumerate(features):
            coef = model.params[i + 1]
            pval = model.pvalues[i + 1]
            ok = np.sign(coef) == expected_sign[feat]
            f.write(f"  {feat:18s}  coef={coef:+.4f}  "
                    f"[{'OK' if ok else 'WRONG'}]  p={pval:.4f}\n")
    print(f"Saved: {pred_csv}")
    print(f"Saved: {OUT / 'experiment_a_summary.txt'}")


if __name__ == "__main__":
    main()
