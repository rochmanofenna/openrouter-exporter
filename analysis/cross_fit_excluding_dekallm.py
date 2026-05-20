"""Cross-fit excluding DekaLLM — is DekaLLM's share explainable by the formula?

PRE-COMMITMENT (written before running on any data):
  Fit the share formula on deepinfra/fireworks/together rows only. Then
  evaluate the fitted model on DekaLLM's lever values and compare predicted
  to observed share.

  Pass criterion: ≥70% of DekaLLM's observed shares fall within the 95%
  prediction interval. That's "DekaLLM is a normal provider with cheap+slow
  characteristics getting predictable share — formula explains DekaLLM."

  Fail criterion: <50% of DekaLLM's shares fall in the PI, AND residuals
  show consistent bias (always over or under). That's "DekaLLM is anomalous;
  there's a non-lever effect (regional routing, brand recognition, captive
  traffic, etc.) the formula doesn't capture."

  In between (50-70%): formula partially explains DekaLLM but a meaningful
  fraction is unexplained.

Why this matters: the v1 fit failed because DekaLLM is collinear with provider
identity (cheapest/slowest/lowest-share on most models). Removing DekaLLM and
asking "does the formula predict DekaLLM out-of-sample" tests the formula
without the identification problem.

Caveat: 3 providers is brittle for fitting. Coefficient values aren't
trustworthy; the *predictions* are still informative because we only need
the formula to interpolate, not to estimate elasticities accurately.

Outputs:
  analysis/out/cross_fit_excluding_dekallm.md
  analysis/out/cross_fit_excluding_dekallm.csv
  analysis/out/cross_fit_excluding_dekallm.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "out" / "share_panel_filled.csv"
OUT = ROOT / "out"


def main() -> None:
    if not PANEL.exists():
        raise SystemExit(f"Panel not found: {PANEL}")
    df = pd.read_csv(PANEL, parse_dates=["date"])

    feat_cols = ["input_price", "output_price", "throughput_p50", "uptime_1d_pct"]
    df = df.dropna(subset=feat_cols + ["token_share", "daily_tokens", "model_id"]).copy()
    df = df[df["token_share"] > 0].copy()
    if df.empty:
        raise SystemExit("No complete rows after dropna.")

    # Logit transform
    eps = 1e-4
    s = df["token_share"].clip(eps, 1 - eps)
    df["y"] = np.log(s / (1 - s))

    # Per-model normalization (same as Experiment A v1)
    df["price_in_ratio"] = (df.groupby("model_id")["input_price"]
                            .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["price_out_ratio"] = (df.groupby("model_id")["output_price"]
                             .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["throughput_ratio"] = (df.groupby("model_id")["throughput_p50"]
                              .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["downtime_pct"] = 100.0 - df["uptime_1d_pct"]

    features = ["price_out_ratio", "throughput_ratio", "downtime_pct"]
    # Dropped price_in_ratio: VIF=9 with price_out_ratio in v1; output price tends
    # to be the larger cost component and is correlated enough with input price
    # that retaining it is sufficient. This is the v2 collinearity fix.

    train = df[df["provider"] != "dekallm"].copy()
    holdout = df[df["provider"] == "dekallm"].copy()
    print(f"Training set : {len(train):,} rows "
          f"({train['provider'].nunique()} providers, "
          f"{train['model_id'].nunique()} models)")
    print(f"DekaLLM holdout: {len(holdout):,} rows "
          f"({holdout['model_id'].nunique()} models)")
    print()

    if train.empty or holdout.empty:
        raise SystemExit("Need both training and holdout data.")

    X_tr = sm.add_constant(train[features].to_numpy())
    y_tr = train["y"].to_numpy()
    w_tr = train["daily_tokens"].to_numpy()
    model = sm.WLS(y_tr, X_tr, weights=w_tr).fit()

    print("=== Fit on non-DekaLLM rows ===")
    print(f"  R² = {model.rsquared:.4f}   adj R² = {model.rsquared_adj:.4f}")
    print(f"  n = {len(train)}")
    for i, feat in enumerate(features):
        coef = model.params[i + 1]
        pval = model.pvalues[i + 1]
        sign = "+" if coef > 0 else ""
        print(f"    {feat:18s} coef={sign}{coef:.4f}  p={pval:.4f}")
    print()

    # Predict on DekaLLM with prediction interval
    X_hold = sm.add_constant(holdout[features].to_numpy())
    pred = model.get_prediction(X_hold)
    pred_summary = pred.summary_frame(alpha=0.05)

    # Convert logit predictions back to share scale.
    # Note: this is a per-row sigmoid of the predicted logit, NOT normalized
    # within (model, date). We're testing "does the formula's predicted
    # *logit* for DekaLLM align with DekaLLM's actual logit", which is
    # cleaner than trying to renormalize a sparse prediction set.
    holdout["y_pred"] = pred_summary["mean"].values
    holdout["y_pred_lower"] = pred_summary["obs_ci_lower"].values
    holdout["y_pred_upper"] = pred_summary["obs_ci_upper"].values

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    holdout["share_pred"] = sigmoid(holdout["y_pred"])
    holdout["share_pred_lower"] = sigmoid(holdout["y_pred_lower"])
    holdout["share_pred_upper"] = sigmoid(holdout["y_pred_upper"])
    holdout["in_pi"] = ((holdout["token_share"] >= holdout["share_pred_lower"])
                       & (holdout["token_share"] <= holdout["share_pred_upper"]))
    holdout["resid"] = holdout["token_share"] - holdout["share_pred"]

    pct_in_pi = 100.0 * holdout["in_pi"].mean()
    mean_resid = holdout["resid"].mean() * 100
    median_resid = holdout["resid"].median() * 100
    print("=== DekaLLM prediction quality ===")
    print(f"  % of DekaLLM obs within 95% PI: {pct_in_pi:.1f}%  (target: ≥70%)")
    print(f"  Mean residual (actual − predicted): {mean_resid:+.2f} pp")
    print(f"  Median residual:                   {median_resid:+.2f} pp")
    print(f"  n = {len(holdout)}")
    print()

    # Per-model breakdown — where exactly does the formula succeed/fail?
    print("=== DekaLLM by model (most recent date per model) ===")
    latest_per_model = holdout.sort_values("date").groupby("model_id").tail(1)
    print(f"  {'Model':<50} {'Actual':>8} {'Pred':>8} {'PI lower':>9} {'PI upper':>9} {'in PI':>6}")
    for _, r in latest_per_model.sort_values("model_id").iterrows():
        print(f"  {r['model_id']:<50} "
              f"{r['token_share']*100:>7.2f}% "
              f"{r['share_pred']*100:>7.2f}% "
              f"{r['share_pred_lower']*100:>8.2f}% "
              f"{r['share_pred_upper']*100:>8.2f}% "
              f"{'YES' if r['in_pi'] else 'no':>6}")
    print()

    # Verdict — must detect the uninformative-PI case where the interval is
    # so wide that "in PI" means nothing. Check PI width on share scale.
    pi_widths = holdout["share_pred_upper"] - holdout["share_pred_lower"]
    median_pi_width = float(pi_widths.median())

    if median_pi_width > 0.80:  # median PI spans >80% of [0,1]
        verdict = (f"INCONCLUSIVE: PI is too wide to be informative "
                   f"(median width = {median_pi_width:.2f} of full [0,1] range). "
                   f"Even 100% of observations 'in PI' wouldn't tell us anything. "
                   f"Better signal is the point-prediction bias: mean residual "
                   f"{mean_resid:+.1f} pp. With 3-provider training data, the fit "
                   f"can't make tight predictions for DekaLLM. This is the structural "
                   f"identification problem playing out — defer to the wider panel.")
    elif pct_in_pi >= 70:
        verdict = ("FORMULA EXPLAINS DEKALLM: ≥70% of DekaLLM observations fall in "
                   "the 95% PI. DekaLLM is a 'normal' provider whose share is predictable "
                   "from its lever values. No DekaLLM-specific effect needed.")
    elif pct_in_pi < 50 and abs(mean_resid) > 5.0:
        verdict = ("DEKALLM IS ANOMALOUS: <50% of observations in PI AND residual is "
                   f"systematically biased ({mean_resid:+.1f} pp). Investigate non-lever "
                   "effects (regional routing, captive Indonesian traffic, brand "
                   "recognition gaps, anything DekaLLM-specific).")
    else:
        verdict = ("PARTIAL: formula predicts DekaLLM's share for some models but misses "
                   f"others ({pct_in_pi:.0f}% in PI, mean residual {mean_resid:+.1f} pp). "
                   "Worth investigating which models the formula nails vs which it doesn't.")
    print(f"VERDICT: {verdict}")
    print()

    holdout.to_csv(OUT / "cross_fit_excluding_dekallm.csv", index=False)

    # Markdown report
    md = [
        "# Cross-fit excluding DekaLLM",
        "",
        f"Fit on {len(train)} non-DekaLLM rows ({train['provider'].nunique()} providers, "
        f"{train['model_id'].nunique()} models), evaluated on {len(holdout)} DekaLLM rows.",
        "",
        f"## Fit summary",
        f"- R² = {model.rsquared:.4f}",
        "",
        f"## Headline result",
        f"- **{pct_in_pi:.1f}%** of DekaLLM observations fall within 95% prediction interval (target: ≥70%)",
        f"- Mean residual: **{mean_resid:+.2f} pp** (actual − predicted, share scale)",
        f"- Median residual: {median_resid:+.2f} pp",
        "",
        f"## Verdict\n\n{verdict}",
        "",
        "## Per-model breakdown (most recent date)",
        "",
        "| Model | Actual | Predicted | PI low | PI high | In PI |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for _, r in latest_per_model.sort_values("model_id").iterrows():
        md.append(
            f"| {r['model_id']} | "
            f"{r['token_share']*100:.2f}% | "
            f"{r['share_pred']*100:.2f}% | "
            f"{r['share_pred_lower']*100:.2f}% | "
            f"{r['share_pred_upper']*100:.2f}% | "
            f"{'✓' if r['in_pi'] else '✗'} |"
        )
    md.extend([
        "",
        "## Caveats",
        "- Fit on only 3 providers; coefficients themselves are not trustworthy.",
        "  This test is about *predictions* for DekaLLM, not coefficient calibration.",
        "- Sigmoid back-transform from logit, not normalized within (model, date).",
        "  Use this for the directional verdict, not for absolute share calibration.",
    ])
    with open(OUT / "cross_fit_excluding_dekallm.md", "w") as f:
        f.write("\n".join(md))
    print(f"Saved md:  {OUT / 'cross_fit_excluding_dekallm.md'}")
    print(f"Saved csv: {OUT / 'cross_fit_excluding_dekallm.csv'}")


if __name__ == "__main__":
    main()
