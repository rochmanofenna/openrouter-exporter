"""fit_decomposition_v1.py — three-factor decomposition of token_share.

  token_share[M, P] ≈ routing_weight[M, P]
                    × tokens_per_request[M, P]
                    × multi_homing_boost[M, P]

PRE-COMMITMENT (written BEFORE running):
  - Routing fit (Step B): expect price_out_ratio NEGATIVE, throughput_ratio
    POSITIVE, downtime_pct NEGATIVE. If all three correct AND p<0.10 even
    with 4 providers, structure validated on existing data.
  - TPR fit (Step C): expect model dummies to explain >40% of log(TPR)
    variance. R² < 0.2 = M-level decomposition assumption is the wrong shape.
  - Multi-homing residual (Step D): expect top-share providers' median boost
    > 1.1 and tail providers (rank ≥ 3) median boost < 0.9 if multi-homing
    is structured. Flat boosts across ranks = noise, not signal.

STRUCTURAL CAVEAT for THIS run:
  Our Prometheus panel only has tokens per (M, P, date) and requests per
  (M, date) — there is NO per-(M, P, date) request count. So
  tokens_per_request collapses to model-level (TPR[M, date]) regardless of
  provider. That means implied_request_share after within-(M, date)
  normalization is mathematically *identical* to observed_token_share — Step
  B's routing fit produces the same coefficients as v1's Experiment A.

  This script runs the decomposition anyway, surfacing where it collapses,
  because:
    1) Step C (cross-model TPR variation) is independent of provider data
       and still informative.
    2) Step D (residual analysis) becomes "what does the routing model
       miss?" rather than a pure multi-homing test — still useful for
       per-provider bias detection.
    3) The degeneracy is itself a meaningful finding: it tells us we need
       per-(M, P) request counts (a new Prometheus scrape) before the
       formula can be properly decomposed.

Inputs:
  analysis/out/share_panel.csv (with model_tpr column from updated dump)

Outputs:
  analysis/out/decomposition_v1_routing_summary.txt
  analysis/out/decomposition_v1_tpr_summary.txt
  analysis/out/decomposition_v1_multihoming.csv
  analysis/out/decomposition_v1_with_predictions.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "out" / "share_panel.csv"
OUT = ROOT / "out"
EPS = 1e-4


def logit(p):
    p_c = np.clip(p, EPS, 1 - EPS)
    return np.log(p_c / (1 - p_c))


def banner(s: str) -> None:
    print("=" * 78)
    print(s)
    print("=" * 78)


def main() -> None:
    if not PANEL.exists():
        raise SystemExit(f"Panel not found: {PANEL}. Run dump_prometheus_panel.py first.")

    df = pd.read_csv(PANEL, parse_dates=["date"])
    print(f"Loaded panel: {len(df):,} rows")

    # Forward-fill features within (provider, model) so 15-day feature window
    # doesn't drop everything that's missing on the edges. Features change
    # slowly so a recent reading carried forward is fine.
    df = df.sort_values(["base_slug", "provider", "date"])
    feat_cols = ["input_price", "output_price", "throughput_p50",
                 "latency_p50_ms", "uptime_1d_pct"]
    df[feat_cols] = df.groupby(["base_slug", "provider"])[feat_cols].ffill()
    # TPR also forward-filled — same justification (model TPR doesn't move wildly day-to-day)
    df["model_tpr"] = df.groupby("base_slug")["model_tpr"].ffill()

    needed = feat_cols + ["token_share", "daily_tokens", "model_id", "model_tpr"]
    before = len(df)
    df = df.dropna(subset=needed).copy()
    df = df[(df["token_share"] > 0) & (df["model_tpr"] > 0)].copy()
    print(f"  Dropped {before - len(df):,} rows missing features/share/TPR")
    print(f"  Fitting on {len(df):,} rows / {df['model_id'].nunique()} models / "
          f"{df['provider'].nunique()} providers / {df['date'].nunique()} dates")
    print()

    # ---- Step A: derive implied_request_share and tokens_per_request --------
    banner("Step A: derive tokens_per_request and implied_request_share")
    # Our TPR is model-level (no provider variation), so this divides out
    # after within-(model, date) normalization. We compute it anyway to make
    # the degeneracy visible.
    df["tokens_per_request"] = df["model_tpr"]  # same for all providers of a model on a date
    df["raw_implied_req"] = df["token_share"] / df["tokens_per_request"]
    df["implied_request_share"] = (
        df.groupby(["model_id", "date"])["raw_implied_req"]
        .transform(lambda x: x / x.sum())
    )

    # Verify the degeneracy is exact (modulo float)
    diff = (df["implied_request_share"] - df["token_share"]).abs()
    print(f"  TPR scale (model-level, all providers per (M, date) share the same value):")
    print(df["tokens_per_request"].describe().round(0).to_string())
    print()
    print(f"  |implied_request_share − token_share| stats:")
    print(f"    mean = {diff.mean():.2e}   max = {diff.max():.2e}")
    if diff.max() < 1e-6:
        print("  CONFIRMED: implied_request_share is identical to token_share")
        print("  (because TPR is model-level, no per-provider variation).")
        print("  Step B's routing fit will reproduce v1's Experiment A coefficients.")
    print()

    # ---- Step B: routing fit on implied_request_share -----------------------
    banner("Step B: routing weight fit on implied_request_share")
    df["price_in_ratio"] = (df.groupby("model_id")["input_price"]
                            .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["price_out_ratio"] = (df.groupby("model_id")["output_price"]
                             .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["throughput_ratio"] = (df.groupby("model_id")["throughput_p50"]
                              .transform(lambda x: x / x.median() if x.median() > 0 else x))
    df["downtime_pct"] = 100.0 - df["uptime_1d_pct"]

    # Drop price_in_ratio per v2 (collinearity with price_out_ratio)
    routing_features = ["price_out_ratio", "throughput_ratio", "downtime_pct"]
    df["y_routing"] = logit(df["implied_request_share"].clip(EPS, 1 - EPS))

    X = sm.add_constant(df[routing_features].to_numpy())
    y = df["y_routing"].to_numpy()
    # Weights: use model_requests where we have them (fewer rows but real volume),
    # else daily_tokens as a proxy.
    weights = df["model_requests"].fillna(df["daily_tokens"]).to_numpy()

    routing = sm.WLS(y, X, weights=weights).fit()
    print(routing.summary(xname=["const"] + routing_features))
    print()

    expected = {"price_out_ratio": -1, "throughput_ratio": +1, "downtime_pct": -1}
    print("Sign check vs pre-commitment:")
    for i, feat in enumerate(routing_features):
        coef = routing.params[i + 1]
        pval = routing.pvalues[i + 1]
        sign_ok = np.sign(coef) == expected[feat]
        match = "OK" if sign_ok else "WRONG SIGN"
        sig = "p<0.10" if pval < 0.10 else f"NS (p={pval:.3f})"
        print(f"  {feat:18s} coef={coef:+.4f}  [{match}]  {sig}")
    print()
    print("VIF (>5 = collinearity warning):")
    for i, feat in enumerate(routing_features):
        v = variance_inflation_factor(X, i + 1)
        print(f"  {feat:18s} VIF={v:.2f}{'  <-- HIGH' if v > 5 else ''}")
    print()

    with open(OUT / "decomposition_v1_routing_summary.txt", "w") as f:
        f.write(str(routing.summary(xname=["const"] + routing_features)))

    df["pred_routing_logit"] = routing.predict(X)
    df["pred_routing_raw"] = 1.0 / (1.0 + np.exp(-df["pred_routing_logit"]))
    df["pred_routing_weight"] = (
        df.groupby(["model_id", "date"])["pred_routing_raw"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    )

    # ---- Step C: TPR fit on model identity ----------------------------------
    banner("Step C: tokens-per-request fit on model-level features")
    print("Target: log(model_tpr). Features: model-id one-hot dummies only.")
    print("Provider features INTENTIONALLY excluded — provider-driven routing of")
    print("long-context requests would create simultaneity bias with the routing fit.")
    print()

    df["log_tpr"] = np.log(df["model_tpr"])
    # One row per (model_id, date) for the TPR fit — TPR doesn't vary by provider
    tpr_rows = df.drop_duplicates(subset=["model_id", "date"]).copy()
    print(f"TPR fit on {len(tpr_rows)} (model, date) observations across "
          f"{tpr_rows['model_id'].nunique()} models.")

    model_dummies = pd.get_dummies(tpr_rows["model_id"], prefix="model", drop_first=True)
    X_tpr = sm.add_constant(model_dummies.astype(float).to_numpy())
    y_tpr = tpr_rows["log_tpr"].to_numpy()
    tpr_model = sm.OLS(y_tpr, X_tpr).fit()
    print(f"  R² = {tpr_model.rsquared:.4f}")
    print(f"  (interpretation: fraction of log(TPR) variance explained by model identity)")
    if tpr_model.rsquared > 0.40:
        print(f"  → above 0.40 threshold: M-level decomposition holds for these models")
    elif tpr_model.rsquared > 0.20:
        print(f"  → 0.20-0.40: partial; some non-model variance remains")
    else:
        print(f"  → <0.20: model identity is NOT the dominant TPR driver; "
              f"decomposition shape may be wrong")
    print()

    with open(OUT / "decomposition_v1_tpr_summary.txt", "w") as f:
        f.write(str(tpr_model.summary()))

    # Predict TPR for all rows
    full_dummies = pd.get_dummies(df["model_id"], prefix="model", drop_first=True)
    # Align columns — any model unseen in tpr_rows will be missing dummies
    for col in model_dummies.columns:
        if col not in full_dummies.columns:
            full_dummies[col] = 0
    full_dummies = full_dummies[model_dummies.columns]
    X_pred = sm.add_constant(full_dummies.astype(float).to_numpy())
    df["pred_log_tpr"] = tpr_model.predict(X_pred)
    df["pred_tokens_per_request"] = np.exp(df["pred_log_tpr"])

    # ---- Step D: implied multi-homing boost as residual ---------------------
    banner("Step D: implied multi-homing boost")
    df["pred_token_share_no_boost_raw"] = (
        df["pred_routing_weight"] * df["pred_tokens_per_request"]
    )
    df["pred_token_share_no_boost"] = (
        df.groupby(["model_id", "date"])["pred_token_share_no_boost_raw"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    )
    df["implied_multihoming_boost"] = (
        df["token_share"] / df["pred_token_share_no_boost"]
    )
    df["provider_rank"] = (
        df.groupby(["model_id", "date"])["token_share"]
        .rank(ascending=False, method="dense").astype(int)
    )

    print("Implied boost distribution by within-(model, date) provider rank:")
    boost_by_rank = (df.groupby("provider_rank")["implied_multihoming_boost"]
                     .describe()[["count", "mean", "50%", "std"]].round(3))
    print(boost_by_rank.to_string())
    print()

    top_boost = df[df["provider_rank"] == 1]["implied_multihoming_boost"].median()
    tail_boost = df[df["provider_rank"] >= 3]["implied_multihoming_boost"].median()
    print(f"Pre-commitment check:")
    print(f"  Top-rank median boost  : {top_boost:.3f}  (expect > 1.1 if multi-homing real)")
    print(f"  Tail (rank≥3) median   : {tail_boost:.3f}  (expect < 0.9 if multi-homing real)")
    if top_boost > 1.1 and tail_boost < 0.9:
        verdict = "SUPPORTED: residual pattern consistent with multi-homing structure"
    elif abs(top_boost - 1.0) < 0.15 and abs(tail_boost - 1.0) < 0.15:
        verdict = "FLAT: boost ~1 everywhere — multi-homing not detectable at this granularity"
    else:
        verdict = "MIXED: directional but not clean; likely identification noise with N=4 providers"
    print(f"  VERDICT: {verdict}")
    print()

    # ---- Step E: combined prediction + per-provider bias --------------------
    banner("Step E: combined prediction quality (no boost applied)")
    df["pred_token_share"] = df["pred_token_share_no_boost"]
    df["resid"] = df["token_share"] - df["pred_token_share"]
    mae = float(np.average(df["resid"].abs(), weights=df["daily_tokens"]))
    corr = float(df[["token_share", "pred_token_share"]].corr().iloc[0, 1])
    print(f"  Volume-weighted MAE on share scale: {mae*100:.2f} pp")
    print(f"  Pearson r (observed, predicted)   : {corr:.3f}")
    print()
    print("Per-provider mean residual (positive = under-predicted):")
    bias = df.groupby("provider").agg(
        n=("resid", "size"),
        mean_resid_pp=("resid", lambda x: x.mean() * 100),
        median_resid_pp=("resid", lambda x: x.median() * 100),
    ).round(2)
    print(bias.to_string())
    print()

    # ---- Oracle row ---------------------------------------------------------
    banner("Oracle: openai/gpt-oss-120b — most recent date")
    oracle = df[df["model_id"] == "openai/gpt-oss-120b"]
    if not oracle.empty:
        latest = oracle["date"].max()
        rows = (oracle[oracle["date"] == latest]
                .sort_values("token_share", ascending=False)
                [["provider", "token_share", "pred_routing_weight",
                  "pred_tokens_per_request", "pred_token_share",
                  "implied_multihoming_boost"]])
        print(rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("  gpt-oss-120b not in fitted set.")
    print()

    # ---- Save ---------------------------------------------------------------
    df.to_csv(OUT / "decomposition_v1_with_predictions.csv", index=False)
    multihoming = df.groupby(["model_id", "provider"]).agg(
        n=("implied_multihoming_boost", "size"),
        median_boost=("implied_multihoming_boost", "median"),
        mean_observed_share=("token_share", "mean"),
    ).reset_index()
    multihoming.to_csv(OUT / "decomposition_v1_multihoming.csv", index=False)
    print(f"Saved: {OUT}/decomposition_v1_*.{{txt,csv}}")


if __name__ == "__main__":
    main()
