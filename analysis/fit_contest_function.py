"""Contest-function refinement of the routing prediction.

Hypothesis (from decomposition_v1 Step D residual pattern):
  OpenRouter's routing isn't proportional to lever-derived routing weights.
  The leader takes share consistent with its lever advantage; runners-up
  lose share faster than levers predict. This is a winner-take-most contest
  function, not a multi-homing boost.

Mechanism (best guesses, not yet tested):
  - OpenRouter's default routing is provider-sticky / picks-a-best rather
    than load-balancing proportionally
  - Brand/trust effect: users manually picking "the obvious one"
  - Caching/latency stickiness: once a user lands on a provider, they don't
    switch unless something breaks → leader compounds

Implementation:
  Apply softmax-with-temperature τ to the predicted routing weights from
  decomposition_v1. With τ < 1 the distribution sharpens toward the leader.
  Fit τ as a single free parameter minimizing MAE on observed token_share.

  Sharpened weight: w_sharp_i = w_i^(1/τ) / sum_j w_j^(1/τ)

  τ = 1   → identity (no sharpening)
  τ < 1   → concentrates toward the highest-weight provider
  τ > 1   → smooths toward uniform
  τ → 0   → winner takes all
  τ → ∞   → uniform

PRE-COMMITMENT (written before running):
  - If optimal τ < 0.8, the contest-function reframe is supported.
  - If optimal τ ≈ 1.0, the routing model's spread is already correct and
    the residual pattern in Step D isn't really concentration — it's
    something else (maybe sample noise).
  - If optimal τ > 1.2, predictions need to be MORE spread out, which would
    be a surprise and suggests the routing fit itself is over-concentrated.

  Beyond τ direction: report MAE improvement. <30% relative improvement =
  contest function exists but isn't the dominant residual; >50% = it IS
  the dominant residual.

Inputs:
  analysis/out/decomposition_v1_with_predictions.csv

Outputs:
  analysis/out/contest_function_summary.md
  analysis/out/contest_function_curve.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "out" / "decomposition_v1_with_predictions.csv"
OUT = ROOT / "out"


def sharpen(weights: np.ndarray, tau: float) -> np.ndarray:
    """Softmax-with-temperature on raw routing weights (already normalized to sum=1)."""
    eps = 1e-9
    log_w = np.log(np.clip(weights, eps, None))
    sharp = np.exp(log_w / tau)
    return sharp / sharp.sum()


def apply_contest(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Return a copy of df with sharpened routing weights and re-derived share predictions."""
    out = df.copy()
    sharpened = (out.groupby(["model_id", "date"])["pred_routing_weight"]
                 .transform(lambda x: sharpen(x.to_numpy(), tau)))
    out["pred_routing_sharp"] = sharpened
    # Recompose: sharpened routing × predicted TPR, normalized to sum=1 within (M, date)
    raw = out["pred_routing_sharp"] * out["pred_tokens_per_request"]
    out["pred_token_share_sharp"] = (
        raw.groupby([out["model_id"], out["date"]])
        .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    )
    out["resid_sharp"] = out["token_share"] - out["pred_token_share_sharp"]
    return out


def mae(df: pd.DataFrame, share_col: str = "pred_token_share_sharp") -> float:
    """Volume-weighted MAE on the share scale."""
    err = (df["token_share"] - df[share_col]).abs()
    w = df["daily_tokens"].to_numpy()
    return float(np.average(err, weights=w))


def main() -> None:
    if not PRED_PATH.exists():
        raise SystemExit(f"Predictions not found: {PRED_PATH}. Run fit_decomposition_v1.py first.")

    df = pd.read_csv(PRED_PATH, parse_dates=["date"])
    print(f"Loaded {len(df):,} prediction rows from decomposition_v1")

    # Baseline (τ=1) MAE for reference
    df_baseline = apply_contest(df, tau=1.0)
    baseline_mae = mae(df_baseline)
    print(f"\nBaseline MAE (τ=1.0, no sharpening): {baseline_mae*100:.2f} pp")

    # ---- 1. Fit optimal τ ---------------------------------------------------
    def loss(tau: float) -> float:
        return mae(apply_contest(df, tau))

    result = minimize_scalar(loss, bounds=(0.05, 5.0), method="bounded",
                             options={"xatol": 1e-3})
    tau_opt = float(result.x)
    mae_opt = float(result.fun)
    improvement = (baseline_mae - mae_opt) / baseline_mae

    print(f"\nOptimal τ = {tau_opt:.3f}")
    print(f"  MAE at optimum:  {mae_opt*100:.2f} pp")
    print(f"  Improvement vs τ=1: {improvement*100:.1f}% relative MAE reduction")
    print()

    # ---- 2. Sweep over τ for the curve --------------------------------------
    taus = np.concatenate([
        np.linspace(0.1, 1.0, 19),
        np.linspace(1.05, 3.0, 20),
    ])
    sweep = [(t, mae(apply_contest(df, t))) for t in taus]
    sweep_df = pd.DataFrame(sweep, columns=["tau", "mae_share"])
    sweep_df["mae_pp"] = sweep_df["mae_share"] * 100

    # ---- 3. Re-run the oracle row with optimal τ ----------------------------
    df_opt = apply_contest(df, tau_opt)
    oracle = df_opt[df_opt["model_id"] == "openai/gpt-oss-120b"]
    if not oracle.empty:
        latest = oracle["date"].max()
        oracle_rows = oracle[oracle["date"] == latest].sort_values(
            "token_share", ascending=False
        )

    # ---- 4. Verdict ---------------------------------------------------------
    print("=" * 70)
    print("Verdict vs pre-commitment")
    print("=" * 70)
    if tau_opt < 0.8:
        direction_verdict = (f"SUPPORTED: τ={tau_opt:.2f} < 0.8 means real distribution "
                             "is more concentrated than levers predict. Contest-function "
                             "reframe holds; OpenRouter routing concentrates toward leader.")
    elif tau_opt < 1.2:
        direction_verdict = (f"WEAK: τ={tau_opt:.2f} is close to 1. Routing model's spread "
                             "is roughly correct; residual pattern from Step D isn't dominantly "
                             "concentration. Look elsewhere for the source.")
    else:
        direction_verdict = (f"SURPRISE: τ={tau_opt:.2f} > 1.2 means predictions need to "
                             "be MORE spread out, not less. Routing fit may be over-concentrated.")
    print(f"  {direction_verdict}")

    if improvement > 0.50:
        mag_verdict = (f"MAE improved {improvement*100:.1f}% — contest function "
                       "is the DOMINANT residual.")
    elif improvement > 0.30:
        mag_verdict = (f"MAE improved {improvement*100:.1f}% — contest function "
                       "is real but not the only residual.")
    elif improvement > 0.05:
        mag_verdict = (f"MAE improved {improvement*100:.1f}% — contest function "
                       "captures a small slice; other residual structure remains.")
    else:
        mag_verdict = (f"MAE barely changed ({improvement*100:.1f}%). The Step D pattern "
                       "wasn't driven by concentration after all.")
    print(f"  {mag_verdict}")
    print()

    # ---- 5. Oracle row comparison ------------------------------------------
    print("=" * 70)
    print(f"Oracle: gpt-oss-120b — actual vs τ=1 vs τ={tau_opt:.2f}")
    print("=" * 70)
    if not oracle.empty:
        for _, r in oracle_rows.iterrows():
            base_pred = float(df_baseline[
                (df_baseline["model_id"] == r["model_id"])
                & (df_baseline["date"] == r["date"])
                & (df_baseline["provider"] == r["provider"])
            ]["pred_token_share_sharp"].iloc[0])
            print(f"  {r['provider']:<10s} actual={r['token_share']*100:>6.2f}%   "
                  f"τ=1 pred={base_pred*100:>6.2f}%   "
                  f"τ={tau_opt:.2f} pred={r['pred_token_share_sharp']*100:>6.2f}%   "
                  f"Δ={r['resid_sharp']*100:+6.2f}pp")
    print()

    # ---- 6. Per-provider residual change ------------------------------------
    print("=" * 70)
    print("Per-provider residual bias (positive = actual exceeds predicted)")
    print("=" * 70)
    print(f"  {'Provider':<12} {'τ=1 mean':>10} {'τ_opt mean':>12} {'Δ in bias':>10}")
    for p in sorted(df["provider"].unique()):
        b1 = df_baseline[df_baseline["provider"] == p]["resid_sharp"].mean() * 100
        bo = df_opt[df_opt["provider"] == p]["resid_sharp"].mean() * 100
        print(f"  {p:<12} {b1:>+9.2f}pp {bo:>+11.2f}pp {bo - b1:>+9.2f}pp")
    print()

    # ---- 7. Plot the sweep --------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sweep_df["tau"], sweep_df["mae_pp"], linewidth=2)
        ax.axvline(tau_opt, color="red", linestyle="--", linewidth=1,
                   label=f"τ* = {tau_opt:.3f}")
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label="τ = 1 (no sharpening)")
        ax.set_xlabel("temperature τ")
        ax.set_ylabel("volume-weighted MAE (pp)")
        ax.set_title(f"Contest-function fit: MAE vs temperature\n"
                     f"baseline {baseline_mae*100:.2f}pp → optimum {mae_opt*100:.2f}pp "
                     f"({improvement*100:+.1f}%)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = OUT / "contest_function_curve.png"
        fig.savefig(png_path, dpi=120)
        print(f"Saved plot: {png_path}")
    except ImportError:
        pass

    # ---- 8. Markdown summary ------------------------------------------------
    md = [
        "# Contest-function refinement results",
        "",
        f"Applied softmax-with-temperature τ to the routing weights from decomposition_v1.",
        f"Optimization minimizes volume-weighted MAE on `token_share`.",
        "",
        "## Headline",
        f"- Baseline MAE (τ=1, proportional routing): **{baseline_mae*100:.2f} pp**",
        f"- Optimal τ = **{tau_opt:.3f}**",
        f"- MAE at optimum: **{mae_opt*100:.2f} pp**",
        f"- Relative MAE reduction: **{improvement*100:.1f}%**",
        "",
        "## Verdict",
        f"- {direction_verdict}",
        f"- {mag_verdict}",
        "",
        "## Oracle (gpt-oss-120b, most recent date)",
        "| Provider | Actual | Predicted (τ=1) | Predicted (τ_opt) |",
        "|---|---:|---:|---:|",
    ]
    if not oracle.empty:
        for _, r in oracle_rows.iterrows():
            base_pred = float(df_baseline[
                (df_baseline["model_id"] == r["model_id"])
                & (df_baseline["date"] == r["date"])
                & (df_baseline["provider"] == r["provider"])
            ]["pred_token_share_sharp"].iloc[0])
            md.append(f"| {r['provider']} | {r['token_share']*100:.2f}% | "
                      f"{base_pred*100:.2f}% | {r['pred_token_share_sharp']*100:.2f}% |")
    md.extend([
        "",
        "## Caveats",
        "- A single global τ is a strong simplification — true contest dynamics likely",
        "  vary by model (popular models may concentrate harder than niche ones).",
        "- Fit on 4-provider panel where DekaLLM is the outlier. With wider panel data,",
        "  τ should be re-estimated and likely lands at a different (less extreme) value.",
        "- This is a *practical* fix for the dashboard's predictions, not a structural",
        "  explanation. The mechanism (provider-stickiness vs brand vs caching) is still open.",
    ])
    with open(OUT / "contest_function_summary.md", "w") as f:
        f.write("\n".join(md))
    print(f"Saved md:   {OUT / 'contest_function_summary.md'}")


if __name__ == "__main__":
    main()
