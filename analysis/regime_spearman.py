"""regime_spearman.py — bimodal/unimodal test for routing-regime detection.

PRE-COMMITMENT (written before running):
  OpenRouter's documented default routing is `weight ∝ 1/price²` filtered by
  uptime. Under that regime, price rank and share rank are tightly inverse
  (Spearman ρ ≈ -1: cheapest provider wins). Apps that pin providers or use
  :nitro / :floor shortcuts break this — pinned share doesn't depend on
  price rank at all, so Spearman ρ ≈ 0.

  If the per-(model, date) Spearman distribution is BIMODAL with peaks
  around ρ ≈ -0.8 (default-routed cluster) and ρ ≈ 0 (pinned/sorted
  cluster), the three-regime mixture story is real and the full classifier
  is worth building.

  If the distribution is UNIMODAL (single peak, regardless of where), then
  routing regime varies more per-request than per-model and we can't
  separate it from public share data alone.

Method:
  For each (model_id, date) with ≥3 providers having both output_price and
  token_share, compute Spearman ρ between price and share. Histogram the
  results. Also aggregate per-model (mean ρ across dates) for the 131-point
  per-model view.

Inputs:
  analysis/out/share_panel_filled.csv

Outputs:
  analysis/out/regime_spearman.csv         (per (model, date) rho)
  analysis/out/regime_spearman_hist.png    (distribution)
  analysis/out/regime_spearman_by_model.csv (per-model summary)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "out" / "share_panel_filled.csv"
OUT = ROOT / "out"


def banner(s: str) -> None:
    print("=" * 78)
    print(s)
    print("=" * 78)


def main() -> None:
    if not PANEL.exists():
        raise SystemExit(f"Panel not found: {PANEL}. Run dump_prometheus_panel.py first.")

    df = pd.read_csv(PANEL, parse_dates=["date"])
    df = df.dropna(subset=["token_share", "output_price", "input_price"]).copy()
    df = df[df["token_share"] > 0].copy()
    print(f"Loaded {len(df):,} rows with share + price across "
          f"{df['model_id'].nunique()} models, {df['provider'].nunique()} providers, "
          f"{df['date'].nunique()} dates")

    # ---- Per (model, date) Spearman -----------------------------------------
    rows = []
    for (model, date), g in df.groupby(["model_id", "date"]):
        if len(g) < 3:
            continue
        # Spearman price vs share; in the data they often have ties (multiple
        # providers at the same advertised price). spearmanr handles ties.
        rho_out, p_out = stats.spearmanr(g["output_price"], g["token_share"])
        rho_in, p_in = stats.spearmanr(g["input_price"], g["token_share"])
        rows.append({
            "model_id": model,
            "date": date,
            "n_providers": len(g),
            "rho_out_price_share": rho_out if not pd.isna(rho_out) else 0.0,
            "rho_in_price_share":  rho_in if not pd.isna(rho_in) else 0.0,
            "p_value_out": p_out if not pd.isna(p_out) else 1.0,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No (model, date) pairs have ≥3 providers — "
                         "need a denser feature panel.")
    res.to_csv(OUT / "regime_spearman.csv", index=False)
    print(f"\nComputed Spearman for {len(res):,} (model, date) pairs "
          f"(each with ≥3 providers)")
    print(f"  median providers per pair: {res['n_providers'].median():.0f}")
    print(f"  max providers per pair:    {res['n_providers'].max()}")
    print()

    # ---- Distribution stats -------------------------------------------------
    banner("Per-(model, date) Spearman ρ (output price vs share)")
    print(res["rho_out_price_share"].describe().round(3).to_string())
    print()

    # ---- Cluster counts vs mentor's pre-committed thresholds ---------------
    rho = res["rho_out_price_share"]
    default_cluster = (rho < -0.5).sum()        # documented-routing-like
    ambiguous = ((rho >= -0.5) & (rho < -0.2)).sum()
    pinned_cluster = (rho >= -0.2).sum()
    n = len(res)
    banner("Cluster counts")
    print(f"  Default-routed-like  (ρ < -0.5): {default_cluster:>4} "
          f"({100*default_cluster/n:.1f}%)")
    print(f"  Ambiguous   (-0.5 ≤ ρ < -0.2):   {ambiguous:>4} "
          f"({100*ambiguous/n:.1f}%)")
    print(f"  Pinned/sort-like   (ρ ≥ -0.2):   {pinned_cluster:>4} "
          f"({100*pinned_cluster/n:.1f}%)")
    print()

    # ---- Bimodality test (Hartigan dip test as a backup if scipy.stats has it,
    # else just compare bin counts at the two predicted peaks) ---------------
    banner("Bimodality check")
    # Count observations in each predicted peak region
    near_default = ((rho < -0.6) & (rho > -1.01)).sum()
    near_pinned = ((rho > -0.3) & (rho < 0.3)).sum()
    middle = ((rho >= -0.6) & (rho <= -0.3)).sum()
    print(f"  Near default peak  (-1.0 < ρ < -0.6): {near_default:>4}")
    print(f"  Middle / valley    (-0.6 ≤ ρ ≤ -0.3): {middle:>4}")
    print(f"  Near pinned peak   (-0.3 < ρ < +0.3): {near_pinned:>4}")
    print()
    # Heuristic: bimodal if both peaks > middle by a margin
    has_two_peaks = (near_default > 1.3 * middle) and (near_pinned > 1.3 * middle)
    has_default_only = (near_default > 1.3 * middle) and (near_pinned <= middle)
    has_pinned_only = (near_pinned > 1.3 * middle) and (near_default <= middle)

    if has_two_peaks:
        verdict = (
            "BIMODAL: distribution has both clusters. The three-regime "
            "mixture story is supported on this data — full classifier is "
            "worth building."
        )
    elif has_default_only:
        verdict = (
            "UNIMODAL at the DEFAULT peak: most models behave like documented "
            "1/price² routing dominates. Pinned/sort traffic might exist but "
            "is a minority slice. Classifier won't help much."
        )
    elif has_pinned_only:
        verdict = (
            "UNIMODAL at the PINNED peak: most models look pin-dominated. "
            "Levers can't predict share for these models at all — provider "
            "identity / app config does. Need OpenRouter telemetry."
        )
    else:
        verdict = (
            "UNIMODAL/FLAT: no clean peaks. Routing regime varies per-request "
            "rather than per-model. Public share data alone can't separate "
            "regimes — need OpenRouter telemetry to make progress."
        )
    print(f"  VERDICT: {verdict}")
    print()

    # ---- Per-model view -----------------------------------------------------
    banner("Per-model summary (mean ρ across dates)")
    per_model = res.groupby("model_id").agg(
        n_dates=("date", "nunique"),
        median_n_providers=("n_providers", "median"),
        mean_rho=("rho_out_price_share", "mean"),
        std_rho=("rho_out_price_share", "std"),
    ).round(3).sort_values("mean_rho")
    print(f"  {len(per_model)} models analyzed")
    print(f"  Bottom 10 (most default-routed-looking, ρ → -1):")
    print(per_model.head(10).to_string())
    print()
    print(f"  Top 10 (most pinned/sort-looking, ρ → 0+):")
    print(per_model.tail(10).to_string())
    per_model.to_csv(OUT / "regime_spearman_by_model.csv")
    print()

    # ---- DekaLLM-specific cut -----------------------------------------------
    banner("DekaLLM portfolio: which regime are our models in?")
    dek_models = sorted(df[df["provider"] == "dekallm"]["model_id"].unique())
    for m in dek_models:
        if m not in per_model.index:
            continue
        row = per_model.loc[m]
        if row["mean_rho"] < -0.5:
            regime = "default-routed (price competition)"
        elif row["mean_rho"] < -0.2:
            regime = "mixed"
        else:
            regime = "pinned/sort"
        print(f"  {m:<55} ρ={row['mean_rho']:+.3f}  n_dates={int(row['n_dates']):>2}  "
              f"{regime}")
    print()

    # ---- Plot ---------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.hist(res["rho_out_price_share"], bins=30, edgecolor="black", alpha=0.75)
        ax1.axvline(-0.5, color="red", linestyle="--", linewidth=1,
                    label="default cluster boundary (-0.5)")
        ax1.axvline(-0.2, color="orange", linestyle="--", linewidth=1,
                    label="pinned cluster boundary (-0.2)")
        ax1.axvline(res["rho_out_price_share"].median(), color="black",
                    linestyle=":", linewidth=1,
                    label=f"median = {res['rho_out_price_share'].median():.2f}")
        ax1.set_xlabel("Spearman ρ (output price vs token share)")
        ax1.set_ylabel("# (model, date) pairs")
        ax1.set_title(f"Per-(model, date) Spearman distribution\nn={len(res)} pairs across "
                      f"{res['model_id'].nunique()} models, {res['date'].nunique()} dates")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.2)

        ax2.hist(per_model["mean_rho"], bins=20, edgecolor="black", alpha=0.75)
        ax2.axvline(-0.5, color="red", linestyle="--", linewidth=1)
        ax2.axvline(-0.2, color="orange", linestyle="--", linewidth=1)
        ax2.axvline(per_model["mean_rho"].median(), color="black", linestyle=":",
                    linewidth=1,
                    label=f"median = {per_model['mean_rho'].median():.2f}")
        ax2.set_xlabel("Mean Spearman ρ across dates")
        ax2.set_ylabel("# models")
        ax2.set_title(f"Per-model mean Spearman ρ\nn={len(per_model)} models")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)

        fig.tight_layout()
        png = OUT / "regime_spearman_hist.png"
        fig.savefig(png, dpi=120)
        print(f"Saved plot: {png}")
    except ImportError:
        pass

    print(f"Saved data: {OUT / 'regime_spearman.csv'}")
    print(f"Saved data: {OUT / 'regime_spearman_by_model.csv'}")


if __name__ == "__main__":
    main()
