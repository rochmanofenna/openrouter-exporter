"""DekaLLM daily forecaster — trained on the analytics API (with optional CSV merge).

Pulls DekaLLM's per-model token history from `/db/providers/dekallm/tokens`
(up to ~32 days of finalized daily totals), builds lag/seasonal features,
and trains a global LightGBM per target with persistence + lag_7 baselines
under walk-forward CV.

Targets:
  - Total Tokens         (always available via API)
  - Requests             (only available when CSVs from the OpenRouter
                          dashboard are present in the repo root)
  - Total Cost           (same — needs CSVs)

The API gives us token counts but not requests or USD. If the historical
CSVs (`dekallm_daily_report_*.csv`) are present in the repo root, those
extra targets are merged in by (Date, Model). Otherwise the script
forecasts Total Tokens only and prints a notice.

Usage:
    pip install -r analysis/requirements.txt
    python analysis/forecast_v2.py

Outputs:
    analysis/out/forecast_v2_backtest_per_target.csv
    analysis/out/forecast_v2_backtest_per_model.csv
    analysis/out/forecast_v2_next_day.csv
    analysis/out/forecast_v2_history.png
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.api_client import require_alive

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Always available via API
PRIMARY_TARGET = "Total Tokens"
# Available iff CSVs are merged in
CSV_ONLY_TARGETS = ["Requests", "Total Cost"]

# Trailing-window LGBM training to stay focused on current regime.
TRAIN_LOOKBACK_DAYS = int(os.environ.get("TRAIN_LOOKBACK_DAYS", "14"))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_from_api() -> pd.DataFrame:
    """Pull DekaLLM history from the analytics API.

    Returns a long DataFrame with columns: Date (datetime), Model (str),
    Total Tokens (int). One row per (date, model).
    """
    client = require_alive()
    rows = client.db_provider_tokens("dekallm")
    flat = []
    for row in rows:
        date = row.get("date")
        if not date:
            continue
        for model, tokens in row.get("tokens", {}).items():
            flat.append({"Date": date, "Model": model, "Total Tokens": int(tokens)})
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Model", "Date"]).reset_index(drop=True)
    return df


def load_csv_supplements() -> pd.DataFrame:
    """Look for OpenRouter dashboard CSVs in repo root. If present, return
    a long DataFrame with Requests + Total Cost merged by (Date, Model).
    Empty DataFrame if no CSVs found.
    """
    paths = sorted(glob.glob(str(ROOT / "dekallm_daily_report_*.csv")))
    if not paths:
        return pd.DataFrame()

    frames = []
    for p in paths:
        with open(p) as f:
            lines = f.readlines()
        try:
            header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
        except StopIteration:
            continue
        df = pd.read_csv(p, skiprows=header_idx)
        df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        for c in ["Requests", "Input Tokens", "Output Tokens", "Cached Tokens",
                  "Total Tokens", "Input Cost", "Output Cost", "Total Cost"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df[["Date", "Model"] + [c for c in df.columns
                                              if c in ("Requests", "Input Tokens",
                                                        "Output Tokens", "Total Cost")]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date", "Model"])
    return out.sort_values(["Model", "Date"]).reset_index(drop=True)


def drop_partial_last_day(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    last_date = df["Date"].max()
    last_day = df[df["Date"] == last_date]
    earlier = df[df["Date"] < last_date]
    if earlier.empty:
        return df
    medians = earlier.groupby("Model")["Total Tokens"].median()
    for _, row in last_day.iterrows():
        m = row["Model"]
        if m in medians.index and row["Total Tokens"] < threshold * medians[m]:
            print(f"  [skip] {last_date.date()} appears partial "
                  f"({m}: {row['Total Tokens']:.2e} vs {medians[m]:.2e} median)")
            return df[df["Date"] < last_date].copy()
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    df = df.copy().sort_values(["Model", "Date"]).reset_index(drop=True)

    for col in targets:
        for k in (1, 2, 3, 7):
            df[f"{col}__lag{k}"] = df.groupby("Model")[col].shift(k)

    g = df.groupby("Model")["Total Tokens"]
    df["tokens_roll3_mean"] = g.shift(1).rolling(3, min_periods=2).mean().values
    df["tokens_roll3_std"]  = g.shift(1).rolling(3, min_periods=2).std().values
    df["tokens_roll7_mean"] = g.shift(1).rolling(7, min_periods=2).mean().values
    df["tokens_roll7_std"]  = g.shift(1).rolling(7, min_periods=2).std().values

    # Cross-model context at t-1
    total_by_day = df.groupby("Date")["Total Tokens"].transform("sum")
    df["dekallm_total_t"] = total_by_day
    df["dekallm_total_lag1"] = df.groupby("Model")["dekallm_total_t"].shift(1)
    df["share_lag1"] = df["Total Tokens__lag1"] / df["dekallm_total_lag1"].replace(0, np.nan)

    # If CSV cost data is present, build per-token rate at t-1
    if "Total Cost" in targets:
        df["cost_per_token_lag1"] = df["Total Cost__lag1"] / df["Total Tokens__lag1"].replace(0, np.nan)

    # Calendar
    df["dow"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_month"] = df["Date"].dt.day
    df["days_since_start"] = (df["Date"] - df["Date"].min()).dt.days
    df["model_idx"] = pd.Categorical(df["Model"]).codes
    return df


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class Result:
    target: str
    method: str
    model_id: str
    mae: float
    smape: float
    rel_mae: float
    n: int


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(100 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def rel_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    base = np.mean(np.abs(y_true))
    return float(np.mean(np.abs(y_pred - y_true)) / base) if base else float("nan")


def evaluate(target: str, method: str, df_test: pd.DataFrame, y_pred: np.ndarray) -> list[Result]:
    out = []
    tmp = df_test.assign(y_pred=y_pred)
    for m, grp in tmp.groupby("Model"):
        yt = grp[target].to_numpy(dtype=float)
        yp = grp["y_pred"].to_numpy(dtype=float)
        if len(yt) == 0:
            continue
        out.append(Result(
            target=target, method=method, model_id=str(m),
            mae=float(np.mean(np.abs(yp - yt))),
            smape=smape(yt, yp),
            rel_mae=rel_mae(yt, yp),
            n=len(yt),
        ))
    return out


def walk_forward_splits(df: pd.DataFrame, target: str, n_splits: int = 5, train_frac: float = 0.5):
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    if n < 10:
        return
    start = max(int(train_frac * n), 7)
    fold_size = max(1, (n - start) // n_splits)
    for i in range(n_splits):
        train_end_i = start + i * fold_size
        test_end_i = min(train_end_i + fold_size, n)
        if train_end_i >= n or test_end_i <= train_end_i:
            break
        train = df[df["Date"] <= dates[train_end_i - 1]]
        test = df[(df["Date"] > dates[train_end_i - 1]) & (df["Date"] <= dates[test_end_i - 1])]
        test = test.dropna(subset=[f"{target}__lag7"])
        if len(train) and len(test):
            yield train, test


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
try:
    from lightgbm import LGBMRegressor
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False


def make_feature_cols(df: pd.DataFrame, targets: list[str]) -> list[str]:
    return [c for c in df.columns
            if c not in ["Date", "Model"] + targets
            and df[c].dtype != "object"
            and not any(c.startswith(t + "__") and not c.endswith("__lag1")
                        and not c.endswith("__lag2") and not c.endswith("__lag3")
                        and not c.endswith("__lag7") for t in targets)]


CAT_COLS = ["model_idx", "dow", "is_weekend"]


def train_lgbm(target: str, train: pd.DataFrame, test: pd.DataFrame,
               feature_cols: list[str]):
    cutoff = train["Date"].max() - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS)
    tr = train[train["Date"] >= cutoff].dropna(subset=[f"{target}__lag1"])
    te = test.dropna(subset=[f"{target}__lag1"]).copy()
    if len(tr) < 20 or len(te) == 0:
        return None, None
    y_tr = np.log1p(tr[target].clip(lower=0).to_numpy(dtype=float))
    X_tr = tr[feature_cols].fillna(-1)
    X_te = te[feature_cols].fillna(-1)
    cats = [c for c in CAT_COLS if c in feature_cols]
    model = LGBMRegressor(
        objective="regression",
        num_leaves=15,
        n_estimators=400,
        learning_rate=0.05,
        min_data_in_leaf=3,
        feature_fraction=0.8,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=cats)
    return te, np.expm1(np.clip(model.predict(X_te), 0, None))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def make_inference_frame(df: pd.DataFrame, targets: list[str]):
    last_date = df["Date"].max()
    next_date = last_date + timedelta(days=1)
    placeholders = []
    for model in df["Model"].unique():
        row = {"Date": next_date, "Model": model}
        for t in targets:
            row[t] = np.nan
        placeholders.append(row)
    full = pd.concat([df, pd.DataFrame(placeholders)], ignore_index=True)
    return build_features(full, targets), next_date


def fit_predict_final(target: str, feat: pd.DataFrame, inf_rows: pd.DataFrame,
                      feature_cols: list[str]):
    cutoff = feat["Date"].max() - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS)
    # last_date in feat = next_date row, so the most recent *actual* data is yesterday.
    actual = feat[feat[target].notna()]
    cutoff = actual["Date"].max() - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS)
    tr = actual[actual["Date"] >= cutoff].dropna(subset=[f"{target}__lag1"])
    if len(tr) < 20:
        return None
    y_tr = np.log1p(tr[target].clip(lower=0).to_numpy(dtype=float))
    X_tr = tr[feature_cols].fillna(-1)
    X_inf = inf_rows[feature_cols].fillna(-1)
    cats = [c for c in CAT_COLS if c in feature_cols]
    model = LGBMRegressor(
        objective="regression",
        num_leaves=15,
        n_estimators=400,
        learning_rate=0.05,
        min_data_in_leaf=3,
        feature_fraction=0.8,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=cats)
    return np.expm1(np.clip(model.predict(X_inf), 0, None))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading DekaLLM history from analytics API...")
    df = load_from_api()
    if df.empty:
        print("API returned no DekaLLM data. Aborting.")
        return
    print(f"  {len(df)} rows, {df['Model'].nunique()} models, "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")

    supplements = load_csv_supplements()
    targets = [PRIMARY_TARGET]
    if not supplements.empty:
        df = df.merge(supplements, on=["Date", "Model"], how="left",
                      suffixes=("", "_csv"))
        # Prefer API tokens (more authoritative), keep CSV-only fields
        if "Total Tokens_csv" in df.columns:
            df = df.drop(columns=["Total Tokens_csv"])
        for t in CSV_ONLY_TARGETS:
            if t in df.columns:
                targets.append(t)
        print(f"  Merged CSV supplements: added {set(targets) - {PRIMARY_TARGET}}")
    else:
        print("  No CSVs found in repo root; forecasting Total Tokens only. "
              "Drop dekallm_daily_report_*.csv files in the repo root to get "
              "Requests + Total Cost forecasts.")

    df = drop_partial_last_day(df)

    zai = df["Model"].str.startswith("z-ai/")
    if zai.any():
        print(f"  [drop] {zai.sum()} z-ai/* rows (retired models)")
        df = df[~zai].copy()

    feat = build_features(df, targets)
    print(f"Features: {feat.shape[1]} cols")

    feature_cols = [c for c in feat.columns
                    if c not in ["Date", "Model"] + targets
                    and feat[c].dtype != "object"]

    results: list[Result] = []
    for target in targets:
        print(f"\n=== target: {target} ===")
        for train, test in walk_forward_splits(feat, target, n_splits=5, train_frac=0.5):
            persist = test[f"{target}__lag1"].fillna(test[target].median()).to_numpy()
            results += evaluate(target, "persistence", test, persist)

            snaive = (test[f"{target}__lag7"].fillna(test[f"{target}__lag1"])
                      .fillna(test[target].median()).to_numpy())
            results += evaluate(target, "lag7_naive", test, snaive)

            if target == "Total Tokens":
                ravg = (test["tokens_roll3_mean"].fillna(test[f"{target}__lag1"])
                        .to_numpy())
                results += evaluate(target, "roll3_mean", test, ravg)

            if not HAVE_LGB:
                continue

            if target == "Total Cost":
                # Derive cost = predicted tokens × yesterday's per-token rate
                te, tok_pred = train_lgbm("Total Tokens", train, test, feature_cols)
                if te is None:
                    continue
                rate = te["cost_per_token_lag1"]
                med_rate = te.groupby("Model")["cost_per_token_lag1"].transform("median")
                rate = rate.fillna(med_rate).fillna(rate.median()).to_numpy()
                cost_pred = tok_pred * rate
                results += evaluate("Total Cost", "derived_from_tokens", te, cost_pred)
                continue

            te, y_pred = train_lgbm(target, train, test, feature_cols)
            if te is None:
                continue
            results += evaluate(target, "lightgbm", te, y_pred)

    if not results:
        print("\nNo backtest results — need at least 10 days of history.")
        return

    res_df = pd.DataFrame([r.__dict__ for r in results])
    agg_global = (res_df.groupby(["target", "method"], as_index=False)
                  .agg(mae=("mae", "mean"), smape=("smape", "mean"),
                       rel_mae=("rel_mae", "mean"))
                  .sort_values(["target", "rel_mae"]))
    print("\n=== Global accuracy per target (lower = better) ===")
    print(agg_global.to_string(index=False))

    agg_per_model = (res_df.groupby(["target", "model_id", "method"], as_index=False)
                     .agg(mae=("mae", "mean"), smape=("smape", "mean"),
                          rel_mae=("rel_mae", "mean"), n=("n", "sum"))
                     .sort_values(["target", "model_id", "method"]))
    agg_per_model.to_csv(OUT / "forecast_v2_backtest_per_model.csv", index=False)
    agg_global.to_csv(OUT / "forecast_v2_backtest_per_target.csv", index=False)

    # Next-day forecast
    if HAVE_LGB:
        inf_feat, next_date = make_inference_frame(df, targets)
        inf_rows = inf_feat[inf_feat["Date"] == next_date].copy()
        print(f"\n=== Next-day forecast for {next_date.date()} ===\n")
        pred_by_target: dict[str, np.ndarray] = {}
        for target in [PRIMARY_TARGET] + [t for t in targets if t != "Total Cost" and t != PRIMARY_TARGET]:
            y_pred = fit_predict_final(target, inf_feat, inf_rows, feature_cols)
            if y_pred is not None:
                pred_by_target[target] = y_pred

        if "Total Cost" in targets and "Total Tokens" in pred_by_target:
            rate = inf_rows["cost_per_token_lag1"]
            med_rate = inf_rows.groupby("Model")["cost_per_token_lag1"].transform("median")
            rate = rate.fillna(med_rate).fillna(rate.median()).to_numpy()
            pred_by_target["Total Cost"] = pred_by_target["Total Tokens"] * rate

        persistence_by_target = {}
        for t in targets:
            persistence_by_target[f"{t} (persistence)"] = inf_rows[f"{t}__lag1"].fillna(0).to_numpy()

        next_day = []
        for i, m in enumerate(inf_rows["Model"].to_numpy()):
            row = {"Date": next_date.date(), "Model": m}
            for t, arr in pred_by_target.items():
                row[t] = float(arr[i])
            for t, arr in persistence_by_target.items():
                row[t] = float(arr[i])
            next_day.append(row)

        pred_df = pd.DataFrame(next_day).sort_values("Total Tokens", ascending=False)
        print(pred_df.to_string(index=False))
        pred_df.to_csv(OUT / "forecast_v2_next_day.csv", index=False)

    # Plot
    try:
        import matplotlib.pyplot as plt
        n = len(targets)
        fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n + 1), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, target in zip(axes, targets):
            for m, grp in df.groupby("Model"):
                ax.plot(grp["Date"], grp[target], label=m, linewidth=1.2)
            ax.set_title(target)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("date (UTC)")
        axes[0].legend(fontsize=7, loc="upper left")
        fig.suptitle("DekaLLM daily history (analytics API)")
        fig.tight_layout()
        fig.savefig(OUT / "forecast_v2_history.png", dpi=120)
        print(f"\nSaved plot → {OUT}/forecast_v2_history.png")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
