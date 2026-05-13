"""DekaLLM daily forecaster — trained on official OpenRouter CSVs.

Ingests the daily reports exported from DekaLLM's OpenRouter operator
dashboard (~40 days of authoritative per-model usage data with
input/output/cost breakdown), builds a feature set with lag_7 weekly
seasonality, and trains a global LightGBM per target.

Three targets predicted simultaneously:
- Total Tokens : raw volume
- Requests     : how many API calls
- Total Cost   : USD billed

For each target, persistence + lag_7 (week-ago) baselines run alongside
LGBM, and walk-forward CV gives an honest accuracy estimate. The script
finishes by retraining on the full history and predicting tomorrow per
model.

Usage:
    pip install -r analysis/requirements.txt
    python analysis/forecast_v2.py

Files it looks for in the repo root (or current dir):
    dekallm_daily_report_2026_*.csv

Output:
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

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["Total Tokens", "Requests", "Total Cost"]

# How many days back from the train-window's end to use for LGBM fitting.
# Keeps the model focused on the current regime; older April low-volume days
# were anchoring predictions toward the median and producing under-estimates.
TRAIN_LOOKBACK_DAYS = int(os.environ.get("TRAIN_LOOKBACK_DAYS", "14"))


# %% Load and concatenate the daily CSVs
def load_daily_csv(path: Path) -> pd.DataFrame:
    """Skip the 'Provider Daily Report: ...' preamble and parse the data block."""
    with open(path) as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    df = pd.read_csv(path, skiprows=header_idx)
    # Drop the trailing "TOTAL" summary row that the dashboard appends
    df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


csv_paths = sorted(glob.glob(str(ROOT / "dekallm_daily_report_*.csv")))
if not csv_paths:
    raise SystemExit(
        "No dekallm_daily_report_*.csv files found in repo root. "
        "Download them from the DekaLLM OpenRouter dashboard first."
    )

print("Loading:")
for p in csv_paths:
    print(f"  {Path(p).name}")

frames = [load_daily_csv(Path(p)) for p in csv_paths]
df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date", "Model"])
df = df.sort_values(["Model", "Date"]).reset_index(drop=True)


def drop_partial_last_day(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """If the latest date's volume is < threshold * 7-day median for any model,
    treat it as a partial-day CSV export and drop it. Otherwise return as-is."""
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
                  f"({m}: {row['Total Tokens']:.2e} vs {medians[m]:.2e} median) — dropping")
            return df[df["Date"] < last_date].copy()
    return df


df = drop_partial_last_day(df)

# Drop hidden z-ai models — the dashboard has them set to Hidden status
# (since ~2026-04-15 / 2026-05-07 respectively) so they no longer serve
# traffic. Training on the trailing zeros pollutes the model.
zai_mask = df["Model"].str.startswith("z-ai/")
if zai_mask.any():
    print(f"  [drop] {zai_mask.sum()} z-ai/* rows (hidden models)")
    df = df[~zai_mask].copy()

# Coerce numerics (CSV quoted everything as strings)
for c in ["Requests", "Input Tokens", "Output Tokens", "Cached Tokens",
          "Total Tokens", "Input Cost", "Output Cost", "Total Cost"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"\nLoaded {len(df)} rows | "
      f"{df['Model'].nunique()} models | "
      f"{df['Date'].min().date()} → {df['Date'].max().date()} "
      f"({(df['Date'].max() - df['Date'].min()).days + 1} days)")


# %% Feature engineering
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Model", "Date"]).reset_index(drop=True)

    # Per-model lags on all three targets (use prev-day values as features)
    for col in TARGETS + ["Input Tokens", "Output Tokens"]:
        for k in (1, 2, 3, 7):
            df[f"{col}__lag{k}"] = df.groupby("Model")[col].shift(k)

    # Rolling stats on Total Tokens (excluding current row)
    g = df.groupby("Model")["Total Tokens"]
    df["tokens_roll3_mean"] = g.shift(1).rolling(3, min_periods=2).mean().values
    df["tokens_roll3_std"]  = g.shift(1).rolling(3, min_periods=2).std().values
    df["tokens_roll7_mean"] = g.shift(1).rolling(7, min_periods=2).mean().values
    df["tokens_roll7_std"]  = g.shift(1).rolling(7, min_periods=2).std().values

    # Derived ratios at t-1 (don't leak)
    prev_in = df["Input Tokens__lag1"]
    prev_out = df["Output Tokens__lag1"]
    df["io_ratio_lag1"] = prev_out / prev_in.replace(0, np.nan)
    df["req_per_token_lag1"] = df["Requests__lag1"] / df["Total Tokens__lag1"].replace(0, np.nan)
    df["cost_per_token_lag1"] = df["Total Cost__lag1"] / df["Total Tokens__lag1"].replace(0, np.nan)

    # Cross-model context at t-1 (total DekaLLM volume yesterday, this model's share)
    total_by_day = df.groupby("Date")["Total Tokens"].transform("sum")
    df["dekallm_total_t"] = total_by_day
    df["dekallm_total_lag1"] = df.groupby("Model")["dekallm_total_t"].shift(1)
    df["share_lag1"] = df["Total Tokens__lag1"] / df["dekallm_total_lag1"].replace(0, np.nan)

    # Calendar
    df["dow"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_month"] = df["Date"].dt.day
    df["days_since_start"] = (df["Date"] - df["Date"].min()).dt.days

    # Model as categorical int
    df["model_idx"] = pd.Categorical(df["Model"]).codes

    return df


feat = build_features(df)
print(f"Features built: {feat.shape[1]} cols")


# %% Baselines + walk-forward
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
    if base == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred - y_true)) / base)


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


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 5, train_frac: float = 0.5):
    """Yield (train, test) DataFrames stepping forward in time across all models."""
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    if n < 10:
        return
    start = max(int(train_frac * n), 7)  # need at least 7 days for lag_7 to be valid in test
    fold_size = max(1, (n - start) // n_splits)
    for i in range(n_splits):
        train_end_i = start + i * fold_size
        test_end_i = min(train_end_i + fold_size, n)
        if train_end_i >= n or test_end_i <= train_end_i:
            break
        train = df[df["Date"] <= dates[train_end_i - 1]]
        test = df[(df["Date"] > dates[train_end_i - 1]) & (df["Date"] <= dates[test_end_i - 1])]
        # Drop rows missing lag_7 (would force baseline to fall back)
        test = test.dropna(subset=[f"{TARGETS[0]}__lag7"])
        if len(train) and len(test):
            yield train, test


# %% LightGBM training per target
try:
    from lightgbm import LGBMRegressor
    HAVE_LGB = True
except ImportError:
    print("WARN: lightgbm not installed; skipping ML baseline. pip install lightgbm")
    HAVE_LGB = False


feature_cols = [c for c in feat.columns
                if c not in ["Date", "Model"] + TARGETS
                and not c.startswith("Input Tokens") and not c.startswith("Output Tokens")
                and feat[c].dtype != "object"]
cat_cols = ["model_idx", "dow", "is_weekend"]

def train_lgbm(target: str, train: pd.DataFrame, test: pd.DataFrame
               ) -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    """Fit LightGBM on log1p(target). Returns (test rows used, predictions in original scale)."""
    # Restrict training to the most recent TRAIN_LOOKBACK_DAYS so the model
    # doesn't get anchored to early-April low-volume days.
    cutoff = train["Date"].max() - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS)
    tr = train[train["Date"] >= cutoff].dropna(subset=[f"{target}__lag1"])
    te = test.dropna(subset=[f"{target}__lag1"]).copy()
    if len(tr) < 20 or len(te) == 0:
        return None, None
    y_tr = np.log1p(tr[target].clip(lower=0).to_numpy(dtype=float))
    X_tr = tr[feature_cols].fillna(-1)
    X_te = te[feature_cols].fillna(-1)
    model = LGBMRegressor(
        objective="regression",
        num_leaves=15,
        n_estimators=400,
        learning_rate=0.05,
        min_data_in_leaf=3,
        feature_fraction=0.8,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    y_pred = np.expm1(np.clip(model.predict(X_te), 0, None))
    return te, y_pred


results: list[Result] = []

for target in TARGETS:
    print(f"\n=== target: {target} ===")
    for train, test in walk_forward_splits(feat, n_splits=5, train_frac=0.5):
        # Persistence
        persist = test[f"{target}__lag1"].fillna(test[target].median()).to_numpy()
        results += evaluate(target, "persistence", test, persist)

        # Seasonal naive: week-ago same day
        snaive = test[f"{target}__lag7"].fillna(test[f"{target}__lag1"]).fillna(test[target].median()).to_numpy()
        results += evaluate(target, "lag7_naive", test, snaive)

        # 3-day rolling average (meaningful for tokens; falls back to lag1 for other targets)
        ravg = test["tokens_roll3_mean"].fillna(test[f"{target}__lag1"]).to_numpy() if target == "Total Tokens" \
               else test[f"{target}__lag1"].fillna(test[target].median()).to_numpy()
        results += evaluate(target, "roll3_mean", test, ravg)

        if not HAVE_LGB:
            continue

        # LightGBM directly for tokens and requests; cost is derived below.
        if target == "Total Cost":
            continue

        te, y_pred = train_lgbm(target, train, test)
        if te is None:
            continue
        results += evaluate(target, "lightgbm", te, y_pred)

    # Derived cost: predict tokens, multiply by yesterday's per-token cost.
    # Cleaner than training LGBM directly on cost because cost = tokens × rate
    # and the rate is very stable per model, while only the volume varies.
    if HAVE_LGB and target == "Total Cost":
        for train, test in walk_forward_splits(feat, n_splits=5, train_frac=0.5):
            te, tok_pred = train_lgbm("Total Tokens", train, test)
            if te is None:
                continue
            # cost_per_token_lag1 = Total Cost / Total Tokens, both from yesterday.
            # Fallback to model-wise median if lag is NaN.
            rate = te["cost_per_token_lag1"].copy()
            med_rate = te.groupby("Model")["cost_per_token_lag1"].transform("median")
            rate = rate.fillna(med_rate).fillna(rate.median()).to_numpy()
            cost_pred = tok_pred * rate
            results += evaluate("Total Cost", "derived_from_tokens", te, cost_pred)

res_df = pd.DataFrame([r.__dict__ for r in results])
if res_df.empty:
    raise SystemExit("No backtest results. Need ≥10 distinct dates of history.")


# %% Summaries
agg_per_model = (
    res_df.groupby(["target", "model_id", "method"], as_index=False)
    .agg(mae=("mae", "mean"), smape=("smape", "mean"), rel_mae=("rel_mae", "mean"), n=("n", "sum"))
    .sort_values(["target", "model_id", "method"])
)

agg_global = (
    res_df.groupby(["target", "method"], as_index=False)
    .agg(mae=("mae", "mean"), smape=("smape", "mean"), rel_mae=("rel_mae", "mean"))
    .sort_values(["target", "rel_mae"])
)

print("\n=== Global accuracy per target (lower = better) ===")
print(agg_global.to_string(index=False))

print("\n=== Per-model accuracy ===")
print(agg_per_model.to_string(index=False))

agg_per_model.to_csv(OUT / "forecast_v2_backtest_per_model.csv", index=False)
agg_global.to_csv(OUT / "forecast_v2_backtest_per_target.csv", index=False)


# %% Final forecast: retrain on all data, predict next day per model
def predict_next_day() -> pd.DataFrame:
    if not HAVE_LGB:
        return pd.DataFrame()

    # Construct the row to predict: each model, date = last_date + 1
    last_date = feat["Date"].max()
    next_date = last_date + timedelta(days=1)
    rows = []
    for model in feat["Model"].unique():
        latest = feat[feat["Model"] == model].sort_values("Date").iloc[-1].to_dict()
        new = dict(latest)
        # Shift all the lag features forward by one day for prediction at next_date
        new["Date"] = next_date
        # lag_k from new's perspective = the value k days ago; we have data up to last_date,
        # so for next_date: lag_1 = value at last_date, lag_2 = value at last_date - 1, ...
        # The cleanest way: re-run build_features on df + a placeholder for next_date.
        rows.append(new)
    return pd.DataFrame(rows)


# More principled: append placeholder rows for next_date and rebuild features so all lags are correct.
def make_inference_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    last_date = df["Date"].max()
    next_date = last_date + timedelta(days=1)
    placeholders = []
    for model in df["Model"].unique():
        placeholders.append({
            "Date": next_date,
            "Model": model,
            "Requests": np.nan,
            "Input Tokens": np.nan,
            "Output Tokens": np.nan,
            "Cached Tokens": np.nan,
            "Total Tokens": np.nan,
            "Input Cost": np.nan,
            "Output Cost": np.nan,
            "Total Cost": np.nan,
        })
    full = pd.concat([df, pd.DataFrame(placeholders)], ignore_index=True)
    return build_features(full), next_date


def fit_predict_final(target: str, inf_rows: pd.DataFrame) -> np.ndarray | None:
    """Retrain LightGBM on the last TRAIN_LOOKBACK_DAYS, predict for inference rows."""
    cutoff = feat["Date"].max() - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS)
    tr = feat[feat["Date"] >= cutoff].dropna(subset=[f"{target}__lag1"])
    if len(tr) < 20:
        return None
    y_tr = np.log1p(tr[target].clip(lower=0).to_numpy(dtype=float))
    X_tr = tr[feature_cols].fillna(-1)
    X_inf = inf_rows[feature_cols].fillna(-1)
    model = LGBMRegressor(
        objective="regression",
        num_leaves=15,
        n_estimators=400,
        learning_rate=0.05,
        min_data_in_leaf=3,
        feature_fraction=0.8,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    return np.expm1(np.clip(model.predict(X_inf), 0, None))


next_day_predictions: list[dict] = []
if HAVE_LGB:
    inf_feat, next_date = make_inference_frame(df)
    inf_rows = inf_feat[inf_feat["Date"] == next_date].copy()

    print(f"\n=== Next-day forecast for {next_date.date()} ===\n")

    # Train direct models for tokens and requests
    pred_by_target: dict[str, np.ndarray] = {}
    for target in ["Total Tokens", "Requests"]:
        y_pred = fit_predict_final(target, inf_rows)
        if y_pred is not None:
            pred_by_target[target] = y_pred

    # Derive cost = predicted tokens × yesterday's per-token rate
    if "Total Tokens" in pred_by_target:
        rate = inf_rows["cost_per_token_lag1"].copy()
        med_rate = inf_rows.groupby("Model")["cost_per_token_lag1"].transform("median")
        rate = rate.fillna(med_rate).fillna(rate.median()).to_numpy()
        pred_by_target["Total Cost"] = pred_by_target["Total Tokens"] * rate

    # Persistence baseline = yesterday's actual value (= lag_1 at the inference row).
    # On May 12 validation this beat LGBM by a wide margin during the growth phase,
    # so it's worth carrying as a sanity-check forecast alongside the ML predictions.
    persistence_by_target = {
        "Total Tokens (persistence)":  inf_rows["Total Tokens__lag1"].fillna(0).to_numpy(),
        "Requests (persistence)":      inf_rows["Requests__lag1"].fillna(0).to_numpy(),
        "Total Cost (persistence)":    inf_rows["Total Cost__lag1"].fillna(0).to_numpy(),
    }

    # Assemble per-model rows
    for i, m in enumerate(inf_rows["Model"].to_numpy()):
        row = {"Date": next_date.date(), "Model": m}
        for target, arr in pred_by_target.items():
            row[target] = float(arr[i])
        for col, arr in persistence_by_target.items():
            row[col] = float(arr[i])
        next_day_predictions.append(row)

    pred_df = pd.DataFrame(next_day_predictions).sort_values("Total Tokens", ascending=False)
    print(pred_df.to_string(index=False))
    pred_df.to_csv(OUT / "forecast_v2_next_day.csv", index=False)


# %% Plot history per target
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, target in zip(axes, TARGETS):
        for m, grp in df.groupby("Model"):
            ax.plot(grp["Date"], grp[target], label=m, linewidth=1.2)
        ax.set_title(target)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("date (UTC)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("DekaLLM daily history (official CSVs)")
    fig.tight_layout()
    fig.savefig(OUT / "forecast_v2_history.png", dpi=120)
    print(f"\nSaved plot → {OUT}/forecast_v2_history.png")
except Exception as e:
    print(f"plot skipped: {e}")
