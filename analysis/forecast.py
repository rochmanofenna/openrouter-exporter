"""
30-min token-volume forecaster for DekaLLM models on OpenRouter.

Pulls history from Prometheus, builds lag/calendar/exogenous features,
runs persistence + seasonal-naive baselines, fits a global LightGBM
model with walk-forward CV, prints a metrics table, and saves
predictions + a plot.

Usage:
    pip install -r analysis/requirements.txt
    # If running on laptop, tunnel Prometheus first:
    #   ssh -L 9090:localhost:9090 dekavm
    python analysis/forecast.py

    # Or open in Jupyter / VSCode and run cell-by-cell.

Env knobs:
    PROM_URL     default http://localhost:9090
    LOOKBACK_H   default 72
    PROVIDER     default dekallm
"""

# %% Imports and config
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
LOOKBACK_H = int(os.environ.get("LOOKBACK_H", "72"))
PROVIDER = os.environ.get("PROVIDER", "dekallm")
STEP_SECONDS = 30 * 60  # 30-min target horizon
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


# %% Prometheus client
def prom_range(query: str, start: datetime, end: datetime, step: int) -> pd.DataFrame:
    """Range-query Prometheus, return long-format frame with columns
    [ts, value, <label1>, <label2>, ...]. Skips empty series."""
    resp = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")

    rows = []
    for series in payload["data"]["result"]:
        labels = series["metric"]
        for ts, val in series["values"]:
            rows.append({"ts": pd.Timestamp(ts, unit="s", tz="UTC"), "value": float(val), **labels})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def assert_prom_alive() -> None:
    r = requests.get(f"{PROM_URL}/-/healthy", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"prometheus not healthy at {PROM_URL}")


# %% Data extraction
assert_prom_alive()
END = datetime.now(timezone.utc).replace(microsecond=0)
START = END - timedelta(hours=LOOKBACK_H)
print(f"Pulling {LOOKBACK_H}h of history: {START.isoformat()} → {END.isoformat()}")

# Target: per-30-min token delta of the daily running total.
# delta() handles UTC-midnight resets cleanly (negative values get filtered below).
# Sum across the `date` label to collapse the ~14 simultaneously-active daily
# series per model down to one. Finalized historical dates contribute 0 deltas;
# today's bucket contributes real movement.
target_q = (
    f'sum by (model_id) (delta(openrouter_provider_tokens_daily{{provider="{PROVIDER}"}}[30m]))'
)
target_df = prom_range(target_q, START, END, STEP_SECONDS)

if target_df.empty:
    raise SystemExit(
        "No target data. Possible causes:\n"
        " - Prometheus has < 30 min of openrouter_provider_tokens_daily history\n"
        " - PROVIDER label doesn't match (try: PROVIDER=...)\n"
        " - PROM_URL not reachable from here (try ssh tunnel)\n"
    )

target_df = target_df.rename(columns={"value": "y"})[["ts", "model_id", "y"]]
# Drop UTC-midnight rollover rows (negative deltas) and clip noise.
target_df = target_df[target_df["y"] >= 0].copy()
target_df = target_df.sort_values(["model_id", "ts"]).reset_index(drop=True)
print(f"target rows: {len(target_df)}, models: {target_df['model_id'].nunique()}")

# Exogenous features: live endpoint metrics (5-min cadence — average to 30-min step).
def pull_exo(promql: str, name: str) -> pd.DataFrame:
    df = prom_range(promql, START, END, STEP_SECONDS)
    if df.empty:
        return pd.DataFrame(columns=["ts", "model_id", name])
    # Endpoint metrics are labeled by (model_id, provider_name, tag, quantization, ...).
    # Aggregate to model_id by mean across providers+tags so we have one feature per (ts, model_id).
    df = df.rename(columns={"value": name})
    df = df.groupby(["ts", "model_id"], as_index=False)[name].mean()
    return df

exo_queries = {
    "latency_p50": 'openrouter_endpoint_latency_milliseconds{quantile="p50"}',
    "throughput_p50": 'openrouter_endpoint_throughput_tokens_per_second{quantile="p50"}',
    "uptime_5m": "openrouter_endpoint_uptime_percentage_last_5m",
    "input_price": "openrouter_model_input_price_dollars_per_million_tokens",
}
exo_frames = {name: pull_exo(q, name) for name, q in exo_queries.items()}
for name, df in exo_frames.items():
    print(f"  exo {name}: {len(df)} rows")


# %% Feature engineering
def build_features(target: pd.DataFrame, exos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = target.copy()
    # Lags (per model)
    for k in (1, 2, 3, 6):
        df[f"lag_{k}"] = df.groupby("model_id")["y"].shift(k)
    # Yesterday-same-time (48 × 30min)
    df["lag_48"] = df.groupby("model_id")["y"].shift(48)
    # Rolling stats (excluding current row)
    for win in (6, 12, 24):
        df[f"roll_{win}_mean"] = (
            df.groupby("model_id")["y"].shift(1).rolling(win, min_periods=2).mean().values
        )
        df[f"roll_{win}_std"] = (
            df.groupby("model_id")["y"].shift(1).rolling(win, min_periods=2).std().values
        )
    # Calendar
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute

    # Exogenous merges (left join — fine if some endpoint metrics are missing)
    for name, exo in exos.items():
        if not exo.empty:
            df = df.merge(exo, on=["ts", "model_id"], how="left")
        else:
            df[name] = np.nan

    # Cross-model context: total target across all DekaLLM models at this timestamp
    total_t = df.groupby("ts")["y"].transform("sum")
    df["total_dekallm_t"] = total_t
    df["share_of_total"] = (df["y"] / total_t.replace(0, np.nan)).fillna(0)

    # Encode model_id as categorical int
    df["model_id_idx"] = pd.Categorical(df["model_id"]).codes
    return df


feat_df = build_features(target_df, exo_frames)
print(
    f"features built: {len(feat_df)} rows, "
    f"{feat_df.columns.size} cols, "
    f"{feat_df['model_id'].nunique()} models, "
    f"timestamps={feat_df['ts'].nunique()}"
)


# %% Baselines
@dataclass
class Result:
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


def relative_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    base = np.mean(np.abs(y_true))
    if base == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred - y_true)) / base)


def evaluate(method: str, df: pd.DataFrame, y_pred: np.ndarray) -> list[Result]:
    out = []
    df = df.assign(y_pred=y_pred)
    for m, grp in df.groupby("model_id"):
        yt = grp["y"].to_numpy()
        yp = grp["y_pred"].to_numpy()
        if len(yt) == 0:
            continue
        out.append(Result(
            method=method,
            model_id=str(m),
            mae=float(np.mean(np.abs(yp - yt))),
            smape=smape(yt, yp),
            rel_mae=relative_mae(yt, yp),
            n=len(yt),
        ))
    return out


# Walk-forward in time: train on first ~60% of timestamps, predict next slice, slide.
def walk_forward_splits(df: pd.DataFrame, n_splits: int = 4, train_frac: float = 0.6):
    timestamps = np.sort(df["ts"].unique())
    n_ts = len(timestamps)
    if n_ts < 8:
        return
    start = max(int(train_frac * n_ts), 4)
    fold_size = max(1, (n_ts - start) // n_splits)
    for i in range(n_splits):
        train_end_idx = start + i * fold_size
        test_end_idx = min(train_end_idx + fold_size, n_ts)
        if train_end_idx >= n_ts or test_end_idx <= train_end_idx:
            break
        t_train = timestamps[train_end_idx - 1]
        t_test_end = timestamps[test_end_idx - 1]
        train = df[df["ts"] <= t_train]
        test = df[(df["ts"] > t_train) & (df["ts"] <= t_test_end)]
        if len(train) and len(test):
            yield train, test


results: list[Result] = []

for train, test in walk_forward_splits(feat_df, n_splits=4, train_frac=0.6):
    # Persistence: ŷ_{t+1} = y_t (= lag_1)
    persist_pred = test["lag_1"].fillna(test["y"].median()).to_numpy()
    results += evaluate("persistence", test, persist_pred)

    # Seasonal-naive: ŷ_{t+1} = y_{t-48} (yesterday same time)
    snaive_pred = test["lag_48"].fillna(test["lag_1"]).fillna(test["y"].median()).to_numpy()
    results += evaluate("seasonal_naive", test, snaive_pred)


# %% LightGBM (skipped if too little data)
USE_LGB = feat_df["ts"].nunique() >= 30
if not USE_LGB:
    print("\nSkipping LightGBM: need >= 30 distinct timestamps; "
          f"have {feat_df['ts'].nunique()}. Re-run after more data accumulates.")
else:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print("lightgbm not installed; skipping. pip install lightgbm")
        USE_LGB = False

if USE_LGB:
    feature_cols = [
        c for c in feat_df.columns
        if c not in ("ts", "y", "model_id") and feat_df[c].dtype != "object"
    ]
    cat_cols = ["model_id_idx"]

    for train, test in walk_forward_splits(feat_df, n_splits=4, train_frac=0.6):
        # Drop rows where any required lag is NaN — needed at start of series.
        tr = train.dropna(subset=["lag_1"])
        te = test.dropna(subset=["lag_1"]).copy()
        if len(tr) < 20 or len(te) == 0:
            continue
        y_tr = np.log1p(tr["y"].clip(lower=0).to_numpy())
        X_tr = tr[feature_cols].fillna(-1)
        X_te = te[feature_cols].fillna(-1)

        model = LGBMRegressor(
            objective="regression",
            num_leaves=15,
            n_estimators=300,
            learning_rate=0.05,
            min_data_in_leaf=5,
            feature_fraction=0.8,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, categorical_feature=cat_cols)
        y_pred_log = model.predict(X_te)
        y_pred = np.expm1(np.clip(y_pred_log, 0, None))
        results += evaluate("lightgbm", te, y_pred)


# %% Summary table + plot
res_df = pd.DataFrame([r.__dict__ for r in results])
if res_df.empty:
    raise SystemExit("No backtest results — likely too little data. Try LOOKBACK_H=24 first.")

# Aggregate across folds: mean of metrics per (method, model_id), then a global summary.
agg = (
    res_df.groupby(["method", "model_id"], as_index=False)
    .agg(mae=("mae", "mean"), smape=("smape", "mean"), rel_mae=("rel_mae", "mean"), n=("n", "sum"))
    .sort_values(["model_id", "method"])
)
print("\nPer-model metrics (lower = better):\n")
print(agg.to_string(index=False))

global_summary = (
    res_df.groupby("method", as_index=False)
    .agg(mae=("mae", "mean"), smape=("smape", "mean"), rel_mae=("rel_mae", "mean"))
    .sort_values("rel_mae")
)
print("\nGlobal averages:\n")
print(global_summary.to_string(index=False))

agg.to_csv(OUT_DIR / "backtest_per_model.csv", index=False)
global_summary.to_csv(OUT_DIR / "backtest_global.csv", index=False)
print(f"\nSaved metrics → {OUT_DIR}/backtest_*.csv")

# Quick sanity plot: target series per model
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for m, grp in target_df.groupby("model_id"):
        ax.plot(grp["ts"], grp["y"] / 1e6, label=m, linewidth=1)
    ax.set_title(f"30-min token deltas per model — last {LOOKBACK_H}h ({PROVIDER})")
    ax.set_ylabel("tokens (millions, per 30-min window)")
    ax.set_xlabel("UTC time")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_series.png", dpi=120)
    print(f"Saved plot     → {OUT_DIR}/target_series.png")
except Exception as e:
    print(f"plot skipped: {e}")
