"""Peak-hour load forecaster for DekaLLM on OpenRouter.

Predicts the next 24h hourly token-volume profile per (provider, model) and
aggregates to a provider-level forecast with heteroscedastic confidence intervals.

Design (see analysis/forecast_peak_README.md for full rationale):

1. Per-model forecasts summed to aggregate. Per-model preserves diurnal shape and
   benefits from variance composition (Var(A+B) = Var(A)+Var(B) under independence).

2. Baseline: hour_of_day_mean. For each (model, hour-of-day), predict tomorrow's
   value as the mean of that hour-of-day over the last 7 days. Captures the strong
   diurnal cycle without needing more data than we have.

3. Heteroscedastic variance via coefficient of variation. We compute residuals
   from walk-forward validation as `residual / point_prediction`, take the std of
   that ratio per model, and apply it multiplicatively at each hour:
       predicted_variance[h] = (rel_std * predicted_mean[h])^2
   So peak hours have wider absolute CI than trough hours — essential for capacity
   planning where the peak bound is the actual deliverable.

4. Negative-delta handling. clamp_min(0) at query time. Then check if the diurnal
   mean at UTC 17 and UTC 23 (where system-wide negatives concentrated in the
   inspection script) is suppressed below neighbors — if so, impute from (t-1, t+1)
   average. One paragraph documents the open question of what causes those events.

5. New-model bucket. Models with <7 days non-zero hourly history are forecast via
   persistence on yesterday's hourly profile, not hour_of_day_mean.

6. Validation block. Walk-forward over the last 5 days, reporting:
   - Peak-hour error (target <=1h median)
   - Peak-rate error (target <=30% median)
   - p95 coverage (target ~95%; far from 95% means the CI is mis-calibrated)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
PROVIDER = os.environ.get("PROVIDER", "dekallm")
HISTORY_DAYS = int(os.environ.get("HISTORY_DAYS", "14"))
WALK_FORWARD_DAYS = int(os.environ.get("WALK_FORWARD_DAYS", "5"))
MIN_DAYS_FOR_HOD = 7  # need at least this many days to use hour_of_day_mean baseline
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Prometheus helpers
# ----------------------------------------------------------------------------
def prom_range(query: str, start: datetime, end: datetime, step: int) -> pd.DataFrame:
    resp = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={"query": query, "start": start.timestamp(), "end": end.timestamp(), "step": step},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"prom error: {payload}")
    rows = []
    for series in payload["data"]["result"]:
        labels = series["metric"]
        for ts, val in series["values"]:
            rows.append({
                "ts": pd.Timestamp(int(float(ts)), unit="s", tz="UTC"),
                "value": float(val),
                **labels,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------
# Data extraction
# ----------------------------------------------------------------------------
def load_history(history_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clamped, raw) hourly history DataFrames with columns
    [ts, model_id, delta, date, hour, dow]."""
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=history_days)
    print(f"Pulling {history_days} days of hourly history: {start.isoformat()} -> {end.isoformat()}")

    q_clamped = (
        f'clamp_min(sum by (model_id) '
        f'(delta(openrouter_provider_tokens_daily{{provider="{PROVIDER}"}}[1h])), 0)'
    )
    q_raw = (
        f'sum by (model_id) '
        f'(delta(openrouter_provider_tokens_daily{{provider="{PROVIDER}"}}[1h]))'
    )

    df_c = prom_range(q_clamped, start, end, 3600).rename(columns={"value": "delta"})
    df_r = prom_range(q_raw, start, end, 3600).rename(columns={"value": "delta"})

    for d in (df_c, df_r):
        if not d.empty:
            d["date"] = d["ts"].dt.date
            d["hour"] = d["ts"].dt.hour
            d["dow"] = d["ts"].dt.dayofweek
    return df_c, df_r


# ----------------------------------------------------------------------------
# Model classification
# ----------------------------------------------------------------------------
def classify_models(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (main_pool, other_bucket, excluded) model ids.
    main_pool: >=MIN_DAYS_FOR_HOD days with any non-zero hourly delta
    other_bucket: has non-zero data but <MIN_DAYS_FOR_HOD days
    excluded: all-zero (hidden models)
    """
    main, other, excluded = [], [], []
    for m, grp in df.groupby("model_id"):
        nonzero = grp[grp["delta"] > 0]
        if nonzero.empty:
            excluded.append(m)
            continue
        n_days = nonzero["date"].nunique()
        if n_days >= MIN_DAYS_FOR_HOD:
            main.append(m)
        else:
            other.append(m)
    return sorted(main), sorted(other), sorted(excluded)


# ----------------------------------------------------------------------------
# Imputation check — handle UTC 17 / 23 suppression after clamping
# ----------------------------------------------------------------------------
def hour_of_day_mean(history: pd.DataFrame, model_id: str, days_back: int = MIN_DAYS_FOR_HOD) -> pd.Series:
    """Mean delta per hour-of-day from the last `days_back` days of `history`."""
    cutoff = history["date"].max() - timedelta(days=days_back - 1)
    sub = history[(history["model_id"] == model_id) & (history["date"] >= cutoff)]
    if sub.empty:
        return pd.Series([0.0] * 24, index=range(24))
    mean = sub.groupby("hour")["delta"].mean()
    mean = mean.reindex(range(24), fill_value=0.0)
    return mean


def impute_correction_hours(profile: pd.Series, suspect_hours: tuple[int, ...] = (17, 23),
                            threshold: float = 0.7) -> tuple[pd.Series, list[int]]:
    """For each suspect_hour h, if profile[h] < threshold * mean(profile[h-1], profile[h+1]),
    replace profile[h] with that average. Returns (new_profile, list of imputed hours)."""
    out = profile.copy()
    imputed = []
    for h in suspect_hours:
        h_prev, h_next = (h - 1) % 24, (h + 1) % 24
        neighbor_avg = (profile[h_prev] + profile[h_next]) / 2
        if neighbor_avg > 0 and profile[h] < threshold * neighbor_avg:
            out[h] = neighbor_avg
            imputed.append(h)
    return out, imputed


# ----------------------------------------------------------------------------
# Walk-forward validation — compute relative residual std per model
# ----------------------------------------------------------------------------
def walk_forward_rel_std(history: pd.DataFrame, model_id: str, days_back: int = WALK_FORWARD_DAYS) -> tuple[float, list[dict]]:
    """For each of the last `days_back` complete days, train on prior history and
    compute residuals on that day. Return (rel_std, per_day_records).
    rel_std = std((actual - predicted) / predicted) for predicted > 0."""
    dates = sorted(history["date"].unique())
    # exclude the partial current day (last date if it's today UTC)
    today = datetime.now(timezone.utc).date()
    dates = [d for d in dates if d < today]
    if len(dates) < days_back + 2:
        return float("nan"), []

    records = []
    rel_resids = []
    for d in dates[-days_back:]:
        train = history[history["date"] < d]
        test = history[(history["date"] == d) & (history["model_id"] == model_id)]
        if test.empty or train.empty:
            continue
        pred_profile = hour_of_day_mean(train, model_id)
        pred_profile, _ = impute_correction_hours(pred_profile)
        for _, row in test.iterrows():
            p = pred_profile[row["hour"]]
            a = row["delta"]
            if p > 0:
                rel_resids.append((a - p) / p)
            records.append({"date": d, "hour": row["hour"], "actual": a, "predicted": p})
    if len(rel_resids) < 5:
        return float("nan"), records
    return float(np.std(rel_resids)), records


# ----------------------------------------------------------------------------
# Forecast generation
# ----------------------------------------------------------------------------
@dataclass
class ModelForecast:
    model_id: str
    method: str  # "hour_of_day_mean" or "persistence"
    rel_std: float  # NaN if unknown
    profile: pd.Series  # length-24 indexed by hour 0..23
    imputed_hours: list[int]


def persistence_profile(history: pd.DataFrame, model_id: str) -> pd.Series:
    """Use the most recent complete UTC day's hourly profile as the forecast."""
    today = datetime.now(timezone.utc).date()
    dates = sorted([d for d in history["date"].unique() if d < today])
    if not dates:
        return pd.Series([0.0] * 24, index=range(24))
    last = dates[-1]
    sub = history[(history["model_id"] == model_id) & (history["date"] == last)]
    profile = sub.set_index("hour")["delta"].reindex(range(24), fill_value=0.0)
    return profile


def forecast_one_model(history: pd.DataFrame, model_id: str, method: str) -> ModelForecast:
    if method == "hour_of_day_mean":
        profile = hour_of_day_mean(history, model_id)
        profile, imputed = impute_correction_hours(profile)
        rel_std, _ = walk_forward_rel_std(history, model_id)
    elif method == "persistence":
        profile = persistence_profile(history, model_id)
        imputed = []
        # No walk-forward CV possible with <MIN_DAYS_FOR_HOD history;
        # use a conservative default CV until enough data accumulates.
        rel_std = 0.50  # 50% relative noise — wide on purpose
    else:
        raise ValueError(f"unknown method {method}")
    return ModelForecast(model_id=model_id, method=method, rel_std=rel_std,
                         profile=profile, imputed_hours=imputed)


# ----------------------------------------------------------------------------
# Aggregate + p95
# ----------------------------------------------------------------------------
def aggregate(forecasts: list[ModelForecast]) -> pd.DataFrame:
    """Sum per-model profiles into provider total + p95 bound.
    Returns DataFrame indexed by hour with columns:
    point_total, variance_total, std_total, p50, p95,
    and one column per model_id with that model's point contribution."""
    hours = range(24)
    out = pd.DataFrame(index=hours)
    out.index.name = "hour"
    point = pd.Series(0.0, index=hours)
    var = pd.Series(0.0, index=hours)
    for f in forecasts:
        # contribution column
        out[f.model_id] = f.profile
        point += f.profile
        rs = f.rel_std if not np.isnan(f.rel_std) else 0.5
        var += (rs * f.profile) ** 2
    out["point_total"] = point
    out["variance_total"] = var
    out["std_total"] = np.sqrt(var)
    out["p50"] = point  # median assumed equal to mean for these symmetric residuals
    out["p95"] = point + 1.96 * out["std_total"]
    return out


# ----------------------------------------------------------------------------
# Validation — walk forward over last 5 days, compute peak metrics + coverage
# ----------------------------------------------------------------------------
def validate(history: pd.DataFrame, main_pool: list[str], other_bucket: list[str],
             days_back: int = WALK_FORWARD_DAYS) -> pd.DataFrame:
    today = datetime.now(timezone.utc).date()
    dates = sorted([d for d in history["date"].unique() if d < today])
    if len(dates) < days_back + 2:
        print(f"  walk-forward skipped: only {len(dates)} complete days, need >= {days_back+2}")
        return pd.DataFrame()

    rows = []
    for d in dates[-days_back:]:
        train = history[history["date"] < d]
        if any(train[train["model_id"] == m].empty for m in main_pool + other_bucket):
            continue
        main_fcs = [forecast_one_model(train, m, "hour_of_day_mean") for m in main_pool]
        other_fcs = [forecast_one_model(train, m, "persistence") for m in other_bucket]
        agg = aggregate(main_fcs + other_fcs)

        actual_grp = history[history["date"] == d].groupby("hour")["delta"].sum()
        actual = actual_grp.reindex(range(24), fill_value=0.0)

        pred_peak_hour = int(agg["point_total"].idxmax())
        actual_peak_hour = int(actual.idxmax())
        # circular hour distance: shortest path around the clock
        peak_hour_err = min(
            abs(pred_peak_hour - actual_peak_hour),
            24 - abs(pred_peak_hour - actual_peak_hour),
        )

        pred_peak_rate = float(agg["point_total"][pred_peak_hour])
        actual_peak_rate = float(actual[actual_peak_hour])
        peak_rate_err = (
            abs(pred_peak_rate - actual_peak_rate) / actual_peak_rate
            if actual_peak_rate > 0 else float("nan")
        )

        # p95 coverage: fraction of hours where actual <= predicted_p95
        coverage = float((actual.values <= agg["p95"].values).mean())

        rows.append({
            "date": d, "pred_peak_hour": pred_peak_hour, "actual_peak_hour": actual_peak_hour,
            "peak_hour_err_h": peak_hour_err,
            "pred_peak_rate": pred_peak_rate, "actual_peak_rate": actual_peak_rate,
            "peak_rate_err": peak_rate_err, "p95_coverage": coverage,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    df_c, df_r = load_history(HISTORY_DAYS)
    if df_c.empty:
        print("No data returned.")
        return

    main_pool, other_bucket, excluded = classify_models(df_c)
    print(f"\nModel classification (>= {MIN_DAYS_FOR_HOD} days non-zero for main pool):")
    print(f"  main pool ({len(main_pool)}): {main_pool}")
    print(f"  other bucket ({len(other_bucket)}): {other_bucket}")
    print(f"  excluded ({len(excluded)}): {excluded}")
    if not main_pool:
        print("\nNo models qualify for main pool. Aborting.")
        return

    # Diagnostic: compare clamped vs raw diurnal mean at suspect hours
    print(f"\nClamped vs raw diurnal mean at UTC 17 and 23 (sanity check):")
    for m in main_pool:
        clamped_prof = hour_of_day_mean(df_c, m)
        raw_prof = hour_of_day_mean(df_r, m)
        for h in (17, 23):
            neighbor_avg = (clamped_prof[(h - 1) % 24] + clamped_prof[(h + 1) % 24]) / 2
            ratio = clamped_prof[h] / neighbor_avg if neighbor_avg > 0 else float("nan")
            print(f"  {m[:30]:30s} h={h:2d}  clamped={clamped_prof[h]:.2e}  raw={raw_prof[h]:.2e}  ratio_to_neighbors={ratio:.2f}")

    # Build per-model forecasts
    print(f"\nBuilding per-model forecasts...")
    main_fcs = [forecast_one_model(df_c, m, "hour_of_day_mean") for m in main_pool]
    other_fcs = [forecast_one_model(df_c, m, "persistence") for m in other_bucket]
    all_fcs = main_fcs + other_fcs

    for f in all_fcs:
        n_zero = int((f.profile == 0).sum())
        peak_h = int(f.profile.idxmax())
        peak_v = float(f.profile.max())
        rs_str = f"rel_std={f.rel_std:.2f}" if not np.isnan(f.rel_std) else "rel_std=NaN"
        imp_str = f" imputed_hours={f.imputed_hours}" if f.imputed_hours else ""
        print(f"  {f.model_id[:30]:30s}  method={f.method:18s}  {rs_str}  peak h={peak_h} val={peak_v:.2e}{imp_str}")

    # Aggregate + p95
    agg = aggregate(all_fcs)
    peak_h = int(agg["point_total"].idxmax())
    peak_p50 = float(agg["point_total"][peak_h])
    peak_p95 = float(agg["p95"][peak_h])

    print(f"\n=== Provider-level peak forecast (next 24h) ===")
    print(f"  Peak hour (UTC):      {peak_h:02d}:00   ({(peak_h+7)%24:02d}:00 GMT+7)")
    print(f"  Peak rate p50:        {peak_p50:.3e} tokens/hour")
    print(f"  Peak rate p95:        {peak_p95:.3e} tokens/hour")
    print(f"  P50 -> p95 widening:  +{100*(peak_p95-peak_p50)/peak_p50:.0f}%")
    print(f"\n  Per-model contribution at peak hour:")
    contrib = sorted([(f.model_id, float(f.profile[peak_h])) for f in all_fcs],
                     key=lambda x: -x[1])
    total_at_peak = sum(v for _, v in contrib)
    for m, v in contrib:
        share = 100 * v / total_at_peak if total_at_peak > 0 else 0
        print(f"    {m[:36]:36s}  {v:.3e}  ({share:5.1f}%)")

    # Save forecast CSV
    fc_csv = OUT / "forecast_peak_24h.csv"
    save = agg.copy()
    save.insert(0, "hour_utc", save.index)
    save["hour_gmt7"] = (save["hour_utc"] + 7) % 24
    save.to_csv(fc_csv, index=False)
    print(f"\n  Saved forecast -> {fc_csv}")

    # Save per-model variance budget
    var_csv = OUT / "forecast_peak_model_uncertainty.csv"
    pd.DataFrame([
        {"model_id": f.model_id, "method": f.method, "rel_std": f.rel_std,
         "peak_contribution": float(f.profile.max()),
         "imputed_hours": ",".join(map(str, f.imputed_hours))}
        for f in all_fcs
    ]).to_csv(var_csv, index=False)
    print(f"  Saved per-model uncertainty budget -> {var_csv}")

    # Validation
    print(f"\n=== Walk-forward validation (last {WALK_FORWARD_DAYS} complete UTC days) ===")
    val_df = validate(df_c, main_pool, other_bucket)
    if val_df.empty:
        print("  (no validation — insufficient history)")
    else:
        for _, r in val_df.iterrows():
            print(f"  {r['date']}  peak: pred h={r['pred_peak_hour']:2d} vs actual h={r['actual_peak_hour']:2d}  "
                  f"err={r['peak_hour_err_h']}h   "
                  f"rate err={r['peak_rate_err']*100:.0f}%   "
                  f"p95 cov={r['p95_coverage']*100:.0f}%")
        print(f"\n  Median peak-hour error:  {val_df['peak_hour_err_h'].median():.1f}h    target <=1h")
        print(f"  Median peak-rate error:  {val_df['peak_rate_err'].median()*100:.0f}%    target <=30%")
        print(f"  Mean p95 coverage:       {val_df['p95_coverage'].mean()*100:.0f}%   target ~95%")
        val_csv = OUT / "forecast_peak_validation.csv"
        val_df.to_csv(val_csv, index=False)
        print(f"  Saved validation -> {val_csv}")

    # Plot: stacked area with p95 ribbon
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        fig, ax = plt.subplots(figsize=(12, 5.5))
        hours = list(range(24))
        bottom = np.zeros(24)
        colors = plt.cm.tab10.colors
        for i, f in enumerate(all_fcs):
            vals = f.profile.values / 1e6
            ax.bar(hours, vals, bottom=bottom, label=f.model_id.split("/")[-1][:24],
                   color=colors[i % len(colors)], width=0.9, alpha=0.85)
            bottom = bottom + vals
        # p95 line
        ax.plot(hours, agg["p95"].values / 1e6, color="red", linewidth=1.5, linestyle="--",
                label="aggregate p95 (capacity bound)")
        ax.plot(hours, agg["point_total"].values / 1e6, color="white", linewidth=1.5,
                label="aggregate p50 (point forecast)")
        ax.set_xlabel(f"hour (UTC) — peak {peak_h:02d}:00 UTC = {(peak_h+7)%24:02d}:00 GMT+7")
        ax.set_ylabel("predicted tokens/hour (millions)")
        ax.set_title(f"DekaLLM peak-hour forecast for next 24h")
        ax.set_xticks(hours)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(alpha=0.15, axis="y")
        fig.tight_layout()
        png = OUT / "forecast_peak_stacked.png"
        fig.savefig(png, dpi=120)
        print(f"\n  Saved chart -> {png}")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()
