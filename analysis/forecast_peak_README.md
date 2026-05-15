# Peak-hour load forecaster — design notes

Script: `analysis/forecast_peak.py`

## What it predicts

Per-model and aggregate **hourly token-volume profile for the next 24 hours**,
with a **p95 upper bound** at each hour and a headline **peak-hour summary**:

> "Tomorrow's peak load is at 22:00 UTC (05:00 GMT+7), expected 850M tokens/hour,
> 95% upper bound 1.1B tokens/hour."

The p95 upper bound is the capacity-planning deliverable. Provision against that.

## Architecture

```
Per-model forecast (hour-of-day mean over last 7 days)
   |
   +----  for each (model, hour): mean -> point estimate
   |
   +----  relative residual std from walk-forward CV
              -> applied multiplicatively at each hour:
                 variance[h] = (rel_std * point[h])^2

Aggregate to provider
   |
   +----  point_total[h] = sum_m point_m[h]
   |
   +----  variance_total[h] = sum_m (rel_std_m * point_m[h])^2
              (independence assumption — see caveat below)
   |
   +----  p95[h] = point_total[h] + 1.96 * sqrt(variance_total[h])
```

## Why per-model, not aggregate

Two reasons, both load-bearing:

1. **Preserves diurnal shape per model.** Different models have different peak
   hours and amplitudes (e.g., mistral peaks differently from qwen).
   Forecasting the aggregate directly is a sum-of-waveforms problem and blurs
   any individual model's shape change into the total. Per-model forecasts
   sum to a sharper aggregate.

2. **Variance composition tightens the bound.** `Var(A+B) = Var(A)+Var(B)+2*Cov(A,B)`.
   If model traffic is roughly independent (different customer pools, different
   schedules), the cross-covariance term is small and per-model variances
   partially cancel when summed — giving a tighter aggregate CI than forecasting
   the aggregate end-to-end.

Bonus: per-model output gives the stacked-area chart that DekaLLM can use to
see *which* model drives peak — actionable for product/customer conversations.

## Model classification rules

For each model:
- **All-zero hours** (e.g., hidden models): **excluded**.
- **>= 7 days of non-zero hourly data**: **main pool**. Forecast with
  `hour_of_day_mean` baseline. Variance from walk-forward CV.
- **< 7 days of non-zero hourly data** (new model): **"other" bucket**.
  Forecast with **persistence on yesterday's hourly profile**. Variance defaulted
  to relative std = 0.50 (50%, wide on purpose) until enough history accumulates.

This is checked at runtime. minimax/glm-4.7 will move between buckets as they
accumulate or lose history.

## Variance estimator — heteroscedastic via coefficient of variation

A flat per-model variance would be too tight at peak (when capacity planning
actually matters) and too loose at trough. Instead:

```
For each model m, walk forward over last 5 complete UTC days:
   for each day d:
      predict day d's 24h profile using hour_of_day_mean trained on days < d
      compute residuals: r_h = actual_h - predicted_h
      compute relative residuals: r_h / predicted_h  (skipping predicted_h == 0)

   rel_std_m = std of relative residuals across all (day, hour) pairs

At forecast time:
   predicted_variance_m[h] = (rel_std_m * predicted_mean_m[h])^2
```

This gives wider CI at high-rate hours, narrower at low-rate hours, from a
single per-model parameter. Assumes constant coefficient of variation — not
perfect (true variance is probably sub-linear in mean), but vastly better than
a flat std for the capacity-planning use case.

## Negative-delta handling

Source data is `clamp_min(sum by (model_id) (delta([1h])), 0)`. Negatives are
rare (~1% of hours in the May 8-15 inspection window) but **not at UTC midnight**,
so they're not counter-reset artifacts.

After clamping, the diurnal mean at UTC 17 and UTC 23 (where most negatives
landed) can be artificially suppressed below their neighbors. The script
auto-detects this and imputes those specific hours from `(h-1 + h+1) / 2` if
the clamped value is < 70% of the neighbor average. Imputed hours are logged
per model.

### What causes the UTC 17 / UTC 23 negative-delta events?

**Currently unresolved.** Three hypotheses:

1. **Provider-side reconciliation.** OpenRouter runs scheduled aggregation jobs
   that retroactively adjust per-model totals. Would explain the system-wide,
   same-time pattern.
2. **Scraper-side counter restart.** If the exporter container restarts at
   those times (cron, k8s liveness, etc.) we'd see synthetic resets. Disprove by
   checking `docker ps --filter name=openrouter-exporter` for restart timestamps.
3. **Billing-cycle close.** Many SaaS bills aggregate on UTC boundaries with
   delayed reporting. UTC 17:00 = US West Coast morning standup; UTC 23:00 = US
   West Coast end-of-day. Plausible "engineer kicks off reconciliation" times.

**One-week followup test.** If the same UTC times show negative deltas next
week, the cause is scheduled (provider or billing). If the hours shift, the cause
is event-driven (manual corrections, releases). The distinction matters for v2
variance modeling — scheduled corrections can be modeled as a fixed offset;
event-driven cannot.

## Validation block — three metrics that matter

Run on every script invocation, against the last 5 complete UTC days:

| Metric | Definition | Target |
|---|---|---|
| **Peak-hour error** | circular distance between predicted and actual peak hour | median <= 1h |
| **Peak-rate error** | abs(predicted_peak - actual_peak) / actual_peak | median <= 30% |
| **p95 coverage** | fraction of hours where actual <= predicted_p95 | mean ~95% |

The **p95 coverage** is the critical one. If it's far from 95% the CI is
mis-calibrated:
- Coverage 70-80%: CI is too tight. Underestimating risk. Increase rel_std.
- Coverage 99%+: CI is too loose. Over-provisioning. Decrease rel_std or
  switch to a tighter quantile.

This is the credibility check. The headline "p95 = X" is just a number unless
this metric says ~95%.

## Caveats — what's wrong with v1

1. **Independence assumption is wrong but useful.** Cross-model correlation
   isn't fitted in v1. Real correlated outages (e.g., a holiday) will see
   actuals drop together more than the aggregate CI suggests. p95 coverage
   will be slightly low on those days. v2: fit residual covariance matrix and
   propagate through the sum.

2. **No weekly seasonality.** Hour-of-day mean over 7 days mixes weekdays and
   weekends. Once we have 14+ days, switch to `(hour_of_day x is_weekend)`
   means for slightly better fit on US workday traffic.

3. **Constant coefficient of variation is an approximation.** True conditional
   variance is probably sub-linear in mean. The heteroscedastic CI is therefore
   slightly too wide at very high rates and slightly too narrow at moderate
   ones. Empirically should still beat flat std handily.

4. **Negative-delta events get clamped + imputed.** If those events represent
   real customer behavior we're missing, we'll under-estimate variance at
   UTC 17 / 23. Tracked via the one-week followup test.

5. **Walk-forward CV uses only last 5 days.** Small sample. Estimates of
   rel_std will have meaningful sampling noise; the actual coverage will
   wander around 95% by a few points. More days helps but also includes more
   non-stationary growth-phase data, so the trade-off is real.

## Outputs

```
analysis/out/forecast_peak_24h.csv             — next-24h forecast (per hour, per model + aggregate)
analysis/out/forecast_peak_model_uncertainty.csv — per-model method, rel_std, peak contribution
analysis/out/forecast_peak_validation.csv       — walk-forward validation rows
analysis/out/forecast_peak_stacked.png          — stacked area chart with p95 ribbon
```

## What DekaLLM should read

The CSV is per-hour and per-model. The chart is the meeting visual. The
headline number is:

> peak rate **p95** at hour H

That's the capacity number. The p50 is for honesty about expected vs.
worst-case but isn't the provisioning bound.

## Production cadence

Re-run daily after the previous UTC day finalizes. The hour-of-day means
naturally roll forward; the rel_std re-estimates each run. No hyperparameter
tuning needed unless the validation metrics drift outside their targets — at
that point we revisit the design.
