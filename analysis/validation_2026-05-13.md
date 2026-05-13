# Forecast validation — 2026-05-13

| | |
|---|---|
| Predictions made | 2026-05-13 (using training data through 2026-05-12 UTC) |
| Predictions target | 2026-05-13 UTC (full day) |
| Forecaster version | `forecast_v2.py` with 14-day training window, z-ai dropped, persistence baseline column |
| Training rows | 95 (6 models × ~16 days, April 28 → May 12) |
| Distinct days | 26 |

## Methodology recap

`forecast_v2.py` now:

1. Drops the partial day at the end of the CSV (`2026-05-13` was partial when this run executed).
2. Drops `z-ai/*` rows (Hidden status on the DekaLLM dashboard since 2026-04-15 / 2026-05-07).
3. Trains LightGBM only on the last 14 days (April 28 → May 12) to avoid anchoring on
   April low-volume days during the current growth phase.
4. Outputs both LightGBM-derived predictions **and** persistence (= yesterday's actual)
   in the next-day CSV, so the two methods can be compared honestly.

## Predictions for 2026-05-13

### LightGBM (with derived cost)

| Model | Predicted Tokens | Predicted Requests | Predicted Cost |
|---|---:|---:|---:|
| mistralai/mistral-nemo | 892,573,700 | 635,359 | $18.24 |
| openai/gpt-oss-120b | 641,063,500 | 437,438 | $31.01 |
| google/gemma-4-26b-a4b-it-20260403 | 509,575,200 | 212,478 | $42.37 |
| nvidia/nemotron-3-super-120b-a12b-20230311 | 390,842,900 | 67,093 | $37.56 |
| minimax/minimax-m2.7-20260318 | 147,500,700 | 20,604 | $41.55 |
| qwen/qwen3.5-35b-a3b-20260224 | 69,634,890 | 21,390 | $20.20 |
| **TOTAL (LGBM)** | **~2.65 B** | **~1,394,362** | **~$191** |

### Persistence (= May 12 finalized actuals)

| Model | Predicted Tokens | Predicted Requests | Predicted Cost |
|---|---:|---:|---:|
| mistralai/mistral-nemo | 4,066,173,404 | 1,928,088 | $83.09 |
| openai/gpt-oss-120b | 1,668,753,725 | 673,778 | $80.71 |
| google/gemma-4-26b-a4b-it-20260403 | 1,089,933,157 | 409,070 | $90.62 |
| nvidia/nemotron-3-super-120b-a12b-20230311 | 1,138,675,921 | 69,085 | $109.42 |
| minimax/minimax-m2.7-20260318 | 991,366,042 | 78,395 | $279.28 |
| qwen/qwen3.5-35b-a3b-20260224 | 96,952,459 | 9,575 | $28.13 |
| **TOTAL (persistence)** | **~9.05 B** | **~3,167,991** | **~$671** |

## Backtest accuracy at time of this run

Walk-forward CV across 5 folds, last 14 days only for LGBM training.

### Global (lower = better)

| Target | Method | rel_mae | sMAPE | Winner |
|---|---|---:|---:|:---:|
| Total Tokens | LightGBM | 0.295 | 27.8% | **LGBM** |
| Total Tokens | persistence | 0.408 | 39.7% | |
| Requests | persistence | 0.552 | 37.4% | **persistence** |
| Requests | LightGBM | 0.726 | 46.3% | |
| Total Cost | derived_from_tokens | 0.308 | 27.9% | **derived** |
| Total Cost | persistence | 0.363 | 35.5% | |

Note the split: **LGBM wins on tokens** (high autocorrelation + features extract signal),
**persistence wins on requests** (low autocorrelation, feature noise hurts). Different
targets have different best estimators given this much data — that's not a contradiction,
it's information.

## Pre-registered decision criterion

To avoid retroactive interpretation tomorrow, the comparison rule is fixed now:

> **Persistence wins May 13** if, against May 13 finalized actuals, persistence has
> lower absolute error than LightGBM on **Total Cost** for at least **4 of 6** models
> AND lower error in aggregate (sum across models).
>
> **LightGBM wins May 13** on the symmetric criterion (4/6 models + aggregate).
>
> Anything in between (e.g., 3/6, or split between per-model and aggregate) is
> **"no decision — wait for May 14."**

Two consecutive days of persistence winning under this rule is sufficient evidence to
ship persistence as the primary production forecast and document LGBM as a secondary
"baseline volume estimate" column. One day isn't enough.

May 12 result (yesterday) under this rule: **persistence won** (5/5 finalized models had
lower error on Cost than LGBM, and aggregate error was −1% vs LGBM's −54%). So May 13
is potentially the second consecutive day; if it confirms, we ship the change.

## Finalization check — fill in once May 13 UTC closes

Run on the VM after a fresh CSV download:

```bash
grep '^"2026-05-13"' dekallm_daily_report_2026_May.csv
```

| Model | Actual Tokens | Actual Requests | Actual Cost | LGBM Cost Err | Persist. Cost Err |
|---|---:|---:|---:|---:|---:|
| mistralai/mistral-nemo |  |  |  |  |  |
| openai/gpt-oss-120b |  |  |  |  |  |
| google/gemma-4-26b-a4b-it-20260403 |  |  |  |  |  |
| nvidia/nemotron-3-super-120b-a12b-20230311 |  |  |  |  |  |
| minimax/minimax-m2.7-20260318 |  |  |  |  |  |
| qwen/qwen3.5-35b-a3b-20260224 |  |  |  |  |  |
| **TOTAL** |  |  |  |  |  |

Errors as `|predicted − actual|`. Lower is better; bolded winner per model.

## Post-hoc analysis (fill in after actuals land)

- Models where persistence had lower Cost error: **___ / 6**
- Models where LGBM had lower Cost error: **___ / 6**
- Aggregate Cost error — LGBM: **$_____**
- Aggregate Cost error — persistence: **$_____**
- Decision per pre-registered rule: **persistence wins / LGBM wins / no decision**

If persistence wins (this would be 2 consecutive days), proceed with the production
change: persistence becomes the primary forecast; LGBM stays as a secondary column for
"expected baseline volume excluding spikes."

If LGBM wins, log the reversal and look at what changed between May 12 and May 13.

If no decision, run May 14 under the same criterion before changing anything.

## Honest framing for the report to DekaLLM

This is the second daily validation point. The story to tell, once May 13 is finalized:

> "LightGBM with log-transformed targets on a 14-day window of trending data is
> structurally biased low — it regresses toward the log-mean of a non-stationary
> series, which on a growing workload sits below today's level. Persistence (use
> yesterday's value as tomorrow's prediction) outperformed on Total Cost for
> N consecutive days during the current growth phase. Recommendation: use persistence
> as the production forecast while volume is climbing; revisit LGBM once daily volume
> stabilizes and the trend flattens."

Don't over-claim either way. The data does the talking.
