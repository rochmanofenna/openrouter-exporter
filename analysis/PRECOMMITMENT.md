# Pre-commitments for Experiment A v2

Written 2026-05-20 BEFORE refitting the share formula on the wider per-model
panel. If predictions disagree with these expectations, debug the fit; if
they agree, the formula is doing what we hoped. No rationalizing after the
fact — this is a falsifiability discipline borrowed from the mentor.

## Expected predictions on the oracle row

For `openai/gpt-oss-120b` with the expanded provider set (~6-10 providers
after we scrape properly), the fitted model should predict shares roughly:

| Provider | Expected predicted share |
|---|---:|
| deepinfra | **80-95%** (dominant) |
| dekallm   | **5-12%** |
| together  | **<5%** |
| any single long-tail provider | **<5%** |

Tolerance: predicted shares within these bands AND the rank order matches
actual = passes the oracle test. A flat ~equal-share prediction or any
provider outside its band = fails.

**Caveat (mentor's catch):** the gpt-oss-120b row is partly a re-prediction
of data we've already seen — we already knew deepinfra dominates from the
4-provider panel. So it's a *consistency* test, not really a falsification.

## Falsification row — providers we've NEVER had token data on

A real test of the formula: take a model where the wider panel will reveal
providers we've never observed, and pre-commit shares for those *before*
seeing the data. The signal is "given price/throughput/uptime we already
know for these providers (from `/api/v1/models`), predict their share."

For `mistralai/mistral-nemo`, the feature data already tells us 4 providers
serve it: DeepInfra, DekaLLM, Mistral, Novita. Of those, **Mistral and Novita
have never appeared in our token panel**.

Pre-committed structural prediction (BEFORE the wider panel arrives):

- **Mistral**: official first-party provider for this model. Expect 5-15%
  share — premium positioning, mid throughput, branding effect. Probably
  not dominant because deepinfra is much cheaper.
- **Novita**: budget commodity provider similar profile to DekaLLM. Expect
  **2-8% share**, likely *below* DekaLLM because their feature numbers don't
  jump out.
- After scraping: **deepinfra + dekallm + Mistral + Novita should sum to >95%**
  of mistral-nemo's market. Tail providers <2% each.

Specifically falsifiable: if Mistral comes in at >30% or <2%, the formula's
price-competitiveness term needs rethinking. If Novita comes in above
DekaLLM despite similar features, the multi-homing factor isn't a scalar
(matches the mentor's earlier critique).

The structural prediction that I'll commit to regardless of which model:
**the provider with the highest `throughput / output_price` ratio will have
share within ±10pp of the largest share observed.** Fill in the number
from the panel once we have it; the prediction is the *band*, made before
seeing the fit.

## Expected coefficient signs

| Lever | Expected sign | Why |
|---|---|---|
| `price_in_ratio` | NEGATIVE | cheaper → more share |
| `price_out_ratio` | NEGATIVE | cheaper → more share |
| `throughput_ratio` | POSITIVE | faster → more share |
| `downtime_pct` | NEGATIVE | downtime → less share |

If any sign is wrong AND significant, debug the fit (likely collinearity or
identification problem), not the formula.

## Validation thresholds

| Outcome | Criteria |
|---|---|
| **Formula validated** | R² > 0.4 on logit AND all signs correct AND oracle row within tolerance above |
| **Formula plausible, needs more features** | R² 0.2-0.4 AND all signs correct AND oracle rank order matches |
| **Structural problem with formula** | R² < 0.2 OR any wrong sign that's also significant |

## What this commits us to

- If the new fit hits validated, report it as such — don't water it down.
- If it hits "needs more features," the next move is to add tool-call quality
  and provider region, not to keep iterating on the same 5 levers.
- If it hits "structural problem," the formula needs rethinking — most likely
  candidates: non-linear interaction between throughput and price, or
  multi-homing factor really is (M, P) and a scalar can't capture it.

## Reading recipe (look at outputs in this order)

1. **Sign check** — non-negotiable first thing to look at.
2. **Oracle row** — does it agree with the table above?
3. **R²** — the absolute number, against the thresholds above.
4. **Per-provider residual bias** — is DekaLLM systematically over/under-predicted?
5. **VIF** — only if signs/R² disagree with expectations.
