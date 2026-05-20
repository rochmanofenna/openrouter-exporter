# Spec: per-(provider, model, date) request counts in Prometheus

## Motivation

The share-formula decomposition (`analysis/fit_decomposition_v1.py`) currently
collapses because our Prometheus panel has tokens per `(provider, model, date)`
but requests only per `(model, date)`. With a single TPR value for all
providers of a model, `implied_request_share` is mathematically identical to
`observed_token_share` after within-(M, date) normalization — Step B of the
decomposition produces zero independent information.

Adding per-(provider, model, date) request counts unblocks the
decomposition's Step B as a real fit and lets us test the per-provider TPR
endogeneity assumption (the mentor's caveat) directly.

## Target metric

```
openrouter_provider_requests_daily{provider, model_id, date} <int64>
```

Labels mirror the existing `openrouter_provider_tokens_daily` exactly so a
join against tokens gives per-row TPR.

## Investigation step (before writing code)

We don't yet know which OpenRouter endpoint exposes this. Candidates, in
priority order:

1. **The existing `/provider/{slug}` chart** (already scraped for tokens by
   `client/activity.go::FetchProviderActivity`). The RSC response *may*
   contain a parallel requests array beyond the `"data":[{"x":"...","ys":{...}}]`
   we currently parse. Inspect the raw body around the chart anchor for any
   sibling array (e.g. `"requests":[{"x":...}]` or `"data2":[...]`).
2. **`/provider/{slug}/analytics`** (best-guess sibling URL). Not yet probed.
3. **`/provider/{slug}/usage`** (best-guess sibling URL). Not yet probed.

Probe procedure (~30 min, requires session cookie):
```bash
# from the VM or wherever the cookie lives
COOKIE='...'  # don't paste; use `read -s COOKIE`
for path in "" "/analytics" "/usage" "/requests"; do
  echo "=== /provider/deepinfra$path ==="
  curl -sH "RSC: 1" -H "Cookie: __session=$COOKIE" \
       "https://openrouter.ai/provider/deepinfra$path" \
       | head -c 5000 | grep -oP '"[a-z_]+":\[\{"x":"[^"]+"' | sort -u
done
```

Whichever candidate returns a chart array with a request-like field wins.

## Implementation outline

Once the endpoint and field name are confirmed:

### `client/activity.go`
- Add a `ProviderRequestPoint` struct (mirroring `ProviderActivityPoint`).
- Add `FetchProviderRequests(ctx, providerSlug, cookie)` that reuses the same
  RSC + session-cookie machinery and the same `findMatchingBracket` parser.
- If the requests data lives on the same `/provider/{slug}` page as tokens
  (Candidate #1), refactor `FetchProviderActivity` to return both arrays in
  one call rather than re-fetching the page.

### `cache/cache.go`
- New field `ProviderRequests map[string][]ProviderRequestPoint` on `CachedData`.
- Wire fetch + cache invalidation alongside the existing `Activity` map.

### `collector/collector.go`
- New `prometheus.Desc` for `openrouter_provider_requests_daily`.
- In the collection loop, emit one sample per `(provider, model_id, date)`
  tuple from the cached `ProviderRequests`, using the same date-string label
  the tokens metric uses.

### `analysis/dump_prometheus_panel.py`
- Add `pull_provider_requests()` mirroring `pull_daily_tokens()`.
- Join via `(provider, base_slug, date)`. Compute true per-(M, P, date) TPR
  as `daily_tokens / provider_requests`.
- Existing `model_tpr` (from `/activity`) stays as a fallback for models we
  don't have provider-level data on.

## Acceptance criteria

1. New metric appears in Prometheus with the labels above.
2. `python3 analysis/dump_prometheus_panel.py` produces a `share_panel.csv`
   where `provider_requests` is non-null for at least the current 4
   configured providers across recent dates.
3. `python3 analysis/fit_decomposition_v1.py` reports that
   `implied_request_share` is **NOT** identical to `token_share` (current
   degeneracy check `max |diff|` should be > 1e-6).
4. Step B's routing fit produces coefficients distinct from a fit on
   `token_share` directly. Whether signs come out correct is a separate
   question — that depends on the wider panel.

## Effort and timing

| Phase | Work | Effort |
|---|---|---|
| Probe | Find the right endpoint + field name | 30 min |
| Go change | Implement scrape + cache + metric | 1.5-2 hr |
| Verify | Redeploy, confirm metric in Prom | 15 min |
| Refresh | Re-run dump + decomposition | 10 min |

**Total: ~3 hours, mostly investigation.**

## Why not block on this

The wider provider panel (`OPENROUTER_ACTIVITY_PROVIDERS` expansion) is the
binding constraint right now. This spec adds a *cleaner target variable* for
Step B but doesn't change the identification problem. Schedule it to land
*after* the wider panel has accumulated 3+ days and the existing fits have
been re-run on that — at that point the request-count addition gives us
genuine per-provider TPR signal on a panel that can actually identify the
formula.

## Open question

If `/provider/{slug}` doesn't expose requests in any chart array, there may
be no scraped path to per-provider request counts at all — OpenRouter might
simply not expose that data publicly. In that case the decomposition's
provider-level TPR fit stays infeasible and we accept the model-level TPR
framing as the operational ceiling.
