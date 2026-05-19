"""Thin client for the local OpenRouter Analytics API (http://localhost:8000).

This service mirrors OpenRouter's undocumented frontend JSON endpoints,
enriches them with provider metadata, and exposes a planner that ranks
GPU deployments. It's hosted on the dekavm internal network and reached
via an SSH LocalForward (`ssh -L 8000:10.250.236.73:8000 dekavm`).

Endpoint groups exposed here:
    scrape/*     — live-ish scrape passthroughs with stale_seconds metadata
    db/*         — DB-cached views (faster, drift up to ~5h from last sync)
    planner/*    — GPU/model deployment optimizer

Design choices:
    - In-process LRU cache so each script-run doesn't refetch the same payload
    - Slug encoding handles forward slashes (mistralai/mistral-nemo)
    - Retries on transient 5xx / connection errors, NOT on 404s
    - Returns raw dicts/lists — callers parse what they need

Caller hygiene:
    - Call `health()` once at script start; treat ConnectionError as
      "API is down" and exit early with a clear message.
    - Prefer db_* methods for analyses (faster, atomic snapshot).
    - Use scrape_* only when you need fresh data or db_* doesn't exist.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30
RETRY_DELAYS = (0.5, 1.5, 3.0)  # seconds; exponential-ish backoff


class APIError(RuntimeError):
    """Raised when the analytics API returns a non-2xx status."""


class APIUnavailable(RuntimeError):
    """Raised when we cannot reach the API at all (likely SSH tunnel dropped)."""


def _encode_slug(slug: str) -> str:
    """URL-encode a model or provider slug. Slashes must be encoded for path segments."""
    return urllib.parse.quote(slug, safe="")


def _request(url: str, method: str = "GET", body: dict | None = None,
             timeout: int = DEFAULT_TIMEOUT) -> Any:
    """GET or POST against the API. Returns parsed JSON. Retries transient failures.

    Raises APIUnavailable if the connection cannot be made after retries.
    Raises APIError for explicit HTTP error responses.
    """
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0.0, *RETRY_DELAYS]):
        if delay:
            time.sleep(delay)
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # 4xx are caller errors — don't retry
            if 400 <= e.code < 500:
                try:
                    detail = e.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                raise APIError(f"HTTP {e.code} {url}: {detail[:300]}") from e
            last_exc = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e

    raise APIUnavailable(f"could not reach {url} after {len(RETRY_DELAYS)+1} attempts: {last_exc}")


class AnalyticsAPI:
    """Single entrypoint into the analytics service.

    Construct one per script; the client memoizes responses inside its lifetime
    so repeated calls (e.g. resolving the same slug multiple times) are free.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ---- core ----------------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return _request(url, timeout=self.timeout)

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self.base}{path}"
        return _request(url, method="POST", body=body, timeout=self.timeout)

    def health(self) -> dict:
        return self._get("/health")

    def is_alive(self) -> bool:
        try:
            self.health()
            return True
        except APIUnavailable:
            return False

    # ---- scrape passthroughs -------------------------------------------------
    def scrape_providers(self) -> list[dict]:
        return self._get("/scrape/providers").get("data", [])

    def scrape_provider_tokens(self, provider_slug: str) -> list[dict]:
        """Per-day, per-model tokens for one provider. 32-day window typical."""
        body = self._get(f"/scrape/providers/{_encode_slug(provider_slug)}/tokens")
        return body.get("data", {}).get("data", [])

    def scrape_models(self, **params) -> list[dict]:
        return self._get("/scrape/models", params or None).get("data", [])

    @lru_cache(maxsize=256)
    def resolve(self, slug: str) -> str:
        """Resolve a slug to its canonical permaslug. Returns input on failure."""
        try:
            body = self._get(f"/scrape/resolve/{_encode_slug(slug)}")
            return body.get("data", {}).get("permaslug") or slug
        except APIError:
            return slug

    def scrape_model_details(self, slug: str) -> dict:
        """Provider endpoints serving this model — pricing, throughput, latency, context."""
        return self._get(f"/scrape/models/{_encode_slug(slug)}/details").get("data", {})

    def scrape_model_uptime(self, slug: str) -> list[dict]:
        return self._get(f"/scrape/models/{_encode_slug(slug)}/uptime").get("data", [])

    def scrape_model_activity(self, slug: str) -> list[dict]:
        return self._get(f"/scrape/models/{_encode_slug(slug)}/activity").get("data", [])

    def scrape_model_provider_tokens(self, slug: str) -> list[dict]:
        body = self._get(f"/scrape/models/{_encode_slug(slug)}/provider-tokens")
        return body.get("data", {}).get("data", [])

    def scrape_rankings(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return self._get("/scrape/rankings", {"limit": limit, "offset": offset}).get("data", [])

    # ---- db views (preferred for analyses) -----------------------------------
    def db_models(self) -> list[dict]:
        return self._get("/db/models").get("data", [])

    def db_providers(self) -> list[dict]:
        return self._get("/db/providers").get("data", [])

    def db_model_details(self, slug: str) -> dict:
        return self._get(f"/db/models/{_encode_slug(slug)}/details").get("data", {})

    def db_model_activity(self, slug: str) -> list[dict]:
        return self._get(f"/db/models/{_encode_slug(slug)}/activity").get("data", [])

    def db_provider_tokens(self, provider_slug: str) -> list[dict]:
        """Per-day token consumption for a provider, broken down by model.

        Returns list of {date: "YYYY-MM-DD", tokens: {model_id: int, ...}}.
        """
        body = self._get(f"/db/providers/{_encode_slug(provider_slug)}/tokens")
        return body.get("data", {}).get("data", [])

    def db_model_provider_tokens(self, slug: str) -> list[dict]:
        """Per-day token consumption for a model, broken down by provider.

        Returns list of {date: "YYYY-MM-DD", providers: {provider_slug: int, ...}}.
        Up to 91 days deep — the widest historical window we have.
        """
        body = self._get(f"/db/models/{_encode_slug(slug)}/provider-tokens")
        return body.get("data", {}).get("data", [])

    def db_rankings(self) -> list[dict]:
        return self._get("/db/rankings").get("data", [])

    def db_sync_status(self) -> list[dict]:
        return self._get("/db/sync-status").get("data", [])

    def db_benchmarks(self) -> list[dict]:
        return self._get("/db/benchmarks").get("data", [])

    # ---- planner -------------------------------------------------------------
    def planner_hardware(self) -> list[dict]:
        return self._get("/planner/hardware").get("data", [])

    def planner_compute(self, body: dict) -> dict:
        """Compute per-model margin for top open-weight models on one hardware.

        Body shape (best-effort; verify against /openapi.json once API is back):
            {"chip": "H100"|"H200"|"L40S"|"B300",
             "dollar_per_gpu_hr": float,        # optional override
             "top_n": int}                      # optional, default ~10
        """
        return self._post("/planner/compute", body)

    def planner_combinations(self, body: dict) -> dict:
        """Find best multi-model deployment combinations on one cluster."""
        return self._post("/planner/combinations", body)

    # ---- convenience helpers -------------------------------------------------
    def dekallm_model_slugs(self) -> list[str]:
        """Return the set of permaslugs DekaLLM has ever served, from db_provider_tokens.

        Computed from the full DekaLLM history so models that came and went
        (e.g. z-ai/glm-4.7-20251222, minimax/minimax-m2.7-20260318) are still
        included. Callers that only want the *current* portfolio should filter
        on the latest date themselves.
        """
        rows = self.db_provider_tokens("dekallm")
        slugs: set[str] = set()
        for row in rows:
            slugs.update(row.get("tokens", {}).keys())
        return sorted(slugs)

    def dekallm_current_model_slugs(self, lookback_days: int = 3) -> list[str]:
        """Return permaslugs DekaLLM served in any of the last `lookback_days` finalized days.

        Drops models that have been retired (e.g. z-ai/glm-4.7) from the result.
        """
        rows = self.db_provider_tokens("dekallm")
        if not rows:
            return []
        # rows arrive newest-first; take the top N finalized days
        recent = rows[:lookback_days]
        slugs: set[str] = set()
        for row in recent:
            slugs.update(row.get("tokens", {}).keys())
        return sorted(slugs)


# ---- module-level helpers ----------------------------------------------------
_default_client: AnalyticsAPI | None = None


def default_client() -> AnalyticsAPI:
    """Return a process-wide default client. Most scripts just want this."""
    global _default_client
    if _default_client is None:
        _default_client = AnalyticsAPI()
    return _default_client


def require_alive(client: AnalyticsAPI | None = None) -> AnalyticsAPI:
    """Use at script entry. Exits the process cleanly if the API isn't reachable.

    Reason for hard-exit: every downstream analysis depends on this data source;
    silently producing empty reports is worse than failing fast with a clear
    message telling the user to check the SSH tunnel.
    """
    c = client or default_client()
    if not c.is_alive():
        import sys
        print(
            "ERROR: analytics API at http://localhost:8000 is not responding.\n"
            "If you're using the SSH tunnel, check it's still up:\n"
            "  ssh -L 8000:10.250.236.73:8000 dekavm\n"
            "Or curl http://localhost:8000/health to verify.",
            file=sys.stderr,
        )
        sys.exit(2)
    return c


if __name__ == "__main__":
    # Quick self-test when run directly.
    c = AnalyticsAPI()
    if not c.is_alive():
        print("API not reachable at", c.base)
        raise SystemExit(2)
    print("health:", c.health())
    sync = c.db_sync_status()
    print(f"sync kinds: {len(sync)}")
    for s in sync:
        print(f"  {s.get('kind'):20s} {s.get('last_status', '?'):10s} "
              f"rows={s.get('rows_written', 0):>6}  "
              f"age={s.get('age_seconds', 0)/3600:.1f}h")
    print(f"\nDekaLLM current models: {c.dekallm_current_model_slugs()}")
    print(f"\nDekaLLM all-time models: {c.dekallm_model_slugs()}")
