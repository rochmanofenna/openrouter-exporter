"""Pull OpenRouter's public model catalog and produce a capability-enriched report.

The /api/v1/models endpoint is unauthenticated and returns every model's:
- pricing (prompt, completion, cache read, image)
- context length / max completion tokens
- architecture (modality, tokenizer, input/output modalities)
- supported_parameters (tools, reasoning, structured outputs, logprobs, ...)
- created timestamp (Unix seconds)

This script:
1. Fetches the full catalog
2. Derives a flat capability profile per model
3. Saves: analysis/out/openrouter_catalog.csv (every model on OpenRouter)
4. Produces: analysis/out/demand_wave_capabilities.md (focused on demand-wave models)
5. Cross-references DekaLLM's portfolio against capability requirements

Usage:
    python analysis/model_catalog.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CATALOG_URL = "https://openrouter.ai/api/v1/models"

# Demand-wave seed authors (per OpenRouter Q1 2026 research).
# The Chinese OSS coding/agent wave + premium OSS reasoning models.
DEMAND_WAVE_AUTHORS = {
    "moonshotai",   # Kimi K2 family
    "deepseek",     # V3.2, R1
    "z-ai",         # GLM 5, 5.1, 4.7-flash
    "minimax",      # M2 family
    "qwen",         # 3.5 / 3-coder / 235B
    "openai",       # gpt-oss family only (filtered below)
    "google",       # gemma-only (filtered below)
    "nvidia",       # nemotron
    "mistralai",    # legacy but DekaLLM serves
}

# Open-weight providers (model authors). Per the research, open-weight = served
# by multiple providers, traffic splits across them. Proprietary = single
# provider (the model author themselves).
OPEN_WEIGHT_AUTHORS = {
    "moonshotai", "deepseek", "z-ai", "minimax", "qwen", "mistralai",
    "meta-llama", "microsoft", "nvidia",
}
# OpenAI and Google are mixed — only their explicitly open-weight families count.
OPEN_WEIGHT_PREFIXES = {
    "openai/gpt-oss",      # OSS family
    "google/gemma",        # Gemma family
}

# DekaLLM's current portfolio (from our earlier audits)
DEKALLM_CURRENT_MODELS_PERMASLUGS = {
    "google/gemma-4-26b-a4b-it-20260403",
    "minimax/minimax-m2.7-20260318",
    "mistralai/mistral-nemo",
    "nvidia/nemotron-3-super-120b-a12b-20230311",
    "openai/gpt-oss-120b",
    "qwen/qwen3.5-35b-a3b-20260224",
    "z-ai/glm-4.7-20251222",
}


def fetch_catalog() -> list[dict]:
    print(f"Fetching {CATALOG_URL} ...")
    r = requests.get(CATALOG_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()
    models = payload.get("data", [])
    print(f"  got {len(models)} models")
    return models


def base_slug(model_id: str) -> str:
    """Strip trailing -YYYYMMDD or -MMDD date suffix if present."""
    parts = model_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) in (4, 6, 8):
        return parts[0]
    return model_id


def is_open_weight(model_id: str) -> bool:
    author = model_id.split("/", 1)[0] if "/" in model_id else ""
    if author in OPEN_WEIGHT_AUTHORS:
        return True
    for prefix in OPEN_WEIGHT_PREFIXES:
        if model_id.startswith(prefix):
            return True
    return False


def is_demand_wave(model_id: str) -> bool:
    if not is_open_weight(model_id):
        return False
    author = model_id.split("/", 1)[0] if "/" in model_id else ""
    return author in DEMAND_WAVE_AUTHORS


def supports(model: dict, param_names: set[str]) -> bool:
    params = set(model.get("supported_parameters") or [])
    return bool(params & param_names)


def parse_pricing(model: dict) -> dict:
    """Extract pricing fields. Values are strings of USD-per-token in the API.
    Convert to USD-per-million-tokens for readability."""
    pricing = model.get("pricing") or {}
    out = {}
    for key in ("prompt", "completion", "image", "request", "input_cache_read",
                "input_cache_write", "web_search", "internal_reasoning"):
        v = pricing.get(key)
        if v is None or v == "":
            out[key] = None
            continue
        try:
            f = float(v)
            # Convert per-token → per-million-tokens for input/output prices
            if key in ("prompt", "completion", "input_cache_read", "input_cache_write"):
                out[key] = f * 1_000_000
            else:
                out[key] = f
        except (TypeError, ValueError):
            out[key] = None
    return out


def profile_model(model: dict) -> dict:
    mid = model.get("id", "")
    arch = model.get("architecture") or {}
    top_provider = model.get("top_provider") or {}
    pricing = parse_pricing(model)
    created = model.get("created")
    days_since_launch = None
    if isinstance(created, (int, float)) and created > 0:
        created_dt = datetime.fromtimestamp(created, tz=timezone.utc)
        days_since_launch = (datetime.now(timezone.utc) - created_dt).days

    profile = {
        "model_id": mid,
        "base_slug": base_slug(mid),
        "name": model.get("name", ""),
        "author": mid.split("/", 1)[0] if "/" in mid else "",
        "created_unix": created,
        "days_since_launch": days_since_launch,
        "is_open_weight": is_open_weight(mid),
        "is_demand_wave_author": is_demand_wave(mid),
        "is_dekallm_current": mid in DEKALLM_CURRENT_MODELS_PERMASLUGS or base_slug(mid) in {base_slug(m) for m in DEKALLM_CURRENT_MODELS_PERMASLUGS},

        # Pricing in $/M tokens
        "input_price_per_M": pricing.get("prompt"),
        "output_price_per_M": pricing.get("completion"),
        "cache_read_price_per_M": pricing.get("input_cache_read"),

        # Context
        "context_length": model.get("context_length") or top_provider.get("context_length"),
        "max_completion_tokens": top_provider.get("max_completion_tokens"),

        # Architecture
        "modality": arch.get("modality", ""),
        "tokenizer": arch.get("tokenizer", ""),
        "input_modalities": ",".join(arch.get("input_modalities") or []),
        "output_modalities": ",".join(arch.get("output_modalities") or []),

        # Capabilities (from supported_parameters)
        "supports_tools": supports(model, {"tools", "tool_choice"}),
        "supports_reasoning": supports(model, {"reasoning", "reasoning_effort", "include_reasoning"}),
        "supports_structured_outputs": supports(model, {"structured_outputs", "response_format"}),
        "supports_logprobs": supports(model, {"logprobs", "top_logprobs"}),
        "supports_web_search": supports(model, {"web_search_options"}),
        "supports_cache": pricing.get("input_cache_read") is not None and (pricing.get("input_cache_read") or 0) > 0,
        "is_multimodal": (arch.get("input_modalities") or ["text"]) != ["text"],

        # Derived flags
        "agent_grade_context": (model.get("context_length") or 0) >= 256_000,
        "long_context": (model.get("context_length") or 0) >= 128_000,
    }
    return profile


def fmt_money(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"${v:.3f}"


def fmt_int(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{int(v):,}"


def fmt_bool(v) -> str:
    return "✓" if v else "·"


def write_report(df: pd.DataFrame) -> str:
    out = []
    out.append("# OpenRouter Model Capability Catalog — Filtered to Demand-Wave Models\n")
    out.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    out.append(f"\nSource: `GET https://openrouter.ai/api/v1/models` (public, unauthenticated).")
    out.append(f"Total models in catalog: {len(df)}\n")

    # Filter to demand-wave models
    demand = df[df["is_demand_wave_author"] & df["is_open_weight"]].copy()
    out.append(f"\n## Demand-wave OSS models ({len(demand)})\n")
    out.append(
        "Models from the demand-wave authors (moonshotai, deepseek, z-ai, "
        "minimax, qwen, OpenAI gpt-oss family, Google gemma, NVIDIA nemotron, "
        "Mistral) — sorted by recency.\n"
    )

    # Useful subset: tool-supporting, agent-grade context
    out.append("\n### Models supporting tools (most relevant for agent workloads)\n")
    tools = demand[demand["supports_tools"]].sort_values("days_since_launch")
    out.append(_format_capability_table(tools))

    out.append("\n### Long-context / agent-grade context (≥128K)\n")
    long_ctx = demand[demand["long_context"]].sort_values("context_length", ascending=False)
    out.append(_format_capability_table(long_ctx))

    out.append("\n### Reasoning-capable models\n")
    reasoning = demand[demand["supports_reasoning"]].sort_values("days_since_launch")
    if reasoning.empty:
        out.append("_None found._")
    else:
        out.append(_format_capability_table(reasoning))

    out.append("\n## DekaLLM's current portfolio — capability check\n")
    dek = df[df["is_dekallm_current"]].copy()
    out.append(
        "What capabilities DekaLLM's current models support. Useful for "
        "knowing whether the existing portfolio competes for agent / tool / "
        "long-context traffic.\n"
    )
    out.append(_format_capability_table(dek))

    # Capability gaps
    dek_tools = dek["supports_tools"].any()
    dek_reasoning = dek["supports_reasoning"].any()
    dek_agent_ctx = dek["agent_grade_context"].any()
    out.append("\n### Capability summary\n")
    out.append(f"- Tool-supporting models in DekaLLM portfolio: **{int(dek['supports_tools'].sum())} of {len(dek)}**")
    out.append(f"- Reasoning-capable: **{int(dek['supports_reasoning'].sum())} of {len(dek)}**")
    out.append(f"- Agent-grade context (≥256K): **{int(dek['agent_grade_context'].sum())} of {len(dek)}**")
    out.append(f"- Long-context (≥128K): **{int(dek['long_context'].sum())} of {len(dek)}**")
    out.append(f"- Cache-pricing enabled: **{int(dek['supports_cache'].sum())} of {len(dek)}**\n")

    # Top demand-wave models DekaLLM does NOT currently serve
    dek_base = {b for b in dek["base_slug"]}
    not_serving = demand[~demand["base_slug"].isin(dek_base)].copy()
    # Prioritize: tool support + reasoning + recent + long-context
    not_serving["fit_score"] = (
        not_serving["supports_tools"].astype(int) * 2 +
        not_serving["supports_reasoning"].astype(int) * 2 +
        not_serving["agent_grade_context"].astype(int) +
        not_serving["supports_cache"].astype(int)
    )
    top_targets = not_serving.sort_values(["fit_score", "days_since_launch"], ascending=[False, True]).head(15)

    out.append("\n## Top portfolio expansion candidates (DekaLLM does NOT serve these)\n")
    out.append(
        "Open-weight demand-wave models DekaLLM doesn't currently serve, "
        "ranked by capability fit (tools + reasoning + agent context + cache).\n"
    )
    out.append(_format_capability_table(top_targets))

    # Cross-reference with our earlier demand_wave_landscape.py
    out.append("\n## Cross-reference with competitive landscape\n")
    out.append(
        "Pair this capability data with `analysis/out/demand_wave_landscape.md` "
        "to get the full picture: a model needs BOTH (a) capability fit and "
        "(b) entry-tier competitive position. The top candidates emerging from "
        "both analyses:\n"
    )
    out.append("- **moonshotai/kimi-k2.6** — high capability (tools, long context, reasoning?), saturated pool (≥18 providers), Auto Exacto risk")
    out.append("- **moonshotai/kimi-k2-0905** — older K2, lower competition, capability profile likely similar")
    out.append("- **deepseek/deepseek-r1** — reasoning model, 2 providers, premium pricing, capability-rich")
    out.append("- **deepseek/deepseek-v3.2** — V3 family, broad capabilities")
    out.append("- **z-ai/glm-5** and **z-ai/glm-5.1** — Z.AI's flagship reasoning + agent models; DekaLLM has existing Z.AI relationship")
    out.append("- **z-ai/glm-4.7-flash** — speed-optimized GLM variant\n")

    out.append("\n## Notes\n")
    out.append(
        "- The `supports_*` flags come from each model's `supported_parameters` "
        "field in the OpenRouter API. They reflect what parameters OpenRouter "
        "documents the model accepts, not necessarily what every provider serving "
        "the model implements correctly (that's the Auto Exacto / tool-call quality story).\n"
        "- `is_open_weight` is heuristic — uses author + name prefix. Imperfect "
        "for edge cases but accurate for the demand-wave authors we care about.\n"
        "- Pricing is OpenRouter's *displayed* price (which usually matches the "
        "model author's blended pass-through price). Individual provider prices "
        "may be lower — see `competitive_position.md` for the actual per-provider "
        "leaderboard.\n"
    )
    return "\n".join(out)


def _format_capability_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_"
    rows = []
    rows.append("| Model | Context | $/M in | $/M out | Tools | Reasoning | Struct | Cache | Multi | Days old |")
    rows.append("|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|---:|")
    for _, r in df.iterrows():
        rows.append(
            f"| `{r['model_id']}` | "
            f"{fmt_int(r['context_length'])} | "
            f"{fmt_money(r['input_price_per_M'])} | "
            f"{fmt_money(r['output_price_per_M'])} | "
            f"{fmt_bool(r['supports_tools'])} | "
            f"{fmt_bool(r['supports_reasoning'])} | "
            f"{fmt_bool(r['supports_structured_outputs'])} | "
            f"{fmt_bool(r['supports_cache'])} | "
            f"{fmt_bool(r['is_multimodal'])} | "
            f"{fmt_int(r['days_since_launch'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    models_raw = fetch_catalog()

    # Save raw JSON for archival
    raw_path = OUT / "openrouter_catalog_raw.json"
    with open(raw_path, "w") as f:
        json.dump(models_raw, f, indent=2)
    print(f"Raw JSON saved -> {raw_path}")

    # Profile each model
    profiles = [profile_model(m) for m in models_raw]
    df = pd.DataFrame(profiles)
    df = df.sort_values(["is_demand_wave_author", "days_since_launch"], ascending=[False, True])

    # Save flat catalog
    csv_path = OUT / "openrouter_catalog.csv"
    df.to_csv(csv_path, index=False)
    print(f"Flat catalog   -> {csv_path}  ({len(df)} models)")

    # Save filtered demand-wave subset
    demand = df[df["is_demand_wave_author"]].copy()
    demand_csv = OUT / "openrouter_catalog_demand_wave.csv"
    demand.to_csv(demand_csv, index=False)
    print(f"Demand-wave    -> {demand_csv}  ({len(demand)} models)")

    # Write markdown report
    md = write_report(df)
    md_path = OUT / "demand_wave_capabilities.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report -> {md_path}")

    # Console summary
    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"Total models on OpenRouter: {len(df)}")
    print(f"Open-weight: {df['is_open_weight'].sum()}")
    print(f"Demand-wave (open-weight + key authors): {(df['is_demand_wave_author'] & df['is_open_weight']).sum()}")
    print(f"Tool-supporting demand-wave: {((df['is_demand_wave_author']) & (df['supports_tools'])).sum()}")
    print(f"Reasoning-capable demand-wave: {((df['is_demand_wave_author']) & (df['supports_reasoning'])).sum()}")
    print(f"Agent-grade context (≥256K) demand-wave: {((df['is_demand_wave_author']) & (df['agent_grade_context'])).sum()}")

    print(f"\nDekaLLM current models found in catalog: {df['is_dekallm_current'].sum()}")
    dek = df[df["is_dekallm_current"]]
    print(f"  Tool-supporting:     {int(dek['supports_tools'].sum())} of {len(dek)}")
    print(f"  Reasoning-capable:   {int(dek['supports_reasoning'].sum())} of {len(dek)}")
    print(f"  Agent-grade context: {int(dek['agent_grade_context'].sum())} of {len(dek)}")


if __name__ == "__main__":
    main()
