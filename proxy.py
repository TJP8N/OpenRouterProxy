"""
OpenRouter Free Model Proxy
Exposes OpenAI-compatible endpoints and automatically routes to the best
available free model on OpenRouter, with fallback on rate-limit / errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("proxy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
LEADERBOARD_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/serve/monitor/leaderboard_table_20240508.csv"
)

MODEL_REFRESH_SECONDS: int = int(os.getenv("MODEL_REFRESH_SECONDS", "300"))
LEADERBOARD_REFRESH_SECONDS: int = int(os.getenv("LEADERBOARD_REFRESH_SECONDS", "3600"))
MAX_FALLBACK_ATTEMPTS: int = int(os.getenv("MAX_FALLBACK_ATTEMPTS", "3"))
PAID_FALLBACK_MODEL: str = os.getenv("PAID_FALLBACK_MODEL", "")
MODEL_BLOCKLIST: set[str] = {
    m.strip() for m in os.getenv("MODEL_BLOCKLIST", "").split(",") if m.strip()
}
ALLOW_PROVIDERS: set[str] = {
    p.strip() for p in os.getenv("ALLOW_PROVIDERS", "").split(",") if p.strip()
}
USE_LEADERBOARD: bool = os.getenv("USE_LEADERBOARD", "true").lower() == "true"
MIN_PARAMS_BILLIONS: float = float(os.getenv("MIN_PARAMS_BILLIONS", "7"))

# ---------------------------------------------------------------------------
# Niche / non-general-purpose model filter
# Models whose IDs match any of these patterns are excluded. We only want
# general-purpose chat/instruction LLMs in the same category as GPT-4 or
# Claude Opus — not code-specialised, vision-only, embedding, math, medical,
# or other narrow-domain models.
# ---------------------------------------------------------------------------
_NICHE_PATTERNS: list[str] = [
    # Task-specific
    r"\bcode\b", r"\bcoder\b", r"\bcoding\b",
    r"\bembed", r"\bembedding",
    r"\bvision\b", r"\bvl\b", r"\bvision-language",
    r"\bspeech\b", r"\basr\b", r"\bwhisper\b", r"\btranscri",
    r"\bmath\b", r"\bmath-",
    r"\bmedical\b", r"\bclinical\b", r"\bbio\b",
    r"\blaw\b", r"\blegal\b",
    r"\bsql\b", r"\btext-to-sql",
    r"\bclassif",
    r"\bmoderat",
    r"\brerank",
    r"\bsummariz",
    r"\btranslat",
    r"\bsentiment",
    # Architecture types that are never general-purpose chat models
    r"\bdiffusion\b",
    r"\bimage\b",
    r"\baudio\b",
    # Very old / low-quality families
    r"\bbloom\b", r"\bopt-\b", r"\bgpt-j\b", r"\bgpt-neo\b",
    r"\bpythia\b", r"\bfalcon-7b\b",
]
_NICHE_RE = re.compile("|".join(_NICHE_PATTERNS), re.IGNORECASE)


def is_general_purpose(model: dict[str, Any], params_b: float) -> bool:
    """
    Return True if the model looks like a general-purpose chat/instruction LLM.
    Excludes niche/specialised models and anything below MIN_PARAMS_BILLIONS.
    """
    model_id: str = model.get("id", "").lower()
    name: str = model.get("name", "").lower()

    # Reject if any niche keyword appears in the id or name
    if _NICHE_RE.search(model_id) or _NICHE_RE.search(name):
        return False

    # Reject if modalities exclude text output (vision/audio-only, image gen)
    architecture = model.get("architecture") or {}
    modality: str = architecture.get("modality", "") or ""
    if modality and "text" not in modality.lower():
        return False

    # Reject models where we can determine they're below the size threshold.
    # If we can't determine size (params_b == 0), we allow through.
    if params_b > 0 and params_b < MIN_PARAMS_BILLIONS:
        return False

    return True


VIRTUAL_MODEL_ID = "auto/best-free"
VIRTUAL_MODEL_NAME = "Best Free Model (Auto-Selected)"

# ---------------------------------------------------------------------------
# Architecture generation multipliers
# Higher = newer / better quality per parameter
# ---------------------------------------------------------------------------
ARCH_GENERATIONS: list[tuple[str, float]] = [
    # pattern (lowercase match), multiplier
    ("deepseek-r1", 1.6),
    ("deepseek-v3", 1.5),
    ("deepseek-v2", 1.3),
    ("llama-3.3", 1.4),
    ("llama-3.2", 1.35),
    ("llama-3.1", 1.3),
    ("llama-3", 1.25),
    ("mistral-large", 1.3),
    ("mistral-nemo", 1.2),
    ("mixtral-8x22b", 1.2),
    ("mixtral-8x7b", 1.15),
    ("gemma-2", 1.25),
    ("gemma-3", 1.3),
    ("qwen-2.5", 1.35),
    ("qwen2.5", 1.35),
    ("qwen-2", 1.2),
    ("phi-4", 1.3),
    ("phi-3.5", 1.2),
    ("phi-3", 1.1),
    ("command-r-plus", 1.2),
    ("command-r", 1.1),
    ("solar", 1.1),
    ("wizardlm-2", 1.1),
    ("nous-hermes", 1.1),
    ("yi-34b", 1.05),
    ("llama-2", 0.9),
    ("llama 2", 0.9),
    ("mistral-7b", 1.0),
    ("mistral", 1.0),
]

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_ranked_models: list[dict[str, Any]] = []
_leaderboard: dict[str, float] = {}  # normalised name -> ELO score
_last_model_refresh: float = 0.0
_last_leaderboard_refresh: float = 0.0
_refresh_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Parameter extraction helpers
# ---------------------------------------------------------------------------
_PARAM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)b"  # MoE: NxMb
    r"|(\d+(?:\.\d+)?)b",                        # dense: Nb
    re.IGNORECASE,
)


def extract_params_billions(model_id: str) -> float:
    """Return estimated effective parameter count in billions from model ID."""
    text = model_id.lower()
    for m in _PARAM_RE.finditer(text):
        if m.group(1) and m.group(2):
            # MoE: count top-2 experts active out of N
            n_experts = float(m.group(1))
            expert_size = float(m.group(2))
            # effective params ≈ 2 active experts * size + small shared portion
            return 2 * expert_size + (n_experts - 2) * expert_size * 0.05
        if m.group(3):
            return float(m.group(3))
    return 0.0


def arch_multiplier(model_id: str) -> float:
    """Return the best matching architecture generation multiplier."""
    text = model_id.lower()
    for pattern, mult in ARCH_GENERATIONS:
        if pattern in text:
            return mult
    return 1.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(
    model: dict[str, Any],
    leaderboard: dict[str, float],
    params: float | None = None,
) -> dict[str, Any]:
    """
    Compute a composite quality score for a free OpenRouter model.
    Returns a dict with the score and breakdown components.
    Optionally accepts pre-computed `params` (billions) to avoid redundant work.
    """
    model_id: str = model.get("id", "")
    lower_id = model_id.lower()

    # --- Parameter score ---
    if params is None:
        params = extract_params_billions(model_id)
    param_score = math.log1p(params) * 10 if params > 0 else 5.0
    arch_mult = arch_multiplier(model_id)
    param_score *= arch_mult

    # --- Context length score ---
    ctx = model.get("context_length") or 0
    ctx_score = min(math.log2(ctx / 1024 + 1) * 3, 15) if ctx > 0 else 0.0

    # --- Recency score ---
    created = model.get("created") or 0
    age_days = (time.time() - created) / 86400 if created else 9999
    recency_score = max(0.0, 10.0 * (1 - age_days / 90)) if age_days < 90 else 0.0

    # --- Specialization score ---
    spec_score = 0.0
    if any(x in lower_id for x in ("-r1", ":r1", "think", "reasoning", "-reason")):
        spec_score = 15.0
    elif any(x in lower_id for x in ("-instruct", "-chat", "-it", "-hf")):
        spec_score = 3.0

    # --- LMSYS ELO score ---
    elo_score: float | None = None
    if leaderboard:
        elo_score = _match_leaderboard(model_id, leaderboard)

    if elo_score is not None:
        # ELO dominates when available — blend 70% ELO, 30% heuristic
        heuristic = param_score + ctx_score + recency_score + spec_score
        total = 0.7 * elo_score + 0.3 * heuristic
        scoring_mode = "elo+heuristic"
    else:
        total = param_score + ctx_score + recency_score + spec_score
        scoring_mode = "heuristic"

    return {
        "score": round(total, 3),
        "params_b": round(params, 2),
        "arch_mult": arch_mult,
        "param_score": round(param_score, 3),
        "ctx_score": round(ctx_score, 3),
        "recency_score": round(recency_score, 3),
        "spec_score": round(spec_score, 3),
        "elo_score": round(elo_score, 3) if elo_score is not None else None,
        "scoring_mode": scoring_mode,
    }


def _match_leaderboard(model_id: str, leaderboard: dict[str, float]) -> float | None:
    """Fuzzy-match a model ID to leaderboard entries; return normalised ELO or None."""
    lower = model_id.lower()

    # Exact match first
    if lower in leaderboard:
        return leaderboard[lower]

    # Extract tokens from model_id: split on / - _ and take longest parts
    tokens = re.split(r"[/\-_: ]+", lower)
    tokens = [t for t in tokens if len(t) >= 3 and not t.endswith("free")]

    best_score: float | None = None
    best_overlap = 0

    for key, elo in leaderboard.items():
        key_tokens = re.split(r"[/\-_: ]+", key)
        overlap = sum(1 for t in tokens if any(t in k or k in t for k in key_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_score = elo

    return best_score if best_overlap >= 2 else None


# ---------------------------------------------------------------------------
# Leaderboard fetching
# ---------------------------------------------------------------------------

async def fetch_leaderboard(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch LMSYS chatbot arena leaderboard and return name→normalised_ELO dict."""
    if not USE_LEADERBOARD:
        return {}
    try:
        resp = await client.get(LEADERBOARD_URL, timeout=15)
        resp.raise_for_status()
        rows = resp.text.strip().splitlines()
        elos: dict[str, float] = {}
        for row in rows[1:]:  # skip header
            parts = row.split(",")
            if len(parts) >= 2:
                name = parts[0].strip().lower()
                try:
                    elos[name] = float(parts[1].strip())
                except ValueError:
                    pass

        if not elos:
            return {}

        min_elo = min(elos.values())
        max_elo = max(elos.values())
        span = max_elo - min_elo or 1.0
        # Normalise to 0–100
        return {k: (v - min_elo) / span * 100 for k, v in elos.items()}
    except Exception as exc:
        log.warning("Leaderboard fetch failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Model list fetching & ranking
# ---------------------------------------------------------------------------

async def fetch_and_rank_models(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Fetch free models from OpenRouter, score and rank them."""
    try:
        resp = await client.get(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        all_models: list[dict] = resp.json().get("data", [])
    except Exception as exc:
        log.error("Failed to fetch models from OpenRouter: %s", exc)
        return []

    free_models = []
    for m in all_models:
        model_id: str = m.get("id", "")

        # Must be free
        pricing = m.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", "1") or "1")
        completion_cost = float(pricing.get("completion", "1") or "1")
        if prompt_cost > 0 or completion_cost > 0:
            continue

        # Apply blocklist
        if model_id in MODEL_BLOCKLIST:
            continue

        # Apply provider filter
        if ALLOW_PROVIDERS:
            provider = model_id.split("/")[0] if "/" in model_id else ""
            if provider not in ALLOW_PROVIDERS:
                continue

        # Pre-compute params for the general-purpose check
        params_b = extract_params_billions(model_id)

        # Skip niche / specialised models
        if not is_general_purpose(m, params_b):
            log.debug("Skipping niche/small model: %s", model_id)
            continue

        breakdown = compute_score(m, _leaderboard, params=params_b)
        free_models.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "context_length": m.get("context_length"),
            "created": m.get("created"),
            **breakdown,
        })

    free_models.sort(key=lambda x: x["score"], reverse=True)
    log.info("Ranked %d free models (top: %s)", len(free_models), free_models[0]["id"] if free_models else "none")
    return free_models


# ---------------------------------------------------------------------------
# Background refresh tasks
# ---------------------------------------------------------------------------

async def _refresh_loop(client: httpx.AsyncClient) -> None:
    """Periodically refresh models and leaderboard."""
    global _ranked_models, _leaderboard, _last_model_refresh, _last_leaderboard_refresh

    while True:
        now = time.time()
        async with _refresh_lock:
            if USE_LEADERBOARD and (now - _last_leaderboard_refresh) >= LEADERBOARD_REFRESH_SECONDS:
                log.info("Refreshing leaderboard…")
                _leaderboard = await fetch_leaderboard(client)
                _last_leaderboard_refresh = time.time()
                log.info("Leaderboard: %d entries loaded", len(_leaderboard))

            if (now - _last_model_refresh) >= MODEL_REFRESH_SECONDS:
                log.info("Refreshing model rankings…")
                _ranked_models = await fetch_and_rank_models(client)
                _last_model_refresh = time.time()

        await asyncio.sleep(30)  # check every 30 s, refresh only when due


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None
_refresh_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _http_client, _refresh_task, _ranked_models, _leaderboard
    global _last_model_refresh, _last_leaderboard_refresh

    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY is not set — proxy will not function correctly")

    _http_client = httpx.AsyncClient(follow_redirects=True)

    # Initial load
    if USE_LEADERBOARD:
        log.info("Initial leaderboard fetch…")
        _leaderboard = await fetch_leaderboard(_http_client)
        _last_leaderboard_refresh = time.time()

    log.info("Initial model ranking…")
    _ranked_models = await fetch_and_rank_models(_http_client)
    _last_model_refresh = time.time()

    _refresh_task = asyncio.create_task(_refresh_loop(_http_client))

    yield

    _refresh_task.cancel()
    await _http_client.aclose()


app = FastAPI(title="OpenRouter Free Proxy", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helper: get HTTP client
# ---------------------------------------------------------------------------

def _client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client not initialised")
    return _http_client


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    top = _ranked_models[0] if _ranked_models else None
    return JSONResponse({
        "status": "ok",
        "model_count": len(_ranked_models),
        "top_model": top["id"] if top else None,
        "top_score": top["score"] if top else None,
        "scoring_mode": top["scoring_mode"] if top else None,
        "leaderboard_entries": len(_leaderboard),
        "last_model_refresh": _last_model_refresh,
        "last_leaderboard_refresh": _last_leaderboard_refresh,
    })


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    now = int(time.time())

    # Virtual model at the top
    virtual = {
        "id": VIRTUAL_MODEL_ID,
        "object": "model",
        "created": now,
        "owned_by": "openrouter-free-proxy",
        "name": VIRTUAL_MODEL_NAME,
        "_proxy": {"type": "virtual", "routes_to": _ranked_models[0]["id"] if _ranked_models else None},
    }

    model_list = [virtual] + [
        {
            "id": m["id"],
            "object": "model",
            "created": m.get("created") or now,
            "owned_by": m["id"].split("/")[0] if "/" in m["id"] else "unknown",
            "name": m.get("name", m["id"]),
            "_proxy": {
                "score": m["score"],
                "params_b": m["params_b"],
                "scoring_mode": m["scoring_mode"],
            },
        }
        for m in _ranked_models
    ]

    return JSONResponse({"object": "list", "data": model_list})


@app.get("/v1/ranking")
async def ranking() -> JSONResponse:
    """Debug endpoint: full ranked list with score breakdowns."""
    return JSONResponse({
        "object": "list",
        "count": len(_ranked_models),
        "data": _ranked_models,
    })


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    requested_model: str = body.get("model", VIRTUAL_MODEL_ID)
    stream: bool = body.get("stream", False)

    # Build candidate list
    if requested_model == VIRTUAL_MODEL_ID or not requested_model:
        candidates = [m["id"] for m in _ranked_models[:MAX_FALLBACK_ATTEMPTS]]
    else:
        # User picked a specific free model — still allow fallback to next best
        candidates = [requested_model] + [
            m["id"] for m in _ranked_models
            if m["id"] != requested_model
        ][:MAX_FALLBACK_ATTEMPTS - 1]

    if PAID_FALLBACK_MODEL:
        candidates.append(PAID_FALLBACK_MODEL)

    if not candidates:
        raise HTTPException(status_code=503, detail="No free models available")

    last_error: str = "Unknown error"
    for model_id in candidates:
        upstream_body = {**body, "model": model_id}

        # Inject proxy metadata via system prompt comment (non-intrusive)
        score_info = next((m for m in _ranked_models if m["id"] == model_id), None)

        try:
            if stream:
                return await _stream_request(upstream_body, model_id, score_info, request)
            else:
                result = await _non_stream_request(upstream_body, model_id, score_info)
                return JSONResponse(result)
        except _RetryableError as exc:
            last_error = str(exc)
            log.warning("Model %s failed (%s), trying next…", model_id, exc)
            continue
        except HTTPException:
            raise
        except Exception as exc:
            last_error = str(exc)
            log.error("Unexpected error for model %s: %s", model_id, exc)
            continue

    raise HTTPException(status_code=503, detail=f"All models exhausted. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

class _RetryableError(Exception):
    pass


async def _non_stream_request(
    body: dict[str, Any],
    model_id: str,
    score_info: dict | None,
) -> dict[str, Any]:
    resp = await _client().post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openrouter-free-proxy",
            "X-Title": "OpenRouter Free Proxy",
        },
        json=body,
        timeout=120,
    )

    if resp.status_code == 429:
        raise _RetryableError(f"Rate limited by {model_id}")
    if resp.status_code >= 500:
        raise _RetryableError(f"Server error {resp.status_code} from {model_id}")
    if resp.status_code >= 400:
        detail = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data: dict[str, Any] = resp.json()

    # Overwrite model field with actual model used
    data["model"] = model_id

    # Inject proxy metadata
    data["_proxy"] = {
        "routed_to": model_id,
        "score": score_info["score"] if score_info else None,
        "params_b": score_info["params_b"] if score_info else None,
        "scoring_mode": score_info["scoring_mode"] if score_info else None,
    }

    return data


async def _stream_request(
    body: dict[str, Any],
    model_id: str,
    score_info: dict | None,
    original_request: Request,
) -> StreamingResponse:
    """Initiate a streaming request, raising _RetryableError on rate limit."""

    # We need to check the first response before returning the StreamingResponse
    # so we can detect 429/5xx before committing to the stream.
    async with _client().stream(
        "POST",
        f"{OPENROUTER_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openrouter-free-proxy",
            "X-Title": "OpenRouter Free Proxy",
        },
        json=body,
        timeout=httpx.Timeout(connect=10, read=120, write=30, pool=5),
    ) as upstream:
        if upstream.status_code == 429:
            raise _RetryableError(f"Rate limited by {model_id}")
        if upstream.status_code >= 500:
            raise _RetryableError(f"Server error {upstream.status_code} from {model_id}")
        if upstream.status_code >= 400:
            body_text = await upstream.aread()
            raise HTTPException(status_code=upstream.status_code, detail=body_text[:500].decode())

        # Read all chunks into memory so we can release the upstream connection cleanly.
        # For very long streams this buffers in memory; acceptable for typical chat use.
        chunks: list[bytes] = []
        async for chunk in upstream.aiter_bytes():
            chunks.append(chunk)

    proxy_meta = json.dumps({
        "routed_to": model_id,
        "score": score_info["score"] if score_info else None,
        "params_b": score_info["params_b"] if score_info else None,
    })

    async def generate() -> AsyncIterator[bytes]:
        # Inject a leading SSE comment with proxy metadata
        yield f": proxy {proxy_meta}\n\n".encode()
        for chunk in chunks:
            if not chunk:
                continue
            # Rewrite model field in data lines
            text = chunk.decode("utf-8", errors="replace")
            lines = text.split("\n")
            out_lines = []
            for line in lines:
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        payload = json.loads(line[6:])
                        payload["model"] = model_id
                        out_lines.append("data: " + json.dumps(payload))
                    except json.JSONDecodeError:
                        out_lines.append(line)
                else:
                    out_lines.append(line)
            yield "\n".join(out_lines).encode("utf-8")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Proxy-Model": model_id,
        },
    )


# ---------------------------------------------------------------------------
# Catch-all for unknown paths
# ---------------------------------------------------------------------------

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(path: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": {"message": f"Unknown endpoint: /{path}", "type": "invalid_request_error"}},
    )
