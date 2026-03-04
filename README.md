# OpenRouter Free Model Proxy

A Docker-deployable proxy that sits between Open WebUI (or any OpenAI-compatible client) and OpenRouter. It automatically routes every request to the best available **general-purpose** free model, falling back to the next best on rate-limit or error.

## Quick Start

```bash
cp .env.example .env
# Edit .env and set your OPENROUTER_API_KEY
docker compose up -d
```

Point Open WebUI (or any OpenAI-compatible client) at:

```
http://<your-host>:8800/v1
```

Select **"Best Free Model (Auto-Selected)"** (`auto/best-free`) as your model. The proxy will route to the highest-scored general-purpose free model automatically.

---

## How It Works

### Model Filtering

Only **general-purpose chat/instruction LLMs** are considered. The following are automatically excluded:

- Niche/specialised models (code-only, vision-only, embedding, moderation, math, medical, etc.)
- Models below a configurable minimum parameter threshold (`MIN_PARAMS_BILLIONS`, default: 7B)

### Scoring (highest wins)

| Component | Weight |
|-----------|--------|
| Parameter count (log-scaled) × architecture multiplier | ~40% |
| Context length bonus (log-scaled, capped) | ~15% |
| Recency (models < 90 days old get up to +10) | ~10% |
| Specialization (reasoning/instruct bonus) | ~10% |
| LMSYS Chatbot Arena ELO (when available, dominates at 70%) | ~70% blend |

Architecture multipliers reward newer model families (DeepSeek R1 = 1.6×, Llama 3.3 = 1.4×, etc.).

### Fallback Chain

On rate-limit (429) or server error (5xx), the proxy tries the next highest-ranked free model, up to `MAX_FALLBACK_ATTEMPTS`. If `PAID_FALLBACK_MODEL` is set, it's tried last.

### Response Transparency

- The `model` field in responses reflects the **actual** model that served the request.
- A `_proxy` block in non-streaming responses includes `routed_to`, `score`, `params_b`, and `scoring_mode`.
- Streaming responses get a leading SSE comment: `: proxy {"routed_to": "...", "score": ...}`.

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Status, top model, model count |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `GET` | `/v1/ranking` | Full ranked list with score breakdowns |
| `POST` | `/v1/chat/completions` | Chat with auto-routing and fallback |

### `/health` example

```json
{
  "status": "ok",
  "model_count": 23,
  "top_model": "meta-llama/llama-3.3-70b-instruct:free",
  "top_score": 87.4,
  "scoring_mode": "elo+heuristic",
  "leaderboard_entries": 142
}
```

### `/v1/ranking` example

```json
{
  "count": 23,
  "data": [
    {
      "id": "meta-llama/llama-3.3-70b-instruct:free",
      "score": 87.4,
      "params_b": 70.0,
      "arch_mult": 1.4,
      "scoring_mode": "elo+heuristic",
      ...
    }
  ]
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | **(required)** | OpenRouter API key |
| `MODEL_REFRESH_SECONDS` | `300` | Model list refresh interval |
| `LEADERBOARD_REFRESH_SECONDS` | `3600` | LMSYS ELO refresh interval |
| `MAX_FALLBACK_ATTEMPTS` | `3` | Free models to try before failing |
| `PAID_FALLBACK_MODEL` | *(empty)* | Paid model as last resort |
| `MODEL_BLOCKLIST` | *(empty)* | Comma-separated model IDs to skip |
| `ALLOW_PROVIDERS` | *(empty)* | Restrict to these providers only |
| `USE_LEADERBOARD` | `true` | Enable LMSYS ELO integration |
| `MIN_PARAMS_BILLIONS` | `7` | Exclude models smaller than this (B params) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Unraid Deployment

1. Copy files to a path on your Unraid server (e.g. `/mnt/user/appdata/openrouter-proxy/`).
2. Create `.env` from `.env.example` and set `OPENROUTER_API_KEY`.
3. `docker compose up -d`
4. In Open WebUI → Settings → Connections → add an OpenAI-compatible connection:
   - **Base URL**: `http://<unraid-ip>:8800/v1`
   - **API Key**: anything (e.g. `proxy`)

---

## Architecture

```
Open WebUI / any client
        │
        │  POST /v1/chat/completions  (model: "auto/best-free")
        ▼
┌─────────────────────────────┐
│   openrouter-free-proxy     │
│                             │
│  ┌─────────────────────┐   │
│  │  Model Ranker        │   │  ← refreshes every 5 min
│  │  (score + filter)    │   │  ← leaderboard every 1 hr
│  └─────────────────────┘   │
│           │                 │
│   ranked free models        │
│   [model-A, model-B, ...]   │
│           │                 │
│  Try model-A ──────────────────────────► OpenRouter API
│    429? Try model-B ───────────────────► OpenRouter API
│    429? Try model-C ───────────────────► OpenRouter API
│    (optional paid fallback)             │
└─────────────────────────────┘
        │
        │  response (model = actual model used)
        ▼
Open WebUI / any client
```
