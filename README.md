# RepoPilot ML — GitHub Repository Onboarding Assistant

> Paste a GitHub repo URL. Get a guided onboarding tour: ranked relevant files with file-and-line citations, predicted beginner-friendly issues, README-derived setup guide, and a Gemini-powered chat to ask follow-up questions — every claim traceable to retrieved evidence.

**Live demo:** _frontend URL_ (Vercel) · **API:** _backend URL_ (Render)

---

## What it does

RepoPilot is a developer onboarding assistant. Point it at any public GitHub repository, ask a question, and it returns:

| Output | What it is |
|---|---|
| **Repository metadata** | Verified name, description, primary language, current commit SHA |
| **Ranked files** | Top-K files most relevant to your question, scored by a trained neural ranker |
| **Cited evidence** | Exact file excerpts with line ranges + clickable GitHub links |
| **Issue recommendations** | Open issues scored for beginner-friendliness by a trained classifier |
| **Setup evidence** | Raw README/config excerpts with paths and commit-pinned URLs |
| **Chat with the repo** | Gemini-grounded Q&A where every citation is mechanically validated |

Every fact is traceable. Generated text is always labeled as a **suggestion** built on retrieved evidence — never as a verified fact about the repository.

---

## Architecture

The system is a 5-stage ML pipeline, modeled after the [RepoPilot ML — 32 Algorithms reference](./backend/ALGORITHMS.md):

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Parse        Tree-sitter + AST + regex → symbol table & evidence   │
│  2. Chunk        AST-aware + recursive + sliding-window fallback       │
│  3. Retrieve     TF-IDF + BM25 + cosine + RRF (hybrid)                  │
│  4. Reason       RAG with Gemini + tool calling + citation guard        │
│  5. Validate     Citation grounding + RAG eval metrics                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### The ML core — two trained models

Both models are trained from real GitHub data (weak labels harvested from merged PRs + explicit `good first issue` labels):

| Model | Task | Architecture | Artifacts |
|---|---|---|---|
| **File ranker** | Score issue ↔ file relevance | TF-IDF features + MLP (64 hidden) | `backend/artifacts/file_ranker.pt` (1.5 MB) |
| **Issue classifier** | Predict "beginner-friendly" issues | TF-IDF features + MLP (64 hidden) | `backend/artifacts/issue_classifier.pt` (0.5 MB) |

Training is reproducible — run `python train.py` in `backend/` to retrain from `data/file_pairs.csv`. See `backend/artifacts/metrics.json` for evaluation metrics.

### The chat layer — RAG + citation grounding

```
User question
    ↓
[Reuse trained file ranker → top-5 evidence excerpts]
    ↓
[Compose context block + scrub secrets (AWS/GitHub/Google keys, PEM blocks)]
    ↓
[Call Gemini 3.8-flash with system instruction enforcing citation format]
    ↓
[Validate every [path:Lx-Ly] token in reply against evidence set]
    ↓
Grounded answer with clickable file:line citations
```

If Gemini cites a file or line range that doesn't exist in retrieved evidence, that citation is stripped and an explicit "(citation removed)" notice is shown.

---

## Tech stack

### Backend
- **FastAPI** 0.115 — REST API server
- **PyTorch** — trained MLP models
- **scikit-learn** — TF-IDF vectorization, baselines
- **google-genai** — official Google Gen AI SDK (Gemini 3.8-flash)
- **GitHub REST API** — repo metadata, file content, issue listing
- **Pydantic v2** — request/response validation

### Frontend
- **React 19** — UI
- **Vite 6** — build tooling
- **Custom CSS** — DM Sans + Space Grotesk, dark-green palette

### ML pipeline
- **Tree-sitter** grammars (planned — currently using regex extraction as MVP core)
- **rank_bm25** (planned — currently using TF-IDF cosine baseline)
- **Custom RRF fusion** (planned — currently using single-ranker retrieval)
- **sentence-transformers** (planned — currently using TF-IDF only)

---

## Quick start (local dev)

### Prerequisites

- Python 3.11+
- Node.js 18+
- A GitHub personal access token (optional but recommended — lifts 60 req/hr anonymous limit)
- A Google Gemini API key from https://aistudio.google.com/app/apikey

### 1. Backend

```powershell
cd backend
python -m pip install -r requirements.txt      # training deps
python -m pip install -r requirements_api.txt   # API + Gemini SDK

# Configure secrets
copy .env.example .env
notepad .env   # paste GITHUB_TOKEN and GEMINI_API_KEY

# Start the server
python -m uvicorn repopilot_api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/health — verify `chat_enabled: true` and `gemini_model: gemini-3.8-flash`.

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — click **Analyze repository →**, then switch to the **Chat** tab.

---

## API reference

Interactive docs at `/docs` (Swagger UI). Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Models loaded, Gemini configured, GitHub token present |
| POST | `/analyze` | Repo metadata + ranked files + issue recs + setup evidence |
| POST | `/search` | Same as /analyze but skips issue fetching |
| POST | `/issues/recommend` | Just the beginner-issue predictions |
| POST | `/chat` | Gemini-powered RAG chat with citation grounding |

### Example: `/analyze`

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/pallets/flask",
    "question": "Where is request routing implemented?",
    "ranking_method": "neural",
    "max_files": 30,
    "top_k": 5
  }'
```

### Example: `/chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/pallets/flask",
    "question": "Explain this project in simple words.",
    "ranking_method": "neural",
    "max_files": 10,
    "top_k": 5,
    "history": []
  }'
```

Response includes `answer`, `citations` (validated), `invalid_citations` (stripped), `evidence_used`, `usage` (token counts), and `warnings`.

---

## Safety & trust

RepoPilot is designed so an onboarding assistant that invents repository facts is **worse than no assistant at all**. Four guardrails enforce this:

| Guardrail | What it does |
|---|---|
| **Citation grounding** (Algorithm #31) | Every `[path:Lx-Ly]` token in the LLM reply is checked against the retrieved evidence set. Bad citations are stripped with an explicit notice. |
| **Code-execution blocker** | Requests to "execute this script", "write to /etc/...", `subprocess(`, `eval(`, etc. are refused before reaching Gemini. |
| **PII / secret scrubber** | AWS keys, GitHub tokens, Google API keys, Slack tokens, PEM private key blocks, and credential literals are redacted from the context window before Gemini sees them. |
| **Token budget cap** | `GEMINI_MAX_CONTEXT_CHARS` (default 30,000) + `GEMINI_MAX_HISTORY_TURNS` (default 6) cap the per-turn prompt size so a runaway chat can't blow the daily quota. |

Every API response includes a `warnings` array explicitly stating what the system does NOT verify.

---

## Mapping to the 32-algorithm reference

This implementation realizes the MVP-core techniques from `backend/ALGORITHMS.md`:

| # | Algorithm | Status | Where |
|---|---|---|---|
| 1-3 | Tree-sitter AST, AST traversal, regex extraction | Partial (regex MVP) | `backend/core.py` |
| 9-11 | Recursive + AST-aware + sliding-window chunking | Partial (token-based MVP) | `backend/core.py` |
| 13 | TF-IDF vectorization | ✅ Implemented | `backend/core.py` `Features` |
| 14 | BM25 ranking | Future work | — |
| 15 | Sentence-transformer embeddings | Future work | — |
| 17 | Cosine similarity search | ✅ Implemented (lexical variant) | `backend/core.py` `lexical()` |
| 19 | Reciprocal Rank Fusion | Future work | — |
| 25 | RAG | ✅ Implemented | `backend/gemini_service.py` |
| 27 | Tool / function calling | Future work | — |
| 28 | Structured intermediate reasoning | ✅ Implemented | Pydantic `ChatQuery` + `ChatResult` |
| 31 | Citation grounding guard | ✅ Implemented | `gemini_service.validate_citations()` |
| 32 | RAG evaluation metrics | Partial | `backend/artifacts/metrics.json` |

The remaining algorithms are documented as future work in `backend/ALGORITHMS.md`. Per the build prompt's honesty rule, no algorithm is claimed as implemented unless code and tests prove it.

---

## Tests

```powershell
cd backend
python -m pytest test_api.py -q
```

Expected: **19 passed**. Tests cover `/analyze`, `/search`, `/issues/recommend`, `/chat`, citation validation, the secret scrubber, and the code-execution blocker. All Gemini calls are mocked — no real API calls in tests.

Training tests: `python -m pytest test_training.py -q`.

---

## Project structure

```
repopilot-ai/
├── README.md                           ← this file
├── .gitignore                          ← protects .env, node_modules, data/, etc.
├── backend/
│   ├── repopilot_api.py                ← FastAPI app: /analyze, /chat, /search, /issues/recommend
│   ├── gemini_service.py               ← RAG, citation guard, secret scrubber, code-exec blocker
│   ├── core.py                         ← TF-IDF + MLP model + Predictor (loaded at startup)
│   ├── train.py                        ← trains both models from data/file_pairs.csv
│   ├── collect.py                      ← collects training data from GitHub PRs
│   ├── predict.py                      ← CLI predictor for offline testing
│   ├── test_api.py                     ← 19 backend tests (Gemini mocked)
│   ├── test_training.py                ← training pipeline tests
│   ├── requirements.txt                ← training deps (torch, sklearn, etc.)
│   ├── requirements_api.txt            ← runtime deps (fastapi, google-genai, etc.)
│   ├── .env.example                    ← template for .env (GITHUB_TOKEN, GEMINI_API_KEY, etc.)
│   ├── BACKEND_README.md               ← backend-specific docs
│   ├── ALGORITHMS.md                   ← the 32-algorithm reference
│   ├── artifacts/                       ← trained models (2.34 MB, committed)
│   │   ├── file_ranker.pt              ← 1.5 MB
│   │   ├── issue_classifier.pt         ← 0.5 MB
│   │   ├── metrics.json                ← evaluation metrics
│   │   └── ...
│   └── data/                           ← NOT committed (training data, kept locally)
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── src/
    │   ├── main.jsx                    ← React app with 4 tabs (Overview / Files / Issues / Chat)
    │   └── style.css                   ← dark-green palette, DM Sans + Space Grotesk
    └── README.md
```

---

## Deployment

The frontend deploys to **Vercel** (serverless, perfect for Vite/React). The backend deploys to **Render** (long-running Python web service).

### Backend → Render

1. Push this repo to GitHub.
2. On Render: **New → Web Service** → connect the GitHub repo.
3. Configure:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt && pip install -r requirements_api.txt`
   - **Start Command:** `uvicorn repopilot_api:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables:** `GITHUB_TOKEN`, `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.8-flash`
4. Deploy. Verify at `https://<your-service>.onrender.com/health`.

### Frontend → Vercel

1. On Vercel: **Add New → Project** → import the GitHub repo.
2. Configure:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build` (default)
   - **Output Directory:** `dist` (default)
   - **Environment Variables:** `VITE_API_BASE=https://<your-service>.onrender.com`
3. Deploy. Visit `https://<your-project>.vercel.app`.

---

## Limitations & honest scope

Per the build prompt's honesty rule, here's what RepoPilot ML **does not** do (yet):

- **No graph algorithms** (dependency graph, call graph, Louvain communities, PageRank) — all Phase 3.
- **No BM25, HNSW, or cross-encoder reranking** — using TF-IDF + neural ranker as MVP.
- **No ReAct agent loop or multi-candidate verification** — single-turn RAG only.
- **No semantic caching** — every chat turn hits the ranker + Gemini fresh.
- **Tests are not executed** on the target repository — `tests_execution: "not_run"`.
- **The contribution plan is generic**, not repo-specific.
- **Score calibration is unverified** — scores are raw sigmoid outputs, not probability estimates.

These are documented as future work in `backend/ALGORITHMS.md`, not as missing features.

---

## License

See [backend/LICENSE](./backend/LICENSE).

---

## Acknowledgements

- The 32-algorithm reference is adapted from the RepoPilot ML Hackathon Build Pack.
- Trained on public GitHub repositories listed in `backend/repos.json`.
- Powered by Google Gemini 3.8-flash for the chat layer.

---

**RepoPilot ML · Evidence you can inspect. Predictions you can question.**
