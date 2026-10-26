# RepoPilot — Step 2: local FastAPI backend

Copy the files in this ZIP into your existing **RepoPilot_Training** folder,
beside `core.py`, `train.py` and your real `artifacts/` folder. No existing
training files or model artifacts are included or overwritten.

## Start on Windows

Use the same PowerShell window where your GitHub token worked, or set
`GITHUB_TOKEN` again securely. The server inherits it at startup. Token
values are never returned by the API.

```powershell
python -m pip install -r requirements_api.txt
python -m uvicorn repopilot_api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/health and http://127.0.0.1:8000/docs .
If port 8000 is occupied by RoadGuard or another app, use `--port 8001`.
Inference uses CPU intentionally: these compact models do not need GPU inference,
and their CUDA-trained weights load on CPU. Training does not run at startup.

In `/docs`, expand POST `/analyze`, click **Try it out**, and use:

```json
{
  "repo_url": "https://github.com/pallets/flask",
  "question": "Where is request routing implemented?",
  "top_k": 5,
  "max_files": 30,
  "ranking_method": "neural"
}
```

Then click **Execute**. First analysis fetches up to 30 files and can take a minute
or more. Repeated analysis reuses up to eight cached file snapshots, keyed by
repository, current commit and file limit. Repository metadata/commit and issues
are checked again. `tfidf_cosine` is also supported for comparison; the test-set
results were not used to silently change the default model.

## Endpoints

- GET `/`: API name and documentation link.
- GET `/health`: models loaded, model version, CPU inference, token-presence boolean.
- POST `/analyze`: metadata, capped file list/directories, ranked files, exact excerpts,
  beginner-issue predictions, README/config excerpts and a generic contribution checklist.
- POST `/search`: same input; question or issue text is required; skips issue fetching.
- POST `/issues/recommend`: same input schema; scores open issues from the latest 100
  updated open issue/PR entries and returns up to `top_k` results. PRs are excluded.

All POST endpoints accept `repo_url`, optional `question`, `issue_title`, `issue_body`,
`top_k` (1–20), `max_files` (5–60), and `ranking_method` (`neural` or `tfidf_cosine`).
Scores are uncalibrated. The issue result includes the validation-selected threshold,
prediction and actual GitHub labels. A high score does not verify beginner suitability.

## Scope and limitations

This is a local prototype, not yet a public deployment. It has no user authentication
or persistent database. CORS allows the local Vite frontend on port 5173. Keep the
server bound to 127.0.0.1 until deployment hardening and access controls are added.

Only https://github.com/OWNER/REPO URLs are accepted. No arbitrary local paths,
URLs, repository scripts, or shell commands are executed. Private repositories
are rejected even if your token could access them. Moved repositories need their
new canonical URL. GitHub failures produce actionable HTTP errors without exposing
tokens. Missing model artifacts cause startup failure instead of fake predictions.

Source is fetched at a fixed commit, with at most 60 eligible text files, 60 KB per
file and 10 MB per GitHub response. README/contributing/config files and shallower
paths are prioritized. Generated directories have basic exclusions; large repos
will have incomplete coverage. Scoring uses the first 12,000 characters per file,
matching the training representation. Evidence excerpts can come from elsewhere
in that fetched file; they are source context, not a neural attribution explanation.
Shared tokens are observable overlap, not a causal explanation of the model score.

Setup evidence is raw cited README/config content, not automatically verified setup
commands. Excerpts have real paths, line ranges and commit-specific links. This
does not prove semantic support for arbitrary claims. The contribution plan is
explicitly generic. Full architecture parsing, BM25/hybrid retrieval, RAG, LLM
explanations and test execution remain future work. GitHub issue text and repository
content must be rendered as untrusted text by the later frontend.

## Tests

After installing the API requirements, from the existing training folder:

```powershell
python -m pytest test_api.py -q
```

Tests use your local trained models, mocked GitHub responses and the real API
lifespan. They make no GitHub network calls. Authoring tests use explicitly marked
synthetic checkpoints because your real trained weights are on your laptop.
Real Windows model-loading and live analysis must be confirmed locally.

Optional `REPOPILOT_ARTIFACTS` selects a trusted artifact directory. Synthetic
artifacts are rejected unless `REPOPILOT_ALLOW_FIXTURE=1` is explicitly set for tests.
Keep `core.py` and the exact training dependency versions with the artifacts.

Reference: https://fastapi.tiangolo.com/advanced/testing-events/
