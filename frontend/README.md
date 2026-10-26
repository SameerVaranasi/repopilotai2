# RepoPilot ML — Step 3: React frontend

Keep this folder separate from RepoPilot_Training. It connects to the existing
Step 2 FastAPI backend; no retraining or model files are needed in this folder.

## Run locally on Windows

1. Keep the authenticated FastAPI server running on http://127.0.0.1:8000.
2. Extract this ZIP. Open a SECOND terminal inside RepoPilot_Frontend_Step3,
   the folder containing package.json.
3. Install Node.js LTS if `node --version` and `npm --version` are unavailable.
4. Run:

```text
npm install
npm run dev
```

Open http://127.0.0.1:5173 . If PowerShell blocks npm.ps1, use `npm.cmd install`
and `npm.cmd run dev`, or open Command Prompt. No execution-policy change is needed.

Do not close either terminal. The backend terminal runs Python; this terminal runs
Vite. To stop a server, press Ctrl+C in its terminal.

## Use the dashboard

1. Verify Models loaded in the left sidebar. Check connection refreshes `/health`.
2. Enter a public GitHub repository URL and a question.
3. Choose neural or TF-IDF cosine, a file cap, and the number of results.
4. Click Analyze repository. First requests fetch files and can take a minute or more.
5. Overview shows metadata, inspected paths and coverage. Relevant files shows scores,
   source excerpts, commit-specific GitHub links and shared tokens.
6. Issues & setup shows model predictions separately from actual issue labels,
   setup excerpts and a session-only contribution checklist.
7. Download result JSON exports the successful response together with its request.

There is no fabricated demo mode. Results require your backend and trained weights.
If the backend reports fixture models, a warning appears. GitHub errors are displayed
with their messages; rate-limit resets are converted to your browser's local time.
Cancel stops waiting in the browser, but an already running backend request can still
finish and consume API requests. New analyses clear old results to prevent mixing them.
No results, credentials, or checklists are persisted by the frontend.

## Connection details

In development, Vite proxies `/api` to http://127.0.0.1:8000 so the browser uses the
same origin. The GitHub token belongs ONLY in the backend environment. Never put
GITHUB_TOKEN in frontend code or any VITE_* variable.

If the backend runs on port 8001, set this in PowerShell BEFORE `npm run dev`:

```powershell
$env:REPOPILOT_BACKEND_URL = 'http://127.0.0.1:8001'
npm.cmd run dev
```

For a future production deployment, `VITE_API_BASE` must point to a deployed API
or a same-origin reverse proxy path. The development proxy does not run in the
production bundle or `vite preview`. Hosted frontend-to-localhost is not deployment.

## Checks

```text
npm run build
```

This creates dist/. The supplied package-lock.json records the installed versions.
The production compilation was checked during preparation. Browser appearance and
connection to your laptop's real backend still need local confirmation. No hosted
deployment has been performed at this milestone.

## Limits visible in the UI

The backend ranks a capped file subset. Your supplied Flask example inspected 30
of 188 eligible files, mostly docs/config files; the neural results do not establish
where request routing is implemented. Larger file caps may increase coverage but
do not guarantee the right source files are included. Source-aware candidate
selection is a backend improvement still needed before claiming comprehensive search.

Scores are not accuracy or calibrated difficulty probabilities. A real source excerpt
does not prove model relevance. Setup snippets may belong to an example subproject;
read the linked file and confirm commands before running them. No repository commands
are executed. The contribution checklist is generic, and LLM generation is not enabled.

Repository strings render as React text, never raw HTML. External result links are
limited to HTTPS github.com and open with noopener/noreferrer. The interface is
responsive and includes loading, cancellation, empty and error states.
