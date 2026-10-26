# RepoPilot ML: Step 1 — data and GPU training

Repository onboarding assistant for the VIT-AP hackathon. This milestone contains
real training code for two compact PyTorch models and their CPU lexical baselines.
Backend, frontend and deployment are the next milestones.

## What is trained

1. **File relevance ranker:** issue/query text + file path/content -> relevance score.
   Training queries currently use merged-PR descriptions as an explicit proxy.
2. **Beginner-label classifier:** issue title/body -> similarity to explicitly
   beginner-labeled issues. Labels themselves are never input features.

Both models use TF-IDF features and a 64-unit MLP with BCEWithLogitsLoss and AdamW.
The file model also uses token overlap, file length and test/config/extension indicators.
TF-IDF and logistic regression run on CPU; the neural models use CUDA when available.
There is no need to train a large language model or 32 separate models.

## Windows: begin here

Extract the ZIP. Open the **RepoPilot_Training** folder in PowerShell/VS Code.
Use the same Python interpreter that already runs your GPU training, if available.
Python 3.11+ is required; 3.12 is the tested runtime for this package.

```powershell
python --version
nvidia-smi
python check_gpu.py
```

If importing torch fails, install a Windows/Pip/Python/CUDA build using the official
selector: https://pytorch.org/get-started/locally/ . Choose a build supported by
your NVIDIA driver, then run `python check_gpu.py` again. Do not reinstall an already
working CUDA environment just for this project.

Expected GPU check: `CUDA available: True`, your NVIDIA GPU name, and a passed
matrix multiplication. This script reports the actual hardware; nothing is hard-coded.

Install the remaining dependencies (the satisfied torch requirement preserves a
compatible existing installation):

```powershell
python -m pip install -r requirements.txt
python check_gpu.py
```

For an isolated environment, optionally use `python -m venv --system-site-packages .venv`
and call `.\.venv\Scripts\python.exe` instead of `python` in every command. This can
reuse a globally installed CUDA torch. No activation or execution-policy change is required.

## Collect real training data

Review `repos.json`: repositories are assigned to train/validation/test **before**
any preprocessing. The defaults target nine public Python ecosystem projects.
Their live availability/label balance has not been assumed; collection and validation
will report failures. Edit this list if necessary, keeping the splits disjoint.

A GitHub token is optional but strongly useful for collecting this dataset without
exhausting unauthenticated API limits. Set `GITHUB_TOKEN` locally in PowerShell.
Use a read-only token for public repository metadata. Never send a token in chat.
The `.env.example` is a reference; scripts read process environment variables and do
not automatically load a `.env` file.

To enter a token without placing the token in PowerShell history:

```powershell
$repoPilotSecret = Read-Host "GitHub token" -AsSecureString
$env:GITHUB_TOKEN = [System.Net.NetworkCredential]::new('', $repoPilotSecret).Password
python collect.py --repos repos.json --prs-per-repo 12 --issues-per-repo 100
```

Without a token, run just the `python collect.py ...` command. If it reaches a rate
limit, set the token or wait for reset and rerun. Successful responses are cached
under `data/cache`; rerunning resumes without refetching those responses. Do not
commit the cache or datasets. Delete the cache when deliberately collecting a fresh
snapshot. Collection can take longer than training; time depends on API/network access.

Outputs: `data/processed/file_pairs.csv`, `issues.csv`, `provenance.json`.
The provenance records repository licenses, actual label mappings, PR base commits
(in the CSV), sampling limits, timestamps, and skipped examples.
Source files are read as text, never imported or executed.

## Train on your GPU

```powershell
python train.py --device cuda --epochs 30 --batch-size 64
```

`--device cuda` fails clearly if CUDA is unavailable, so it cannot silently train
on CPU. `--device auto` chooses CUDA or falls back to CPU. For an out-of-memory
error, retry with `--batch-size 16 --max-features 1024`.

Both classes must exist in every split for each task. If this validation fails,
collect more examples or revise the assigned repositories; do not randomly mix
rows across splits. MLP checkpoints are selected by validation loss with early
stopping. Issue thresholds are selected on validation F1. Test data is only used
for the final metrics, with a separate save/load prediction consistency check.
Do not keep tuning against the test report; reserve new repositories for a new final evaluation.

```powershell
Get-Content artifacts\metrics.json
python predict.py
```

## Saved artifacts

- `file_ranker.pt`, `issue_classifier.pt`: trained PyTorch weights and architecture metadata.
- `file_vectorizer.joblib`, `issue_vectorizer.joblib`: train-fitted TF-IDF vectorizers.
- `file_feature_config.joblib`, `issue_feature_config.joblib`: full shared preprocessing.
- `file_baseline.joblib`, `issue_baseline.joblib`: logistic regression comparisons.
- `metrics.json`: test metrics, class counts, repository splits and limitations.
- `training_config.json`: seed, parameters, library versions and data SHA-256 hashes.
- `label_mapping.json`, `data_provenance.json`, per-task training-history CSVs.

Keep `core.py` and dependency versions with the artifacts. The later FastAPI app
will use `core.Predictor` so preprocessing remains identical. Only load your own
trusted joblib files. Record exact local dependencies with:

```powershell
python -m pip freeze > artifacts\requirements-lock.txt
```

File metrics: Precision@5 (denominator 5), Recall@5, MRR, MAP, compared with
TF-IDF cosine and logistic regression. Issue metrics: precision, recall, F1 and ROC-AUC,
compared with logistic regression. Neural scores are uncalibrated label predictions.
There is no single overall "accuracy" for these two different tasks.

## Smoke test without GitHub

```powershell
python make_fixture.py
python train.py --data data/fixture --out artifacts_fixture --allow-fixture --epochs 3 --device cpu
python -m pytest -q
python predict.py --data data/fixture --artifacts artifacts_fixture
```

These synthetic fixtures only verify wiring. Their metrics MUST NOT be presented
as performance on real repositories. Fixture artifacts are kept separate from
real artifacts. No pretrained weights or real training data are included in this ZIP.

## Data schema and label limitations

`file_pairs.csv`: repo, issue_number (PR number for the current proxy queries),
issue_text, file_path, file_text, label, split_group, split, source_pr_url, base_sha,
label_method. Positives are modified files present at the PR base commit. Added files
are excluded because they have no base content; removed files may be included.
Negatives are sampled base files outside the complete changed-file set. Up to 5
positives and 15 negatives per PR; PRs over 100 changed files and truncated trees
are skipped. Source blobs are capped at 60 KB, text at 12,000 characters; large or
unsupported/generated/vendor files are excluded with the documented filters.

`issues.csv`: repo, issue_number, title, body, labels, created_at, comments_count,
label, split_group, split, source_url, label_method. An explicit normalized mapping
in `collect.BEGINNER` defines positives. Unlabeled/other-label rows are **weak negatives**,
not proof of difficulty. Current sampling selects up to half the cap from positives;
this changes class priors, so model scores are not calibrated real-world probabilities.
Collection reads up to 500 repository labels, the latest 100 issues plus the latest
100 per recognized beginner label, and up to 100 recent closed PRs to find merged PRs.

PR text can mention paths; unchanged files can be relevant. Ranking evaluation uses
sampled candidates and only the retained positives, not full-repository relevance.
Repository separation reduces leakage but does not eliminate copied boilerplate
or similar text across projects. Near-duplicate removal and a manually curated query
test set remain future work. Defaults are Python-heavy, so multilingual generalization
is unmeasured. Keep repository license/attribution metadata and inspect terms before
redistributing any collected source snapshot. This release distributes no snapshots.

## Status and next milestones

Implemented now: GitHub data collection, explicit weak label provenance, disjoint
repository splits, TF-IDF, cosine baseline, logistic baselines, two PyTorch MLPs,
validation model/threshold selection, metrics, versioned save/load, CPU fallback,
CUDA selection, and synthetic integration checks.

All other techniques in the uploaded 32-algorithm reference are future work in this
milestone: Tree-sitter/AST/regex repository parsing; graphs/symbol resolution/community
detection/PageRank; recursive/AST/sliding-window chunking; near-duplicate removal;
BM25; sentence/code embeddings; HNSW; RRF; cross-encoder reranking; MMR; query
expansion; HyDE; semantic caching; RAG; ReAct; tool calling; structured plans;
map-reduce; multi-candidate verification; citation validation; RAG evaluation.
RAG metrics are explicitly unavailable until that pipeline exists.

Next: FastAPI `/health`, `/`, `/analyze`, `/search`, `/issues/recommend` loading real
saved weights; then evidence retrieval/citation checks; React UI; integration tests;
deployment. An external explanation model remains optional.

## Validation in the authoring environment

See `VALIDATION.md` for the actual checks run. Local Windows CUDA training and
real GitHub data collection must be run on your machine before claiming success.

## References

- PyTorch install selector: https://pytorch.org/get-started/locally/
- GitHub pull requests API: https://docs.github.com/en/rest/pulls/pulls
- GitHub issues API: https://docs.github.com/en/rest/issues/issues
- GitHub labels API: https://docs.github.com/en/rest/issues/labels

## License

New project source code is MIT licensed; see `LICENSE`. Collected repository
contents retain their original licenses and are not relicensed by this project.
