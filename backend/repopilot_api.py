"""Local API overlay: place beside core.py and your real artifacts/ folder."""
import base64
import json
import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit, quote
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from core import Predictor, VERSION, tokens

# Load .env at import time so GEMINI_API_KEY / GITHUB_TOKEN are visible to every
# helper. python-dotenv is optional at runtime: if the package is missing we
# fall back to plain os.getenv and surface a warning in /health.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

import gemini_service

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.rst', '.toml', '.json', '.yaml', '.yml', '.ini', '.txt'}


class Query(BaseModel):
    repo_url: str = Field(max_length=250)
    question: str = Field(default='', max_length=6000)
    issue_title: str = Field(default='', max_length=500)
    issue_body: str = Field(default='', max_length=5500)
    top_k: int = Field(default=5, ge=1, le=20)
    max_files: int = Field(default=30, ge=5, le=60)
    ranking_method: Literal['neural', 'tfidf_cosine'] = 'neural'

    @field_validator('repo_url')
    @classmethod
    def public_url(cls, value):
        parsed = urlsplit(value)
        if (parsed.scheme != 'https' or parsed.netloc != 'github.com' or parsed.query or parsed.fragment
                or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?', parsed.path)):
            raise ValueError('Use https://github.com/OWNER/REPO without extra paths or query parameters')
        parts = parsed.path.strip('/').split('/')
        parts[-1] = parts[-1].removesuffix('.git')
        if any(p in {'', '.', '..'} for p in parts):
            raise ValueError('Invalid repository name')
        return 'https://github.com/' + '/'.join(parts)


class ChatTurn(BaseModel):
    role: Literal['user', 'assistant']
    text: str = Field(max_length=12000)


class ChatQuery(BaseModel):
    """POST /chat input. The repo_url + question are forwarded to /analyze
    internally so the ranker decides what evidence Gemini sees."""
    repo_url: str = Field(max_length=250)
    question: str = Field(min_length=1, max_length=6000)
    issue_title: str = Field(default='', max_length=500)
    issue_body: str = Field(default='', max_length=5500)
    top_k: int = Field(default=5, ge=1, le=20)
    max_files: int = Field(default=30, ge=5, le=60)
    ranking_method: Literal['neural', 'tfidf_cosine'] = 'neural'
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)

    @field_validator('repo_url')
    @classmethod
    def public_url(cls, value):
        # Same rule as Query.public_url — kept duplicated so each Pydantic
        # model is self-contained.
        parsed = urlsplit(value)
        if (parsed.scheme != 'https' or parsed.netloc != 'github.com' or parsed.query or parsed.fragment
                or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?', parsed.path)):
            raise ValueError('Use https://github.com/OWNER/REPO without extra paths or query parameters')
        parts = parsed.path.strip('/').split('/')
        parts[-1] = parts[-1].removesuffix('.git')
        if any(p in {'', '.', '..'} for p in parts):
            raise ValueError('Invalid repository name')
        return 'https://github.com/' + '/'.join(parts)


def github(path, **params):
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'RepoPilot-Local'}
    token = os.getenv('GITHUB_TOKEN', '').strip()
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        with requests.get('https://api.github.com/' + path, params=params, headers=headers,
                          timeout=(10, 30), allow_redirects=False, stream=True) as r:
            if r.status_code == 401:
                raise HTTPException(502, 'GitHub rejected GITHUB_TOKEN. Set a valid token and restart the server.')
            if r.status_code in {403, 429}:
                raise HTTPException(503, {'message': 'GitHub rate limit or access restriction',
                                          'remaining': r.headers.get('X-RateLimit-Remaining'),
                                          'reset_epoch': r.headers.get('X-RateLimit-Reset')})
            if r.status_code == 404:
                raise HTTPException(404, 'GitHub resource not found or unavailable')
            if 300 <= r.status_code < 400:
                raise HTTPException(422, 'Repository moved. Use its current canonical GitHub URL.')
            if r.status_code != 200:
                raise HTTPException(502, f'GitHub returned HTTP {r.status_code}')
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > 10_000_000:
                    raise HTTPException(413, 'GitHub response exceeds the prototype size limit')
                chunks.append(chunk)
            return json.loads(b''.join(chunks))
    except (requests.RequestException, ValueError):
        raise HTTPException(502, 'Unable to read GitHub response; check network and retry')


@lru_cache(maxsize=8)
def snapshot(repo, sha, limit):
    root = 'repos/' + repo
    tree = github(root + '/git/trees/' + sha, recursive='1')
    candidates = [x for x in tree.get('tree', []) if x.get('type') == 'blob'
                  and 0 < x.get('size', 0) <= 60000
                  and PurePosixPath(x['path']).suffix.lower() in EXTENSIONS
                  and not any(p in {'node_modules', 'vendor', 'dist', '.git'} for p in x['path'].split('/'))]
    def priority(x):
        name = PurePosixPath(x['path']).name.lower()
        return (0 if name.startswith(('readme', 'contributing')) else
                1 if name in {'pyproject.toml', 'package.json', 'requirements.txt'} else 2,
                x['path'].count('/'), x['path'])
    candidates.sort(key=priority)
    files, skipped = [], []
    for entry in candidates[:limit]:
        blob = github(root + '/git/blobs/' + entry['sha'])
        try:
            if blob.get('encoding') != 'base64':
                raise ValueError('encoding')
            raw = base64.b64decode(blob['content'], validate=False)
            if len(raw) > 60000 or b'\0' in raw:
                raise ValueError('binary or oversized')
            text = raw.decode('utf-8')
        except (ValueError, UnicodeError, KeyError):
            skipped.append(entry['path'])
            continue
        files.append({'path': entry['path'], 'text': text})
    return {'files': files, 'eligible_files': len(candidates), 'tree_truncated': tree.get('truncated', False), 'skipped': skipped}


def evidence(repo, sha, file, question=''):
    lines = file['text'].splitlines()
    if not lines:
        return None
    wanted = set(tokens(question))
    counts = [len(wanted & set(tokens(line))) for line in lines]
    center = max(range(len(lines)), key=lambda i: counts[i]) if wanted else 0
    start, end = max(0, center - 2), min(len(lines), center + 8)
    return {'kind': 'verified_evidence', 'path': file['path'], 'start_line': start + 1, 'end_line': end,
            'excerpt': '\n'.join(lines[start:end]), 'commit_sha': sha,
            'url': f'https://github.com/{repo}/blob/{sha}/{quote(file["path"], safe="/")}#L{start+1}-L{end}'}


@asynccontextmanager
async def lifespan(app):
    directory = Path(os.getenv('REPOPILOT_ARTIFACTS', str(ROOT / 'artifacts')))
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    if metrics.get('fixture_only') and os.getenv('REPOPILOT_ALLOW_FIXTURE') != '1':
        raise RuntimeError('Use real trained artifacts. Synthetic artifacts are only allowed explicitly for tests.')
    app.state.models = {task: Predictor(directory, task, device='cpu') for task in ('file', 'issue')}
    app.state.fixture = metrics.get('fixture_only', False)
    yield
    app.state.models.clear()
    snapshot.cache_clear()


app = FastAPI(title='RepoPilot ML API', version='0.3.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


@app.get('/')
def root():
    return {'name': 'RepoPilot ML', 'docs': '/docs', 'model_version': VERSION,
            'chat_enabled': gemini_service.is_configured()}


@app.get('/health')
def health():
    return {'status': 'ok', 'models_loaded': sorted(app.state.models), 'model_version': VERSION,
            'inference_device': 'cpu', 'fixture_only': app.state.fixture,
            'github_token_configured': bool(os.getenv('GITHUB_TOKEN')),
            'chat_enabled': gemini_service.is_configured(),
            'gemini_sdk_available': gemini_service.is_available(),
            'gemini_model': gemini_service.get_model_name() if gemini_service.is_configured() else None}


def metadata(query):
    repo = query.repo_url.removeprefix('https://github.com/')
    meta = github('repos/' + repo)
    if meta.get('private', True):
        raise HTTPException(422, 'Only public repositories are supported')
    return repo, meta


def issues_for(repo, limit):
    rows = github('repos/' + repo + '/issues', state='open', per_page=100, sort='updated')
    rows = [r for r in rows if 'pull_request' not in r]
    if not rows:
        return []
    frame = pd.DataFrame([{'title': r['title'], 'body': (r.get('body') or '')[:12000]} for r in rows])
    predictor = app.state.models['issue']
    scores = predictor.predict(frame)
    result = [{'kind': 'model_prediction', 'number': r['number'], 'title': r['title'],
               'url': f'https://github.com/{repo}/issues/{r["number"]}', 'score': float(score),
               'predicted_beginner': bool(score >= predictor.threshold),
               'threshold': predictor.threshold, 'observed_labels': [x['name'] for x in r.get('labels', [])]}
              for r, score in zip(rows, scores)]
    return sorted(result, key=lambda r: r['score'], reverse=True)[:limit]


def analyze(query, include_issues=True):
    repo, meta = metadata(query)
    commit = github('repos/' + repo + '/commits/' + quote(meta['default_branch'], safe=''))
    sha = commit['sha']
    data = snapshot(repo, sha, query.max_files)
    question = '\n'.join(x for x in (query.question, query.issue_title, query.issue_body) if x).strip()
    ranked = []
    if question and data['files']:
        frame = pd.DataFrame([{'issue_text': question[:12000], 'file_path': f['path'],
                               'file_text': f['text'][:12000]} for f in data['files']])
        predictor = app.state.models['file']
        scores = predictor.predict(frame) if query.ranking_method == 'neural' else predictor.features.lexical(frame)
        for f, score in sorted(zip(data['files'], scores), key=lambda x: x[1], reverse=True)[:query.top_k]:
            ranked.append({'kind': 'model_prediction' if query.ranking_method == 'neural' else 'lexical_score',
                           'score': float(score), 'method': query.ranking_method,
                           'shared_tokens': sorted(set(tokens(question)) & set(tokens(f['path'] + ' ' + f['text'][:12000])))[:20],
                           'evidence': evidence(repo, sha, f, question)})
    setup = [evidence(repo, sha, f, 'install setup test') for f in data['files']
             if PurePosixPath(f['path']).name.lower().startswith(('readme', 'contributing'))
             or PurePosixPath(f['path']).name in {'pyproject.toml', 'package.json', 'requirements.txt'}][:8]
    warnings = ['Scores are uncalibrated predictions. Issue difficulty is not verified.',
                'Only a capped subset of files is ranked; this is not a complete architecture analysis.']
    if not question:
        warnings.append('Provide a question or issue text to rank relevant files.')
    if data['tree_truncated']:
        warnings.append('GitHub returned a truncated repository tree.')
    recommendations = []
    if include_issues:
        try:
            recommendations = issues_for(repo, query.top_k)
        except HTTPException as exc:
            warnings.append(f'Issue recommendations unavailable: {exc.detail}')
    return {'repository': {'kind': 'verified_metadata', 'name': meta['full_name'], 'url': query.repo_url,
                           'description': meta.get('description'), 'primary_language': meta.get('language'),
                           'commit_sha': sha},
            'files': [f['path'] for f in data['files']],
            'directories': sorted({f['path'].split('/')[0] for f in data['files'] if '/' in f['path']}),
            'ranked_files': ranked, 'issue_recommendations': recommendations,
            'setup_evidence': [e for e in setup if e],
            'contribution_plan': {'kind': 'generic_suggestion', 'steps': [
                'Read the linked README and contributing excerpts.',
                'Review the suggested files and issue discussion before choosing a change.',
                'Confirm setup and test commands in the repository documentation before running them.']},
            'coverage': {'indexed_files': len(data['files']), 'eligible_files': data['eligible_files'],
                         'max_files': query.max_files, 'skipped_files': data['skipped'],
                         'issue_scope': 'open issues among the latest 100 updated issue/PR entries'},
            'model_version': VERSION, 'fixture_only': app.state.fixture, 'warnings': warnings,
            'tests_execution': 'not_run',
            'llm_generation': 'enabled' if gemini_service.is_configured() else 'not_configured'}


@app.post('/analyze')
def analyze_endpoint(query: Query):
    return analyze(query)


@app.post('/search')
def search_endpoint(query: Query):
    if not any(x.strip() for x in (query.question, query.issue_title, query.issue_body)):
        raise HTTPException(422, 'Provide question or issue text')
    return analyze(query, include_issues=False)


@app.post('/issues/recommend')
def recommend_endpoint(query: Query):
    repo, _ = metadata(query)
    return {'issues': issues_for(repo, query.top_k), 'model_version': VERSION,
            'fixture_only': app.state.fixture, 'scope': 'latest 100 updated open issue/PR entries',
            'warning': 'Uncalibrated predictions of explicit beginner labels, not verified difficulty'}


@app.post('/chat')
def chat_endpoint(query: ChatQuery):
    """Chat with a repository using Gemini (RAG).

    Flow: validate request -> refuse blocked requests -> reuse the trained
    file ranker to fetch top-K evidence excerpts -> call Gemini with the
    evidence as context -> validate every citation in the reply against the
    evidence set -> return the grounded answer.
    """
    if not gemini_service.is_available():
        raise HTTPException(503, 'Gemini SDK not installed. Run '
                                 '`python -m pip install -r requirements_api.txt` and restart.')
    if not gemini_service.is_configured():
        raise HTTPException(503, 'GEMINI_API_KEY is not set. Copy .env.example to .env, '
                                 'paste your key from https://aistudio.google.com/app/apikey, '
                                 'and restart the server.')

    # Reuse the existing /analyze pipeline so Gemini sees exactly what the
    # trained ranker thinks is most relevant for this question.
    base = Query(repo_url=query.repo_url, question=query.question,
                 issue_title=query.issue_title, issue_body=query.issue_body,
                 top_k=query.top_k, max_files=query.max_files,
                 ranking_method=query.ranking_method)
    analysis = analyze(base, include_issues=False)

    history = [{'role': t.role, 'text': t.text} for t in query.history]
    try:
        result = gemini_service.chat(
            question=query.question,
            analysis=analysis,
            history=history,
        )
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))

    return {
        'kind': 'generated_suggestion',
        'answer': result.answer,
        'citations': result.citations,
        'invalid_citations': result.invalid_citations,
        'grounded': result.grounded,
        'model': result.model,
        'evidence_used': result.evidence_used,
        'blocked_reason': result.blocked_reason,
        'usage': result.usage,
        'repository': analysis.get('repository'),
        'commit_sha': analysis.get('repository', {}).get('commit_sha'),
        'warnings': [
            'Generated text is a suggestion built on retrieved evidence, never a verified fact.',
            'Citations are mechanically validated against retrieved excerpts, not against the full repository.',
        ],
    }
