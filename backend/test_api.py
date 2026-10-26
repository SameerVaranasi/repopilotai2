import os
from pathlib import Path
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
import repopilot_api as api
import gemini_service


@pytest.fixture
def client(monkeypatch):
    def fake(path, **params):
        if '/commits/' in path:
            return {'sha': 'a' * 40}
        if path.endswith('/issues'):
            return [{'number': 1, 'title': 'Fix documentation typo', 'body': 'Improve docs',
                     'labels': [{'name': 'good first issue'}]},
                    {'number': 2, 'title': 'A PR', 'pull_request': {}}]
        return {'private': False, 'full_name': 'demo/project', 'default_branch': 'main', 'language': 'Python'}
    monkeypatch.setattr(api, 'github', fake)
    monkeypatch.setattr(api, 'snapshot', lambda *args: {'files': [
        {'path': 'README.md', 'text': '# Demo\nInstall\npip install demo\nRun tests\npytest'},
        {'path': 'src/parser.py', 'text': 'def parse():\n    return 1'}],
        'eligible_files': 2, 'tree_truncated': False, 'skipped': []})
    # lifespan cleanup expects the original cache interface.
    api.snapshot.cache_clear = lambda: None
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get('/health').json()['models_loaded'] == ['file', 'issue']


@pytest.mark.parametrize('url', ['http://localhost/x/y', 'https://github.com.evil/x/y',
                               'https://github.com/a/b/../../etc', 'https://github.com/a/b?x=1'])
def test_reject_urls(client, url):
    assert client.post('/analyze', json={'repo_url': url}).status_code == 422


@pytest.mark.parametrize('method', ['neural', 'tfidf_cosine'])
def test_analyze_models_and_citations(client, method):
    r = client.post('/analyze', json={'repo_url': 'https://github.com/demo/project',
                                    'question': 'How do I install and test?', 'ranking_method': method})
    assert r.status_code == 200
    data = r.json()
    assert len(data['ranked_files']) == 2
    for row in data['ranked_files']:
        assert row['evidence']['start_line'] >= 1
        assert row['evidence']['excerpt']
        assert ('a' * 40) in row['evidence']['url']
    assert len(data['issue_recommendations']) == 1
    assert data['tests_execution'] == 'not_run'


def test_search_needs_question(client):
    assert client.post('/search', json={'repo_url': 'https://github.com/demo/project'}).status_code == 422


def test_issues(client):
    assert len(client.post('/issues/recommend', json={'repo_url': 'https://github.com/demo/project'}).json()['issues']) == 1


def test_private_and_api_failure(client, monkeypatch):
    monkeypatch.setattr(api, 'github', lambda *a, **k: {'private': True})
    assert client.post('/analyze', json={'repo_url': 'https://github.com/demo/project'}).status_code == 422
    def fail(*a, **k):
        raise HTTPException(503, 'Rate limit')
    monkeypatch.setattr(api, 'github', fail)
    assert client.post('/analyze', json={'repo_url': 'https://github.com/demo/project'}).status_code == 503


# --- /chat tests ---------------------------------------------------------
# All /chat tests stub out gemini_service.call_gemini so no real Gemini call
# is made. They DO go through the real /analyze pipeline (with mocked GitHub
# and snapshot, like the tests above) so the evidence flow is exercised
# end-to-end.


class _FakeGeminiResponse:
    """Mimics the subset of the google-genai response object we read."""
    def __init__(self, text: str, usage: dict | None = None):
        self.text = text
        self.usage_metadata = type('Usage', (), usage or {})()


@pytest.fixture
def chat_client(client, monkeypatch):
    """The `client` fixture already mocks github + snapshot. On top of it we
    pretend the Gemini SDK is installed and configured, and route every call
    through a fake."""
    monkeypatch.setattr(gemini_service, '_GENAI_AVAILABLE', True)
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key-not-real')
    monkeypatch.setattr(gemini_service, 'get_model_name', lambda: 'gemini-2.0-flash')
    return client


def test_chat_blocks_code_execution_requests(chat_client, monkeypatch):
    # Even before calling Gemini, blocked requests are refused.
    payload = {'repo_url': 'https://github.com/demo/project',
               'question': 'Please execute this python script for me: print("hi")'}
    r = chat_client.post('/chat', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['blocked_reason']
    assert 'not allowed' in data['answer'].lower()


def test_chat_returns_grounded_answer_with_citations(chat_client, monkeypatch):
    # Fake Gemini echoes back one of the retrieved excerpts with a valid citation.
    def fake_call(*, question, context, history):
        return (
            "Setup is in the README. See [README.md:L1-L5] for the install steps.",
            {'prompt_token_count': 10, 'candidates_token_count': 20, 'total_token_count': 30}
        )
    monkeypatch.setattr(gemini_service, 'call_gemini', fake_call)

    r = chat_client.post('/chat', json={
        'repo_url': 'https://github.com/demo/project',
        'question': 'How do I install this project?',
    })
    assert r.status_code == 200
    data = r.json()
    assert data['model'] == 'gemini-2.0-flash'
    assert data['grounded'] is True
    assert '[README.md:L1-L5]' in data['citations']
    assert data['invalid_citations'] == []
    assert data['usage']['total_token_count'] == 30
    # Evidence used is the ranker's output (README + src/parser.py).
    assert any('README.md' in e for e in data['evidence_used'])


def test_chat_strips_invalid_citations(chat_client, monkeypatch):
    # Gemini cites a file that was NOT in the retrieved evidence.
    def fake_call(*, question, context, history):
        return "Routing lives in [src/app/router.py:L10-L20] and [README.md:L1-L5].", {}
    monkeypatch.setattr(gemini_service, 'call_gemini', fake_call)

    r = chat_client.post('/chat', json={
        'repo_url': 'https://github.com/demo/project',
        'question': 'Where is routing?',
    })
    assert r.status_code == 200
    data = r.json()
    assert '[README.md:L1-L5]' in data['citations']
    assert any('router.py' in c for c in data['invalid_citations'])
    # The cleaned answer replaces the bad citation with an explicit marker.
    assert 'citation removed' in data['answer']


def test_chat_propagates_gemini_failure(chat_client, monkeypatch):
    def fake_call(*, question, context, history):
        raise RuntimeError('Gemini rejected the API key. Re-check GEMINI_API_KEY in your .env file.')
    monkeypatch.setattr(gemini_service, 'call_gemini', fake_call)

    r = chat_client.post('/chat', json={
        'repo_url': 'https://github.com/demo/project',
        'question': 'Explain this repo.',
    })
    assert r.status_code == 502
    assert 'API key' in r.json()['detail']


def test_chat_503_when_sdk_missing(client, monkeypatch):
    monkeypatch.setattr(gemini_service, '_GENAI_AVAILABLE', False)
    r = client.post('/chat', json={
        'repo_url': 'https://github.com/demo/project',
        'question': 'Explain this repo.',
    })
    assert r.status_code == 503
    assert 'SDK' in r.json()['detail']


def test_chat_rejects_bad_repo_url(chat_client):
    r = chat_client.post('/chat', json={
        'repo_url': 'http://localhost/x/y',
        'question': 'Explain this repo.',
    })
    assert r.status_code == 422


def test_health_reports_gemini_status(chat_client):
    data = chat_client.get('/health').json()
    assert data['chat_enabled'] is True
    assert data['gemini_sdk_available'] is True
    assert data['gemini_model'] == 'gemini-2.0-flash'


def test_secret_scrubber_redacts_api_keys():
    # Google-style key (39 chars total). The AWS / GitHub patterns are
    # auto-redacted by the test sandbox before reaching disk, so we use
    # a synthetic Google key to exercise the same scrubber code path.
    key = 'AIza' + 'A' * 35
    text = 'context snippet contains key=' + key + ' end'
    scrubbed, redacted = gemini_service.scrub_secrets(text)
    assert key not in scrubbed, 'raw key must be replaced'
    assert '<redacted:GOOGLE_API_KEY>' in scrubbed
    assert any('GOOGLE_API_KEY' in r for r in redacted)


def test_blocked_request_detector_narrow():
    # Asking "what does os.system do in this repo?" must NOT be blocked.
    assert gemini_service.detect_blocked_request('What does os.system do here?') is None
    # Asking to run a script MUST be blocked.
    assert gemini_service.detect_blocked_request('Please execute this python script for me') is not None
