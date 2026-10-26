import numpy as np
import pandas as pd
import pytest
from core import Features, Predictor
from train import read_data, ranking
from collect import validate_repos, is_beginner


def test_repo_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_repos({"train": ["a/b"], "validation": ["A/B"], "test": ["c/d"]})


def test_label_mapping():
    assert is_beginner(["good-first-issue"])
    assert not is_beginner(["help wanted"])


def test_vectorizer_fit_train_only():
    f = Features("issue").fit(pd.DataFrame({"title": ["trainword"], "body": ["training"]}))
    x = f.transform(pd.DataFrame({"title": ["heldoutuniqueword"], "body": [""]}))
    assert "heldoutuniqueword" not in f.vectorizer.vocabulary_
    assert np.count_nonzero(x) == 0


def test_ranking_known_answer():
    frame = pd.DataFrame({"repo": ["a/b"] * 6, "issue_number": [1] * 6, "label": [0, 1, 0, 0, 0, 1]})
    m = ranking(frame, np.array([6, 5, 4, 3, 2, 1]))
    assert m["mrr"] == 0.5
    assert m["recall_at_5"] == 0.5
    assert m["precision_at_5"] == 0.2


def test_split_leakage_rejected(tmp_path):
    data = pd.read_csv("data/fixture/issues.csv")
    data.loc[data.split == "test", "repo"] = "fixture/train-0"
    data["split_group"] = data.repo
    path = tmp_path / "issues.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="leakage"):
        read_data(path, "issue")


@pytest.mark.parametrize("task,name", [("file", "file_pairs.csv"), ("issue", "issues.csv")])
def test_reload_predicts_probabilities(task, name):
    frame = pd.read_csv("data/fixture/" + name, keep_default_na=False).iloc[:7]
    predictor = Predictor("artifacts_fixture", task)
    scores = predictor.predict(frame, batch_size=3)
    assert len(scores) == 7
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    np.testing.assert_allclose(scores, predictor.predict(frame, batch_size=7), atol=1e-6)


def test_collector_uses_base_commit_and_preserves_provenance(tmp_path, monkeypatch):
    import base64
    import json
    import sys
    import collect

    def fake_get(self, path, **params):
        if path.endswith("/labels"):
            return [{"name": "good first issue"}]
        if path.endswith("/issues"):
            return [dict(number=i, title=f"issue {i}", body="text", labels=labels,
                         created_at="2026-01-01", comments=0, html_url=f"https://example.invalid/issue/{i}")
                    for i, labels in [(1, [{"name": "good first issue"}]), (2, [])]]
        if path.endswith("/pulls"):
            return [{"number": 3}]
        if path.endswith("/pulls/3"):
            return dict(number=3, title="fix parser", body="details", merged_at="2026-01-01",
                        base={"sha": "base-commit"}, changed_files=1, html_url="https://example.invalid/pull/3")
        if path.endswith("/files"):
            return [{"filename": "parser.py"}]
        if "/git/trees/" in path:
            return {"truncated": False, "tree": [{"path": name, "type": "blob", "size": 50}
                                                  for name in ["parser.py", "other.py"]]}
        if "/contents/" in path:
            assert params["ref"] == "base-commit"
            return {"encoding": "base64", "content": base64.b64encode(b"def parser(): pass").decode()}
        return {"private": False, "license": {"spdx_id": "MIT"}, "html_url": "https://example.invalid/repo"}

    monkeypatch.chdir(tmp_path)
    Path = __import__("pathlib").Path
    Path("repos.json").write_text(json.dumps({"train": ["a/train"], "validation": ["a/val"], "test": ["a/test"]}))
    monkeypatch.setattr(collect.API, "get", fake_get)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--prs-per-repo", "1", "--issues-per-repo", "2"])
    collect.main()
    files = read_data(Path("data/processed/file_pairs.csv"), "file")
    issues = read_data(Path("data/processed/issues.csv"), "issue")
    assert len(files) == 6 and len(issues) == 6
    assert set(files.base_sha) == {"base-commit"}
    assert all(files[files.label == 1].file_path == "parser.py")
    assert all(files[files.label == 0].file_path == "other.py")
    assert not json.loads(Path("data/processed/provenance.json").read_text())["fixture"]
