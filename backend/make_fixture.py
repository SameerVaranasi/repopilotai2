"""Synthetic plumbing test. Never use its metrics as hackathon accuracy."""
import json
from pathlib import Path
import pandas as pd


def main():
    out = Path("data/fixture")
    out.mkdir(parents=True, exist_ok=True)
    pairs, issues = [], []
    for split in ["train", "validation", "test"]:
        for n in range(3):
            repo = f"fixture/{split}-{n}"
            for i in range(20):
                topic = ["login", "parser", "cache", "route"][i % 4]
                for j in range(8):
                    positive = j == i % 4
                    other = ["login", "parser", "cache", "route"][j % 4]
                    pairs.append(dict(repo=repo, issue_number=i, issue_text=f"fix {topic} error",
                                      file_path=f"src/{other}{j}.py", file_text=f"def {other}(): return '{other}'",
                                      label=int(positive), split_group=repo, split=split,
                                      source_pr_url=f"https://example.invalid/{repo}/pull/{i}", label_method="synthetic_fixture"))
                beginner = i % 2
                issues.append(dict(repo=repo, issue_number=i,
                                   title=("update documentation typo" if beginner else "redesign concurrent event scheduler") + f" {topic}",
                                   body="synthetic fixture only", label=beginner,
                                   labels='["good first issue"]' if beginner else "[]", created_at="2026-01-01",
                                   comments_count=0, split_group=repo, split=split, label_method="synthetic_fixture"))
    pd.DataFrame(pairs).to_csv(out / "file_pairs.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "issues.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({"fixture": True, "source": "make_fixture.py", "seed": 42}))
    print("Synthetic fixture created. Not real training evidence.")


if __name__ == "__main__":
    main()
