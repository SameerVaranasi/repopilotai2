"""Collect real weak supervision. Never executes code from a repository."""
import argparse
import base64
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests

BEGINNER = {"good first issue", "beginner", "beginner friendly", "first timers only"}
EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".toml", ".json", ".yaml", ".yml", ".ini"}


def is_beginner(labels):
    return bool({re.sub(r"[-_]", " ", x.lower()).strip() for x in labels} & BEGINNER)


def validate_repos(config):
    seen = set()
    for split in ("train", "validation", "test"):
        if not config.get(split):
            raise ValueError(f"Missing repositories for {split}")
        for repo in config[split]:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
                raise ValueError("Use owner/repository names")
            if repo.lower() in seen:
                raise ValueError(f"Repository overlap: {repo}")
            seen.add(repo.lower())


class API:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github+json", "User-Agent": "RepoPilot-Training"})
        if os.getenv("GITHUB_TOKEN"):
            self.session.headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]

    def get(self, path, **params):
        key = hashlib.sha256((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        target = self.cache / f"{key}.json"
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
        for attempt in range(4):
            response = self.session.get("https://api.github.com/" + path, params=params, timeout=45)
            if response.status_code in {403, 429}:
                raise RuntimeError("GitHub rate limit/permission error. Check token locally; rerun later to resume cached data.")
            if response.status_code >= 500 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            result = response.json()
            target.write_text(json.dumps(result), encoding="utf-8")
            time.sleep(0.15)
            return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repos", default="repos.json")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--prs-per-repo", type=int, default=12)
    p.add_argument("--issues-per-repo", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.prs_per_repo < 1 or args.issues_per_repo < 2:
        p.error("Use at least one PR and two issues per repository")
    config = json.loads(Path(args.repos).read_text())
    validate_repos(config)
    api = API("data/cache")
    rng = random.Random(args.seed)
    pairs, issues, snapshots, skipped = [], [], [], []
    for split, repos in config.items():
        for repo in repos:
            print(f"Collecting {repo} ({split})", flush=True)
            root = f"repos/{repo}"
            meta = api.get(root)
            if meta.get("private"):
                raise ValueError("Only public repositories are supported")
            if not meta.get("license"):
                raise ValueError(f"{repo} has no detected license. Choose another licensed public repository.")
            snapshots.append({"repo": repo, "split": split, "license": meta["license"],
                              "html_url": meta["html_url"]})
            # Query known beginner labels separately so recent unlabeled issues do not crowd them out.
            labels = []
            for page in range(1, 6):
                batch = api.get(root + "/labels", per_page=100, page=page)
                labels.extend(x["name"] for x in batch)
                if len(batch) < 100:
                    break
            mapped = [label for label in labels if is_beginner([label])]
            snapshots[-1]["beginner_label_mapping"] = mapped
            candidates = []
            for label in [None] + mapped:
                query = {"state": "all", "per_page": 100}
                if label:
                    query["labels"] = label
                candidates.extend(api.get(root + "/issues", **query))
            unique = {x["number"]: x for x in candidates if "pull_request" not in x}
            positive, ordinary = [], []
            for item in unique.values():
                names = [x["name"] for x in item["labels"]]
                row = dict(repo=repo, issue_number=item["number"], title=item["title"],
                           body=(item.get("body") or "")[:12000], labels=json.dumps(names),
                           created_at=item["created_at"], comments_count=item["comments"],
                           label=int(is_beginner(names)), split_group=repo, split=split,
                           source_url=item["html_url"],
                           label_method="explicit_beginner_label" if is_beginner(names) else "not_explicitly_beginner_weak_negative")
                (positive if row["label"] else ordinary).append(row)
            # Keep both classes when available, recording the sampling strategy.
            rng.shuffle(positive)
            rng.shuffle(ordinary)
            cap = args.issues_per_repo // 2
            issues.extend(positive[:cap] + ordinary[:args.issues_per_repo - min(cap, len(positive))])
            pulls = api.get(root + "/pulls", state="closed", per_page=100, sort="updated", direction="desc")
            used = 0
            for summary in pulls:
                if used >= args.prs_per_repo:
                    break
                pr = api.get(root + f"/pulls/{summary['number']}")
                if not pr.get("merged_at"):
                    continue
                sha = pr["base"]["sha"]
                tree = api.get(root + f"/git/trees/{sha}", recursive="1")
                if tree.get("truncated"):
                    skipped.append({"repo": repo, "pr": pr["number"], "reason": "truncated tree"})
                    continue
                eligible = {x["path"] for x in tree["tree"] if x["type"] == "blob" and
                            x.get("size", 0) <= 60000 and Path(x["path"]).suffix in EXTENSIONS and
                            not any(t in x["path"].split("/") for t in ("vendor", "node_modules", "dist", ".github"))}
                changed = api.get(root + f"/pulls/{pr['number']}/files", per_page=100)
                if pr.get("changed_files", 0) > 100:
                    skipped.append({"repo": repo, "pr": pr["number"], "reason": "over 100 changed files"})
                    continue
                changed_names = {f["filename"] for f in changed}
                positives = sorted(eligible & changed_names)
                negatives = sorted(eligible - changed_names)
                if not positives or not negatives:
                    continue
                chosen = [(f, 1) for f in rng.sample(positives, min(5, len(positives)))]
                chosen += [(f, 0) for f in rng.sample(negatives, min(15, len(negatives)))]
                for filename, label in chosen:
                    try:
                        blob = api.get(root + "/contents/" + quote(filename, safe="/"), ref=sha)
                        if blob.get("encoding") != "base64":
                            raise ValueError("unsupported content encoding")
                        text = base64.b64decode(blob["content"]).decode("utf-8")[:12000]
                    except (requests.HTTPError, UnicodeError, ValueError) as exc:
                        skipped.append({"repo": repo, "path": filename, "reason": type(exc).__name__})
                        continue
                    pairs.append(dict(repo=repo, issue_number=pr["number"],
                                      issue_text=(pr["title"] + "\n" + (pr.get("body") or ""))[:12000],
                                      file_path=filename, file_text=text, label=label, split_group=repo,
                                      split=split, source_pr_url=pr["html_url"], base_sha=sha,
                                      label_method="merged_pr_description_proxy_changed_base_file" if label else "sampled_unchanged_base_file_weak_negative"))
                used += 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not pairs or not issues:
        raise ValueError("No usable examples. Inspect repository choices and collection limits.")
    pd.DataFrame(pairs).to_csv(out / "file_pairs.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "issues.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({"fixture": False, "collected_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args), "repositories": snapshots, "skipped": skipped,
        "limitations": ["PR text is a proxy query, not necessarily a linked issue.",
                         "Changed files are weak positives; unchanged files can still be relevant.",
                         "Unlabeled issues are not known to be difficult.",
                         "Recent capped samples are not representative; ranking uses sampled candidates.",
                         "PR descriptions may explicitly name changed paths; this can inflate ranking metrics."]}, indent=2), encoding="utf-8")
    print(f"Saved {len(pairs)} file pairs and {len(issues)} issues to {out}")


if __name__ == "__main__":
    main()
