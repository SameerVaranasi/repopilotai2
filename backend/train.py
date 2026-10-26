import argparse
import copy
import hashlib
import json
import platform
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from core import Features, MLP, Predictor, VERSION


def classification(y, scores, threshold=0.5):
    pred = scores >= threshold
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
            "roc_auc_note": None if len(set(y)) == 2 else "Only one class present"}


def ranking(frame, scores, k=5):
    frame = frame.copy()
    frame["score"] = scores
    values = []
    for _, group in frame.groupby(["repo", "issue_number"]):
        relevant = int(group.label.sum())
        if relevant == 0:
            continue
        y = group.sort_values("score", ascending=False, kind="stable").label.to_numpy()
        hits = np.flatnonzero(y == 1) + 1
        values.append([float(y[:k].sum() / k), float(y[:k].sum() / relevant),
                       float(1 / hits[0]), float(np.mean(np.arange(1, len(hits) + 1) / hits))])
    result = {key: float(value) for key, value in zip(
        ["precision_at_5", "recall_at_5", "mrr", "map"], np.mean(values, axis=0))} if values else {}
    return {**result, "evaluated_queries": len(values),
            "candidate_scope": "sampled files per PR; not full-repository ranking",
            "precision_denominator": k}


def read_data(path, task):
    data = pd.read_csv(path, keep_default_na=False)
    required = {"repo", "issue_number", "label", "split", "split_group", "label_method"}
    required |= {"issue_text", "file_path", "file_text", "source_pr_url"} if task == "file" else {
        "title", "body", "labels", "created_at", "comments_count"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns in {path}: {required - set(data.columns)}")
    if set(data.split) != {"train", "validation", "test"}:
        raise ValueError("Need train, validation and test rows")
    if not set(data.label).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    if (data.repo != data.split_group).any():
        raise ValueError("split_group must be the repository")
    groups = data.groupby(data.repo.str.lower()).split.nunique()
    if (groups > 1).any():
        raise ValueError("Repository leakage across splits")
    identity = ["repo", "issue_number", "file_path"] if task == "file" else ["repo", "issue_number"]
    if data.duplicated(identity).any():
        raise ValueError("Duplicate examples detected")
    for split in ("train", "validation", "test"):
        part = data[data.split == split]
        if set(part.label) != {0, 1}:
            raise ValueError(f"{task}/{split} needs both classes. Collect more data or revise repo choices; do not shuffle rows.")
    return data


def train_task(data, task, out, args, device):
    train, val, test = [data[data.split == s].reset_index(drop=True) for s in ("train", "validation", "test")]
    features = Features(task, args.max_features).fit(train)
    x, xv, xt = [features.transform(f) for f in (train, val, test)]
    y, yv, yt = [f.label.to_numpy(dtype="float32") for f in (train, val, test)]
    baseline = LogisticRegression(max_iter=500, random_state=args.seed).fit(x, y)
    model = MLP(x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Use unweighted loss; validation selects the issue decision threshold.
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                                         batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(yb)
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(torch.from_numpy(xv).to(device)), torch.from_numpy(yv).to(device)).cpu())
        history.append({"epoch": epoch, "train_loss": total / len(y), "validation_loss": validation_loss})
        print(f"{task} epoch {epoch}: train={total/len(y):.4f} validation={validation_loss:.4f}", flush=True)
        if validation_loss < best_loss - 0.00001:
            best_loss, best_state, stale, best_epoch = validation_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(torch.from_numpy(xv).to(device))).cpu().numpy()
        test_scores = torch.sigmoid(model(torch.from_numpy(xt).to(device))).cpu().numpy()
    grid = np.linspace(0.05, 0.95, 37)
    threshold = float(max(grid, key=lambda t: f1_score(yv, val_scores >= t, zero_division=0))) if task == "issue" else 0.5
    prefix = "file_ranker" if task == "file" else "issue_classifier"
    torch.save({"version": VERSION, "input_dim": x.shape[1], "hidden": args.hidden, "threshold": threshold,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, out / f"{prefix}.pt")
    joblib.dump(features, out / f"{task}_feature_config.joblib")
    joblib.dump(features.vectorizer, out / f"{task}_vectorizer.joblib")
    joblib.dump(baseline, out / f"{task}_baseline.joblib")
    restored = Predictor(out, task).predict(test)
    if not np.allclose(restored, test_scores, atol=1e-5):
        raise RuntimeError("Save/load prediction mismatch")
    if task == "file":
        report = {"neural": ranking(test, test_scores),
                  "logistic_regression": ranking(test, baseline.predict_proba(xt)[:, 1]),
                  "tfidf_cosine": ranking(test, features.lexical(test))}
    else:
        bpv = baseline.predict_proba(xv)[:, 1]
        bt = float(max(grid, key=lambda t: f1_score(yv, bpv >= t, zero_division=0)))
        report = {"neural": classification(yt, test_scores, threshold),
                  "logistic_regression": classification(yt, baseline.predict_proba(xt)[:, 1], bt),
                  "baseline_threshold": bt, "neural_threshold": threshold}
    pd.DataFrame(history).to_csv(out / f"{task}_training_history.csv", index=False)
    report["best_epoch"] = best_epoch
    report["reload_check"] = "passed"
    report["splits"] = {s: {"rows": len(data[data.split == s]),
                             "positives": int(data[data.split == s].label.sum()),
                             "negatives": int((data[data.split == s].label == 0).sum()),
                             "repositories": sorted(data[data.split == s].repo.unique())} for s in ("train", "validation", "test")}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-features", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-fixture", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.max_features, args.hidden, args.patience) < 1:
        p.error("Training sizes and epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Verify your PyTorch installation and NVIDIA driver.")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    print(f"Training on {device}" + (": " + torch.cuda.get_device_name(0) if device == "cuda" else ""), flush=True)
    data_dir, out = Path(args.data), Path(args.out)
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    fixture = provenance.get("fixture", False)
    if fixture and not args.allow_fixture:
        raise ValueError("Synthetic fixtures require --allow-fixture and are not real performance evidence")
    if fixture and out.name == "artifacts":
        raise ValueError("Use a separate --out artifacts_fixture directory for fixture models")
    # Validate both tasks and their shared split assignments before writing models.
    frames = {task: read_data(data_dir / name, task) for task, name in
              [("file", "file_pairs.csv"), ("issue", "issues.csv")]}
    combined = pd.concat([f[["repo", "split"]] for f in frames.values()])
    if (combined.groupby(combined.repo.str.lower()).split.nunique() > 1).any():
        raise ValueError("Cross-task repository leakage")
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"version": VERSION, "fixture_only": fixture, "device": device,
               "interpretation": "Synthetic wiring test only" if fixture else "Held-out repositories; weak proxy labels and sampled file candidates",
               "rag_metrics": {"status": "unavailable", "reason": "RAG is not implemented in this training milestone"}}
    for task, frame in frames.items():
        metrics[task] = train_task(frame, task, out, args, device)
    config = {**vars(args), "version": VERSION, "device_used": device,
              "versions": {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__},
              "dataset_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                                  for name in ["file_pairs.csv", "issues.csv", "provenance.json"]}}
    for name, content in [("metrics.json", metrics), ("training_config.json", config),
                          ("data_provenance.json", provenance),
                          ("label_mapping.json", {"file": {"0": "sampled unchanged weak negative", "1": "changed file weak positive"},
                                                  "issue": {"0": "not explicitly beginner-labeled", "1": "explicitly beginner-labeled"}})]:
        (out / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Both models saved and reload-checked. Results: {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
