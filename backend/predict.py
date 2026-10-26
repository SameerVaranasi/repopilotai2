"""Exercise saved models on existing held-out examples, before API integration."""
import argparse
import pandas as pd
from core import Predictor

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed")
p.add_argument("--artifacts", default="artifacts")
args = p.parse_args()
files = pd.read_csv(args.data + "/file_pairs.csv", keep_default_na=False)
files = files[files.split == "test"]
first = files.iloc[0]
group = files[(files.repo == first.repo) & (files.issue_number == first.issue_number)].copy()
group["prediction_score"] = Predictor(args.artifacts, "file").predict(group)
print("Sample held-out PR query:", first.issue_text[:300])
print(group.sort_values("prediction_score", ascending=False)[["file_path", "prediction_score", "label"]].head(5).to_string(index=False))
issues = pd.read_csv(args.data + "/issues.csv", keep_default_na=False)
issues = issues[issues.split == "test"].head(5).copy()
predictor = Predictor(args.artifacts, "issue")
issues["beginner_score"] = predictor.predict(issues)
issues["predicted_beginner"] = issues.beginner_score >= predictor.threshold
print("\nIssue predictions (scores are not calibrated probabilities of real-world difficulty):")
print(issues[["title", "beginner_score", "predicted_beginner", "label"]].to_string(index=False))
