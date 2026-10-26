"""Shared, versioned preprocessing and inference for training and the later API."""
import re
from pathlib import PurePosixPath, Path
import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

VERSION = "repopilot-step1-v1"


def tokens(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    return re.findall(r"[a-z][a-z0-9]*", text.lower())


class Features:
    def __init__(self, task, max_features=2048):
        self.task = task
        self.vectorizer = TfidfVectorizer(tokenizer=tokens, token_pattern=None,
                                         max_features=max_features, sublinear_tf=True)

    def fit(self, frame):
        if self.task == "file":
            texts = frame.issue_text.tolist() + (frame.file_path + " " + frame.file_text).tolist()
        else:
            texts = (frame.title + " " + frame.body).tolist()
        self.vectorizer.fit(texts)
        return self

    def transform(self, frame):
        if self.task == "issue":
            return self.vectorizer.transform(frame.title + " " + frame.body).toarray().astype("float32")
        q = self.vectorizer.transform(frame.issue_text).toarray().astype("float32")
        d = self.vectorizer.transform(frame.file_path + " " + frame.file_text).toarray().astype("float32")
        numeric = []
        for row in frame.itertuples():
            a, b = set(tokens(row.issue_text)), set(tokens(row.file_path + " " + row.file_text))
            p = PurePosixPath(row.file_path)
            numeric.append([len(a & b) / max(1, len(a)),
                            np.log1p(len(row.file_text)) / 12,
                            float("test" in row.file_path.lower()),
                            float(p.suffix in {".toml", ".json", ".yaml", ".yml", ".ini"}),
                            float(p.suffix == ".py")])
        return np.concatenate([q, d, q * d, np.asarray(numeric, dtype="float32")], axis=1)

    def lexical(self, frame):
        q = self.vectorizer.transform(frame.issue_text)
        d = self.vectorizer.transform(frame.file_path + " " + frame.file_text)
        return np.asarray(q.multiply(d).sum(axis=1)).ravel()


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden),
                                          torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class Predictor:
    """Only load artifacts produced by your own trusted training run."""
    def __init__(self, directory, task, device="cpu"):
        directory = Path(directory)
        prefix = "file_ranker" if task == "file" else "issue_classifier"
        self.features = joblib.load(directory / f"{task}_feature_config.joblib")
        saved = torch.load(directory / f"{prefix}.pt", map_location=device, weights_only=True)
        if saved["version"] != VERSION:
            raise ValueError("Model/preprocessing version mismatch")
        self.model = MLP(saved["input_dim"], saved["hidden"]).to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.device = device
        self.threshold = saved["threshold"]

    def predict(self, frame, batch_size=128):
        outputs = []
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                x = self.features.transform(frame.iloc[start:start + batch_size])
                outputs.extend(torch.sigmoid(self.model(torch.from_numpy(x).to(self.device))).cpu().tolist())
        return np.asarray(outputs)
