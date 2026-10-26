# Actual validation for Step 1

Authoring environment: Linux, Python 3.12.14, PyTorch 2.14.1+cu130.
CUDA hardware was unavailable here. No Windows or GPU training is claimed.

Completed:

- Generated explicitly synthetic fixtures with repository-disjoint splits.
- Trained both neural models for 3 CPU epochs and fitted logistic baselines.
- Saved both checkpoints and preprocessing, reloaded each model, and checked
  numerical prediction agreement within 1e-5.
- Exercised the prediction CLI for both tasks and checked the GPU diagnostic.
- Ran `python -m pytest -q`: **8 passed**.
- Tests cover repository overlap rejection, explicit beginner-label mapping,
  train-only vocabulary fitting, ranking metrics on a known example, leakage
  rejection, probability/batching consistency for both reloaded models, and a
  mocked end-to-end collector checking base-commit content and provenance.
- Live read-only GitHub connectivity check for `pallets/flask`: HTTP 200.

Not completed in this environment:

- Full real GitHub dataset collection and live endpoint integration across all
  configured repositories (collector integration used mocked API responses).
- Real-data training or accuracy/generalization measurement.
- CUDA execution on the user's GPU or validation on Windows.
- FastAPI, frontend, deployment, or any RAG component.

Synthetic checkpoints/metrics are intentionally excluded from the deliverable.
Run the documented real collection and CUDA training steps locally to create the
actual artifacts. Preserve the real metrics JSON and training configuration for
the next backend milestone.
