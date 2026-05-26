# Trained Policy v4 Demo

This demo trains a PyTorch classifier and plugs it into `roboeval` as `policy_v4_trained`.

## Flow

```text
generate synthetic robot states
  -> label them with expert safety rules
  -> clean/validate rows
  -> train PyTorch classifier
  -> save weights + metadata
  -> load as policy_v4_trained
  -> run SDK eval report with metadata and highlights
```

## Run

This example requires PyTorch. The SDK core itself stays dependency-free.

```bash
python3 examples/trained_policy/train.py
python3 examples/trained_policy/self_check.py
python3 examples/trained_policy/run_eval.py
```

Outputs:

```text
examples/trained_policy/data/synthetic_policy_data.csv
examples/trained_policy/data/clean_policy_data.csv
examples/trained_policy/artifacts/policy_v4_trained.pt
examples/trained_policy/artifacts/metadata.json
runs/trained_policy_v4/report.md
```

The generated Markdown report should show which policy improved, which policy failed, and a short story of the trained policy behavior, such as reversing, escaping, and reaching the goal.
