# Reports

SDK v1 writes four report artifacts:

```text
runs/latest/decision_logs.jsonl
runs/latest/episode_results.json
runs/latest/comparison_report.json
runs/latest/report.md
```

## Decision Logs

`decision_logs.jsonl` contains one line per step:

- episode id
- scenario name
- policy version
- step
- state
- action
- next state
- outcome
- failure label
- terminal marker
- policy debug info

This is the lowest-level debugging file.

## Episode Results

`episode_results.json` contains each full episode with logs, rule results, first failure step, and scenario metadata.

Use this when you want to inspect one policy on one scenario.

## Comparison Report

`comparison_report.json` contains the cross-policy summary:

- run metadata
- policy summary
- regressions
- improvements
- failure cases
- action divergences
- grouped metrics
- human-readable highlights

Run metadata includes:

- SDK name
- SDK version
- generated timestamp
- baseline policy
- policy versions
- scenario count
- episode count
- environment name

## Markdown Report

`report.md` is the human-readable report for review.

It starts with run metadata and highlights, then shows metrics and deeper debug sections.

Example highlight:

```text
policy_v4_trained improved dead_end_reverse_needed; baseline policy_v1_cautious failed with unsafe_forward_action.
policy_v1_cautious moved forward unsafely on dead_end_reverse_needed at step 0.
policy_v4_trained reversed, escaped, then reached goal on dead_end_reverse_needed.
```

