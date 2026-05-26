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

The environment receives the raw policy action, but report files store a
JSON-safe copy. Strings remain strings. Lists, tuples, dicts, dataclasses,
numpy-like arrays, and tensor-like objects are converted into serializable
values; unknown objects fall back to `repr(...)`.

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
- outcome counts
- metric summaries
- human-readable highlights

Action divergences include both serialized action values and stable action keys
so non-string actions can still be compared across policy versions.

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
arm_policy_v2 improved grasp_cube; baseline arm_policy_v1 failed rule object_grasped.
arm_policy_v2 outcome trace: object_aligned -> object_grasped.
arm_policy_v2 chose close_gripper while baseline chose move_arm_down at step 1.
```
