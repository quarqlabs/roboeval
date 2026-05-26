# Policy Adapter

The policy adapter standardizes many policy shapes into one SDK decision format.

Users can pass a function, an object with `decide(state)`, or a `PolicyAdapter`.

## Simple Function Policy

```python
def policy_v1(state):
    return {"action": "close_gripper"}
```

Actions are generic. A policy can return a string action, a discrete integer,
or a continuous/vector action such as a list, tuple, array, or tensor-like
object. The SDK passes the raw action to the environment.

```python
def continuous_arm_policy(state):
    return {"action": [0.12, -0.04, 0.8]}
```

## Policy With Debug Info

```python
def policy_v2(state):
    return {
        "action": "scan_target",
        "debug_info": {
            "reason": "front obstacle detected",
        },
    }
```

## Trained Model Policy

Model policies can include standard ML debug fields:

```python
def policy_v4_trained(state):
    return {
        "action": "close_gripper",
        "probabilities": {
            "close_gripper": 0.82,
            "move_arm_down": 0.04,
        },
        "logits": {
            "close_gripper": 3.2,
            "move_arm_down": -0.8,
        },
        "confidence": 0.82,
        "model_version": "policy_v4_trained",
    }
```

The SDK preserves these fields in decision logs and failure cases so debugging can connect actions back to model behavior.

## Accepted Return Shapes

- `"close_gripper"`
- `[0.12, -0.04, 0.8]`
- `{"joint_delta": [0.1, -0.2], "grip": 0.7}`
- `("close_gripper", {"reason": "object aligned"})`
- `{"action": "close_gripper"}`
- `{"action": [0.12, -0.04, 0.8], "confidence": 0.91}`
- `{"action": "close_gripper", "debug_info": {...}}`
- `Decision(action="close_gripper", debug_info={...})`

For JSON/Markdown reports, non-string actions are converted into a safe
serialized form. The environment still receives the original raw action object.
