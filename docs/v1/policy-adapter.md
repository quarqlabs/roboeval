# Policy Adapter

The policy adapter standardizes many policy shapes into one SDK decision format.

Users can pass a function, an object with `decide(state)`, or a `PolicyAdapter`.

## Simple Function Policy

```python
def policy_v1(state):
    return {"action": "move_forward"}
```

## Policy With Debug Info

```python
def policy_v2(state):
    return {
        "action": "turn_right",
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
        "action": "reverse",
        "probabilities": {
            "reverse": 0.82,
            "move_forward": 0.04,
        },
        "logits": {
            "reverse": 3.2,
            "move_forward": -0.8,
        },
        "confidence": 0.82,
        "model_version": "policy_v4_trained",
    }
```

The SDK preserves these fields in decision logs and failure cases so debugging can connect actions back to model behavior.

## Accepted Return Shapes

- `"move_forward"`
- `("move_forward", {"reason": "clear path"})`
- `{"action": "move_forward"}`
- `{"action": "move_forward", "debug_info": {...}}`
- `Decision(action="move_forward", debug_info={...})`

