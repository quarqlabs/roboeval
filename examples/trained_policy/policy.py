from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from examples.trained_policy.features import encode_state
from examples.trained_policy.model import PolicyNet, torch


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "policy_v4_trained.pt"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


def policy_v4_trained(state: dict[str, Any]) -> dict[str, Any]:
    model, metadata = _load_model()
    features = torch.tensor([encode_state(state)], dtype=torch.float32)
    with torch.no_grad():
        logits = model(features)[0]
        probabilities = torch.softmax(logits, dim=0)
        action_index = int(probabilities.argmax().item())
    action = metadata["actions"][action_index]
    return {
        "action": action,
        "debug_info": {
            "policy_type": "trained_pytorch_model",
            "version": metadata["model_version"],
            "model_version": metadata["model_version"],
            "model_path": str(MODEL_PATH),
            "probabilities": {
                action_name: round(float(probabilities[index].item()), 4)
                for index, action_name in enumerate(metadata["actions"])
            },
            "logits": {
                action_name: round(float(logits[index].item()), 4)
                for index, action_name in enumerate(metadata["actions"])
            },
        },
    }


@lru_cache(maxsize=1)
def _load_model() -> tuple[PolicyNet, dict[str, Any]]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Trained policy artifacts are missing. Run `python3 examples/trained_policy/train.py` first."
        )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = PolicyNet(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        output_dim=int(metadata["output_dim"]),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, metadata
