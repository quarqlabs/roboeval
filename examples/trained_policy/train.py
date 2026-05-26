from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.trained_policy.actions import ACTIONS, ACTION_TO_INDEX
from examples.trained_policy.data import clean_rows, generate_synthetic_rows, split_rows, write_rows_csv
from examples.trained_policy.features import FEATURE_NAMES, encode_state
from examples.trained_policy.model import PolicyNet, torch, nn


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"


def main() -> None:
    rows = generate_synthetic_rows(count=1500, seed=21)
    raw_path = DATA_DIR / "synthetic_policy_data.csv"
    clean_path = DATA_DIR / "clean_policy_data.csv"
    rejected_path = DATA_DIR / "rejected_policy_data.csv"
    write_rows_csv(rows, raw_path)

    clean, rejected = clean_rows(rows)
    write_rows_csv(clean, clean_path)
    if rejected:
        write_rows_csv(rejected, rejected_path)

    train_rows, val_rows = split_rows(clean, train_ratio=0.8, seed=99)
    x_train, y_train = _tensorize(train_rows)
    x_val, y_val = _tensorize(val_rows)

    torch.manual_seed(42)
    model = PolicyNet(input_dim=len(FEATURE_NAMES), hidden_dim=32, output_dim=len(ACTIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 180
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    train_accuracy = _accuracy(model, x_train, y_train)
    val_accuracy = _accuracy(model, x_val, y_val)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "policy_v4_trained.pt"
    metadata_path = ARTIFACT_DIR / "metadata.json"
    torch.save(model.state_dict(), model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "policy_v4_trained",
                "model_type": "pytorch_mlp_classifier",
                "input_dim": len(FEATURE_NAMES),
                "hidden_dim": 32,
                "output_dim": len(ACTIONS),
                "actions": ACTIONS,
                "feature_names": FEATURE_NAMES,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "epochs": epochs,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "raw_data_path": str(raw_path.relative_to(REPO_ROOT)),
                "clean_data_path": str(clean_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("trained policy_v4_trained")
    print(f"rows: clean={len(clean)} rejected={len(rejected)}")
    print(f"accuracy: train={train_accuracy:.3f} val={val_accuracy:.3f}")
    print(f"saved weights: {model_path}")
    print(f"saved metadata: {metadata_path}")


def _tensorize(rows):
    features = [encode_state(row) for row in rows]
    labels = [ACTION_TO_INDEX[row["expert_action"]] for row in rows]
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def _accuracy(model: PolicyNet, features, labels) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
        return float((predictions == labels).float().mean().item())


if __name__ == "__main__":
    main()
