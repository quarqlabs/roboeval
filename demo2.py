from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.generic_robots.run_eval import run_all


def main() -> None:
    run_all(output_root="runs/demo", name_prefix="")
    print("\nDemo complete. Open runs/demo/robot_arm/report.md to start.")


if __name__ == "__main__":
    main()
