#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_env.sh [--all | --minimal | --extras dev,gymnasium,mujoco,trained]

Creates .venv and installs RoboEval in editable mode.

Options:
  --all        Install all repo extras: dev, gymnasium, mujoco, trained. Default.
  --minimal    Install only the core SDK with no optional simulator/training deps.
  --extras X   Install a custom comma-separated extras list.

Environment variables:
  PYTHON_BIN   Python executable to use. Default: python3
  VENV_DIR     Virtual environment directory. Default: .venv
EOF
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
EXTRAS="dev,gymnasium,mujoco,trained"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      EXTRAS="dev,gymnasium,mujoco,trained"
      shift
      ;;
    --minimal)
      EXTRAS=""
      shift
      ;;
    --extras)
      if [[ $# -lt 2 ]]; then
        echo "error: --extras requires a comma-separated value" >&2
        exit 2
      fi
      EXTRAS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} was not found. Set PYTHON_BIN to a valid Python executable." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Using existing virtual environment: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Python: $(python --version)"
echo "Upgrading pip, setuptools, and wheel"
python -m pip install --upgrade pip "setuptools<82" wheel

if [[ -n "${EXTRAS}" ]]; then
  PACKAGE_SPEC=".[${EXTRAS}]"
else
  PACKAGE_SPEC="."
fi

echo "Installing RoboEval: pip install -e \"${PACKAGE_SPEC}\""
python -m pip install -e "${PACKAGE_SPEC}"

echo "Verifying RoboEval import"
python - <<'PY'
import sys
import roboeval

print(f"roboeval loaded from: {roboeval.__file__}")
print(f"python executable: {sys.executable}")
PY

cat <<EOF

Setup complete.

Activate the environment with:
  source ${VENV_DIR}/bin/activate

Run tests with:
  python -m unittest discover -s tests
EOF
