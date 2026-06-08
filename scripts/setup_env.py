#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_EXTRAS = "dev,gymnasium,mujoco,trained"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a virtual environment and install RoboEval locally.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Install all repo extras: dev, gymnasium, mujoco, trained. Default.",
    )
    group.add_argument(
        "--minimal",
        action="store_true",
        help="Install only the core SDK with no optional simulator/training deps.",
    )
    group.add_argument(
        "--extras",
        help="Install a custom comma-separated extras list, e.g. dev,gymnasium.",
    )
    parser.add_argument(
        "--venv-dir",
        default=os.environ.get("VENV_DIR", ".venv"),
        help="Virtual environment directory. Default: .venv or VENV_DIR.",
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable used to create the virtual environment. Default: current Python or PYTHON_BIN.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = Path(args.venv_dir)
    if not venv_dir.is_absolute():
        venv_dir = repo_root / venv_dir

    extras = _selected_extras(args)
    package_spec = f".[{extras}]" if extras else "."

    print(f"Repository: {repo_root}")
    print(f"Virtual environment: {venv_dir}")

    if not venv_dir.exists():
        print("Creating virtual environment")
        _run([args.python_bin, "-m", "venv", str(venv_dir)], cwd=repo_root)
    else:
        print("Using existing virtual environment")

    venv_python = _venv_python(venv_dir)
    print(f"Python: {_capture([str(venv_python), '--version'], cwd=repo_root)}")

    print("Upgrading pip, setuptools, and wheel")
    _run([
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools<82",
        "wheel",
    ], cwd=repo_root)

    print(f"Installing RoboEval: pip install -e \"{package_spec}\"")
    _run([str(venv_python), "-m", "pip", "install", "-e", package_spec], cwd=repo_root)

    print("Verifying RoboEval import")
    _run([
        str(venv_python),
        "-c",
        "import sys, roboeval; print(f'roboeval loaded from: {roboeval.__file__}'); print(f'python executable: {sys.executable}')",
    ], cwd=repo_root)

    print("\nSetup complete.\n")
    _print_next_steps(venv_dir, repo_root)


def _selected_extras(args: argparse.Namespace) -> str:
    if args.minimal:
        return ""
    if args.extras is not None:
        return args.extras.strip()
    return DEFAULT_EXTRAS


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(command: list[str], cwd: Path) -> None:
    print(f"$ {_format_command(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def _capture(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip() or result.stderr.strip()


def _format_command(command: list[str]) -> str:
    return " ".join(_quote(part) for part in command)


def _quote(value: str) -> str:
    return shlex.quote(value)


def _print_next_steps(venv_dir: Path, repo_root: Path) -> None:
    try:
        display_venv = venv_dir.relative_to(repo_root)
    except ValueError:
        display_venv = venv_dir

    if os.name == "nt":
        print("Activate in PowerShell with:")
        print(f"  .\\{display_venv}\\Scripts\\Activate.ps1")
        print("\nActivate in cmd.exe with:")
        print(f"  {display_venv}\\Scripts\\activate.bat")
    else:
        print("Activate with:")
        print(f"  source {display_venv}/bin/activate")

    print("\nRun tests with:")
    print("  python -m unittest discover -s tests")


if __name__ == "__main__":
    main()
