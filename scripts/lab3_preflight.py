"""Diagnose Lab 3 setup and run its tests from any working directory."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/test_model_data.py", "tests/test_ner_alignment.py", "tests/test_qa.py",
    "tests/test_lab3_regressions.py", "tests/test_lab3_setup.py",
)
REQUIRED = (*TESTS, "src/bayan/models/training.py", "data/raw/bayan_feedback.csv",
            "data/models/bayan_ner.conll", "data/models/bayan_qa.json",
            "data/eval/qa_smoke_set.json", "requirements.txt", "pyproject.toml")


def missing_files(project):
    return [name for name in REQUIRED if not (project / name).is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    print(f"Python: {sys.version.split()[0]}\nExecutable: {sys.executable}\nProject: {project}", flush=True)
    if sys.version_info[:2] != (3, 12):
        print("SETUP ERROR: select a Colab runtime with Python 3.12, then rerun setup.")
        return 2
    missing = missing_files(project)
    if missing:
        print("CHECKOUT ERROR: missing files:\n" + "\n".join(missing))
        print("Run the notebook's repository setup/update cell first. Do not delete your work.")
        return 2
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    failures = []
    for name in ("torch", "transformers", "accelerate", "sentencepiece", "datasets",
                 "sklearn", "seqeval", "pytest"):
        try:
            module = importlib.import_module(name)
            print(f"OK {name}: {getattr(module, '__version__', 'installed')}")
        except Exception as error:
            failures.append(name)
            print(f"IMPORT ERROR {name}: {type(error).__name__}: {error}")
    if failures:
        print("Run the notebook's dependency installation cell, then rerun this check.")
        return 2
    import torch
    print("CUDA GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable (unit tests can use CPU)")
    if args.require_gpu and not torch.cuda.is_available():
        print("GPU ERROR: select T4 GPU in Colab before training.")
        return 2
    if args.check_only:
        print("LAB3 SETUP READY")
        return 0
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q"], cwd=project,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(result.stdout, flush=True)
    if result.returncode:
        print(f"LAB3 TESTS FAILED (exit {result.returncode}). Copy the complete output above.")
    else:
        print("LAB3 TESTS PASSED")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
