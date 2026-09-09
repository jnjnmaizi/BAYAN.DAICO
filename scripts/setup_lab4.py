"""Install only the CAMeL MSA morphology + MLE resources used by Lab 4."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    path = Path(os.environ.setdefault("CAMELTOOLS_DATA", str(root / "artifacts/camel_tools")))
    path.mkdir(parents=True, exist_ok=True)
    print("CAMeL resources:", path, flush=True)
    subprocess.run([sys.executable, "-m", "camel_tools.cli.camel_data", "-i",
                    "disambig-mle-calima-msa-r13"], check=True)
    from bayan.preprocessing.arabic import segment
    print("Segmentation smoke:", segment("وبالرياض"), flush=True)


if __name__ == "__main__":
    main()
