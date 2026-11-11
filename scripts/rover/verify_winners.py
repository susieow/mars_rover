"""Verify explicitly accepted dataset episodes with the independent verifier."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    verifier = Path(__file__).with_name("verify_dataset.py")
    result = subprocess.run([sys.executable, str(verifier), "--accepted", *sys.argv[1:]])
    raise SystemExit(result.returncode)
