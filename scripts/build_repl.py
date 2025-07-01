""" 
This script is part of pyproject.toml setup.

It builds the Lean REPL.
"""

import subprocess
import sys
from pathlib import Path


def main():
    lean_repl_dir = Path(__file__).parent.parent / "lean-repl"
    try:
        subprocess.run(["lake", "build"], cwd=lean_repl_dir, check=True)
    except subprocess.CalledProcessError as e:
        print("Error building Lean REPL:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
