import tempfile
from pathlib import Path
import pytest

from leantree import LeanProject
from leantree.utils import Logger, LogLevel

REPL_EXE = Path("../lean-repl/.lake/build/bin/repl")

@pytest.fixture
def fixture_project():
    project_path = Path("../leantree_project")
    if not project_path.exists():
        raise FileNotFoundError(
            f"Project path {project_path} does not exist. Please follow the Development section in README to create it."
        )
    project = LeanProject(project_path, repl_path=REPL_EXE, logger=Logger(LogLevel.DEBUG))
    yield project


@pytest.fixture
def fixture_env(fixture_project: LeanProject):
    with fixture_project.environment() as env:
        yield env


@pytest.fixture
def fixture_file_with_theorem():
    path = Path(tempfile.mktemp(suffix=".lean"))
    with open(path, "w") as f:
        f.write("import Mathlib\n\ntheorem sub_zero' (a b : ℕ) (h : b = 0) : a - b = a := by\nrw [h]\nrw [Nat.sub_zero]")
    yield path
    path.unlink()
