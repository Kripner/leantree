import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import requests

from leantree import LeanContext


@dataclass
class MiniF2F:
    val_theorems: list[str]
    test_theorems: list[str]
    global_context: LeanContext


class InterLMMiniF2FAdapter:
    DATA_URL: Final[str] = \
        "https://raw.githubusercontent.com/InternLM/InternLM-Math/refs/heads/main/minif2f/data/minif2f-lean4.7.0.jsonl"
    VALIDATION_LEAN_FILE_URL: Final[str] = \
        "<anonymized>"
    IMPORTS_URL: Final[str] = \
        "<anonymized>"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    @property
    def data_path(self) -> Path:
        return self.cache_dir / "minif2f" / "minif2f-lean4.7.0.jsonl"

    @property
    def validation_lean_file_path(self) -> Path:
        return self.cache_dir / "minif2f" / "Validation.lean"

    @property
    def imports_path(self) -> Path:
        return self.cache_dir / "minif2f" / "Minif2fImport.lean"

    def fetch_minif2f(self) -> MiniF2F:
        print(f"Loading MiniF2F from: {self.cache_dir}")
        self._fetch_if_absent()
        theorems = self._load_theorems()
        return MiniF2F(
            theorems["valid"],
            theorems["test"],
            LeanContext(
                # imports=self._load_imports(),
                imports=["import Mathlib"],
                # open_namespaces=self._load_open_namespaces(),
                open_namespaces=["open BigOperators Real Nat Topology"],
            ),
        )

    def _fetch_if_absent(self):
        if not self.data_path.exists():
            print(f"Fetching MiniF2F theorems from: {self.DATA_URL}")
            response = requests.get(self.DATA_URL)
            response.raise_for_status()
            self.data_path.parent.mkdir(exist_ok=True, parents=True)
            self.data_path.write_text(response.text)

        # if not self.validation_lean_file_path.exists():
        #     print(f"Fetching MiniF2F validationfile from: {self.VALIDATION_LEAN_FILE_URL}")
        #     response = requests.get(self.VALIDATION_LEAN_FILE_URL)
        #     response.raise_for_status()
        #     self.validation_lean_file_path.parent.mkdir(exist_ok=True, parents=True)
        #     self.validation_lean_file_path.write_text(response.text)

        # if not self.imports_path.exists():
        #     print(f"Fetching MiniF2F imports from: {self.IMPORTS_URL}")
        #     response = requests.get(self.IMPORTS_URL)
        #     response.raise_for_status()
        #     self.imports_path.parent.mkdir(exist_ok=True, parents=True)
        #     self.imports_path.write_text(response.text)

    def _load_theorems(self) -> dict[str, list[str]]:
        text = self.data_path.read_text()
        theorems = {"valid": [], "test": []}
        for line in text.splitlines():
            data = json.loads(line)
            statement = data["statement"] + " := by sorry"
            theorems[data["split"]].append(statement)
        return theorems

    def _load_imports(self) -> list[str]:
        text = self.imports_path.read_text()
        imports = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("import "):
                imports.append(line)
        return imports

    def _load_open_namespaces(self) -> list[str]:
        text = self.validation_lean_file_path.read_text()
        namespaces = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("open "):
                namespaces.append(line)
        return namespaces
