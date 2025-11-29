import traceback
from typing import Self
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, replace

from leantree import utils
from leantree.core.lean_file import LeanFile, LeanTheorem, LeanTacticBlock, StoredError
from leantree.data_extraction.tree_builder import ProofTreeBuilder
from leantree.data_extraction.tree_postprocessor import ProofTreePostprocessor
from leantree.repl_adapter.data import SingletonProofTree
from leantree.repl_adapter.interaction import LeanProcess, LeanInteractionException
from leantree.repl_adapter.data_extraction import LeanFileParser
from leantree.repl_adapter.singleton_trees import SingletonTreeBuilder


class LeanProject:
    def __init__(
            self,
            path: Path | str | None = None,
            *,
            repl_path: Path | str | None = None,
            create: bool = False,
            logger: utils.Logger | None = None,
    ):
        self.path = Path(path) if path else Path("leantree_project")
        if not self.path.exists():
            if create:
                self.create(path, repl_path=repl_path, logger=logger)
            else:
                raise FileNotFoundError(f"Lean project not found: {self.path}")

        self.repl_path = self._get_repl_path(repl_path)
        self.logger = logger

    def lean_toolchain(self) -> str:
        path = self.path / "lean-toolchain"
        if not path.exists():
            raise Exception(f"lean-toolchain file does not exist: {path}")
        return path.read_text().strip()

    def environment(self) -> LeanProcess:
        return LeanProcess(self.repl_path, self.path, self.logger)

    def load_theorem(
            self,
            theorem: str,
            env: LeanProcess,
    ) -> LeanTheorem:
        checkpoint = env.checkpoint()
        loaded_unit = env.send_theorem(theorem)
        env.rollback_to(checkpoint)

        loaded_unit.trees = SingletonTreeBuilder.build_singleton_trees(loaded_unit)
        for tree in loaded_unit.trees:
            ProofTreePostprocessor.transform_proof_tree(tree)
        result = ProofTreeBuilder.run_proof_trees(theorem, loaded_unit, env)
        env.rollback_to(checkpoint)
        return result

    def load_file(
            self,
            path: Path | str,
            use_cache: bool = True,
            store_assertion_errors: bool = True,
    ) -> LeanFile:
        path = Path(path).absolute()
        assert path.is_file()
        assert str(path).endswith(".lean")

        loaded_file = LeanFileParser.load_lean_file(self.repl_path, self.path, path, use_cache)
        for unit in loaded_file.units:
            if len(unit.proof_steps) == 0:
                continue

            try:
                unit.trees = SingletonTreeBuilder.build_singleton_trees(unit)
                for tree in unit.trees:
                    ProofTreePostprocessor.transform_proof_tree(tree)
            except (AssertionError, LeanInteractionException) as e:
                traceback.print_exc()
                if store_assertion_errors:
                    unit.trees = e
                    continue
                else:
                    raise

        file = LeanFile(
            path=Path(loaded_file.path),
            imports=loaded_file.imports,
            theorems=[],
        )
        block_to_tree: dict[LeanTacticBlock, SingletonProofTree] = {}
        for unit in loaded_file.units:
            if unit.trees is None:
                continue
            if isinstance(unit.trees, BaseException):
                file.theorems.append(StoredError.from_exception(unit.trees))
                continue
            by_blocks = []
            theorem = LeanTheorem(
                span=unit.span,
                file=file,
                by_blocks=by_blocks,
                context=unit.global_context,
            )
            file.theorems.append(theorem)
            for singleton_tree in unit.trees:
                by_block = LeanTacticBlock(
                    theorem=theorem,
                    tree=None,
                    span=singleton_tree.span,
                )
                by_blocks.append(by_block)
                block_to_tree[by_block] = singleton_tree

        failed_theorems = []
        with self.environment() as env:
            for theorem, init_proof_states in env.file_proofs(file):
                if isinstance(init_proof_states, BaseException):
                    failed_theorems.append((theorem, init_proof_states))
                    continue
                init_proof_states = list(init_proof_states)
                assert len(theorem.by_blocks) == len(init_proof_states)
                for by_block, init_proof_state in zip(theorem.by_blocks, init_proof_states):
                    singleton_tree = block_to_tree[by_block]
                    try:
                        tree = ProofTreeBuilder.run_proof_tree(singleton_tree, init_proof_state)
                    except (AssertionError, LeanInteractionException) as e:
                        # traceback.print_exc()
                        if store_assertion_errors:
                            by_block.tree = StoredError.from_exception(e)
                            continue
                        else:
                            raise
                    by_block.tree = tree

        for i in range(len(file.theorems)):
            error = [err for t, err in failed_theorems if t == file.theorems[i]]
            if error:
                file.theorems[i] = StoredError.from_exception(error[0])
        return file

    def check_file(self, path: Path):
        # TODO
        pass

    # TODO: user should be able to choose which libraries get included
    @classmethod
    def create(
            cls,
            path: Path | str | None = None,
            lean_version: str | None = "v4.19.0",
            *,
            repl_path: Path | str | None = None,
            logger: utils.Logger | None = None,
            suppress_output: bool = False,
            libraries: "list[LeanLibrary | str] | None" = None,
    ) -> Self:
        if path is None:
            path = "leantree_project"

        path = Path(path)
        # Check that `lake` is in PATH.
        if shutil.which("lake") is None:
            raise RuntimeError(
                "Unable to find 'lake' in PATH. Please install Lean 4 and lake before creating a Lean project. "
                "We recommend using elan. See: "
                "https://docs.lean-lang.org/lean4/doc/setup.html"
            )

        if path.exists():
            if any(path.iterdir()):
                raise FileExistsError(
                    f"Cannot create Lean project: directory exists and is not empty: {path}"
                )
        else:
            path.mkdir(parents=True)

        def run_command(args):
            """Helper to run a command with or without suppressed output."""
            if suppress_output:
                result = subprocess.run(args, cwd=path, text=True, capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Command {args} failed with code {result.returncode}\n"
                        f"stderr:\n{result.stderr}"
                    )
            else:
                result = subprocess.run(args, cwd=path)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Command {args} failed with code {result.returncode}"
                    )

        if lean_version:
            # The lean-toolchain file has to be present before running lake, so that the correct version of lake does
            # the project initialization (the format of lakefile differs across versions).
            toolchain_file = path / "lean-toolchain"
            toolchain_file.write_text(f"leanprover/lean4:{lean_version}\n")

        # Note: We could use `math` instead of `lib` to get mathlib dependency automatically. However, this overwrites
        # lean-toolchain with that of the latest mathlib. So instead, we add mathlib manually.
        run_command(["lake", "init", ".", "lib.toml"])

        libraries = libraries or []
        libraries = [LeanLibraries.from_name(library) if isinstance(library, str) else library for library in libraries]
        require_blocks = []
        for library in libraries:
            if library.name == "mathlib" and library.rev is None and lean_version:
                # Specify Mathlib version - otherwise, the latest version will be used, which may not be compatible with the Lean version.
                library.rev = lean_version

            block = f"""
[[require]]
name = "{library.name}"
scope = "{library.scope}"
git = "{library.git}"
            """.strip()
            if library.rev:
                block += f"\nrev = \"{library.rev}\""
            require_blocks.append(block)

        if require_blocks:
            lakefile = path / "lakefile.toml"
            lakefile_text = lakefile.read_text().strip() + "\n\n" + "\n\n".join(require_blocks) + "\n"
            lakefile.write_text(lakefile_text)

        # Note: We could disable automatic toolchain updates with --keep-toolchain, but that is not available in old
        # versions. For more info, see:
        # https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/#automatic-toolchain-updates
        run_command(["lake", "build"])

        return cls(path, repl_path=repl_path, logger=logger)

    @classmethod
    def _get_repl_path(cls, repl_path: Path | str | None = None) -> Path:
        if repl_path is None:
            repl_path = cls._get_default_repl_exe_path()
        repl_path = Path(repl_path)
        if not repl_path.exists():
            raise Exception(
                f"REPL executable does not exist: {repl_path}.\nPlease run `lake build` in: {cls._get_default_repl_path()}")
        return repl_path

    @classmethod
    def _get_default_repl_path(cls) -> Path:
        return Path(__file__).parent.parent.parent / "lean-repl"

    @classmethod
    def _get_default_repl_exe_path(cls) -> Path:
        return cls._get_default_repl_path() / ".lake/build/bin/repl"


@dataclass
class LeanLibrary:
    name: str
    scope: str
    # Note that the `git` attribute is only necessary for old lake versions.
    git: str
    rev: str | None = None

    def with_(self, **changes):
        return replace(self, **changes)


class LeanLibraries:
    MATHLIB = LeanLibrary(
        name="mathlib",
        scope="leanprover-community",
        git="https://github.com/leanprover-community/mathlib4",
    )

    @classmethod
    def from_name(cls, name: str) -> "LeanLibrary":
        if name == "mathlib":
            return cls.MATHLIB
        else:
            raise ValueError(f"Unknown library: {name}")