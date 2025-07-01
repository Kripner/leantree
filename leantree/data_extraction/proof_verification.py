from pathlib import Path

from leantree import utils
from leantree.core.lean import LeanProofState
from leantree.core.lean_file import LeanTacticBlock, LeanFile
from leantree.data_extraction.proof_tree import ProofTreeNode, ProofTree
from leantree.environment.interaction import LeanEnvironment, LeanInteractionException, LeanProofBranch


class ProofTreeVerifier:
    def __init__(self, project_path: Path, lean_repl_exe: Path, logger: utils.Logger):
        self.project_path = project_path
        self.lean_repl_exe = lean_repl_exe
        self.logger = logger

    # TODO: this shouldn't modify the LeanFile but rather return a new one!
    def verify_proofs_in_file(self, file: LeanFile, env: LeanEnvironment | None = None) -> list[tuple[LeanTacticBlock, str]]:
        if env is not None:
            return self._verify_proofs_in_file(file, env)
        else:
            with LeanEnvironment(self.lean_repl_exe, self.project_path, self.logger) as env:
                return self._verify_proofs_in_file(file, env)

    def _verify_proofs_in_file(self, file: LeanFile, env: LeanEnvironment) -> list[tuple[LeanTacticBlock, str]]:
        errors: list[tuple[LeanTacticBlock, str]] = []
        for by_block, init_proof_state in env.file_proofs(file):
            debug_infos = [
                f"in file: {file.path}",
                f"in unit:\n'{by_block.theorem.load_source()}'",
                f"in by-block:\n'{by_block.span.read_from_file(file.path)}'"
            ]
            if isinstance(init_proof_state, Exception):
                errors.append((by_block, "\n".join([*debug_infos, str(init_proof_state)])))
                by_block.valid = False
                continue

            try:
                self.verify_proof(by_block.tree, init_proof_state)
                by_block.valid = True
            except AssertionError as e:
                errors.append((by_block, "\n".join([*debug_infos, *(str(arg) for arg in e.args)])))
                by_block.valid = False
            except LeanInteractionException as e:
                errors.append((by_block, "\n".join([*debug_infos, str(e)])))
                by_block.valid = False
        return errors

    def verify_theorem_proof(self, theorem_with_sorry: str, proof: ProofTree, env: LeanEnvironment | None = None):
        if env is not None:
            self._verify_theorem_proof(theorem_with_sorry, proof, env)
        else:
            with LeanEnvironment(self.lean_repl_exe, self.project_path, self.logger) as env:
                self._verify_theorem_proof(theorem_with_sorry, proof, env)

    def _verify_theorem_proof(self, theorem_with_sorry: str, proof: ProofTree, env: LeanEnvironment):
        init_states = list(env.proofs_from_sorries(theorem_with_sorry))
        assert len(init_states) == 1
        init_state = init_states[0]
        self.verify_proof(proof, init_state)

    def verify_proof(self, tree: ProofTree, init_proof_branch: LeanProofBranch):
        def visitor(node: ProofTreeNode):
            assert node.tactic is not None
            assert node in proof_branches

            branch = proof_branches[node]
            assert node.state.semantic_equals(branch.state)

            expected_children = node.tactic.all_children()
            actual_branches = branch.apply_tactic(node.tactic.tactic_string)

            valid = all(
                child.state is None or child.state.semantic_equals(branch.state)
                for child, branch in zip(expected_children, actual_branches)
            )
            if not valid:
                raise AssertionError(self._states_differs_error(
                    expected_children,
                    actual_branches,
                    message=f"After tactic: {node.tactic.tactic_string}"
                ))

            proof_branches.update(
                {child: branch for child, branch in zip(expected_children, actual_branches)}
            )
            for child, branch in zip(expected_children, actual_branches):
                if child.state is None:
                    child.state = branch.state

        if not tree.root.state.semantic_equals(init_proof_branch.state):
            raise AssertionError([tree.root], [init_proof_branch])
        proof_branches = {tree.root: init_proof_branch}
        tree.traverse_preorder(visitor)

    @classmethod
    def _states_differs_error(
            cls, expected_children: list[ProofTreeNode], actual_branches: list[LeanProofBranch], message: str = None,
    ) -> str:
        def get_state_str(g: LeanProofState | None):
            if g is None:
                return ProofTreeNode.StateHoleString
            return g.__repr__()

        def get_debug_string():
            debug_str = "VERIFICATION FAILED\n"
            debug_str += f"EXPECTED ({len(expected_children)}):\n"
            debug_str += "\n".join(f"{get_state_str(c.state)}\n" for c in expected_children)

            debug_str += f"\nACTUAL ({len(actual_branches)}):\n"
            debug_str += "\n".join(f"{get_state_str(b.state)}\n" for b in actual_branches)

            if message:
                debug_str += "\nMESSAGE:\n"
                debug_str += f"{message}\n"

            return debug_str

        return f"REPL produced different children than listed in the tree:\n{get_debug_string()}"
