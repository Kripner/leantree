from leantree.file_span import FileSpan
from leantree.repl_adapter.data import ReplCompilationUnit, SingletonProofTreeNode, SingletonProofTree, \
    SingletonProofTreeEdge, ReplProofStepInfo

# TODOO (!): add proof_runtime

# TODO: for cases/induction, account for something like this:
# example (e p q: Prop) (hep : e = p) (h : p ∨ q) : q ∨ e := by
#   cases h with rw[hep]
#   | inl hp => exact Or.inr hp
#   | inr hq => exact Or.inl hq

# TODO: maybe replace `simp(a) only` with `simp(a)` (see ABEL paper, appendix E Induced distribution shift via tactic post-processing)
# TODO: probably just replace all newlines with spaces in all tacticStrings

# TODO: context should also include private (and protected?) theorems

# TODO: add enum whether the compilation unit is `instance`, `theorem`, `example`, `def`, `lemma` etc. (+ maybe the name)
#  + visibility modifiers (private, protected)  ?
#  + attributes like @[simp]  ?

# TODO: for `theorem`, `example`, etc., compute span of the whole proof (the whole term, not just tactic blocks) - probably from the AST

# TODO: once we solve `calc` (and `conv`), include also proofs in the form `:= calc` (not just `:= by calc`)

# TODO: handle `open Classical in ...`


# TODO (rw will close the goal automatically by rfl if possible):
#  fix: if the last step is rw [rfl] in prettifySteps, remove it (it will be done automatically) - but NOT if it is rewrite [rfl]
# Sending to REPL: '{"cmd": "theorem geom_sum₂_with_one (x : α) (n : ℕ) :\n    ∑ i ∈ range n, x ^ i * 1 ^ (n - 1 - i) = ∑ i ∈ range n, x ^ i :=\n  sum_congr rfl fun i _ => by sorry", "env": 8}'
# Received from REPL: '{"sorries": [{"proofState": 25, "pos": {"line": 3, "column": 30}, "goal": "α : Type u\ninst✝ : Semiring α\nx : α\nn i : ℕ\nx✝ : i ∈ range n\n⊢ x ^ i * 1 ^ (n - 1 - i) = x ^ i", "endPos": {"line": 3, "column": 35}}], "messages": [{"severity": "warning", "pos": {"line": 2, "column": 8}, "endPos": {"line": 2, "column": 20}, "data": "declaration uses 'sorry'"}, {"severity": "warning", "pos": {"line": 1, "column": 8}, "endPos": {"line": 1, "column": 26}, "data": "declaration uses 'sorry'"}], "env": 9}'
# theorem geom_sum₂_with_one (x : α) (n : ℕ) :
#     ∑ i ∈ range n, x ^ i * 1 ^ (n - 1 - i) = ∑ i ∈ range n, x ^ i :=
#   sum_congr rfl fun i _ => by rw [one_pow, mul_one]
#
# rw [one_pow, mul_one]
# Sending to REPL: '{"tactic": "focus rw [one_pow]", "proofState": 25}'
# Received from REPL: '{"proofState": 26, "goals": ["α : Type u\ninst✝ : Semiring α\nx : α\nn i : ℕ\nx✝ : i ∈ range n\n⊢ x ^ i * 1 = x ^ i"]}'
# Sending to REPL: '{"tactic": "focus rw [mul_one]", "proofState": 26}'
# Received from REPL: '{"proofState": 27, "goals": []}'
# VERIFICATION FAILED
# EXPECTED (1):
# case None
# α : Type u
# inst✝ : Semiring α
# x : α
# n : ℕ
# i : ℕ
# x✝ : i ∈ range n
# x ^ i = x ^ i
#
#
#
# ACTUAL (0):
# MESSAGE:
# After tactic: 'rw [mul_one]'

# TODO: `path` of each file has to be relative!

# For `calc`, see Hitchhiker's guide section 4.4 Calculational Proofs


class SingletonTreeBuilder:
    @classmethod
    def build_singleton_trees(cls, unit: ReplCompilationUnit) -> list[SingletonProofTree]:
        """Piece together loaded proof steps based on metavariable IDs."""

        def create_proof_tree(start_idx: int) -> tuple[SingletonProofTree, int]:
            root = SingletonProofTreeNode.from_goal(unit.proof_steps[start_idx].goal_before)
            all_goals: dict[str, SingletonProofTreeNode] = {root.id: root}
            i = start_idx
            while i < len(unit.proof_steps):
                step = unit.proof_steps[i]
                if step.goal_before.mvar_id not in all_goals:
                    # Root's ID always is in `all_goals`, so `i` will get incremented at least once. We assert it to be sure.
                    assert i > start_idx
                    break
                i += 1

                goal_before = all_goals[step.goal_before.mvar_id]
                assert goal_before.tactic is None, (
                    "Reusing closed goal!\n"
                    f"ID: {goal_before.id}\n"
                    f"Already assigned tactic: {goal_before.tactic.tactic_string if goal_before.tactic is not None else "None"}\n"
                    f"New tactic: {step.tactic_string}\n"
                    # f"{root.pretty_print()}"  # TODO
                )

                for goal in step.all_children():
                    if goal.mvar_id not in all_goals:
                        all_goals[goal.mvar_id] = SingletonProofTreeNode.from_goal(goal)
                tactic = SingletonProofTreeEdge.from_step_info(step, all_goals)
                assert not any(child.parent is not None for child in tactic.all_children()), (
                    "Reusing a child!\n"
                    f"ID: {goal_before.id}\n"
                    f"Tactic: {tactic.tactic_string}\n"
                    f"... with children:\n"
                    f"{"\n".join(f"{child.id} ---> {child.parent.id if child.parent else "None"}\n" for child in tactic.all_children())}"
                    # f"\n{root.pretty_print()}"  # TODO
                )
                goal_before.set_edge(tactic)
            tree = SingletonProofTree(
                root,
                span=FileSpan.get_containing_span([
                    n.tactic.span for n in root.get_subtree_nodes()
                    if n.tactic is not None and n.tactic.span is not None
                ])
            )
            return tree, i

        cls._check_if_unit_supported(unit)
        trees = []
        step_idx = 0
        while step_idx < len(unit.proof_steps):
            new_tree, step_idx = create_proof_tree(step_idx)
            trees.append(new_tree)
        return trees

    @classmethod
    def _check_if_unit_supported(cls, unit: ReplCompilationUnit):
        unsupported_tactics = [
            "calc",
            "conv",
        ]
        for tactic in unsupported_tactics:
            if any(edge.tactic_string.strip().startswith(tactic) for edge in unit.proof_steps):
                raise AssertionError(f"`{tactic}` tactic is not yet supported")
