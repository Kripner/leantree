import re

import leantree.utils
from leantree.repl_adapter.data import SingletonProofTree, SingletonProofTreeNode, SingletonProofTreeEdge
from leantree.file_span import FileSpan
from leantree.repl_adapter.ast_parser import LeanASTObject, LeanASTArray


class ProofTreePostprocessor:
    @classmethod
    def transform_proof_tree(cls, tree: SingletonProofTree):
        def visitor(node: SingletonProofTreeNode):
            assert node.tactic is not None, "Transforming an unsolved node."
            if node.tactic.is_synthetic():
                return

            # Note: the order is important here, because tacticStrings are being modified.
            cls._replace_nested_tactics_with_sorries(node)
            cls._remove_by_sorry_in_have(node)
            cls._transform_with_cases(node)
            cls._transform_case_tactic(node)
            cls._transform_simp_rw(node)
            cls._transform_rw(node)

            node.tactic.tactic_string = leantree.utils.remove_empty_lines(leantree.utils.remove_comments(
                node.tactic.tactic_string
            ))

        cls._add_missing_assumption_tactics(tree)
        tree.traverse_preorder(visitor)

    @classmethod
    def _add_missing_assumption_tactics(cls, tree: SingletonProofTree):
        if tree.is_solved():
            return

        # e.g. `by` or `suffices` tactics seem to transform the spawned goal to a state where the goal trivially follows
        # from a hypothesis, but then no `assumption` tactic follows. Unfortunately it is not enough to compare the goal
        # type with all hypotheses syntactically - consider "x ≠ 1" and "¬x = 1".
        # This fix seems reckless, but the resulting trees are later verified for correctness.
        def visitor(node: SingletonProofTreeNode):
            if node.tactic is None:
                node.set_edge(SingletonProofTreeEdge.create_synthetic(
                    tactic_string="assumption",
                    goal_before=node.goal,
                    goals_after=[],
                    spawned_goals=[],
                ))

        tree.traverse_preorder(visitor)

    # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/Tactic-Reference/#cases
    @classmethod
    def _transform_with_cases(cls, node: SingletonProofTreeNode):
        tactic_str = node.tactic.tactic_string
        cases_match = re.search(r"^(cases\s+[^\n]+)with\s+", tactic_str)
        induction_match = re.search(r"^(induction\s+[^\n]+)with\s+", tactic_str)
        if cases_match:
            constructors = cls._extract_cases_constructors(node)
            match = cases_match
        elif induction_match:
            constructors = cls._extract_induction_constructors(node)
            match = induction_match
        else:
            return

        assert len(constructors) == len(node.tactic.spawned_goals),\
            "Different number of constructors and spawned goal for cases/induction."
        intermezzo_nodes = []
        for constructor, child in zip(constructors, node.tactic.spawned_goals):
            # We do not want to synthesize the state before the renaming of constructor variables, so we leave that to
            # Lean during tree verification.
            intermezzo_node = SingletonProofTreeNode.create_synthetic(
                parent=node,
            )
            # The `case` tactic handles renaming of inaccessible hypotheses.
            intermezzo_node.set_edge(SingletonProofTreeEdge.create_synthetic(
                tactic_string=f"case {constructor}",
                goal_before=intermezzo_node.goal,
                spawned_goals=[child],
                goals_after=[],
            ))
            intermezzo_nodes.append(intermezzo_node)
        node.tactic.spawned_goals = intermezzo_nodes
        node.tactic.tactic_string = match.group(1)

        # Alternatively, we could use the explicit `rename_i` tactic in each branch to not depend on Mathlib.
        # https://lean-lang.org/doc/reference/latest/Tactic-Proofs/The-Tactic-Language/#rename_i

        # Another idea would be to use the cases' tactic from Mathlib.
        # https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Cases.html#Mathlib.Tactic.cases'
        # However, cases' doesn't work because not all argument names in the constructors of "cases ... with" need to be specified,
        # so the constructor arguments names and the cases' arguments would be misaligned. We could align them by using "_"
        # in `cases'`, but for that we would need to know the number of arguments for each constructor (which is not visible
        # from the AST)

    # Note that `case` tactics are still present in the tree because they handle variable renaming (not just goal selection).
    @classmethod
    def _transform_case_tactic(cls, node: SingletonProofTreeNode):
        # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/The-Tactic-Language/#case
        tactic_str = node.tactic.tactic_string
        pattern = r"case'?[ \t]+([^\n]+?)[ \t]+=>"
        match = re.match(pattern, tactic_str)
        if not match:
            return

        # Note `case'` doesn't force the goal to be solved immediately, but `case` seems to work as well in the REPL.
        new_tactic = f"case {match.group(1)}"
        node.tactic.tactic_string = new_tactic

    # TODO: e.g. `have` doesn't need the `:= by sorry` - without it, it correctly spawns a goal
    # TODO: expand the spans by any whitespaces at the sides
    @classmethod
    def _replace_nested_tactics_with_sorries(cls, node: SingletonProofTreeNode):
        ancestors = [n for n in node.get_subtree_nodes() if n != node]
        # By blocks are present e.g. in
        # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/The-Tactic-Language/#have
        # and replacing the is in accordance with the examples in the official repo. See e.g.:
        # https://github.com/leanprover-community/repl/blob/master/test/name_generator.in
        # By blocks are also in any number of other places, like `exact sum_congr rfl fun x _ ↦ by ac_rfl`.
        sub_spans = []
        for ancestor in ancestors:
            if not ancestor.tactic.is_synthetic() and node.tactic.span.contains(ancestor.tactic.span):
                sub_spans.append(ancestor.tactic.span.relative_to(node.tactic.span.start))
        if sub_spans:
            sub_spans = FileSpan.merge_contiguous_spans(
                sub_spans,
                node.tactic.tactic_string,
                lambda inbetween: len(inbetween.strip()) == 0,
            )
            new_tactic = FileSpan.replace_spans(
                base_string=node.tactic.tactic_string,
                replacement="sorry",
                spans=sub_spans,
            )
            node.tactic.tactic_string = new_tactic

    @classmethod
    def _remove_by_sorry_in_have(cls, node: SingletonProofTreeNode):
        match = re.match(r"(have[ \t]+[^\n]+?)[ \t]+:=[ \t]+by[ \t\n]+sorry", node.tactic.tactic_string)
        if not match:
            return
        if len(node.tactic.spawned_goals) != 1:
            return

        node.tactic.tactic_string = match.group(1)
        node.tactic.goals_after.insert(0, node.tactic.spawned_goals[0])
        node.tactic.spawned_goals = []

    # TODO: deduplicate?
    @classmethod
    def _extract_cases_constructors(cls, node: SingletonProofTreeNode) -> list[str]:
        ast_node = node.tactic.ast.root
        assert (
                isinstance(ast_node, LeanASTObject) and
                len(ast_node.args) == 4 and
                ast_node.args[0].pretty_print() == "cases"
        )
        alts_array = ast_node.args[3]
        assert isinstance(alts_array, LeanASTArray) and len(alts_array.items) == 1
        alts_node = alts_array.items[0]
        assert (
                isinstance(alts_node, LeanASTObject) and
                alts_node.type == "Tactic.inductionAlts" and
                len(alts_node.args) == 3 and
                alts_node.args[0].pretty_print() == "with"
        )
        alts = alts_node.args[2]
        assert isinstance(alts, LeanASTArray)

        constructors = []
        for alt in alts.items:
            assert (
                    isinstance(alt, LeanASTObject) and
                    alt.type == "Tactic.inductionAlt" and
                    len(alt.args) == 3 and
                    alt.args[1].pretty_print() == "=>"
            )
            constructor_tokens = alt.args[0].get_tokens()
            assert constructor_tokens[0] == "|"
            constructor = " ".join(constructor_tokens[1:])
            constructors.append(constructor)
        return constructors

    @classmethod
    def _extract_induction_constructors(cls, node: SingletonProofTreeNode) -> list[str]:
        ast_node = node.tactic.ast.root
        assert (
                isinstance(ast_node, LeanASTObject) and
                len(ast_node.args) == 5 and
                ast_node.args[0].pretty_print() == "induction"
        )
        alts_array = ast_node.args[4]
        assert isinstance(alts_array, LeanASTArray) and len(alts_array.items) == 1
        alts_node = alts_array.items[0]
        assert (
                isinstance(alts_node, LeanASTObject) and
                alts_node.type == "Tactic.inductionAlts" and
                len(alts_node.args) == 3 and
                alts_node.args[0].pretty_print() == "with"
        )
        alts = alts_node.args[2]
        assert isinstance(alts, LeanASTArray)

        constructors = []
        for alt in alts.items:
            assert (
                    isinstance(alt, LeanASTObject) and
                    alt.type == "Tactic.inductionAlt" and
                    len(alt.args) == 3 and
                    alt.args[1].pretty_print() == "=>"
            )
            constructor_tokens = alt.args[0].get_tokens()
            assert constructor_tokens[0] == "|"
            constructor = " ".join(constructor_tokens[1:])
            constructors.append(constructor)
        return constructors

    @classmethod
    def _transform_simp_rw(cls, node: SingletonProofTreeNode):
        match = re.match(r"simp_rw \[([^\n]+)]( at [^\n]+)?", node.tactic.tactic_string)
        if not match:
            return
        assert len(node.tactic.spawned_goals) == 0, "`simp_rw` has spawned goals"

        rules_list = match.group(1)
        at_clause = match.group(2) or ""

        def simp_only(rule: str) -> str:
            return f"simp only [{rule}]{at_clause}"

        rules = [rule.strip() for rule in rules_list.split(",")]
        assert len(rules) > 0, "No rules in a `simp_rw`"
        if len(rules) == 1:
            return

        node.tactic.tactic_string = simp_only(rules[0])
        goals_after = node.tactic.goals_after
        curr_node = node
        for rule in rules[1:]:
            child = SingletonProofTreeNode.create_synthetic(
                parent=curr_node,
            )
            child.set_edge(SingletonProofTreeEdge.create_synthetic(
                tactic_string=simp_only(rule),
                goal_before=child.goal,
                spawned_goals=[],
                goals_after=[],  # Will be filled in.
            ))
            curr_node.tactic.goals_after = [child]
            child.parent = curr_node

            curr_node = child
        curr_node.tactic.goals_after = goals_after
        for g in goals_after:
            g.parent = curr_node

    # @classmethod
    # def _transform_exacts(cls, node: SingletonProofTreeNode):
    #     match = re.match(r"exacts \[([^\n]+)]", node.tactic.tactic_string.strip())
    #     if not match:
    #         return
    #     print(node.goal)
    #     print()
    #     print(f"tactic: {node.tactic.tactic_string}")
    #     print()
    #     for g in node.parent.tactic.spawned_goals:
    #         print(f"parent spawned: {g.goal}")
    #         print()
    #     for g in node.parent.tactic.goals_after:
    #         print(f"parent after: {g.goal}")
    #         print()
    #     print("------")
    #
    #     terms = match.group(1).split(",")
    #     assert len(node.parent.tactic.all_children()) == len(terms),\
    #         "`exacts` has different number of terms then open goals"
    #     term_idx = [
    #         i for i, child in enumerate(node.parent.tactic.all_children())
    #         if child.goal.semantic_equals(node.goal, ignore_metavars=True)
    #     ]
    #     assert len(term_idx) == 1, "Ambiguous or duplicated open goals for `exacts`"
    #
    #     node.tactic.tactic_string = f"exact {terms[term_idx[0]].strip()}"

    @classmethod
    def _transform_rw(cls, node: SingletonProofTreeNode):
        if node.tactic.tactic_string.strip() == "rw [rfl]":
            node.tactic.tactic_string = "rfl"
