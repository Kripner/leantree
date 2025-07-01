import re
from pathlib import Path

from leantree.core.lean import LeanProofState, LeanTactic
from leantree.core.lean_file import LeanFile, LeanTheorem, LeanTacticBlock
from leantree.core.proof_tree import ProofTree, ProofTreeNode, ProofTreeEdge
from leantree.file_span import FilePosition
from leantree.repl_adapter.data import ReplLoadedLeanFile, SingletonProofTree, SingletonProofTreeNode, \
    ReplCompilationUnit
from leantree.repl_adapter.interaction import LeanProofBranch, LeanServer
from leantree.utils import get_source_with_sorries, replace_with_sorries


class ProofTreeBuilder:
    @classmethod
    def run_proof_trees(cls, theorem_str: str, unit: ReplCompilationUnit, env: LeanServer) -> LeanTheorem:
        block_to_tree: dict[LeanTacticBlock, SingletonProofTree] = {}
        by_blocks = []
        theorem = LeanTheorem(
            span=unit.span,
            file=None,
            by_blocks=by_blocks,
            context=unit.global_context,
        )
        for singleton_tree in unit.trees:
            by_block = LeanTacticBlock(
                theorem=theorem,
                tree=None,
                span=singleton_tree.span,
            )
            by_blocks.append(by_block)
            block_to_tree[by_block] = singleton_tree

        thm_with_sorries = replace_with_sorries(theorem_str, [block.span for block in theorem.by_blocks])
        init_proof_states = list(env.proofs_from_sorries(thm_with_sorries))
        assert len(theorem.by_blocks) == len(init_proof_states)
        for by_block, init_proof_state in zip(theorem.by_blocks, init_proof_states):
            singleton_tree = block_to_tree[by_block]
            tree = cls.run_proof_tree(singleton_tree, init_proof_state)
            by_block.tree = tree
        return theorem

    @classmethod
    def run_file_proof_trees(cls, loaded_file: ReplLoadedLeanFile, env: LeanServer) -> LeanFile:
        theorems = []
        file = LeanFile(
            path=Path(loaded_file.path),
            imports=loaded_file.imports,
            theorems=theorems,
        )
        block_to_tree: dict[LeanTacticBlock, SingletonProofTree] = {}
        for unit in loaded_file.units:
            if not unit.trees:
                continue
            by_blocks = []
            theorem = LeanTheorem(
                span=unit.span,
                file=file,
                by_blocks=by_blocks,
                context=unit.global_context,
            )
            theorems.append(theorem)
            for singleton_tree in unit.trees:
                by_block = LeanTacticBlock(
                    theorem=theorem,
                    tree=None,
                    span=singleton_tree.span,
                )
                by_blocks.append(by_block)
                block_to_tree[by_block] = singleton_tree

        for theorem, init_proof_states in env.file_proofs(file):
            for by_block, init_proof_state in zip(theorem.by_blocks, init_proof_states):
                singleton_tree = block_to_tree[by_block]
                tree = cls.run_proof_tree(singleton_tree, init_proof_state)
                by_block.tree = tree
        return file

    @classmethod
    def run_proof_tree(cls, src_tree: SingletonProofTree, init_proof_branch: LeanProofBranch) -> ProofTree:
        root = ProofTreeNode.from_state(LeanProofState([src_tree.root.goal]))
        to_prove: list[tuple[LeanProofBranch, ProofTreeNode, list[SingletonProofTreeNode]]] = [
            (init_proof_branch, root, [src_tree.root])
        ]

        solved_nodes: set[str] = set()
        while len(to_prove) > 0:
            branch, node, src_nodes = to_prove.pop(0)

            assert len(branch.state.goals) == len(node.state.goals) == len(src_nodes)
            # for g1, g2, g3 in zip(branch.state.goals, node.state.goals,
            #                       [src_node.tactic.goal_before for src_node in src_nodes]):
            #     print(g1)
            #     print(g2)
            #     print(g3)
            #     print()
            for g1, g2, g3 in zip(branch.state.goals, node.state.goals,
                                  [src_node.tactic.goal_before for src_node in src_nodes]):
                # TODO: uncomment!!!!!
                pass
                # assert g1.semantic_equals(g2)
                # # Synthetic nodes in the SingletonTree can have None goal (to be filled in).
                # if g3 is not None:
                #     # The `state_before` of a src_child was captured right before it's tactic evaluation, at which point
                #     # some metavariables might have been assigned which have not yet been assigned now.
                #     assert g2.semantic_equals(g3, ignore_metavars=True)

            expansion_idx = min(
                range(len(src_nodes)),
                key=lambda i: (
                    # We do not know the order of synthetic tactics here. For now, we just evaluate them first.
                    FilePosition.beginning_of_file()
                    if src_nodes[i].tactic.is_synthetic()
                    else src_nodes[i].tactic.span.start
                )
            )
            expansion_node = src_nodes[expansion_idx]
            assert expansion_node.id not in solved_nodes  # This assures termination.
            goal = expansion_node.goal
            assert goal is None or goal.mvar_id is not None
            tactic_info = expansion_node.tactic

            if expansion_idx == 0 or re.match(r"^case'? ", tactic_info.tactic_string):
                # The goal is the main goal or the tactic already selects the goal using `case` / `case'`.
                tactic = tactic_info.tactic_string
            elif (
                    goal is not None and
                    goal.tag is not None and
                    not any(n.goal is None or n.goal.tag == goal.tag for n in src_nodes[:expansion_idx])
            ):
                # The goal has a tag and can be unambiguously selected using it.
                tactic = f"case {goal.tag} => {tactic_info.tactic_string}"
            else:
                # Here we could do rotate_left or pick_goal.
                raise AssertionError("Applying tactic to a non-main unnamed goal is not yet supported.")

            sub_branches = branch.apply_tactic(tactic)

            src_siblings = [src_nodes[i] for i in range(len(src_nodes)) if i != expansion_idx]
            # Note that the order here is important and reflects behavior of the REPL.
            src_all_children = expansion_node.tactic.goals_after + expansion_node.tactic.spawned_goals + src_siblings

            children = []
            for sub_branch in sub_branches:
                child = ProofTreeNode.from_state(sub_branch.state)
                branch_size = len(sub_branch.state.goals)
                assert len(src_all_children) >= branch_size, "Not enough singleton nodes to use in the proof."
                src_children, src_all_children = src_all_children[:branch_size], src_all_children[branch_size:]

                # print("!!", sub_branch._proof_state_id)
                # print("\n".join([str(c.tactic.goal_before) for c in src_children]))
                # print()
                # print("\n".join([str(g) for g in sub_branch.state.goals]))
                # print("-")

                # It is tempting to use `goal_before` directly to match singleton nodes to goals-to-be-solved. However,
                # that would case problems when some goals are duplicated in the singleton tree. Additionally,
                # metavariables would complicate that.
                # TODO: uncomment!!!!!
                # assert all(
                #     (
                #             src_child.tactic.goal_before is None or
                #             src_child.tactic.goal_before.semantic_equals(
                #                 branch_goal, ignore_metavars=True, ignore_tags=True
                #             )
                #     )
                #     for src_child, branch_goal in zip(src_children, sub_branch.state.goals)
                # )

                children.append(child)
                to_prove.append((
                    sub_branch,
                    child,
                    src_children,
                ))
            node.set_tactic(ProofTreeEdge(
                tactic=LeanTactic(tactic),
                span=tactic_info.span,
                parent=node,
                children=children,
                tactic_depends_on=tactic_info.tactic_depends_on,
            ))
            # TODO: look into this error
            # assert len(src_all_children) == 0, \
                # f"Some {len(src_all_children)} singleton nodes were not used in the proof."

            solved_nodes.add(expansion_node.id)
        return ProofTree(root)
