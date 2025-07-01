import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Callable

from leantree.core.lean import LeanGoal, LeanHypothesis
from leantree.file_span import FileSpan, FilePosition
from leantree.metavar_graph import MetavarGraph
from leantree.repl_adapter.ast_parser import LeanAST
from leantree import utils


@dataclass
class ReplProofStepInfo:
    tactic_string: str
    goal_before: LeanGoal
    goals_after: list[LeanGoal]
    spawned_goals: list[LeanGoal]
    mctx_before: MetavarGraph | None
    mctx_after: MetavarGraph | None
    tactic_depends_on: list[str]
    span: FileSpan
    ast: LeanAST | None = None

    def all_children(self) -> list[LeanGoal]:
        return self.goals_after + self.spawned_goals

    @classmethod
    def from_repl_data(cls, data: dict, file_line_lengths: list[int]) -> Self:
        return ReplProofStepInfo(
            tactic_string=data["tacticString"],
            goal_before=ReplGoalInfo.goal_from_repl_data(data["goalBefore"]),
            goals_after=[ReplGoalInfo.goal_from_repl_data(g) for g in data["goalsAfter"]],
            spawned_goals=[ReplGoalInfo.goal_from_repl_data(g) for g in data["spawnedGoals"]],
            mctx_before=MetavarGraph.from_dict(data["mctxBefore"]) if data["mctxBefore"] else None,
            mctx_after=MetavarGraph.from_dict(data["mctxAfter"]) if data["mctxAfter"] else None,
            tactic_depends_on=data["tacticDependsOn"],
            span=FilePositionParser.create_file_span(data, file_line_lengths) if data["start"] else None,
            ast=LeanAST.parse_from_string(data["infoTree"]["node"]["stx"]["raw"]) if data["infoTree"] else None,
        )


class ReplGoalInfo:
    @classmethod
    def goal_from_repl_data(cls, goal_info: dict) -> LeanGoal:
        username = goal_info["username"]
        if username == "[anonymous]":
            username = None
        else:
            # For some reason, Lean sometimes automatically adds parts like `._@._hyg.590` or `.Init.Data.Subtype`.
            # We ignore these.
            def tag_is_nonsense(tag: str) -> bool:
                return tag.startswith("_") or tag[0] in string.digits or tag[0] in string.ascii_uppercase

            tag_parts = goal_info["username"].split(".")
            nonsense_indices = [i for i, tag in enumerate(tag_parts) if tag_is_nonsense(tag)]
            first_nonsense = min(nonsense_indices + [len(tag_parts)])
            username = ".".join(tag_parts[:first_nonsense])

        return LeanGoal(
            goal_info["type"],
            [
                # Note: hyp["isProof"] is ignored.
                LeanHypothesis(hyp["type"], hyp["username"], hyp["value"], hyp["id"])
                for hyp in goal_info["hyps"]
            ],
            username,
            goal_info["id"],
        )


class FilePositionParser:
    @classmethod
    def create_file_position(cls, data: dict, file_line_lengths: list[int]) -> FilePosition:
        line, column = data["line"], data["column"]
        # In lean-repl, lines are indexed from 1 and columns are indexed from 0.
        return FilePosition(sum(file_line_lengths[:line - 1]) + column)

    @classmethod
    def create_file_span(cls, data: dict, file_line_lengths: list[int]) -> FileSpan:
        return FileSpan(
            cls.create_file_position(data["start"], file_line_lengths),
            cls.create_file_position(data["finish"], file_line_lengths),
        )


@dataclass
class ReplCompilationUnit:
    proof_steps: list[ReplProofStepInfo]
    pretty_print: str | None
    span: FileSpan
    global_context: list[str] | None
    trees: "list[SingletonProofTree] | None | Exception" = None


@dataclass
class ReplLoadedLeanFile:
    path: Path
    units: list[ReplCompilationUnit]
    imports: list[str]
    file_line_lengths: list[int]


@dataclass
class SingletonProofTreeEdge:
    tactic_string: str
    goal_before: LeanGoal | None
    spawned_goals: "list[SingletonProofTreeNode]"
    goals_after: "list[SingletonProofTreeNode]"

    span: FileSpan | None
    ast: LeanAST | None
    tactic_depends_on: list[str] | None

    @classmethod
    def from_step_info(cls, step: ReplProofStepInfo, all_nodes: "dict[str, SingletonProofTreeNode]") -> Self:
        return SingletonProofTreeEdge(
            step.tactic_string,
            goal_before=step.goal_before,
            spawned_goals=[all_nodes[goal.mvar_id] for goal in step.spawned_goals],
            goals_after=[all_nodes[goal.mvar_id] for goal in step.goals_after],
            span=step.span,
            ast=step.ast,
            tactic_depends_on=step.tactic_depends_on,
        )

    @classmethod
    def create_synthetic(
            cls,
            tactic_string: str,
            goal_before: LeanGoal | None,
            spawned_goals: "list[SingletonProofTreeNode]",
            goals_after: "list[SingletonProofTreeNode]",
    ) -> Self:
        return SingletonProofTreeEdge(tactic_string, goal_before, spawned_goals, goals_after, None, None, None)

    def is_synthetic(self) -> bool:
        return self.span is None

    def all_children(self) -> "list[SingletonProofTreeNode]":
        return self.spawned_goals + self.goals_after


@dataclass
class SingletonProofTreeNode:
    # When transforming the proof tree, some goals are creating whose shape we don't know, so we assign GoalHole. These
    # are then filled in by Lean when we do tree verification.
    goal: LeanGoal | None
    # If `goal` is not None, `id` is equal to `goal.mvar_id`.
    id: str
    parent: Self | None = None
    tactic: SingletonProofTreeEdge | None = None

    @classmethod
    def from_goal(cls, goal: LeanGoal) -> Self:
        return SingletonProofTreeNode(goal, goal.mvar_id)

    @classmethod
    def create_synthetic(cls, parent: Self | None = None, tactic: SingletonProofTreeEdge | None = None) -> Self:
        return SingletonProofTreeNode(
            goal=None,
            id="synthetic_" + "".join(str(random.randint(0, 9)) for _ in range(10)),
            parent=parent,
            tactic=tactic,
        )

    def is_synthetic(self) -> bool:
        return self.goal is None

    def traverse_preorder(self, consumer: Callable[[Self], None]):
        consumer(self)
        if not isinstance(self.tactic, SingletonProofTreeEdge):
            return
        for child in self.tactic.all_children():
            child.traverse_preorder(consumer)

    def traverse(self, node_fn: Callable[[Self], list[Self]]):
        children = node_fn(self)
        if not isinstance(self.tactic, SingletonProofTreeEdge):
            return
        for child in children:
            child.traverse(node_fn)

    def get_subtree_nodes(self) -> list[Self]:
        nodes = []
        self.traverse_preorder(lambda n: nodes.append(n))
        return nodes

    def set_edge(self, edge: SingletonProofTreeEdge):
        assert self.tactic is None
        self.tactic = edge
        for child in edge.all_children():
            child.parent = self

    def is_solved(self) -> bool:
        return self.tactic is not None and all(
            child.is_solved() for child in self.tactic.all_children()
        )


@dataclass
class SingletonProofTree:
    root: SingletonProofTreeNode | None
    span: FileSpan

    def traverse_preorder(self, consumer: Callable[[SingletonProofTreeNode], None]):
        if self.root is None:
            return
        self.root.traverse_preorder(consumer)

    def traverse(self, node_fn: Callable[[SingletonProofTreeNode], list[SingletonProofTreeNode]]):
        if self.root is None:
            return
        self.root.traverse(node_fn)

    def get_nodes(self) -> list[SingletonProofTreeNode]:
        if self.root is None:
            return []
        return self.root.get_subtree_nodes()

    def is_solved(self) -> bool:
        return self.root.is_solved()

    def pretty_print(self) -> str:
        @dataclass
        class IntermediateNode:
            tactic: SingletonProofTreeEdge | None
            children: list[SingletonProofTreeNode]

        def get_children(node: SingletonProofTreeNode | IntermediateNode):
            if isinstance(node, IntermediateNode):
                return node.children
            assert isinstance(node, SingletonProofTreeNode)
            if node.tactic is None:
                return [IntermediateNode(None, [])]
            children = node.tactic.all_children()
            if len(children) == 1:
                return children
            return [IntermediateNode(node.tactic, children)]

        def get_node_label(node: SingletonProofTreeNode | IntermediateNode):
            if isinstance(node, IntermediateNode):
                if node.tactic is None:
                    return "UNEXPANDED"
                return "■" if len(node.children) == 0 else " "
            assert isinstance(node, SingletonProofTreeNode)
            if node.goal is None:
                return f"{node.id}\n<SYNTHETIC>"
            return f"{node.id}\n{node.goal.tag + ": " if node.goal.tag else ""}{node.goal.type}"

        def get_edge_label(node: SingletonProofTreeNode | IntermediateNode):
            if isinstance(node, IntermediateNode):
                if node.tactic is None:
                    return None
                return node.tactic.tactic_string
            assert isinstance(node, SingletonProofTreeNode)
            if node.parent is None:
                return None
            if len(node.parent.tactic.all_children()) > 1:
                return None
            return node.parent.tactic.tactic_string

        return utils.pretty_print_tree(self.root, get_children, get_node_label, get_edge_label)
