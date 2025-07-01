import random
from dataclasses import dataclass
from functools import cached_property
from typing import Self, Callable

from leantree import utils
from leantree.core.lean import LeanProofState, LeanTactic
from leantree.file_span import FileSpan


@dataclass(eq=False)
class ProofTreeEdge:
    tactic: LeanTactic
    span: FileSpan | None
    parent: "ProofTreeNode"
    children: "list[ProofTreeNode]"
    tactic_depends_on: list[str] | None = None

    def is_synthetic(self) -> bool:
        return self.span is None

    # TODO: the decision what to serialize should be in the dataset generator, not here
    def serialize(self) -> dict:
        return {
            "tactic_string": self.tactic.tactic,
            "span": self.span.serialize() if self.span is not None else None,
            "parent": self.parent.id,
            "children": [child.id for child in self.children],
            # Enables e.g. data augmentation of removing irrelevant hypotheses.
            "tactic_depends_on": self.tactic_depends_on,
        }

    @classmethod
    def deserialize(cls, data: dict, all_nodes: "dict[str, ProofTreeNode]") -> "ProofTreeEdge":
        return ProofTreeEdge(
            tactic=LeanTactic(data["tactic_string"]),
            span=FileSpan.deserialize(data["span"]) if data["span"] is not None else None,
            parent=all_nodes[data["parent"]],
            children=[all_nodes[node_id] for node_id in data["children"]],
            tactic_depends_on=data["tactic_depends_on"],
        )


@dataclass
class ProofTreeNode:
    # When transforming the proof tree, some goals are created whose shape we don't know, so we assign None. These
    # are then filled in by Lean when we do tree verification.
    id: str
    state: LeanProofState | None
    # TODO: rename to edge
    tactic: ProofTreeEdge | None = None
    parent: Self | None = None

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other) -> bool:
        return isinstance(other, ProofTreeNode) and other.id == self.id

    @classmethod
    def from_state(cls, state: LeanProofState) -> Self:
        return cls(
            id="node_" + "".join(str(random.randint(0, 9)) for _ in range(10)),
            state=state,
        )

    def set_tactic(self, tactic: ProofTreeEdge):
        assert self.tactic is None
        self.tactic = tactic
        for child in tactic.children:
            assert child.parent is None
            child.parent = self

    @cached_property
    def proof_size(self) -> int:
        assert self.tactic is not None
        return 1 + sum(
            child.proof_size for child in self.tactic.children
        )

    @cached_property
    def proof_depth(self) -> int:
        assert self.tactic is not None
        return 1 + max(
            [0] + [child.proof_depth for child in self.tactic.children]
        )

    # TODO
    # def proof_runtime(self):

    def traverse_preorder(self, consumer: Callable[[Self], None]):
        consumer(self)
        if self.tactic is None:
            return
        for child in self.tactic.children:
            child.traverse_preorder(consumer)

    def get_subtree_nodes(self) -> list[Self]:
        nodes = []
        self.traverse_preorder(lambda n: nodes.append(n))
        return nodes

    def is_solved(self) -> bool:
        return self.tactic is not None and all(
            child.is_solved() for child in self.tactic.children
        )

    def serialize(self) -> dict:
        data = {
            "id": self.id,
            "state": self.state.serialize()
        }
        if self.tactic is not None:
            data = {
                **data,
                "tactic": self.tactic.serialize(),
                "proof_size": self.proof_size,
                "proof_depth": self.proof_depth,
            }
        return data

    @classmethod
    def deserialize(
            cls, data: dict, including_edges=False, all_nodes: dict[str, Self] | None = None
    ) -> Self:
        result = ProofTreeNode(
            id=data["id"],
            state= LeanProofState.deserialize(data["state"]),
        )
        if including_edges:
            result.deserialize_edges(data, all_nodes)
        return result

    def deserialize_edges(self, data: dict, all_nodes: dict[str, Self] | None = None):
        assert "tactic" in data
        self.tactic = ProofTreeEdge.deserialize(data["tactic"], all_nodes)
        for child in self.tactic.children:
            child.parent = self


@dataclass(eq=False)
class ProofTree:
    root: ProofTreeNode

    def traverse_preorder(self, consumer: Callable[[ProofTreeNode], None]):
        self.root.traverse_preorder(consumer)

    def is_solved(self) -> bool:
        return self.root.is_solved()

    def serialize(self) -> dict:
        return {
            "nodes": [n.serialize() for n in self.get_nodes()],
            "root_id": self.root.id,
        }

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        all_nodes = {}
        for node_data in data["nodes"]:
            node = ProofTreeNode.deserialize(node_data)
            all_nodes[node.id] = node
        for node_data in data["nodes"]:
            all_nodes[node_data["id"]].deserialize_edges(node_data, all_nodes)
        root = all_nodes[data["root_id"]]
        return ProofTree(root)

    def get_nodes(self) -> list[ProofTreeNode]:
        return self.root.get_subtree_nodes()

    def pretty_print(self) -> str:
        @dataclass
        class IntermediateNode:
            tactic: ProofTreeEdge | None
            children: list[ProofTreeNode]

        def get_children(node: ProofTreeNode | IntermediateNode):
            if isinstance(node, IntermediateNode):
                return node.children
            assert isinstance(node, ProofTreeNode)
            if node.tactic is None:
                return [IntermediateNode(None, [])]
            children = node.tactic.children
            if len(children) == 1:
                return children
            return [IntermediateNode(node.tactic, children)]

        def get_node_label(node: ProofTreeNode | IntermediateNode):
            if isinstance(node, IntermediateNode):
                if node.tactic is None:
                    return "UNEXPANDED"
                return "■" if len(node.children) == 0 else " "
            assert isinstance(node, ProofTreeNode)
            descriptor = node.id + "\n" + "\n".join(
                f"{goal.tag + ": " if goal.tag else ""}{goal.type}" for goal in node.state.goals
            )
            if node.is_solved():
                return f"({node.proof_size}) {descriptor}"
            return descriptor

        def get_edge_label(node: ProofTreeNode | IntermediateNode):
            if isinstance(node, IntermediateNode):
                if node.tactic is None:
                    return None
                return node.tactic.tactic.tactic
            assert isinstance(node, ProofTreeNode)
            if node.parent is None:
                return None
            if len(node.parent.tactic.children) > 1:
                return None
            return node.parent.tactic.tactic.tactic

        return utils.pretty_print_tree(self.root, get_children, get_node_label, get_edge_label)
