from dataclasses import dataclass
from pathlib import Path
from typing import Self

from leantree.core.proof_tree import ProofTree
from leantree.file_span import FileSpan

@dataclass
class StoredError:
    error: str

    def serialize(self) -> dict:
        return {
            "error": self.error
        }

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        return cls(data["error"])

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        return cls(str(e))


@dataclass(eq=False)
class LeanTacticBlock:
    theorem: "LeanTheorem"
    tree: ProofTree | StoredError | None
    span: FileSpan

    # TODO: the decision what to serialize should be in the dataset generator, not here
    def serialize(self) -> dict:
        return {
            "tree": self.tree.serialize(),
            "span": self.span.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict, theorem: "LeanTheorem") -> "LeanTacticBlock":
        return LeanTacticBlock(
            theorem=theorem,
            tree=ProofTree.deserialize(data["tree"]) if data["tree"].get("error") is None else StoredError.deserialize(data["tree"]),
            span=FileSpan.deserialize(data["span"]),
        )

# TODO: start_proof(env) method
@dataclass(eq=False)
class LeanTheorem:
    span: FileSpan
    file: "LeanFile | None"

    # Single theorem can contain multiple `by` clauses.
    by_blocks: list[LeanTacticBlock]

    # Can contain clauses `open` (including `hiding`, `renaming`, etc.), `variable`, `universe`
    context: list[str]
    name: str | None = None

    def serialize(self) -> dict:
        data = {
            "span": self.span.serialize(),
            "by_blocks": [b.serialize() for b in self.by_blocks],
            "context": self.context,
        }
        if self.name is not None:
            data["name"] = self.name
        return data

    @classmethod
    def deserialize(cls, data: dict, file: "LeanFile | None" = None) -> "LeanTheorem":
        by_blocks = []
        thm = LeanTheorem(
            span=FileSpan.deserialize(data["span"]),
            file=file,
            by_blocks=by_blocks,
            context=data["context"],
            name=data.get("name"),
        )
        for block_data in data["by_blocks"]:
            if block_data.get("error") is not None:
                by_blocks.append(StoredError.deserialize(block_data))
            else:
                by_blocks.append(LeanTacticBlock.deserialize(block_data, thm))
        return thm

    def load_source(self) -> str:
        return self.span.read_from_file(self.file.path)

@dataclass(eq=False)
class LeanFile:
    path: Path
    imports: list[str]
    theorems: list[LeanTheorem | StoredError]

    def serialize(self) -> dict:
        return {
            "path": str(self.path),
            "imports": self.imports,
            "theorems": [t.serialize() for t in self.theorems]
        }

    @classmethod
    def deserialize(cls, data: dict) -> "LeanFile":
        theorems = []
        file = LeanFile(
            path=Path(data["path"]),
            imports=data["imports"],
            theorems=theorems,
        )
        for thm_data in data["theorems"]:
            # TODO: error should never be empty when it's non-None
            if thm_data.get("error") is not None:
                theorems.append(StoredError.deserialize(thm_data))
            else:
                theorems.append(LeanTheorem.deserialize(thm_data, file))
        return file
