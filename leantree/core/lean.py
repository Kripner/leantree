import re
from dataclasses import dataclass, replace
from typing import ClassVar, Self

from leantree.core.abstraction import ProofState, ProofTactic, ProofGoal


@dataclass(eq=False, frozen=True)
class LeanHypothesis:
    type: str
    user_name: str
    value: str | None
    mvar_id: str | None = None

    # Matches: "name : type", then optionally ":= value"
    HypothesisLineRegex: ClassVar[re.Pattern] = re.compile(r"^([^:]+):(.+)")

    # TODO: the decision what to serialize should be in the dataset generator, not here
    def serialize(self) -> dict:
        data = {
            "type": self.type,
            "user_name": self.user_name,
        }
        if self.value is not None:
            data["value"] = self.value
        return data

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        return LeanHypothesis(
            type=data["type"],
            user_name=data["user_name"],
            value=data.get("value")
        )

    def __str__(self):
        return f"{self.user_name} : {self.type}" + (f" := {self.value}" if self.value else "")

    def with_(self, **changes) -> Self:
        return replace(self, **changes)

    @classmethod
    def from_string(cls, s: str) -> list[Self]:
        # Example of more complicated hypotheses:
        #
        # hu : u ∈ { carrier := {u | E.IsSolution u}, add_mem' := ⋯, zero_mem' := ⋯ }.carrier
        # ih :
        #   ∀ (x : α),
        #     traverse F ({ head := x, tail := tl } * { head := y, tail := L2 }) =
        #       (fun x1 x2 => x1 * x2) <$> traverse F { head := x, tail := tl } <*> traverse F { head := y, tail := L2 }

        s = re.sub(r"\s+", " ", s.strip())

        match = LeanHypothesis.HypothesisLineRegex.match(s)
        names_str, hyp_type = match.group(1).strip(), match.group(2).strip()
        names = names_str.split()

        value = None
        assign_signs_indices = cls._find_unbracketed_assign_signs(hyp_type)
        assert len(assign_signs_indices) <= 1
        if assign_signs_indices:
            hyp_type, value = (
                hyp_type[:assign_signs_indices[0]].strip(),
                hyp_type[assign_signs_indices[0] + 2:].strip()
            )

        return [LeanHypothesis(hyp_type, name, value) for name in names]

    # TODO: unit test
    @classmethod
    def _find_unbracketed_assign_signs(cls, s: str) -> list[int]:
        brackets = ["()", "[]", "{}", "⦃⦄", "⟨⟩", "⁅⁆"]
        depths = {b: 0 for b in brackets}
        result = []
        for i in range(len(s) - 1):
            for b in brackets:
                if s[i] == b[0]:
                    depths[b] += 1
                elif s[i] == b[1]:
                    assert depths[b] > 0
                    depths[b] -= 1
            if s[i:i + 2] == ":=" and all(d == 0 for d in depths.values()):
                result.append(i)
        return result


@dataclass(eq=False, frozen=True)
class LeanGoal(ProofGoal):
    type: str
    hypotheses: list[LeanHypothesis]
    tag: str | None
    mvar_id: str | None = None

    TagRegex: ClassVar[re.Pattern] = re.compile(r"^case\s+(\S+)")
    TargetSymbol: ClassVar[str] = "⊢"
    MetavarRegex: ClassVar[re.Pattern] = re.compile(r"^\?[A-Za-z0-9._]+$")

    def serialize(self) -> dict:
        data = {
            "type": self.type,
            "hypotheses": [h.serialize() for h in self.hypotheses],
        }
        if self.tag:
            data["tag"] = self.tag
        return data

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        return LeanGoal(
            type=data["type"],
            tag=data.get("tag"),
            hypotheses=[LeanHypothesis.deserialize(h) for h in data["hypotheses"]],
        )

    def __str__(self):
        return (
                (f"case {self.tag}\n" if self.tag else "") +
                "\n".join(h.__str__() for h in self.hypotheses) +
                f"\n{self.TargetSymbol} {self.type}"
        )

    def with_(self, **changes) -> Self:
        return replace(self, **changes)

    # TODO: unit tests
    @classmethod
    def from_string(cls, goal_str: str) -> Self:
        goal_str = goal_str.strip()

        lines = goal_str.splitlines()
        tag = None
        if lines[0].startswith("case ") and ":" not in lines[0]:
            tag_match = LeanGoal.TagRegex.match(lines[0])
            tag = tag_match.group(1).strip()
            goal_str = goal_str[len(tag_match.group(0)):].strip()

        assert goal_str.count(LeanGoal.TargetSymbol) == 1
        hypotheses_str, type_str = goal_str.split(LeanGoal.TargetSymbol)

        curr_hypothesis = ""
        hypotheses = []
        for line in [line for line in hypotheses_str.splitlines() if len(line.strip()) != 0]:
            # We use the fact that subsequent lines of the same hypothesis are indented.
            if curr_hypothesis and line[0] != " ":
                hypotheses.extend(LeanHypothesis.from_string(curr_hypothesis))
                curr_hypothesis = ""
            curr_hypothesis += line
        if curr_hypothesis:
            hypotheses.extend(LeanHypothesis.from_string(curr_hypothesis))

        return cls(type_str.strip(), hypotheses, tag)

    def semantic_equals(self, other: Self, ignore_metavars: bool = False, ignore_tags: bool = False) -> bool:
        def normalize_str(s: str) -> str:
            return re.sub(r"\s+", " ", s)

        def hyp_to_str(h: LeanHypothesis) -> str:
            return f"{h.user_name} : {normalize_str(h.type)} := {h.value}"

        self_hyps = sorted([hyp_to_str(h) for h in self.hypotheses])
        other_hyps = sorted([hyp_to_str(h) for h in other.hypotheses])

        # The problem with direct comparison of tags would be that there is no easy way to rename a goal, so we cannot
        # ensure that the names always match. E.g. when we break down a `cases _ with | A => X | B => Y` into `cases _`,
        # `case A; X`, and `case B; Y`, the goal name in X is `something` in the first case but `something.A` in the
        # second case. Fortunately, it seems that tactics that operate with the goal name are OK when only some suffix
        # of the name parts (separated by dots) is specified, which solves the problem in our case (because we are only
        # removing the names, not changing them).
        self_tags = self.tag.split(".") if self.tag else []
        other_tags = other.tag.split(".") if other.tag else []
        tag_equals = all(
            self_tag_part == other_tag_part
            for self_tag_part, other_tag_part
            in zip(reversed(self_tags), reversed(other_tags))
        )
        if not ignore_metavars:
            return (
                    normalize_str(self.type) == normalize_str(other.type) and
                    self_hyps == other_hyps and
                    (tag_equals or ignore_tags)
            )
        else:
            return (
                    self._equal_up_to_metavar(normalize_str(self.type), normalize_str(other.type)) and
                    all(
                        self._equal_up_to_metavar(self_hyp, other_hyp)
                        for self_hyp, other_hyp in zip(self_hyps, other_hyps)
                    ) and
                    (tag_equals or ignore_tags)
            )

    # TODO!!: this is a much harder problem! The metavar substitutions can contain spaces!
    @classmethod
    def _equal_up_to_metavar(cls, str_a, str_b):
        """
        Compare two strings token-by-token. Each token must either:
          - be identical, or
          - be a 'metavar' in one string (i.e. matches ^\\?[A-Za-z0-9._]+$) which can match any token in the other string.
        """

        tokens_a = str_a.split()
        tokens_b = str_b.split()
        if len(tokens_a) != len(tokens_b):
            return False

        for t_a, t_b in zip(tokens_a, tokens_b):
            if t_a == t_b:
                continue
            if cls.MetavarRegex.match(t_a) or cls.MetavarRegex.match(t_b):
                continue
            return False
        return True


@dataclass(eq=False, frozen=True)
class LeanProofState(ProofState):
    goals: list[LeanGoal]

    def is_solved(self):
        return len(self.goals) == 0

    def semantic_equals(self, other: Self) -> bool:
        return (
                len(self.goals) == len(other.goals) and
                all(g1.semantic_equals(g2) for g1, g2 in zip(self.goals, other.goals))
        )

    def serialize(self) -> dict:
        return {
            "goals": [g.serialize() for g in self.goals]
        }

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        return LeanProofState([LeanGoal.deserialize(g) for g in data["goals"]])

    def __str__(self):
        return "\n\n".join(str(g) for g in self.goals)


@dataclass(frozen=True)
class LeanTactic(ProofTactic):
    tactic: str

    def __str__(self):
        return self.tactic

    def __eq__(self, other):
        return isinstance(other, LeanTactic) and self.tactic == other.tactic


@dataclass(eq=False, frozen=True)
class LeanStep:
    tactic: LeanTactic
    children: list[LeanGoal]


@dataclass(eq=False, frozen=True)
class LeanContext:
    imports: list[str]
    open_namespaces: list[str]
