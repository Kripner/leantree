from dataclasses import dataclass
from typing import Self

from leantree.core.lean import LeanGoal


@dataclass
class MetavarInfo:
    userName: str
    type: str
    mvarsInType: list[str]
    id: str


class MetavarGraph:
    def __init__(self, mvars: dict[str, MetavarInfo]):
        self.mvars = mvars

    def goals_connected(self, goal_a: LeanGoal, goal_b: LeanGoal) -> bool:
        """Reflexive, symmetric, not transitive."""

        def _get_metavars(goal: LeanGoal) -> list[str]:
            # Note that hypotheses can be shared between goals.
            return [
                goal.mvar_id,
                *self.mvars[goal.mvar_id].mvarsInType,
                *[
                    hyp_mvar
                    for h in goal.hypotheses if h.mvar_id in self.mvars
                    for hyp_mvar in self.mvars[h.mvar_id].mvarsInType
                ],
            ]

        return len(set(_get_metavars(goal_a)).intersection(set(_get_metavars(goal_b)))) != 0

    @classmethod
    def from_list(cls, mvars: list[MetavarInfo]) -> Self:
        return cls({mvar.id: mvar for mvar in mvars})

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls.from_list([
            MetavarInfo(mvar["userName"], mvar["type"], mvar["mvarsInType"], mvar["mvarId"])
            for mvar in data["decls"]
        ])

    def partition_independent_goals(self, goals: list[LeanGoal]) -> list[list[LeanGoal]]:
        """
        Split the goals into independent branches. Goals sharing a metavariable (in type or in a hypothesis) cannot be
        separated.
        """
        if len(goals) == 0:
            return []
        if len(goals) == 1:
            return [goals]
        branch_tags = [object() for _ in range(len(goals))]
        for i in range(len(goals)):
            for j in range(i + 1, len(goals)):
                if branch_tags[i] != branch_tags[j] and self.goals_connected(goals[i], goals[j]):
                    # Merge the two branches (replace branch_tags[j] with branch_tags[i] everywhere).
                    branch_tags = [
                        branch_tags[i] if tag == branch_tags[j] else tag
                        for tag in branch_tags
                    ]
        result = []
        for tag in list(dict.fromkeys(branch_tags)):  # dict keeps order
            result.append([goals[i] for i in range(len(goals)) if branch_tags[i] == tag])
        return result
