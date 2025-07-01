from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Self

from leantree.utils import ValueOrError


class ProofGoal(ABC):
    pass


class ProofTactic(ABC):
    @abstractmethod
    def __eq__(self, other):
        pass


@dataclass(eq=False, frozen=True)
class ProofState:
    goals: list[ProofGoal]

    def is_solved(self) -> bool:
        return len(self.goals) == 0

    def __str__(self):
        if self.is_solved():
            return "No goals"
        return "\n".join(str(goal) for goal in self.goals)


@dataclass
class ProofStep:
    tactic: ProofTactic
    children: list[ProofState]


class ProofBranch[TGoal: ProofGoal, TTactic: ProofTactic](ABC):
    # Empty result means that the tactic is solving.
    @abstractmethod
    def apply_tactic(self, tactic: TTactic) -> list[Self]:
        pass

    @abstractmethod
    async def apply_tactic_async(self, tactic: TTactic) -> list[Self]:
        pass

    @abstractmethod
    def try_apply_tactic(self, tactic: TTactic) -> ValueOrError[list[Self]]:
        pass

    @abstractmethod
    async def try_apply_tactic_async(self, tactic: TTactic) -> ValueOrError[list[Self]]:
        pass

    @property
    @abstractmethod
    def state(self) -> ProofState:
        pass
