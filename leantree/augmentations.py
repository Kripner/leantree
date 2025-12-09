import random
import traceback
from datasets import load_dataset

from leantree import LeanGoal, ProofTreeNode, ProofTree, LeanProofState, LeanFile, StoredError


class ShuffleGoalsAndHypotheses:
    def __init__(self, seed: int = None):
        self.seed = seed
        self.rng = random.Random(seed)

    def run_on_goal(self, goal: LeanGoal) -> LeanGoal:
        shuffled = list(goal.hypotheses)
        self.rng.shuffle(shuffled)
        return goal.with_(hypotheses=shuffled)

    def run(self, node: ProofTreeNode) -> ProofTreeNode:
        shuffled_goals = [self.run_on_goal(goal) for goal in node.state.goals]
        self.rng.shuffle(shuffled_goals)
        return node.with_(state=LeanProofState(shuffled_goals))


def random_drop_irrelevant_hypotheses(node: ProofTreeNode):
    pass


class RandomAddHypothesis:
    def collect_hypotheses(self, corpus: list[ProofTree]):
        pass

    def run(self, node: ProofTreeNode):
        pass


def random_change_names(node: ProofTreeNode):
    # TODO: variable names, goal tags (if not used in tactic)
    pass


def _main():
    print("Loading dataset...")
    ds = load_dataset("ufal/leantree", split="train", streaming=True)

    shuffler = ShuffleGoalsAndHypotheses(seed=0)

    count = 0
    for sample in ds:
        # if sample.get("path") == "None":
        #     continue

        lean_file = LeanFile.deserialize(sample)

        for theorem in lean_file.theorems:
            if isinstance(theorem, StoredError):
                continue

            for block in theorem.by_blocks:
                if isinstance(block, StoredError) or isinstance(block.tree, StoredError):
                    continue

                tree = block.tree
                if not tree:
                    continue

                nodes = tree.get_nodes()

                for i, node in enumerate(nodes):
                    if i > 2: break  # Limit to first 3 nodes per tree to avoid spam

                    if node.state:
                        print(f"--- Node {node.id} ---")
                        print("BEFORE:")
                        print(str(node.state))

                        new_node = shuffler.run(node)

                        print("\nAFTER:")
                        print(str(new_node.state))
                        print("-" * 40)

                        count += 1

                        if count >= 10:
                            return



if __name__ == "__main__":
    _main()
