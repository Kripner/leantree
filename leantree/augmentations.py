import random
import string
from dataclasses import replace
from datasets import load_dataset

from leantree import LeanGoal, ProofTreeNode, ProofTree, LeanProofState, LeanFile, StoredError, LeanTactic
from leantree.core.lean_file import LeanTheorem


class ShuffleGoalsAndHypotheses:
    def __init__(self, shuffle_prob: float = 0.8, seed: int = None):
        self.shuffle_prob = shuffle_prob
        self.seed = seed
        self.rng = random.Random(seed)

    def run_on_goal(self, goal: LeanGoal) -> LeanGoal:
        shuffled = list(goal.hypotheses)
        self.rng.shuffle(shuffled)
        return goal.with_(hypotheses=shuffled)

    def run(self, node: ProofTreeNode) -> ProofTreeNode:
        if self.rng.random() < self.shuffle_prob:
            shuffled_goals = [self.run_on_goal(goal) for goal in node.state.goals]
            self.rng.shuffle(shuffled_goals)
            return node.with_(state=LeanProofState(shuffled_goals))
        else:
            return node


class RandomRename:
    def __init__(self, seed: int = None):
        self.seed = seed
        self.rng = random.Random(seed)

    def run(self, node: ProofTreeNode) -> ProofTreeNode:
        node = random_rename_variables(node, rng=self.rng)
        node = random_rename_goals(node, rng=self.rng)
        return node


def random_drop_irrelevant_hypotheses(node: ProofTreeNode):
    print("tactic_depends_on: ", node.tactic.tactic_depends_on)
    print("hypotheses: ", [h.mvar_id for g in node.state.goals for h in g.hypotheses])
    print("goals: ", [g.mvar_id for g in node.state.goals])


    return node


class RandomAddHypothesis:
    def collect_hypotheses(self, corpus: list[ProofTree]):
        pass

    def run(self, node: ProofTreeNode):
        pass


SUBSCRIPTS = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]


def _generate_random_name(length: int, avoid_names: set[str], rng=random) -> str:
    chars = string.ascii_letters
    for _ in range(100):
        new_name = "".join(rng.choice(chars) for _ in range(length))
        if rng.random() < 0.2:
            new_name += rng.choice(SUBSCRIPTS)
        if new_name not in avoid_names:
            return new_name
    raise Exception("Infinite loop detected in _generate_random_name.")


def _replace_name(text: str, old_name: str, new_name: str) -> str:
    def is_identifier_like(c):
        return c.isalnum() or c == "_" or c == "'"

    result = []
    i = 0
    n = len(text)
    m = len(old_name)
    
    while i < n:
        if text[i:i+m] == old_name:
            is_start_ok = (i == 0) or not is_identifier_like(text[i-1])
            is_end_ok = (i + m >= n) or not is_identifier_like(text[i+m])
            
            if is_start_ok and is_end_ok:
                result.append(new_name)
                i += m
                continue
        
        result.append(text[i])
        i += 1
        
    return "".join(result)


def _random_rename_variables_in_goal(goal: LeanGoal, rng=random) -> tuple[LeanGoal, dict[str, str]]:
    avoid_names = set(h.user_name for h in goal.hypotheses)
    if goal.tag:
        avoid_names.add(goal.tag)
    
    current_hypotheses = list(goal.hypotheses)
    current_goal_type = goal.type
    replacements = {}
    
    for i in range(len(current_hypotheses)):
        h = current_hypotheses[i]
        old_name = h.user_name
        
        if rng.random() < 0.5:
            new_name = _generate_random_name(len(old_name), avoid_names, rng=rng)
            if "✝" in old_name:
                # ✝ marks that the name is not accessible, which has semantic meaning
                assert len(old_name) >= 2
                new_name = new_name[:-1] + "✝"
            avoid_names.add(new_name)
            replacements[old_name] = new_name
            
            # Update the hypothesis itself
            h = h.with_(user_name=new_name)
            current_hypotheses[i] = h
            
            # Propagate to all types/values (hypotheses and goal)
            for j in range(len(current_hypotheses)):
                target_h = current_hypotheses[j]
                new_type = _replace_name(target_h.type, old_name, new_name)
                new_val = _replace_name(target_h.value, old_name, new_name) if target_h.value else None
                
                if new_type != target_h.type or new_val != target_h.value:
                    current_hypotheses[j] = target_h.with_(type=new_type, value=new_val)
            
            current_goal_type = _replace_name(current_goal_type, old_name, new_name)
            
    return goal.with_(hypotheses=current_hypotheses, type=current_goal_type), replacements

def random_rename_variables(node: ProofTreeNode, rng=random) -> ProofTreeNode:
    new_goals = []
    all_replacements = {}
    
    for g in node.state.goals:
        new_g, replacements = _random_rename_variables_in_goal(g, rng=rng)
        new_goals.append(new_g)
        for k, v in replacements.items():
            if k not in all_replacements:
                all_replacements[k] = v
                
    new_node = node.with_(state=LeanProofState(new_goals))
    
    if node.tactic and node.tactic.tactic:
        tactic_str = node.tactic.tactic.tactic
        for old_name, new_name in all_replacements.items():
            tactic_str = _replace_name(tactic_str, old_name, new_name)
            
        new_lean_tactic = replace(node.tactic.tactic, tactic=tactic_str)
        new_edge = replace(node.tactic, tactic=new_lean_tactic)
        new_node = new_node.with_(tactic=new_edge)
        
    return new_node

def random_rename_goals(node: ProofTreeNode, rng=random) -> ProofTreeNode:
    avoid_names = set()
    for g in node.state.goals:
        if g.tag:
            avoid_names.add(g.tag)
        for h in g.hypotheses:
            avoid_names.add(h.user_name)

    new_goals = []
    replacements = {}
    
    tactic_str = node.tactic.tactic.tactic if node.tactic and node.tactic.tactic else ""

    for g in node.state.goals:
        old_name = g.tag
        length = min(len(old_name), 6) if old_name else rng.randint(1, 5)
        new_name = _generate_random_name(length, avoid_names, rng=rng)
        avoid_names.add(new_name)
        
        updated_goal = g
        renamed = False
        
        if old_name is None:
            if rng.random() < 0.5:
                updated_goal = g.with_(tag=new_name)
        else:
            new_type = _replace_name(g.type, old_name, new_name)
            used_in_type = (new_type != g.type)
            
            used_in_tactic = False
            if tactic_str:
                # Check if replacing would change the tactic string
                if _replace_name(tactic_str, old_name, new_name) != tactic_str:
                    used_in_tactic = True
            
            if used_in_type or used_in_tactic:
                if rng.random() < 0.5:
                    updated_goal = g.with_(tag=new_name, type=new_type)
                    renamed = True
            else:
                rand_val = rng.random()
                if rand_val < 1/3:
                    updated_goal = g.with_(tag=new_name)
                    renamed = True
                elif rand_val < 2/3:
                    pass
                else:
                    updated_goal = g.with_(tag=None)
        
        if renamed:
            replacements[old_name] = new_name
            
        new_goals.append(updated_goal)
        
    new_node = node.with_(state=LeanProofState(new_goals))
    
    if replacements and tactic_str:
        for old_name, new_name in replacements.items():
            tactic_str = _replace_name(tactic_str, old_name, new_name)
            
        new_lean_tactic = replace(node.tactic.tactic, tactic=tactic_str)
        new_edge = replace(node.tactic, tactic=new_lean_tactic)
        new_node = new_node.with_(tactic=new_edge)

    return new_node


# TODO: there should be a column for source (mathlib/DeepSeekProver)
def _main():
    print("Loading dataset...")
    ds = load_dataset("ufal/leantree", split="train", streaming=True)
    
    # get the mathlib samples
    ds = ds.filter(lambda sample: sample.get("path") != "None")

    # ds = ds.shuffle(seed=42, buffer_size=10000)

    # shuffler = ShuffleGoalsAndHypotheses(seed=0)
    renamer = RandomRename(seed=0)

    print("Iterating...")
    count = 0
    for sample in ds:
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
                    if i > 2: break  # Limit to first 2 nodes per tree to avoid spam

                    if node.state:
                        print(f"--- Node {node.id} ---")
                        print("BEFORE:")
                        print(str(node.state))
                        print("->", str(node.tactic.tactic))

                        # new_node = shuffler.run(node)
                        new_node = renamer.run(node)

                        print("\nAFTER:")
                        print(str(new_node.state))
                        print("->", str(new_node.tactic.tactic))
                        print("-" * 40)

                        count += 1
                        if count >= 10:
                            return



if __name__ == "__main__":
    _main()
