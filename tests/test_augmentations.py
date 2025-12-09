
from leantree import LeanGoal, LeanHypothesis, ProofTreeNode, LeanProofState, LeanTactic
from leantree.augmentations import ShuffleGoalsAndHypotheses, RandomRename, _generate_random_name, _replace_name
from leantree.core.proof_tree import ProofTreeEdge
import random

def test_shuffle_hypotheses():
    # Setup
    h1 = LeanHypothesis(type="Nat", user_name="n", value=None)
    h2 = LeanHypothesis(type="n > 0", user_name="h", value=None)
    h3 = LeanHypothesis(type="Prop", user_name="p", value=None)
    goal = LeanGoal(type="True", hypotheses=[h1, h2, h3], tag="case_tag")
    
    # Execute
    # We want to check that it eventually shuffles.
    shuffled_at_least_once = False
    for i in range(100):
        shuffler = ShuffleGoalsAndHypotheses(shuffle_prob=1.0, seed=i)
        new_goal = shuffler.run_on_goal(goal)
        
        # Invariants
        assert len(new_goal.hypotheses) == 3
        assert set(h.user_name for h in new_goal.hypotheses) == {"n", "h", "p"}
        assert new_goal.type == goal.type
        assert new_goal.tag == goal.tag
        
        current_order = [h.user_name for h in new_goal.hypotheses]
        original_order = [h.user_name for h in goal.hypotheses]
        if current_order != original_order:
            shuffled_at_least_once = True
            
    assert shuffled_at_least_once

def test_shuffle_goals():
    # Setup
    g1 = LeanGoal(type="True", hypotheses=[], tag="g1")
    g2 = LeanGoal(type="False", hypotheses=[], tag="g2")
    g3 = LeanGoal(type="1=1", hypotheses=[], tag="g3")
    state = LeanProofState(goals=[g1, g2, g3])
    node = ProofTreeNode(id="1", state=state)
    
    # Execute
    shuffled_at_least_once = False
    for i in range(100):
        shuffler = ShuffleGoalsAndHypotheses(shuffle_prob=1.0, seed=i)
        new_node = shuffler.run(node)
        
        # Invariants
        assert len(new_node.state.goals) == 3
        tags = [g.tag for g in new_node.state.goals]
        assert set(tags) == {"g1", "g2", "g3"}
        
        if tags != ["g1", "g2", "g3"]:
            shuffled_at_least_once = True
            
    assert shuffled_at_least_once

def test_shuffle_no_op():
    g1 = LeanGoal(type="True", hypotheses=[], tag="g1")
    state = LeanProofState(goals=[g1])
    node = ProofTreeNode(id="1", state=state)
    
    # Even with many tries, if prob is 0.0, it should never shuffle (which is trivial here since list size 1)
    # But let's test that it returns the exact same object or structually equal
    shuffler = ShuffleGoalsAndHypotheses(shuffle_prob=0.0, seed=42)
    new_node = shuffler.run(node)
    assert new_node == node

def test_rename_variables_basic():
    # Setup: h1: Nat, h2: h1 > 0 ⊢ h1 = h1
    h1 = LeanHypothesis(type="Nat", user_name="MyVar", value=None)
    h2 = LeanHypothesis(type="MyVar > 0", user_name="h_ineq", value=None)
    goal = LeanGoal(type="MyVar = MyVar", hypotheses=[h1, h2], tag="tag")
    state = LeanProofState(goals=[goal])
    node = ProofTreeNode(id="1", state=state)

    renamed_at_least_once = False
    
    for i in range(100):
        renamer = RandomRename(seed=i)
        new_node = renamer.run(node)
        new_goal = new_node.state.goals[0]
        new_hyps = new_goal.hypotheses

        assert len(new_hyps) == 2
        
        # Find the hypothesis corresponding to h1 (the one with type Nat)
        new_h1 = next(h for h in new_hyps if h.type == "Nat")
        new_h1_name = new_h1.user_name
        
        # Check invariants
        if new_h1_name != "MyVar":
            renamed_at_least_once = True
            
            # Check propagation if renamed
            new_h2 = next(h for h in new_hyps if h != new_h1)
            assert new_h1_name in new_h2.type
            assert "MyVar" not in new_h2.type
            assert new_h1_name in new_goal.type
            assert "MyVar" not in new_goal.type

    assert renamed_at_least_once

def test_rename_variables_with_tactic():
    # Setup: h: a = b ⊢ True. Tactic: "rw [h]"
    h = LeanHypothesis(type="a = b", user_name="my_hypothesis", value=None)
    goal = LeanGoal(type="True", hypotheses=[h], tag="tag")
    state = LeanProofState(goals=[goal])
    
    tactic = LeanTactic("rw [my_hypothesis]")
    child_node = ProofTreeNode(id="2", state=LeanProofState([]))
    
    edge = ProofTreeEdge(tactic=tactic, span=None, parent=None, children=[child_node])
    node = ProofTreeNode(id="1", state=state)
    node.tactic = edge
    edge.parent = node
    
    renamed_at_least_once = False
    
    for i in range(100):
        renamer = RandomRename(seed=i)
        new_node = renamer.run(node)
        
        new_goal = new_node.state.goals[0]
        new_h_name = new_goal.hypotheses[0].user_name
        
        if new_h_name != "my_hypothesis":
            renamed_at_least_once = True
            
            # Check tactic string update
            new_tactic_str = new_node.tactic.tactic.tactic
            assert new_h_name in new_tactic_str
            assert "my_hypothesis" not in new_tactic_str
            
    assert renamed_at_least_once

def test_rename_goals_basic():
    # Setup
    g1 = LeanGoal(type="True", hypotheses=[], tag="case_one")
    state = LeanProofState(goals=[g1])
    node = ProofTreeNode(id="1", state=state)
    
    renamed_at_least_once = False
    dropped_at_least_once = False
    
    # We loop more times because there are 3 outcomes: keep, rename, drop (if not used)
    for i in range(200):
        renamer = RandomRename(seed=i)
        new_node = renamer.run(node)
        new_tag = new_node.state.goals[0].tag
        
        assert new_tag is None or isinstance(new_tag, str)
        
        if new_tag != "case_one":
            if new_tag is None:
                dropped_at_least_once = True
            else:
                renamed_at_least_once = True
                
    assert renamed_at_least_once
    assert dropped_at_least_once

def test_rename_goal_used_in_tactic():
    # Setup: Goal tag used in tactic "case my_tag => ..."
    g1 = LeanGoal(type="True", hypotheses=[], tag="my_tag")
    state = LeanProofState(goals=[g1])
    
    tactic = LeanTactic("case my_tag => exact True")
    child = ProofTreeNode(id="2", state=LeanProofState([]))
    edge = ProofTreeEdge(tactic=tactic, span=None, parent=None, children=[child])
    node = ProofTreeNode(id="1", state=state, tactic=edge)
    edge.parent = node
    
    renamed_at_least_once = False
    
    for i in range(100):
        renamer = RandomRename(seed=i)
        new_node = renamer.run(node)
        
        new_tag = new_node.state.goals[0].tag
        new_tactic_str = new_node.tactic.tactic.tactic
        
        # Invariants: if renamed, must update tactic
        if new_tag and new_tag != "my_tag":
            renamed_at_least_once = True
            assert new_tag in new_tactic_str
            assert "my_tag" not in new_tactic_str
            
    assert renamed_at_least_once

def test_generate_random_name():
    avoid = {"a", "b"}
    name = _generate_random_name(length=3, avoid_names=avoid)
    assert len(name) >= 3 
    assert name not in avoid
    assert isinstance(name, str)

def test_replace_name_boundaries():
    text = "n + n*2 + n_1 + an"
    new_text = _replace_name(text, "n", "x")
    assert new_text == "x + x*2 + n_1 + an"

    text2 = "foo(bar, bar)"
    new_text2 = _replace_name(text2, "bar", "baz")
    assert new_text2 == "foo(baz, baz)"

    text3 = "n' + n"
    new_text3 = _replace_name(text3, "n", "x")
    assert new_text3 == "n' + x"
