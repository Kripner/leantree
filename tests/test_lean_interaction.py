from pathlib import Path

from leantree.repl_adapter.interaction import LeanProofBranch, LeanServer

from conftest import *
import pytest


def print_state(s: LeanProofBranch):
    print("proof_state:", s._proof_state_id)
    print("all_goals:", s._all_goals)
    print("goals_mask:", s._goals_mask)
    print()


def print_tactic(s: LeanProofBranch, t: str):
    print(f"Applying '{t}' to:")
    print_state(s)
    result = s.apply_tactic(t)
    print("Resulting states:")
    if len(result) == 0:
        print("SOLVED")
    for r in result:
        print_state(r)
    print("-" * 60)
    return result


def apply_tactics(env: LeanServer, thm_with_sorry: str, tactics: list[str]):
    states = list(env.proofs_from_sorries(thm_with_sorry))
    assert len(states) == 1
    s = states[0]
    print_state(s)

    for tactic in tactics:
        next_states = print_tactic(s, tactic)
        if len(next_states) == 0:
            break
        s = next_states[0]


def test_thm_2_le_5(fixture_env):
    thm_2_le_5 = """
    import Mathlib
    open Nat

    example : 2 ≤ 5 := by sorry
    """

    apply_tactics(fixture_env, thm_2_le_5, [
        "apply Nat.le_trans",
        "case m => use 3",
        "decide",
    ])


def test_thm_disj_comm(fixture_env):
    thm_disj_comm = """
    example : ∀ (p q: Prop), p ∨ q → q ∨ p := by sorry
    """

    apply_tactics(fixture_env, thm_disj_comm, [
        "intro p q h",
        "cases h",
        "apply Or.inr",
        "assumption",
    ])


def test_apply_cases_tactic(fixture_env: LeanServer):
    env = fixture_env
    env.send_command("import Mathlib\nopen BigOperators Real Nat Topology Rat")
    branch = env.proof_from_sorry("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
    zero, succ = branch.apply_tactic("cases n")
    print("Factorized proof states after `cases n`:")
    print(zero.state)
    print(succ.state)
    assert not zero.is_solved


def test_load_theorem_from_file(fixture_project: LeanProject, fixture_file_with_theorem: Path):
    file = fixture_project.load_file(fixture_file_with_theorem)
    with fixture_project.environment() as env:
        print("Found theorems", file.theorems)
        print(file.theorems[0].context)
        for thm, branch in env.file_proofs(file):
            if isinstance(branch, Exception):
                print(f"Could not start theorem '{thm}' due to exception: {branch}")


@pytest.mark.asyncio
async def test_apply_cases_tactic_2(fixture_project: LeanProject):
    async with fixture_project.environment() as env:
        await env.send_command_async("import Mathlib\nopen BigOperators Real Nat Topology Rat")
        branch = await env.proof_from_sorry_async("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
        zero, succ = await branch.apply_tactic_async("cases n")
