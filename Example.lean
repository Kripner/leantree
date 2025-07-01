import Mathlib

open Nat

theorem succ_less_double_succ (n : ℕ) : n > 0 → n < 2 * n := by
  intro h
  cases n with
  | zero => apply h
  | succ n' =>
    apply Nat.le_trans
    case m => exact n' + 2
    · rfl
    · rw [two_mul, add_succ]
      have h1 : 1 ≤ n' + 1
      · apply Nat.succ_le_succ (Nat.zero_le _)
      · exact add_le_add_left h1 (n' + 1)
