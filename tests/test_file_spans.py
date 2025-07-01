from leantree.file_span import FilePosition, FileSpan

s1 = """
/-- `E.tupleSucc` maps `![s₀, s₁, ..., sₙ]` to `![s₁, ..., sₙ, ∑ (E.coeffs i) * sᵢ]`,
  where `n := E.order`. -/
def tupleSucc : (Fin E.order → α) →ₗ[α] Fin E.order → α where
  toFun X i := if h : (i : ℕ) + 1 < E.order then X ⟨i + 1, h⟩ else ∑ i, E.coeffs i * X i
  map_add' x y := by
    ext i
    simp only
    split_ifs with h <;> simp [h, mul_add, sum_add_distrib]
  map_smul' x y := by
    ext i
    simp only
    split_ifs with h <;> simp [h, mul_sum]
    exact sum_congr rfl fun x _ ↦ by ac_rfl
""".strip()

s2 = """
protected theorem Commute.mul_geom_sum₂_Ico [Ring α] {x y : α} (h : Commute x y) {m n : ℕ} (hmn : m ≤ n) :
    ((x - y) * ∑ i ∈ Finset.Ico m n, x ^ i * y ^ (n - 1 - i)) = x ^ n - x ^ m * y ^ (n - m) :=
  by
  rw [sum_Ico_eq_sub _ hmn]
  have : ∑ k ∈ range m, x ^ k * y ^ (n - 1 - k) = ∑ k ∈ range m, x ^ k * (y ^ (n - m) * y ^ (m - 1 - k)) :=
    by
    refine sum_congr rfl fun j j_in => ?_
    rw [← pow_add]
    congr
    rw [mem_range] at j_in
    omega
  rw [this]
  simp_rw [pow_mul_comm y (n - m) _]
  simp_rw [← mul_assoc]
  rw [← sum_mul, mul_sub, h.mul_geom_sum₂, ← mul_assoc, h.mul_geom_sum₂, sub_mul, ← pow_add, add_tsub_cancel_of_le hmn,
    sub_sub_sub_cancel_right (x ^ n) (x ^ m * y ^ (n - m)) (y ^ n)]
"""

def test_merge_and_replace():
    by_block = """
        refine sum_congr rfl fun j j_in => ?_
        rw [← pow_add]
        rw [mem_range] at j_in
        omega
    """
    tactics = [line.strip() for line in by_block.strip().splitlines()]
    spans = [
        FileSpan(FilePosition(s2.index(tactic)), FilePosition(s2.index(tactic) + len(tactic)))
        for tactic in tactics
    ]
    print(spans)
    replacement1 = FileSpan.replace_spans(s2, "sorry", spans)
    print(replacement1)
    merged_spans = FileSpan.merge_contiguous_spans(
        spans,
        s2,
        lambda inbetween: len(inbetween.strip()) == 0,
    )
    replacement2 = FileSpan.replace_spans(s2, "sorry", merged_spans)
    print(replacement2)
