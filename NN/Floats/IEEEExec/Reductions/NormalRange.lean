/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Reductions
public import NN.Floats.IEEEExec.Semantics.ErrorBounds
import Mathlib.Tactic.Linarith

/-!
# Reductions in the normal range

`RelativeLocalAddBound (fun a b => fp32Round (a + b)) u` does not hold for binary32 with
`u = 2⁻²⁴`: a sum that rounds into the subnormal range has an absolute error the relative
envelope does not cover (`Reductions`), so the enclosures there take it as a hypothesis, and
no statement about all real `a`, `b` can discharge it. This file discharges it where it is
true: at an addition whose exact result is zero or of magnitude at least `minNormal`, the
rounded sum is within `u · (|a| + |b|)` of the exact one (`fp32Round_add_error_of_normal`).
A reduction tree whose intermediate sums are all zero or normal (`NormalRangeSumTree`,
`NormalRangeDot`) therefore satisfies the tree-local bound (`LocalAddBoundOn`), and the
enclosures follow with hypotheses about the tree's values alone: `evalIEEE_enclosure`,
`evalDotIEEE_enclosure`. Those are the forms a user can instantiate, since the hypotheses are
about data.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

open TorchLean.Floats
open ReductionBound

noncomputable section

/-- The unit roundoff of binary32, `2⁻²⁴`. -/
def unitRoundoff32 : ℝ := neuralBpow binaryRadix (-24)

theorem unitRoundoff32_nonneg : 0 ≤ unitRoundoff32 :=
  le_of_lt (neuralBpow.pos binaryRadix (-24))

/-- An exact result is zero or normal: the case in which binary32 rounding has relative error
at most `u`. -/
def ZeroOrNormal (x : ℝ) : Prop := x = 0 ∨ FP32.minNormal ≤ _root_.abs x

/-- **Rounding a zero-or-normal value** has relative error at most `u`. -/
theorem fp32Round_error_of_normal (x : ℝ) (h : ZeroOrNormal x) :
    _root_.abs (fp32Round x - x) ≤ unitRoundoff32 * _root_.abs x := by
  rcases h with heq | hnormal
  · rw [heq, fp32Round_zero]; simp
  · have hmin : 0 < FP32.minNormal := by
      rw [FP32.minNormal_eq_bpow]; exact neuralBpow.pos binaryRadix (-126)
    have hne : x ≠ 0 := by
      intro h0
      rw [h0, abs_zero] at hnormal
      linarith
    have hrel := fp32Round_relative_error_of_normal x hne hnormal
    rwa [ErrorBounds.relativeError, div_le_iff₀ (abs_pos.mpr hne)] at hrel

/-- **Rounding a zero-or-normal sum** is within `u · (|a| + |b|)` of the exact sum. -/
theorem fp32Round_add_error_of_normal (a b : ℝ) (h : ZeroOrNormal (a + b)) :
    _root_.abs (fp32Round (a + b) - (a + b)) ≤
      unitRoundoff32 * (_root_.abs a + _root_.abs b) :=
  (fp32Round_error_of_normal (a + b) h).trans
    (mul_le_mul_of_nonneg_left (abs_add_le a b) unitRoundoff32_nonneg)

/-! ## The executable operations, on the finite branch and in the normal range -/

/-- Addition: within `u · (|x| + |y|)` of the exact sum. -/
theorem toReal_add_error_of_normal (x y : IEEE32Exec) (hfin : isFinite (add x y) = true)
    (h : ZeroOrNormal (toReal x + toReal y)) :
    _root_.abs (toReal (add x y) - (toReal x + toReal y)) ≤
      unitRoundoff32 * (_root_.abs (toReal x) + _root_.abs (toReal y)) := by
  rw [toReal_add_eq_fp32Round_of_isFinite x y hfin]
  exact fp32Round_add_error_of_normal _ _ h

/-- Subtraction: within `u · (|x| + |y|)` of the exact difference. -/
theorem toReal_sub_error_of_normal (x y : IEEE32Exec) (hx : isFinite x = true)
    (hy : isFinite y = true) (hfin : isFinite (sub x y) = true)
    (h : ZeroOrNormal (toReal x - toReal y)) :
    _root_.abs (toReal (sub x y) - (toReal x - toReal y)) ≤
      unitRoundoff32 * (_root_.abs (toReal x) + _root_.abs (toReal y)) := by
  rw [toReal_sub_eq_fp32Round_of_isFinite x y hx hy hfin]
  have := fp32Round_add_error_of_normal (toReal x) (-toReal y) (by rwa [← sub_eq_add_neg])
  rwa [← sub_eq_add_neg, abs_neg] at this

/-- Multiplication: relative error at most `u`. -/
theorem toReal_mul_error_of_normal (x y : IEEE32Exec) (hfin : isFinite (mul x y) = true)
    (h : ZeroOrNormal (toReal x * toReal y)) :
    _root_.abs (toReal (mul x y) - toReal x * toReal y) ≤
      unitRoundoff32 * _root_.abs (toReal x * toReal y) := by
  rw [toReal_mul_eq_fp32Round_of_isFinite x y hfin]
  exact fp32Round_error_of_normal _ h

/-! ## Sums -/

/-- Every intermediate sum of the tree, read on the executable values, is zero or normal. -/
def NormalRangeSumTree : SumTree IEEE32Exec → Prop
  | .leaf _ => True
  | .node a b =>
      NormalRangeSumTree a ∧ NormalRangeSumTree b ∧
      ZeroOrNormal (toReal (evalIEEE a) + toReal (evalIEEE b))

/-- A finite tree with zero-or-normal intermediate sums satisfies the tree-local bound. -/
theorem localAddBoundOn_of_normalRange :
    ∀ t : SumTree IEEE32Exec, FiniteEvalSumTree t → NormalRangeSumTree t →
      LocalAddBoundOn (fun a b => fp32Round (a + b)) toReal unitRoundoff32 t
  | .leaf _, _, _ => trivial
  | .node a b, hfin, hnorm => by
      obtain ⟨hfa, hfb, _⟩ := hfin
      obtain ⟨hna, hnb, hab⟩ := hnorm
      refine ⟨localAddBoundOn_of_normalRange a hfa hna,
        localAddBoundOn_of_normalRange b hfb hnb, ?_⟩
      have ha : evalRealIEEE a = toReal (evalIEEE a) :=
        (toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree a hfa).symm
      have hb : evalRealIEEE b = toReal (evalIEEE b) :=
        (toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree b hfb).symm
      show _root_.abs (fp32Round (evalRealIEEE a + evalRealIEEE b) -
          (evalRealIEEE a + evalRealIEEE b)) ≤
        unitRoundoff32 * (_root_.abs (evalRealIEEE a) + _root_.abs (evalRealIEEE b))
      rw [ha, hb]
      exact fp32Round_add_error_of_normal _ _ hab

/-- **The enclosure of a sum, from hypotheses about the tree**: a finite tree with
zero-or-normal intermediate sums evaluates within `(growth u n − 1) · Σ|leaf|` of the exact
sum of its leaves, `n` the leaf count, `u = 2⁻²⁴`. -/
theorem evalIEEE_enclosure (t : SumTree IEEE32Exec)
    (hfin : FiniteEvalSumTree t) (hnorm : NormalRangeSumTree t) :
    _root_.abs (toReal (evalIEEE t) - exactSumIEEE t) ≤
      (growth unitRoundoff32 t.leafCount - 1) * sumAbsIEEE t := by
  rw [toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree t hfin]
  exact evalRound_enclosure_of_localAddBoundOn _ _ _ unitRoundoff32_nonneg t
    (localAddBoundOn_of_normalRange t hfin hnorm)

/-! ## Sums of products -/

/-- Every intermediate accumulation of the dot tree is zero or normal. -/
def NormalRangeDot : SumTree (IEEE32Exec × IEEE32Exec) → Prop
  | .leaf _ => True
  | .node a b =>
      NormalRangeDot a ∧ NormalRangeDot b ∧
      ZeroOrNormal (toReal (evalDotIEEE a) + toReal (evalDotIEEE b))

theorem localAddBoundOn_of_normalRangeDot :
    ∀ t : SumTree (IEEE32Exec × IEEE32Exec), FiniteEvalDot t → NormalRangeDot t →
      LocalAddBoundOn (fun a b => fp32Round (a + b)) (fun p => toReal (mul p.1 p.2))
        unitRoundoff32 t
  | .leaf _, _, _ => trivial
  | .node a b, hfin, hnorm => by
      obtain ⟨hfa, hfb, _⟩ := hfin
      obtain ⟨hna, hnb, hab⟩ := hnorm
      refine ⟨localAddBoundOn_of_normalRangeDot a hfa hna,
        localAddBoundOn_of_normalRangeDot b hfb hnb, ?_⟩
      have ha : evalRealDotIEEE a = toReal (evalDotIEEE a) :=
        (toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot a hfa).symm
      have hb : evalRealDotIEEE b = toReal (evalDotIEEE b) :=
        (toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot b hfb).symm
      show _root_.abs (fp32Round (evalRealDotIEEE a + evalRealDotIEEE b) -
          (evalRealDotIEEE a + evalRealDotIEEE b)) ≤
        unitRoundoff32 * (_root_.abs (evalRealDotIEEE a) + _root_.abs (evalRealDotIEEE b))
      rw [ha, hb]
      exact fp32Round_add_error_of_normal _ _ hab

/-- Every leaf product of a finite dot tree is finite. -/
theorem isFinite_mul_of_mem_leaves_of_FiniteEvalDot :
    ∀ (t : SumTree (IEEE32Exec × IEEE32Exec)), FiniteEvalDot t →
      ∀ q ∈ t.leaves, isFinite (mul q.1 q.2) = true
  | .leaf (x, y), h, q, hq => by
    simp [SumTree.leaves] at hq; subst hq; exact h
  | .node a b, h, q, hq => by
    simp only [SumTree.leaves, List.mem_append] at hq
    rcases hq with hq | hq
    · exact isFinite_mul_of_mem_leaves_of_FiniteEvalDot a h.1 q hq
    · exact isFinite_mul_of_mem_leaves_of_FiniteEvalDot b h.2.1 q hq

/-- Every leaf of a finite sum tree is finite. -/
theorem isFinite_of_mem_leaves_of_FiniteEvalSumTree :
    ∀ (t : SumTree IEEE32Exec), FiniteEvalSumTree t → ∀ x ∈ t.leaves, isFinite x = true
  | .leaf y, h, x, hx => by
    simp [SumTree.leaves] at hx; subst hx; exact h
  | .node a b, h, x, hx => by
    simp only [SumTree.leaves, List.mem_append] at hx
    rcases hx with hx | hx
    · exact isFinite_of_mem_leaves_of_FiniteEvalSumTree a h.1 x hx
    · exact isFinite_of_mem_leaves_of_FiniteEvalSumTree b h.2.1 x hx

/-- **The enclosure of a sum of products, from hypotheses about the tree**: the accumulation
error relative to the sum of the rounded products, as in `dotTreeResult_enclosure`. -/
theorem evalDotIEEE_enclosure (t : SumTree (IEEE32Exec × IEEE32Exec))
    (hfin : FiniteEvalDot t) (hnorm : NormalRangeDot t) :
    _root_.abs (toReal (evalDotIEEE t) - exactSumDotIEEE t) ≤
      (growth unitRoundoff32 t.leafCount - 1) * sumAbsDotIEEE t := by
  rw [toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot t hfin]
  exact evalRound_enclosure_of_localAddBoundOn _ _ _ unitRoundoff32_nonneg t
    (localAddBoundOn_of_normalRangeDot t hfin hnorm)

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
