/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics

/-!
# Shape-transfer lemmas for IR evaluation helpers

Each helper function in `NN.IR.Semantics` that operates on `DVal α` has its success determined
by shapes (first projections), not by tensor values (second projections). The lemmas here
make this compositional: if two inputs agree on shape, the helper either succeeds for both
with matching output shapes, or fails for both.

The compositional invariant is: if `f v₁ = ok r₁` and `v₂.1 = v₁.1`, then
`∃ r₂, f v₂ = ok r₂ ∧ r₂.1 = r₁.1`.
-/

@[expose] public section

namespace NN.Proofs.IR.EvalShapeTransfer

open NN.IR
open _root_.Spec
open _root_.Spec.Tensor

variable {α : Type} [Context α] [DecidableEq Shape]

/-! ## Except bind utilities -/

omit [DecidableEq Shape] in
private theorem inner_ok_of_bind_ok {ε β γ : Type}
    {x : Except ε β} {f : β → Except ε γ} {c : γ}
    (h : Bind.bind x f = Except.ok c) : ∃ a, x = Except.ok a := by
  cases x with
  | error e => simp [Bind.bind, Except.bind] at h
  | ok a => exact ⟨a, rfl⟩

omit [DecidableEq Shape] in
private theorem bind_ok_split {ε β γ : Type}
    {x : Except ε β} {f : β → Except ε γ} {c : γ}
    (h : Bind.bind x f = Except.ok c) : ∃ a, x = Except.ok a ∧ f a = Except.ok c := by
  cases x with
  | error e => simp [Bind.bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, by simpa [Bind.bind, Except.bind] using h⟩

/-! ## expectShape -/

theorem expectShape_ok_shape {s : Shape} {v : DVal α} {t : Tensor α s}
    (h : Graph.expectShape (α := α) s v = Except.ok t) : v.1 = s := by
  unfold Graph.expectShape at h
  split at h
  · assumption
  · simp at h

theorem expectShape_ok_of_shape_eq {s : Shape} {v : DVal α}
    (h : v.1 = s) : ∃ t, Graph.expectShape (α := α) s v = Except.ok t := by
  unfold Graph.expectShape
  have hEq : DVal.shape v = s := h
  simp only [hEq, dite_true]
  exact ⟨_, rfl⟩

/-! ## permuteDVal -/

omit [DecidableEq Shape] in
private theorem foldl_swap_shape_eq {v₁ v₂ : PackedTensor α} {swaps : List Nat}
    (hS : v₂.1 = v₁.1) :
    (swaps.foldl (fun acc d => PackedTensor.swapAdjacentAtDepth acc d) v₂).1 =
    (swaps.foldl (fun acc d => PackedTensor.swapAdjacentAtDepth acc d) v₁).1 := by
  induction swaps generalizing v₁ v₂ with
  | nil => simpa using hS
  | cons s swaps ih =>
    simp only [List.foldl_cons]
    apply ih
    simp [PackedTensor.swapAdjacentAtDepth, hS]

omit [DecidableEq Shape] in
theorem permuteDVal_ok_of_shape_eq {v₁ v₂ : DVal α} {perm : List Nat} {r₁ : DVal α}
    (hS : v₂.1 = v₁.1) (h : Graph.permuteDVal v₁ perm = Except.ok r₁) :
    ∃ r₂, Graph.permuteDVal v₂ perm = Except.ok r₂ ∧ r₂.1 = r₁.1 := by
  unfold Graph.permuteDVal at h ⊢
  simp only [DVal.shape] at h ⊢
  split at h
  · simp at h
  · rename_i val₁ hPerm₁
    have hPermEq : Shape.permute? v₂.1 perm = some val₁ := by rw [hS]; exact hPerm₁
    simp only [hPermEq]
    simp only [Bind.bind, Except.bind, Pure.pure, Except.pure] at h ⊢
    split at h
    · simp at h
    · rename_i swaps₁ hSwaps₁
      have hRank : Shape.rank v₂.1 = Shape.rank v₁.1 := by rw [hS]
      rw [hRank, hSwaps₁]
      simp only [Pure.pure, Except.pure] at h ⊢
      cases h
      exact ⟨_, rfl, foldl_swap_shape_eq hS⟩

/-! ## Payload-dependent helpers -/

omit [DecidableEq Shape] in
theorem evalConst_ok_of_compat {p₁ p₂ : Payload α} {id : Nat} {s : Shape}
    {t₁ : Tensor α s}
    (h : Graph.evalConst p₁ id s = Except.ok t₁)
    (hc : ∀ c₁, p₁.const? id = some c₁ → ∃ c₂, p₂.const? id = some c₂ ∧ c₂.n = c₁.n) :
    ∃ t₂, Graph.evalConst (α := α) p₂ id s = Except.ok t₂ := by
  unfold Graph.evalConst at h ⊢
  match hm₁ : p₁.const? id with
  | none => rw [hm₁] at h; simp at h
  | some c₁ =>
    rw [hm₁] at h
    obtain ⟨c₂, hc₂, hcn⟩ := hc c₁ hm₁
    rw [hc₂]
    simp only [Bind.bind, Except.bind] at h ⊢
    split at h
    · rename_i hn₁
      have hn₂ : c₂.n = Shape.size s := by rw [hcn]; exact hn₁
      simp only [hn₂, dite_true]
      exact ⟨_, rfl⟩
    · simp at h

theorem evalLinear_ok_of_compat {p₁ p₂ : Payload α} {id : Nat}
    {x₁ x₂ : DVal α} {outShape : Shape} {r₁ : DVal α}
    (hS : x₂.1 = x₁.1)
    (h : Graph.evalLinear (α := α) p₁ id x₁ outShape = Except.ok r₁)
    (hl : ∀ l₁, p₁.linear? id = some l₁ →
          ∃ l₂, p₂.linear? id = some l₂ ∧ l₂.inDim = l₁.inDim ∧ l₂.outDim = l₁.outDim) :
    ∃ r₂, Graph.evalLinear (α := α) p₂ id x₂ outShape = Except.ok r₂ := by
  unfold Graph.evalLinear at h ⊢
  match hm₁ : p₁.linear? id with
  | none => rw [hm₁] at h; simp at h
  | some l₁ =>
    rw [hm₁] at h
    obtain ⟨l₂, hl₂, hIn, hOut⟩ := hl l₁ hm₁
    rw [hl₂]
    simp only [Bind.bind, Except.bind] at h ⊢
    match hes₁ : Graph.expectShape (α := α) (Shape.dim l₁.inDim Shape.scalar) x₁ with
    | .error _ => rw [hes₁] at h; simp at h
    | .ok t₁ =>
      rw [hes₁] at h
      have hx₁s : x₁.1 = Shape.dim l₁.inDim Shape.scalar := expectShape_ok_shape hes₁
      have hx₂s : x₂.1 = Shape.dim l₂.inDim Shape.scalar := by rw [hS, hx₁s, hIn]
      obtain ⟨t₂, ht₂⟩ := expectShape_ok_of_shape_eq hx₂s
      rw [ht₂]
      simp only [Pure.pure, Except.pure] at h ⊢
      split at h
      · rename_i hOutEq
        have hOutEq₂ : outShape = Shape.dim l₂.outDim Shape.scalar := by rw [hOutEq, hOut]
        simp only [hOutEq₂, dite_true]
        exact ⟨_, rfl⟩
      · simp at h

theorem evalConv2D_ok_of_compat {p₁ p₂ : Payload α} {id : Nat}
    {x₁ x₂ : DVal α} {r₁ : DVal α}
    (hS : x₂.1 = x₁.1)
    (h : Graph.evalConv2D (α := α) p₁ id x₁ = Except.ok r₁)
    (hc : ∀ c₁, p₁.conv2d? id = some c₁ →
          ∃ c₂, p₂.conv2d? id = some c₂ ∧
          c₂.inC = c₁.inC ∧ c₂.outC = c₁.outC ∧
          c₂.kH = c₁.kH ∧ c₂.kW = c₁.kW ∧
          c₂.stride = c₁.stride ∧ c₂.padding = c₁.padding ∧
          c₂.inH = c₁.inH ∧ c₂.inW = c₁.inW) :
    ∃ r₂, Graph.evalConv2D (α := α) p₂ id x₂ = Except.ok r₂ ∧ r₂.1 = r₁.1 := by
  -- Phase 1: extract from h
  unfold Graph.evalConv2D at h
  match hm₁ : p₁.conv2d? id with
  | none => rw [hm₁] at h; simp at h
  | some cfg₁ =>
    rw [hm₁] at h
    obtain ⟨cfg₂, hcfg₂, hIC, hOC, hkH, hkW, hStride, hPad, hIH, hIW⟩ := hc cfg₁ hm₁
    simp only [Bind.bind, Except.bind] at h
    match hinf₁ : OpContracts.inferConv2dCHWOutShape cfg₁.inC cfg₁.outC cfg₁.kH cfg₁.kW
        cfg₁.stride cfg₁.padding (DVal.shape x₁) with
    | .error _ => rw [hinf₁] at h; simp at h
    | .ok infVal =>
      rw [hinf₁] at h
      match hes₁ : Graph.expectShape (α := α)
          (Shape.dim cfg₁.inC (Shape.dim cfg₁.inH (Shape.dim cfg₁.inW Shape.scalar))) x₁ with
      | .error _ => rw [hes₁] at h; simp at h
      | .ok t₁ =>
        rw [hes₁] at h
        simp only [Pure.pure, Except.pure] at h
        -- Phase 2: construct the goal
        unfold Graph.evalConv2D
        simp only [hcfg₂, Bind.bind, Except.bind]
        have hinf₂ : OpContracts.inferConv2dCHWOutShape cfg₂.inC cfg₂.outC cfg₂.kH cfg₂.kW
            cfg₂.stride cfg₂.padding (DVal.shape x₂) = .ok infVal := by
          rw [hIC, hOC, hkH, hkW, hStride, hPad, show DVal.shape x₂ = DVal.shape x₁ from hS]
          exact hinf₁
        simp only [hinf₂]
        have hx₁s := expectShape_ok_shape hes₁
        have hx₂s : x₂.1 = Shape.dim cfg₂.inC (Shape.dim cfg₂.inH (Shape.dim cfg₂.inW Shape.scalar)) := by
          rw [hS, hx₁s, hIC, hIH, hIW]
        obtain ⟨t₂, ht₂⟩ := expectShape_ok_of_shape_eq hx₂s
        simp only [ht₂, Pure.pure, Except.pure, DVal.mk, DVal.shape]
        cases h
        exact ⟨_, rfl, by simp [hOC, hkH, hkW, hStride, hPad, hIH, hIW]⟩

theorem evalBatchNorm_ok_of_compat {p₁ p₂ : Payload α} {id : Nat}
    {x₁ x₂ : DVal α} {r₁ : DVal α}
    (hS : x₂.1 = x₁.1)
    (h : Graph.evalBatchNorm2DNchwEval (α := α) p₁ id x₁ = Except.ok r₁)
    (hb : ∀ b₁, p₁.batchNorm2dNchwEval? id = some b₁ →
          ∃ b₂, p₂.batchNorm2dNchwEval? id = some b₂ ∧ b₂.c = b₁.c) :
    ∃ r₂, Graph.evalBatchNorm2DNchwEval (α := α) p₂ id x₂ = Except.ok r₂ ∧ r₂.1 = r₁.1 := by
  -- Phase 1: extract from h
  unfold Graph.evalBatchNorm2DNchwEval at h
  match hm₁ : p₁.batchNorm2dNchwEval? id with
  | none => rw [hm₁] at h; simp at h
  | some cfg₁ =>
    rw [hm₁] at h
    obtain ⟨cfg₂, hcfg₂, hC⟩ := hb cfg₁ hm₁
    match hshape : DVal.shape x₁ with
    | .dim n (.dim c (.dim hh (.dim w .scalar))) =>
      rw [hshape] at h
      simp only [] at h
      split at h
      · rename_i hc₁
        simp only [Bind.bind, Except.bind] at h
        match hes₁ : Graph.expectShape (α := α)
            (.dim n (.dim cfg₁.c (.dim hh (.dim w .scalar)))) x₁ with
        | .error _ => rw [hes₁] at h; simp at h
        | .ok t₁ =>
          rw [hes₁] at h
          simp only [Pure.pure, Except.pure] at h
          -- Phase 2: construct the goal
          unfold Graph.evalBatchNorm2DNchwEval
          simp only [hcfg₂]
          have hx₂shape : DVal.shape x₂ = .dim n (.dim c (.dim hh (.dim w .scalar))) := by
            rw [show DVal.shape x₂ = DVal.shape x₁ from hS, hshape]
          simp only [hx₂shape]
          have hc₂ : c = cfg₂.c := by rw [hc₁, hC]
          simp only [hc₂, dite_true, Bind.bind, Except.bind]
          have hx₁s := expectShape_ok_shape hes₁
          have hx₂s : x₂.1 = .dim n (.dim cfg₂.c (.dim hh (.dim w .scalar))) :=
            hS.trans (hx₁s.trans (by simp [← hC]))
          obtain ⟨t₂, ht₂⟩ := expectShape_ok_of_shape_eq hx₂s
          simp only [ht₂, Pure.pure, Except.pure, DVal.mk, DVal.shape]
          cases h
          exact ⟨_, rfl, by simp [← hC]⟩
      · simp at h
    | .scalar => rw [hshape] at h; simp at h
    | .dim _ .scalar => rw [hshape] at h; simp at h
    | .dim _ (.dim _ .scalar) => rw [hshape] at h; simp at h
    | .dim _ (.dim _ (.dim _ .scalar)) => rw [hshape] at h; simp at h
    | .dim _ (.dim _ (.dim _ (.dim _ (.dim _ _)))) => rw [hshape] at h; simp at h

/-! ## normalizeNodeOutput -/

theorem normalizeNodeOutput_ok_of_shape_eq {i : Nat} {n : Node} {s : Shape} {t : Tensor α s}
    (h : s = n.outShape) :
    ∃ v, Graph.normalizeNodeOutput i n (DVal.mk s t) = .ok v ∧ v.1 = n.outShape := by
  subst h; exact ⟨_, Graph.normalizeNodeOutput_declared i n t, rfl⟩

/-! ## mseLossDVal -/

theorem mseLossDVal_ok_of_shape_eq {y₁ t₁ y₂ t₂ : DVal α} {r₁ : DVal α} {i : Nat}
    (hY : y₂.1 = y₁.1) (hT : t₂.1 = t₁.1)
    (h : Graph.mseLossDVal (α := α) i y₁ t₁ = Except.ok r₁) :
    ∃ r₂, Graph.mseLossDVal (α := α) i y₂ t₂ = Except.ok r₂ ∧ r₂.1 = r₁.1 := by
  unfold Graph.mseLossDVal at h ⊢
  simp only [DVal.shape] at h ⊢
  split at h
  · rename_i heq₁
    have heq₂ : y₂.1 = t₂.1 := by rw [hY, hT, heq₁]
    simp only [heq₂, dite_true]
    simp only [Pure.pure, Except.pure] at h ⊢
    cases h
    exact ⟨_, rfl, rfl⟩
  · simp at h

/-! ## evalConcat -/

omit [DecidableEq Shape] in
private theorem expectLeadingAxisInput_ok_of_shape_eq
    {v₁ v₂ : DVal α} {i : Nat} {rest : Shape}
    {sig₁ : Σ n, Tensor α (Shape.dim n rest)}
    (hS : DVal.shape v₂ = DVal.shape v₁)
    (h : Graph.expectLeadingAxisInput (α := α) i rest v₁ = Except.ok sig₁) :
    ∃ sig₂, Graph.expectLeadingAxisInput (α := α) i rest v₂ = Except.ok sig₂ ∧
      sig₂.1 = sig₁.1 := by
  unfold Graph.expectLeadingAxisInput at h ⊢
  rcases v₁ with ⟨s₁, t₁⟩
  rcases v₂ with ⟨s₂, t₂⟩
  simp only [DVal.shape] at hS
  cases s₁ with
  | scalar => simp at h
  | dim size rest' =>
    subst hS
    simp only [] at h ⊢
    split at h
    · rename_i hRest
      simp only [hRest, dite_true, Pure.pure, Except.pure] at h ⊢
      exact ⟨_, rfl, by cases h; rfl⟩
    · simp at h

omit [DecidableEq Shape] in
private theorem mapM_expectLeadingAxis_ok_of_shapes_eq
    {l₁ l₂ : List (DVal α)} {i : Nat} {rest : Shape}
    {sigs₁ : List (Σ n, Tensor α (Shape.dim n rest))}
    (hShapes : l₂.map DVal.shape = l₁.map DVal.shape)
    (h : l₁.mapM (Graph.expectLeadingAxisInput (α := α) i rest) = Except.ok sigs₁) :
    ∃ sigs₂, l₂.mapM (Graph.expectLeadingAxisInput (α := α) i rest) = Except.ok sigs₂ ∧
      sigs₂.map Sigma.fst = sigs₁.map Sigma.fst := by
  induction l₁ generalizing l₂ sigs₁ with
  | nil =>
    rcases l₂ with _ | ⟨_, _⟩
    · simp [List.mapM, Pure.pure, Except.pure] at h ⊢
      cases h; exact ⟨[], rfl, rfl⟩
    · simp [List.map] at hShapes
  | cons a₁ rest₁ ih =>
    rcases l₂ with _ | ⟨a₂, rest₂⟩
    · simp [List.map] at hShapes
    · simp only [List.map, List.cons.injEq] at hShapes
      obtain ⟨hHead, hTail⟩ := hShapes
      simp only [List.mapM_cons, Bind.bind, Except.bind] at h ⊢
      obtain ⟨b₁, hb₁, h'⟩ := bind_ok_split h
      obtain ⟨bs₁, hbs₁, hr₁⟩ := bind_ok_split h'
      obtain ⟨b₂, hb₂, hb_fst⟩ := expectLeadingAxisInput_ok_of_shape_eq hHead hb₁
      obtain ⟨bs₂, hbs₂, hbs_fst⟩ := ih hTail hbs₁
      rw [hb₂]
      simp only [Except.bind]
      rw [hbs₂]
      simp only [Pure.pure, Except.pure, List.map]
      cases hr₁
      exact ⟨b₂ :: bs₂, rfl, by simp [hb_fst, hbs_fst]⟩

omit [DecidableEq Shape] in
private theorem foldl_concat_fst_eq {rest : Shape}
    {sigs₁ sigs₂ : List (Σ n, Tensor α (Shape.dim n rest))}
    {s0₁ s0₂ : Σ n, Tensor α (Shape.dim n rest)}
    (hInit : s0₂.1 = s0₁.1)
    (hFsts : sigs₂.map Sigma.fst = sigs₁.map Sigma.fst) :
    (sigs₂.foldl (fun acc nxt => match acc, nxt with
      | ⟨n1, t1⟩, ⟨n2, t2⟩ => ⟨n1 + n2,
          Tensor.concatLeadingAxisSpec (α := α) (n := n1) (m := n2) (s := rest) t1 t2⟩)
      s0₂).1 =
    (sigs₁.foldl (fun acc nxt => match acc, nxt with
      | ⟨n1, t1⟩, ⟨n2, t2⟩ => ⟨n1 + n2,
          Tensor.concatLeadingAxisSpec (α := α) (n := n1) (m := n2) (s := rest) t1 t2⟩)
      s0₁).1 := by
  induction sigs₁ generalizing sigs₂ s0₁ s0₂ with
  | nil =>
    rcases sigs₂ with _ | ⟨_, _⟩
    · simpa using hInit
    · simp [List.map] at hFsts
  | cons x₁ xs₁ ih =>
    rcases sigs₂ with _ | ⟨x₂, xs₂⟩
    · simp [List.map] at hFsts
    · simp only [List.map, List.cons.injEq] at hFsts
      obtain ⟨hx, hxs⟩ := hFsts
      simp only [List.foldl_cons]
      apply ih _ hxs
      rcases s0₁ with ⟨n1₁, t1₁⟩
      rcases s0₂ with ⟨n1₂, t1₂⟩
      rcases x₁ with ⟨n2₁, t2₁⟩
      rcases x₂ with ⟨n2₂, t2₂⟩
      simp only [Sigma.fst] at hInit hx ⊢
      simp only [hInit, hx]

omit [DecidableEq Shape] in
private theorem evalConcatLeadingAxisFold_ok_of_shapes_eq
    {i nOut : Nat} {rest : Shape}
    {parents₁ parents₂ : List (DVal α)} {r₁ : DVal α}
    (hShapes : parents₂.map DVal.shape = parents₁.map DVal.shape)
    (h : Graph.evalConcatLeadingAxisFold (α := α) i nOut rest parents₁ = Except.ok r₁) :
    ∃ r₂, Graph.evalConcatLeadingAxisFold (α := α) i nOut rest parents₂ = Except.ok r₂ ∧
      r₂.1 = r₁.1 := by
  unfold Graph.evalConcatLeadingAxisFold at h ⊢
  simp only [Bind.bind, Except.bind] at h ⊢
  obtain ⟨sigs₁, hsigs₁, h'⟩ := bind_ok_split h
  obtain ⟨sigs₂, hsigs₂, hfsts⟩ :=
    mapM_expectLeadingAxis_ok_of_shapes_eq hShapes hsigs₁
  rw [hsigs₂]
  simp only [Except.bind] at h' ⊢
  rcases sigs₁ with _ | ⟨s0₁, srest₁⟩
  · simp at h'
  · rcases sigs₂ with _ | ⟨s0₂, srest₂⟩
    · simp [List.map] at hfsts
    · simp only [List.map, List.cons.injEq] at hfsts
      obtain ⟨hInit, hTail⟩ := hfsts
      have hFoldFst := foldl_concat_fst_eq (rest := rest) hInit hTail
      simp only [] at h' ⊢
      split at h'
      · rename_i hEq₁
        have hEq₂ := hFoldFst.trans hEq₁
        rw [dif_pos hEq₂] at ⊢
        simp only [Pure.pure, Except.pure] at h' ⊢
        cases h'
        exact ⟨_, rfl, rfl⟩
      · simp at h'

omit [DecidableEq Shape] in
private theorem mapM_permuteDVal_ok_of_shapes_eq
    {l₁ l₂ : List (DVal α)} {perm : List Nat}
    {rs₁ : List (DVal α)}
    (hShapes : l₂.map DVal.shape = l₁.map DVal.shape)
    (h : l₁.mapM (fun pv => Graph.permuteDVal (α := α) pv perm) = Except.ok rs₁) :
    ∃ rs₂, l₂.mapM (fun pv => Graph.permuteDVal (α := α) pv perm) = Except.ok rs₂ ∧
      rs₂.map DVal.shape = rs₁.map DVal.shape := by
  induction l₁ generalizing l₂ rs₁ with
  | nil =>
    rcases l₂ with _ | ⟨_, _⟩
    · simp [List.mapM, Pure.pure, Except.pure] at h ⊢
      cases h; exact ⟨[], rfl, rfl⟩
    · simp [List.map] at hShapes
  | cons a₁ rest₁ ih =>
    rcases l₂ with _ | ⟨a₂, rest₂⟩
    · simp [List.map] at hShapes
    · simp only [List.map, List.cons.injEq] at hShapes
      obtain ⟨hHead, hTail⟩ := hShapes
      simp only [List.mapM_cons, Bind.bind, Except.bind] at h ⊢
      obtain ⟨b₁, hb₁, h'⟩ := bind_ok_split h
      obtain ⟨bs₁, hbs₁, hr₁⟩ := bind_ok_split h'
      obtain ⟨b₂, hb₂, hb_shape⟩ := permuteDVal_ok_of_shape_eq hHead hb₁
      obtain ⟨bs₂, hbs₂, hbs_shape⟩ := ih hTail hbs₁
      rw [hb₂]
      simp only [Except.bind]
      rw [hbs₂]
      simp only [Pure.pure, Except.pure, List.map]
      cases hr₁
      exact ⟨b₂ :: bs₂, rfl, by simp [hb_shape, hbs_shape]⟩

set_option maxHeartbeats 1600000 in
theorem evalConcat_ok_of_shapes_eq {n : Node} {axis : Nat}
    {parents₁ parents₂ : List (DVal α)} {r₁ : DVal α} {i : Nat}
    (hShapes : parents₂.map DVal.shape = parents₁.map DVal.shape)
    (h : Graph.evalConcat (α := α) i n axis parents₁ = Except.ok r₁) :
    ∃ r₂, Graph.evalConcat (α := α) i n axis parents₂ = Except.ok r₂ ∧ r₂.1 = r₁.1 := by
  unfold Graph.evalConcat at h ⊢
  rw [show parents₂.map (fun pv => pv.shape) = parents₁.map (fun pv => pv.shape) from hShapes] at ⊢
  match hInfer : OpContracts.inferConcatOutShape axis
      (List.map (fun (pv : DVal α) => pv.shape) parents₁) with
  | .error msg =>
    revert h; rw [hInfer]; simp [Bind.bind, Except.bind]
  | .ok expected =>
    revert h; rw [hInfer]
    simp only [Bind.bind, Except.bind, Pure.pure, Except.pure]; intro h
    by_cases hCond : (expected != n.outShape) = true
    · rw [if_pos hCond] at h
      simp only [MonadExcept.throw, throwThe, MonadExceptOf.throw] at h; simp at h
    · rw [if_neg hCond] at h ⊢
      by_cases hAxis : axis = 0
      · simp only [hAxis, ite_true] at h ⊢
        cases hOS : n.outShape with
        | scalar =>
          simp only [hOS, MonadExcept.throw, throwThe, MonadExceptOf.throw] at h; simp at h
        | dim nOut rest =>
          simp only [hOS] at h ⊢
          exact evalConcatLeadingAxisFold_ok_of_shapes_eq hShapes h
      · rw [if_neg hAxis] at h ⊢
        match hPerm : OpContracts.permMoveAxisToFront axis n.outShape with
        | .error msg =>
          simp only [hPerm, MonadExcept.throw, throwThe, MonadExceptOf.throw] at h; simp at h
        | .ok permFront =>
          simp only [hPerm] at h ⊢
          match hInv : OpContracts.inversePerm permFront with
          | .error msg =>
            simp only [hInv, MonadExcept.throw, throwThe, MonadExceptOf.throw] at h; simp at h
          | .ok permBack =>
            simp only [hInv] at h ⊢
            match hPS : Spec.Shape.permute? n.outShape permFront with
            | none =>
              simp only [hPS, MonadExcept.throw, throwThe, MonadExceptOf.throw] at h; simp at h
            | some outPermShape =>
              simp only [hPS] at h ⊢
              obtain ⟨pp₁, hpp₁, h'⟩ := bind_ok_split h
              obtain ⟨pp₂, hpp₂, hppS⟩ := mapM_permuteDVal_ok_of_shapes_eq hShapes hpp₁
              rw [hpp₂]; simp only [Except.bind] at h' ⊢
              cases outPermShape with
              | scalar =>
                simp only [MonadExcept.throw, throwThe, MonadExceptOf.throw] at h'; simp at h'
              | dim nOut rest =>
                simp only [Bind.bind, Except.bind] at h' ⊢
                obtain ⟨mid₁, hmid₁, h''⟩ := bind_ok_split h'
                obtain ⟨mid₂, hmid₂, hmidS⟩ :=
                  evalConcatLeadingAxisFold_ok_of_shapes_eq hppS hmid₁
                rw [hmid₂]; simp only [Except.bind] at h'' ⊢
                obtain ⟨o₁, ho₁, h'''⟩ := bind_ok_split h''
                obtain ⟨o₂, ho₂, hoS⟩ := permuteDVal_ok_of_shape_eq hmidS ho₁
                rw [ho₂]; simp only [Except.bind] at h''' ⊢
                obtain ⟨y₁, hy₁, hFin⟩ := bind_ok_split h'''
                have hoS₁ : o₁.1 = n.outShape := expectShape_ok_shape hy₁
                obtain ⟨y₂, hy₂⟩ := expectShape_ok_of_shape_eq
                  (show o₂.1 = n.outShape from hoS.trans hoS₁)
                rw [hy₂]; simp only [Pure.pure, Except.pure] at hFin ⊢
                cases hFin; exact ⟨_, rfl, rfl⟩


end NN.Proofs.IR.EvalShapeTransfer
