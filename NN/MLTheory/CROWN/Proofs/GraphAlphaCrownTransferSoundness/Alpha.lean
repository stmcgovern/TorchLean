/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common

/-!
# α-CROWN Graph Transfer Soundness

Pointwise soundness theorem for the graph-dialect `alphaCrownStepNode?` transfer rule.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open scoped BigOperators
open Proofs.TensorAlgebra

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness

/-! ## Main transfer theorem -/

/--
Pointwise soundness of the graph-dialect α-CROWN transfer rule.

Fix a graph `g`, parameters `ps`, an input point `x`, and a locally-consistent value semantics
array `vals` (i.e. `vals[id]` agrees with evaluating node `id` from its parents’ values).

Assume:

- the designated input node in `inputs` matches `x` (`InputsMatch`),
- the IBP boxes `ibp` enclose the semantic values in `vals` (`IBPEnclosesVals`), and
- the α parameters are well-formed (`AlphaOK`).

Then the concrete step function `alphaCrownStepNode?` satisfies the abstract
`CrownTransferSound` requirement: whenever every parent `p` is enclosed by its certificate entry,
the current node `id` is enclosed by the step-produced certificate entry as well.

This is the key lemma that lets `alphaCrownStepNode?` plug into the generic end-to-end checker
theorem in `NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness`.
-/
theorem alphaCrown_transfer_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatVec ℝ)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ (.dim ctx.inputDim .scalar))
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha)) :
    CrownTransferSound
      (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
      (ctx := ctx) (x := x)
      (step := stepAlpha g ps ibp alpha ctx) (cert := cert) := by
  classical
  intro id hid hparents
  cases hs : stepAlpha g ps ibp alpha ctx cert id <;> cases hv : vals[id]!
  all_goals simp
  case some.some b v =>
    -- Semantic evaluation at this node.
    have hEvalEq : vals[id]! = evalNode? g.nodes ps inputs vals id := hsem.2 id hid
    have hEvalSome : evalNode? g.nodes ps inputs vals id = some v := by
      have : some v = evalNode? g.nodes ps inputs vals id := by simpa [hv] using hEvalEq
      simpa using this.symm

    -- Helper: get parent enclosure when both cert/val are present.
    have parentEnc :
        ∀ p : Nat, p ∈ (g.nodes[id]!).parents →
          ∀ (bp : FlatAffineBounds ℝ) (vp : Val),
            cert[p]! = some bp → vals[p]! = some vp → EnclosesAtInput (α := ℝ) ctx x bp vp := by
      intro p hp bp vp hbp hvp
      have h := hparents p hp
      simpa [hbp, hvp] using h

    -- Split by node kind, mirroring `alphaCrownStepNode?`.
    refine (match hk : (g.nodes[id]!).kind with
    | .input => by
        -- Step success forces `id = ctx.inputId` and `b = bounds_identity`.
        have hidCtx : id = ctx.inputId := by
          by_contra hne
          have hnone : stepAlpha g ps ibp alpha ctx cert id = none := by
            simp [stepAlpha, alphaCrownStepNode?, hk, hne]
          have : False := by
            have : (some b) = none := by
              simpa [hs.symm] using hnone
            cases this
          exact this
        subst hidCtx
        have hb : b = Cert.boundsIdentity (α := ℝ) ctx.inputDim := by
          have : some (Cert.boundsIdentity (α := ℝ) ctx.inputDim) = some b := by
            simpa [stepAlpha, alphaCrownStepNode?, hk] using hs
          cases this
          rfl
        subst hb
        -- Identify the semantic input value and relate it to `x`.
        rcases hinputs with ⟨vin, hmap, ⟨hdim, hxEq⟩⟩
        have hin : inputs[ctx.inputId]? = some vin := hmap
        have hvIn : v = vin := by
          have hiv : inputs[ctx.inputId]? = some v := by
            -- Unfold the evaluator for `.input`.
            simpa [CertSoundness.evalNode?, hk] using hEvalSome
          have : some vin = some v := by
            calc
              some vin = inputs[ctx.inputId]? := by simpa using hin.symm
              _ = some v := hiv
          injection this with h
          exact h.symm
        subst hvIn
        -- Prove enclosure of the point box at `x`.
        refine ⟨rfl, ?_⟩
        dsimp [CrownCertSoundness.EnclosesAtInput]
        simp [Cert.boundsIdentity]
        refine ⟨hdim.symm, ?_⟩
        -- `v.v` is definitionally `x` by `InputsMatch`.
        have hx : castDimScalar (α := ℝ) hdim v.v = x := by
          simpa using hxEq
        -- Show enclosure by unfolding `boundsEvalAt` and simplifying the identity affine map.
        have hC :
            (Cert.affIdentity (α := ℝ) ctx.inputDim).c =
              Spec.fill (α := ℝ) (0 : ℝ) (.dim ctx.inputDim .scalar) := by
          simp [Cert.affIdentity]
        have hMat :
            Spec.matVecMulSpec (α := ℝ) (Cert.affIdentity (α := ℝ) ctx.inputDim).A x = x :=
          mat_vec_mul_spec_aff_identity (n := ctx.inputDim) x
        have hGoalX : Theorems.Semantics.encloses (α := ℝ)
            (boundsEvalAt (α := ℝ) (Cert.boundsIdentity (α := ℝ) ctx.inputDim) x) x := by
          -- Unfold to a record and prove coordinatewise using `encloses_iff_toVec`.
          dsimp [boundsEvalAt, Cert.boundsIdentity]
          refine (encloses_iff_toVec (n := ctx.inputDim)
              (lo := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := ctx.inputDim)
                (Cert.affIdentity (α := ℝ) ctx.inputDim) x)
              (hi := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := ctx.inputDim)
                (Cert.affIdentity (α := ℝ) ctx.inputDim) x)
              (x := x)).2 ?_
          intro i
          constructor <;> simp [affineEvalAt, hMat, hC]
        have hGoalCast :
            Theorems.Semantics.encloses (α := ℝ)
              (boundsEvalAt (α := ℝ) (Cert.boundsIdentity (α := ℝ) ctx.inputDim) x)
              (castDimScalar (α := ℝ) hdim v.v) :=
          sem_encloses_value_eq (B := boundsEvalAt (α := ℝ)
            (Cert.boundsIdentity (α := ℝ) ctx.inputDim) x) hx.symm hGoalX
        simpa [Cert.boundsIdentity] using hGoalCast

    | .const _ => by
        -- Both semantics and the step read `ps.constVals[id]?`.
        cases hcv : ps.constVals[id]? with
        | none =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hcv] at hs
        | some vc =>
            have hs' : some (Cert.boundsConst (α := ℝ) ctx.inputDim vc.n vc.v vc.v) = some b := by
              simpa [stepAlpha, alphaCrownStepNode?, hk, hcv] using hs
            have hv' : v = vc := by
              have hev : ps.constVals[id]? = some v := by
                simpa [CertSoundness.evalNode?, hk] using hEvalSome
              have : some v = some vc := by
                calc
                  some v = ps.constVals[id]? := by simpa using hev.symm
                  _ = some vc := hcv
              cases this
              rfl
            cases hs'
            -- Keep `vc` (substitute `v`, not `vc`).
            subst v
            -- Now `b` is the constant affine enclosure and `v = vc`.
            refine ⟨rfl, ?_⟩
            dsimp [CrownCertSoundness.EnclosesAtInput]
            refine ⟨by simp [boundsEvalAt, Cert.boundsConst], ?_⟩
            -- The evaluated affine bounds are the point box `{vc.v}`.
            have hGoalV : Theorems.Semantics.encloses (α := ℝ)
                (boundsEvalAt (α := ℝ) (Cert.boundsConst (α := ℝ) ctx.inputDim vc.n vc.v vc.v) x) vc.v
                  := by
              -- Unfold to a record and prove coordinatewise using `encloses_iff_toVec`.
              dsimp [boundsEvalAt, Cert.boundsConst]
              refine (encloses_iff_toVec (n := vc.n)
                  (lo := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := vc.n)
                    { A := Spec.fill (α := ℝ) 0 (.dim vc.n (.dim ctx.inputDim .scalar)), c := vc.v }
                      x)
                  (hi := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := vc.n)
                    { A := Spec.fill (α := ℝ) 0 (.dim vc.n (.dim ctx.inputDim .scalar)), c := vc.v }
                      x)
                  (x := vc.v)).2 ?_
              intro i
              constructor <;> simp [affineEvalAt, mat_vec_mul_spec_fill_zero]
            have hGoalVCast :
                ∀ hOut : vc.n = vc.n,
                  Theorems.Semantics.encloses (α := ℝ)
                    (boundsEvalAt (α := ℝ)
                      (Cert.boundsConst (α := ℝ) ctx.inputDim vc.n vc.v vc.v) x)
                    (castDimScalar (α := ℝ) hOut vc.v) := by
              intro hOut
              exact sem_encloses_value_eq
                (hxy := (castDimScalar_self hOut vc.v).symm) hGoalV
            exact hGoalVCast _

    | .detach => by
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            -- Semantics returns the parent value; the step returns the parent affine bound.
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
            -- From step success: `getAff? cert p1 = some b`.
            by_cases hltC : p1 < cert.size
            · have hbp : cert[p1]! = some b := by
                simpa [stepAlpha, alphaCrownStepNode?, hk, hps, NN.MLTheory.CROWN.Cert.getAff?,
                  hltC] using hs
              -- From semantic success: `getVal? vals p1 = some v`, so `vals[p1]! = some v`.
              have hval : CertSoundness.getVal? vals p1 = some v := by
                simpa [CertSoundness.evalNode?, hk, hps] using hEvalSome
              by_cases hltV : p1 < vals.size
              · have hvp : vals[p1]! = some v := by
                  simpa [CertSoundness.getVal?, hltV] using hval
                have hpar : EnclosesAtInput (α := ℝ) ctx x b v := parentEnc p1 hpMem b v hbp hvp
                simpa using hpar
              · have : CertSoundness.getVal? vals p1 = none := by
                  simp [CertSoundness.getVal?, hltV]
                have : False := by
                  simp [this] at hval
                exact False.elim this
            · -- If `p1` is out of bounds, `getAff?` is `none`, contradiction.
              have hnone : stepAlpha g ps ibp alpha ctx cert id = none := by
                simp [stepAlpha, alphaCrownStepNode?, hk, hps, NN.MLTheory.CROWN.Cert.getAff?, hltC]
              have : False := by
                have : (some b) = none := by
                  simpa [hs.symm] using hnone
                cases this
              exact False.elim this

    | .reshape _ _ => by
          -- Same as detach, but with an out-dimension cast (step checks it; semantics checks it).
          cases hps : (g.nodes[id]!).parents with
          | nil =>
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
          | cons p1 _ =>
              have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
              -- Extract step-side parent affine bound and the out-dimension check.
              have hs' := hs
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
              cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
              | none =>
                  -- Step would be `none`, contradicting `hs : some b = ...`.
                  have : False := by
                    simp [hxin] at hs'
                  exact False.elim this
              | some xin =>
                  by_cases hout : xin.outDim = (g.nodes[id]!).outShape.size
                  ·
                    have hbEq :
                        some
                          { inDim := xin.inDim
                            outDim := (g.nodes[id]!).outShape.size
                            loAff := Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                              (m' := (g.nodes[id]!).outShape.size) hout xin.loAff
                            hiAff := Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                              (m' := (g.nodes[id]!).outShape.size) hout xin.hiAff } = some b := by
                      simpa [hxin, hout] using hs'
                    cases hbEq
                    -- Extract semantic-side parent value and the out-dimension check.
                    have hEval' := hEvalSome
                    simp [CertSoundness.evalNode?, hk, hps] at hEval'
                    cases hgv : CertSoundness.getVal? vals p1 with
                    | none =>
                        have : False := by
                          simp [hgv] at hEval'
                        exact False.elim this
                    | some vp =>
                        -- Peel the evaluator's size check.
                        simp [hgv] at hEval'
                        by_cases hvout : vp.n = (g.nodes[id]!).outShape.size
                        · have hvEq :
                            some { n := (g.nodes[id]!).outShape.size
                                   v := castDimScalar (α := ℝ) hvout vp.v } = some v := by
                            simpa [hvout] using hEval'
                          cases hvEq
                          -- Convert `getAff?/getVal?` equalities into array lookup equalities for
                          -- `parentEnc`.
                          have hcertp : cert[p1]! = some xin := by
                            by_cases hltC : p1 < cert.size
                            · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                            · have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                              have : False := by
                                simp [this] at hxin
                              exact False.elim this
                          have hvpp : vals[p1]! = some vp := by
                            by_cases hltV : p1 < vals.size
                            · simpa [CertSoundness.getVal?, hltV] using hgv
                            · have : CertSoundness.getVal? vals p1 = none := by
                                simp [CertSoundness.getVal?, hltV]
                              have : False := by
                                simp [this] at hgv
                              exact False.elim this
                          have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                            parentEnc p1 hpMem xin vp hcertp hvpp
                          -- Reshape/flatten are value-preserving casts: transport the enclosure
                          -- across output-dim casts.
                          simpa using (enclosesAtInput_castOut (ctx := ctx) (x := x) (xin := xin)
                            (vp := vp)
                            (hout := hout) (hvout := hvout) hpar)
                        · -- Semantic reshape would be `none`, contradicting `hEvalSome`.
                          have : False := by
                            simp [hvout] at hEval'
                          exact False.elim this
                  ·
                    -- Step reshape would be `none`, contradicting `hs`.
                    have hcontra : (none : Option (FlatAffineBounds ℝ)) = some b := by
                      have hs'' := hs'
                      simp [hxin, hout] at hs''
                    cases hcontra

    | .flatten _ => by
          cases hps : (g.nodes[id]!).parents with
          | nil =>
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
          | cons p1 _ =>
              have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
              have hs' := hs
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
              cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
              | none =>
                  have : False := by
                    simp [hxin] at hs'
                  exact False.elim this
              | some xin =>
                  by_cases hout : xin.outDim = (g.nodes[id]!).outShape.size
                  · have hbEq :
                      some
                        { inDim := xin.inDim
                          outDim := (g.nodes[id]!).outShape.size
                          loAff := Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                            (m' := (g.nodes[id]!).outShape.size) hout xin.loAff
                          hiAff := Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                            (m' := (g.nodes[id]!).outShape.size) hout xin.hiAff } = some b := by
                      simpa [hxin, hout] using hs'
                    cases hbEq
                    have hEval' := hEvalSome
                    simp [CertSoundness.evalNode?, hk, hps] at hEval'
                    cases hgv : CertSoundness.getVal? vals p1 with
                    | none =>
                        have : False := by
                          simp [hgv] at hEval'
                        exact False.elim this
                    | some vp =>
                        simp [hgv] at hEval'
                        by_cases hvout : vp.n = (g.nodes[id]!).outShape.size
                        · have hvEq :
                            some { n := (g.nodes[id]!).outShape.size
                                   v := castDimScalar (α := ℝ) hvout vp.v } = some v := by
                            simpa [hvout] using hEval'
                          cases hvEq
                          have hcertp : cert[p1]! = some xin := by
                            by_cases hltC : p1 < cert.size
                            · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                            · have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                              have : False := by
                                simp [this] at hxin
                              exact False.elim this
                          have hvpp : vals[p1]! = some vp := by
                            by_cases hltV : p1 < vals.size
                            · simpa [CertSoundness.getVal?, hltV] using hgv
                            · have : CertSoundness.getVal? vals p1 = none := by
                                simp [CertSoundness.getVal?, hltV]
                              have : False := by
                                simp [this] at hgv
                              exact False.elim this
                          have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                            parentEnc p1 hpMem xin vp hcertp hvpp
                          -- Reshape/flatten are value-preserving casts: transport the enclosure
                          -- across output-dim casts.
                          simpa using (enclosesAtInput_castOut (ctx := ctx) (x := x) (xin := xin)
                            (vp := vp)
                            (hout := hout) (hvout := hvout) hpar)
                        · have : False := by
                            simp [hvout] at hEval'
                          exact False.elim this
                  ·
                    have : False := by
                      simp [hxin, hout] at hs'
                    exact False.elim this

    | .linear => by
        -- Linear layer: sound by sign-splitting + parent enclosure.
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
            -- Step-side: extract the parent affine bound, parameters, and the out-dimension check.
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                cases hwb : ps.linearWB[id]? with
                | none =>
                    have : False := by
                      simp [hxin, hwb] at hs'
                    exact False.elim this
                | some p =>
                    by_cases hout : xin.outDim = p.n
                    ·
                      have hbEq :
                          some (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := p.n)
                            (m := p.m) p.w p.b xin hout) =
                            some b := by
                        simpa [hxin, hwb, hout] using hs'
                      cases hbEq

                      -- Semantic-side: extract parent value and the in-dimension check used by
                      -- `.linear`.
                      have hEval' := hEvalSome
                      simp [CertSoundness.evalNode?, hk, hps, hwb] at hEval'
                      cases hgv : CertSoundness.getVal? vals p1 with
                      | none =>
                          have : False := by
                            simp [hgv] at hEval'
                          exact False.elim this
                      | some vp =>
                          simp [hgv] at hEval'
                          by_cases hvIn : vp.n = p.n
                          ·
                            have hvEq :
                                some
                                  { n := p.m
                                    v :=
                                      Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b }
                                        (castDimScalar (α := ℝ) hvIn vp.v) } = some v := by
                              simpa [hvIn] using hEval'
                            cases hvEq

                            -- Turn `getAff?/getVal?` equalities into array lookup equalities to use
                            -- `parentEnc`.
                            have hcertp : cert[p1]! = some xin := by
                              by_cases hltC : p1 < cert.size
                              · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                              ·
                                have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                  simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                                have : False := by
                                  simp [this] at hxin
                                exact False.elim this
                            have hvpp : vals[p1]! = some vp := by
                              by_cases hltV : p1 < vals.size
                              · simpa [CertSoundness.getVal?, hltV] using hgv
                              ·
                                have : CertSoundness.getVal? vals p1 = none := by
                                  simp [CertSoundness.getVal?, hltV]
                                have : False := by
                                  simp [this] at hgv
                                exact False.elim this
                            have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                              parentEnc p1 hpMem xin vp hcertp hvpp

                            -- Unfold the parent enclosure to obtain a semantic enclosure of
                            -- `boundsEvalAt xin x'`.
                            rcases hpar with ⟨hinDim, hvec⟩
                            dsimp at hvec
                            rcases hvec with ⟨hdimB, hencB⟩
                            let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                              castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                hinDim.symm x

                            -- Identify `l/u` as in `boundsEvalAt_linear_bounds_from_affine`.
                            let l : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n)
                                (NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim)
                                  (m := xin.outDim) (m' := p.n) hout xin.loAff) x'
                            let u : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n)
                                (NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim)
                                  (m := xin.outDim) (m' := p.n) hout xin.hiAff) x'
                            -- Cast the parent box to dimension `p.n` (using `hout`), and rewrite it
                            -- into `[l,u]`.
                            have hxCast :
                                Theorems.Semantics.encloses (α := ℝ) { dim := p.n, lo := l, hi := u
                                  }
                                  (castDimScalar (α := ℝ) hvIn vp.v) := by
                              have hx0 :=
                                sem_encloses_castDim
                                  (B := boundsEvalAt (α := ℝ) xin x')
                                  (h := hout)
                                  (x := castDimScalar (α := ℝ) hdimB.symm vp.v)
                                  hencB
                              have hvIn' : Eq.trans hdimB.symm hout = hvIn := by
                                exact Subsingleton.elim _ _
                              have hxCastVec :
                                  castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                    vp.v) =
                                    castDimScalar (α := ℝ) hvIn vp.v := by
                                have hx1 :
                                    castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                      vp.v) =
                                      castDimScalar (α := ℝ) (Eq.trans hdimB.symm hout) vp.v := by
                                  exact (castDimScalar_trans (h₁ := hdimB.symm) (h₂ := hout)
                                    (t := vp.v)).symm
                                exact hx1.trans (castDimScalar_proof_irrel (Eq.trans hdimB.symm hout)
                                  hvIn vp.v)
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo = l
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, l] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi = u
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, u] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              -- Avoid rewriting dependent boxes: reason componentwise via `toVec`.
                              have hx0' :=
                                (encloses_iff_toVec (n := p.n)
                                  (lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').lo)
                                  (hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').hi)
                                  (x := castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ)
                                    hdimB.symm vp.v))).1 hx0
                              refine (encloses_iff_toVec (n := p.n) (lo := l) (hi := u)
                                (x := castDimScalar (α := ℝ) hvIn vp.v)).2 ?_
                              intro i
                              have hi := hx0' i
                              constructor
                              · exact (by simpa [hl, hxCastVec] using hi.1)
                              · exact (by simpa [hu, hxCastVec] using hi.2)

                            -- Apply sign-splitting enclosure at the point `xv`.
                            let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) hvIn vp.v
                            have hy :
                                Theorems.Semantics.encloses (α := ℝ)
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        p.b
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        p.b }
                                  (Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv) := by
                              -- `encloses_linear_signSplit` encloses `W·x + b`, and `linear_spec`
                              -- is definitional.
                              simpa [Spec.linearSpec, xv] using
                                (encloses_linear_signSplit (m := p.m) (n := p.n) (W := p.w) (b :=
                                  p.b)
                                  (lo := l) (hi := u) (x := xv) hxCast)

                            -- Rewrite the computed bounds to `boundsEvalAt
                            -- (linear_bounds_from_affine ...) x'`.
                            have hBE :
                                boundsEvalAt (α := ℝ)
                                  (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                                    p.n) (m := p.m) p.w p.b xin hout) x' =
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        p.b
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        p.b } := by
                              -- This is exactly `boundsEvalAt_linear_bounds_from_affine`.
                              simpa [l, u, x'] using
                                (boundsEvalAt_linear_bounds_from_affine (n := p.n) (m := p.m) (W :=
                                  p.w) (b := p.b) (xB := xin)
                                  (hout := hout) (x := x'))

                            -- Package the result as `EnclosesAtInput`.
                            refine ⟨hinDim, ?_⟩
                            dsimp [CrownCertSoundness.EnclosesVec]
                            refine ⟨rfl, ?_⟩
                            -- Transport the enclosure `hy` across the `boundsEvalAt` equality
                            -- without rewriting in a dependent motive.
                            have hyCast :=
                              sem_encloses_of_eq (h := hBE.symm)
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv)
                                  hy
                            exact sem_encloses_value_eq
                              (hxy := castDimScalar_proof_irrel _ _ _) hyCast
                          ·
                            have : False := by
                              simp [hvIn] at hEval'
                            exact False.elim this
                    ·
                      have : False := by
                        simp [hxin, hwb, hout] at hs'
                      exact False.elim this

    | .matmul => by
        -- Matmul is linear with zero bias; same proof strategy as `.linear`.
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                cases hwb : ps.matmulW[id]? with
                | none =>
                    have : False := by
                      simp [hxin, hwb] at hs'
                    exact False.elim this
                | some p =>
                    by_cases hout : xin.outDim = p.n
                    ·
                      have hbEq :
                          some (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := p.n)
                            (m := p.m) p.w
                            (Spec.fill (α := ℝ) Zero.zero (.dim p.m .scalar)) xin hout) = some b := by
                        simpa [hxin, hwb, hout, Numbers.zero] using hs'
                      cases hbEq

                      have hEval' := hEvalSome
                      simp [CertSoundness.evalNode?, hk, hps, hwb] at hEval'
                      cases hgv : CertSoundness.getVal? vals p1 with
                      | none =>
                          have : False := by
                            simp [hgv] at hEval'
                          exact False.elim this
                      | some vp =>
                          simp [hgv] at hEval'
                          by_cases hvIn : vp.n = p.n
                          ·
                            have hvEq :
                                some
                                  { n := p.m
                                    v :=
                                      Spec.linearSpec (α := ℝ)
                                        { weights := p.w, bias := Spec.fill (α := ℝ) 0 (.dim p.m
                                          .scalar) }
                                        (castDimScalar (α := ℝ) hvIn vp.v) } = some v := by
                              simpa [hvIn] using hEval'
                            cases hvEq

                            have hcertp : cert[p1]! = some xin := by
                              by_cases hltC : p1 < cert.size
                              · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                              ·
                                have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                  simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                                have : False := by
                                  simp [this] at hxin
                                exact False.elim this
                            have hvpp : vals[p1]! = some vp := by
                              by_cases hltV : p1 < vals.size
                              · simpa [CertSoundness.getVal?, hltV] using hgv
                              ·
                                have : CertSoundness.getVal? vals p1 = none := by
                                  simp [CertSoundness.getVal?, hltV]
                                have : False := by
                                  simp [this] at hgv
                                exact False.elim this
                            have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                              parentEnc p1 hpMem xin vp hcertp hvpp

                            rcases hpar with ⟨hinDim, hvec⟩
                            dsimp at hvec
                            rcases hvec with ⟨hdimB, hencB⟩
                            let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                              castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                hinDim.symm x
                            let z : Tensor ℝ (.dim p.m .scalar) := Spec.fill (α := ℝ) 0 (.dim p.m
                              .scalar)
                            let l : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n)
                                (NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim)
                                  (m := xin.outDim) (m' := p.n) hout xin.loAff) x'
                            let u : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n)
                                (NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim)
                                  (m := xin.outDim) (m' := p.n) hout xin.hiAff) x'
                            have hxCast :
                                Theorems.Semantics.encloses (α := ℝ) { dim := p.n, lo := l, hi := u
                                  }
                                  (castDimScalar (α := ℝ) hvIn vp.v) := by
                              have hx0 :=
                                sem_encloses_castDim
                                  (B := boundsEvalAt (α := ℝ) xin x')
                                  (h := hout)
                                  (x := castDimScalar (α := ℝ) hdimB.symm vp.v)
                                  hencB
                              have hvIn' : Eq.trans hdimB.symm hout = hvIn := by
                                exact Subsingleton.elim _ _
                              have hxCastVec :
                                  castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                    vp.v) =
                                    castDimScalar (α := ℝ) hvIn vp.v := by
                                have hx1 :
                                    castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                      vp.v) =
                                      castDimScalar (α := ℝ) (Eq.trans hdimB.symm hout) vp.v := by
                                  exact (castDimScalar_trans (h₁ := hdimB.symm) (h₂ := hout)
                                    (t := vp.v)).symm
                                exact hx1.trans (castDimScalar_proof_irrel (Eq.trans hdimB.symm hout)
                                  hvIn vp.v)
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo = l
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, l] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi = u
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, u] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              -- Avoid rewriting dependent boxes: reason componentwise via `toVec`.
                              have hx0' :=
                                (encloses_iff_toVec (n := p.n)
                                  (lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').lo)
                                  (hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').hi)
                                  (x := castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ)
                                    hdimB.symm vp.v))).1 hx0
                              refine (encloses_iff_toVec (n := p.n) (lo := l) (hi := u)
                                (x := castDimScalar (α := ℝ) hvIn vp.v)).2 ?_
                              intro i
                              have hi := hx0' i
                              constructor
                              · exact (by simpa [hl, hxCastVec] using hi.1)
                              · exact (by simpa [hu, hxCastVec] using hi.2)

                            let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) hvIn vp.v
                            have hy :
                                Theorems.Semantics.encloses (α := ℝ)
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        z
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        z }
                                  (Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv) := by
                              simpa [Spec.linearSpec, z, xv] using
                                (encloses_linear_signSplit (m := p.m) (n := p.n) (W := p.w) (b := z)
                                  (lo := l) (hi := u) (x := xv) hxCast)

                            have hBE :
                                boundsEvalAt (α := ℝ)
                                  (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                                    p.n) (m := p.m) p.w z xin hout) x' =
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        z
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        z } := by
                              simpa [l, u, x', z] using
                                (boundsEvalAt_linear_bounds_from_affine (n := p.n) (m := p.m) (W :=
                                  p.w) (b := z) (xB := xin)
                                  (hout := hout) (x := x'))

                            refine ⟨hinDim, ?_⟩
                            dsimp [CrownCertSoundness.EnclosesVec]
                            refine ⟨rfl, ?_⟩
                            -- Transport the enclosure `hy` across the `boundsEvalAt` equality
                            -- without rewriting in a dependent motive.
                            have hyCast :=
                              sem_encloses_of_eq (h := hBE.symm)
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv) hy
                            -- Keep the `linear_spec` form (it matches the semantic evaluator for
                            -- `.matmul`).
                            exact sem_encloses_value_eq
                              (hxy := castDimScalar_proof_irrel _ _ _) hyCast
                          ·
                            have : False := by
                              simp [hvIn] at hEval'
                            exact False.elim this
                    ·
                      have : False := by
                        simp [hxin, hwb, hout] at hs'
                      exact False.elim this

    | .sum => by
        -- Sum is a 1×n linear layer with all-ones weights and zero bias.
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]

            -- Step-side: extract the parent affine bound.
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                -- Step forces `b` to be `linear_bounds_from_affine onesRow 0 xin`.
                let onesRow : Tensor ℝ (.dim 1 (.dim xin.outDim .scalar)) :=
                  Spec.fill (α := ℝ) (Numbers.one : ℝ) (.dim 1 (.dim xin.outDim .scalar))
                let zb : Tensor ℝ (.dim 1 .scalar) :=
                  Spec.fill (α := ℝ) (Numbers.zero : ℝ) (.dim 1 .scalar)
                have hbEq :
                    some
                        (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := xin.outDim)
                          (m := 1)
                          onesRow zb xin (by rfl)) =
                      some b := by
                  simpa [hxin, onesRow, zb, Numbers.zero, Numbers.one] using hs'
                cases hbEq

                -- Semantic-side: extract the parent value.
                have hEval' := hEvalSome
                simp [CertSoundness.evalNode?, hk, hps] at hEval'
                cases hgv : CertSoundness.getVal? vals p1 with
                | none =>
                    have : False := by
                      simp [hgv] at hEval'
                    exact False.elim this
                | some vp =>
                    -- `v` is exactly the mat-vec multiply by an all-ones row.
                    have hvEq :
                        some
                            { n := 1
                              v :=
                                Spec.matVecMulSpec (α := ℝ)
                                  (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v } =
                          some v := by
                      simpa [hgv] using hEval'
                    cases hvEq

                    -- Connect `getAff?/getVal?` equalities to array lookups to use `parentEnc`.
                    have hcertp : cert[p1]! = some xin := by
                      by_cases hltC : p1 < cert.size
                      · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                      ·
                        have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                          simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                        have : False := by
                          simp [this] at hxin
                        exact False.elim this
                    have hvpp : vals[p1]! = some vp := by
                      by_cases hltV : p1 < vals.size
                      · simpa [CertSoundness.getVal?, hltV] using hgv
                      ·
                        have : CertSoundness.getVal? vals p1 = none := by
                          simp [CertSoundness.getVal?, hltV]
                        have : False := by
                          simp [this] at hgv
                        exact False.elim this
                    have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                      parentEnc p1 hpMem xin vp hcertp hvpp

                    -- Unpack parent enclosure.
                    rcases hpar with ⟨hinDim, hvec⟩
                    dsimp at hvec
                    rcases hvec with ⟨hdimB, hencB⟩
                    let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                      castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim) hinDim.symm x
                    let l : Tensor ℝ (.dim xin.outDim .scalar) :=
                      affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := xin.outDim)
                        (Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                          (m' := xin.outDim) rfl xin.loAff) x'
                    let u : Tensor ℝ (.dim xin.outDim .scalar) :=
                      affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := xin.outDim)
                        (Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                          (m' := xin.outDim) rfl xin.hiAff) x'
                    let xv : Tensor ℝ (.dim xin.outDim .scalar) :=
                      castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdimB.symm vp.v

                    have hxCast :
                        Theorems.Semantics.encloses (α := ℝ)
                          { dim := xin.outDim, lo := l, hi := u } xv := by
                      -- `boundsEvalAt` is definitional in terms of `affineEvalAt`.
                      simpa [CrownCertSoundness.boundsEvalAt, CrownCertSoundness.affineEvalAt,
                        Cert.castAffineOut, l, u, xv, x'] using hencB

                    -- Apply sign-splitting enclosure (bias is zero, so use the mat-vec form).
                    have hy0 :
                        Theorems.Semantics.encloses (α := ℝ)
                          { dim := 1
                            lo :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u))
                                zb
                            hi :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l))
                                zb }
                          (Spec.linearSpec (α := ℝ) { weights := onesRow, bias := zb } xv) := by
                      simpa [Spec.linearSpec, onesRow, zb, xv] using
                        (encloses_linear_signSplit (m := 1) (n := xin.outDim) (W := onesRow) (b :=
                          zb)
                          (lo := l) (hi := u) (x := xv) hxCast)
                    have hy :
                        Theorems.Semantics.encloses (α := ℝ)
                          { dim := 1
                            lo :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u))
                                zb
                            hi :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l))
                                zb }
                            (Spec.matVecMulSpec (α := ℝ) onesRow xv) := by
                      -- Convert `linear_spec` to mat-vec since the bias is zero.
                      have hlin :
                          Spec.linearSpec (α := ℝ) { weights := onesRow, bias := zb } xv =
                            Spec.matVecMulSpec (α := ℝ) onesRow xv := by
                        simpa [onesRow, zb, Numbers.zero] using
                          (linear_spec_bias_zero_eq_matvec (W := onesRow) (x := xv))
                      exact sem_encloses_value_eq (hxy := hlin) hy0

                    -- Rewrite the computed bounds to `boundsEvalAt (linear_bounds_from_affine ...)
                    -- x'`.
                    have hBE :
                        boundsEvalAt (α := ℝ)
                          (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := xin.outDim)
                            (m := 1)
                            onesRow zb xin (by rfl)) x' =
                          { dim := 1
                            lo :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u))
                                zb
                            hi :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l))
                                zb } := by
                      simpa [l, u, x'] using
                        (boundsEvalAt_linear_bounds_from_affine (n := xin.outDim) (m := 1) (W :=
                          onesRow) (b := zb)
                          (xB := xin) (hout := (by rfl)) (x := x'))

                    refine ⟨hinDim, ?_⟩
                    dsimp [CrownCertSoundness.EnclosesVec]
                    refine ⟨rfl, ?_⟩
                    -- Finally, match the semantic evaluator's `onesRow` (built from `vp.n`).
                    have hvOut :
                        Spec.matVecMulSpec (α := ℝ) onesRow xv =
                          Spec.matVecMulSpec (α := ℝ)
                            (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v := by
                      have hdn : xin.outDim = vp.n := by
                        simpa [CrownCertSoundness.boundsEvalAt] using hdimB
                      have hxv :
                          xv = castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdn.symm vp.v := by
                        have : hdimB.symm = hdn.symm := by exact Subsingleton.elim _ _
                        simp [xv]
                      have hcast :=
                        mat_vec_mul_fill1_castDimScalar (h := hdn.symm) (v := vp.v)
                      simpa [onesRow, Numbers.one, hxv] using hcast.symm
                    have hyVal :
                        Theorems.Semantics.encloses (α := ℝ)
                          { dim := 1
                            lo :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u))
                                zb
                            hi :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l))
                                zb }
                          (Spec.matVecMulSpec (α := ℝ)
                            (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v) :=
                      sem_encloses_value_eq (hxy := hvOut) hy
                    have hyBout :
                        Theorems.Semantics.encloses (α := ℝ)
                          (boundsEvalAt (α := ℝ)
                            (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                              xin.outDim) (m := 1)
                              onesRow zb xin (by rfl)) x')
                            (Spec.matVecMulSpec (α := ℝ)
                              (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v) := by
                        let yv : Tensor ℝ (.dim 1 .scalar) :=
                          Spec.matVecMulSpec (α := ℝ)
                            (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v
                        have hyBoxCast :=
                          sem_encloses_of_eq (h := hBE.symm) (x := yv) hyVal
                        have hcastY :
                            castDimScalar (α := ℝ) (congrArg FlatBox.dim hBE.symm) yv = yv := by
                          exact castDimScalar_self _ yv
                        exact sem_encloses_value_eq (hxy := hcastY) hyBoxCast
                    exact hyBout

    | .relu => by
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]

            -- Step-side: extract parent affine bounds and IBP pre-activation box.
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                cases hpre : ibp[p1]! with
                | none =>
                    have : False := by
                      simp [hxin, hpre] at hs'
                    exact False.elim this
                | some preB =>
                    -- Semantic-side: extract the parent value `vp`.
                    have hEval' := hEvalSome
                    simp [CertSoundness.evalNode?, hk, hps] at hEval'
                    cases hgv : CertSoundness.getVal? vals p1 with
                    | none =>
                        have : False := by
                          simp [hgv] at hEval'
                        exact False.elim this
                    | some vp =>
                        have hvEq :
                            some { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } = some v :=
                              by
                          simpa [hgv] using hEval'
                        cases hvEq

                        -- Connect `getAff?/getVal?` equalities to array lookups to use `parentEnc`.
                        have hcertp : cert[p1]! = some xin := by
                          by_cases hltC : p1 < cert.size
                          · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                          ·
                            have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                              simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                            have : False := by
                              simp [this] at hxin
                            exact False.elim this
                        have hvpp : vals[p1]! = some vp := by
                          by_cases hltV : p1 < vals.size
                          · simpa [CertSoundness.getVal?, hltV] using hgv
                          ·
                            have : CertSoundness.getVal? vals p1 = none := by
                              simp [CertSoundness.getVal?, hltV]
                            have : False := by
                              simp [this] at hgv
                            exact False.elim this
                        have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                          parentEnc p1 hpMem xin vp hcertp hvpp

                        -- IBP enclosure for the pre-activation box `preB` at parent `p1`.
                        have hp1ltId : p1 < id := htopo id hid p1 hpMem
                        have hp1ltNodes : p1 < g.nodes.size := lt_trans hp1ltId hid
                        have hp1ltVals : p1 < vals.size := by simpa [hsem.1] using hp1ltNodes
                        have hibpP := hibp p1 hp1ltVals
                        have hEncIbp : CertSoundness.EnclosesBox preB vp := by
                          simpa [hpre, hvpp] using hibpP
                        rcases hEncIbp with ⟨hdimIbp, hencIbp⟩

                        -- Unpack parent enclosure.
                        rcases hpar with ⟨hinDim, hvec⟩
                        dsimp at hvec
                        rcases hvec with ⟨hdimB, hencB⟩
                        have hdn : xin.outDim = vp.n := by
                          simpa [CrownCertSoundness.boundsEvalAt] using hdimB

                        -- Both α-cases share the same main proof, only α-vector differs.
                        cases hαopt : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id with
                        | none =>
                            by_cases hout : xin.outDim = preB.dim
                            ·
                              let αt : Tensor ℝ (.dim preB.dim .scalar) :=
                                defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                              let bout : FlatAffineBounds ℝ :=
                                { inDim := xin.inDim
                                  outDim := preB.dim
                                  loAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                                        αt)
                                      (Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                        (m' := preB.dim) hout xin.loAff)
                                  hiAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n :=
                                        preB.dim) preB.lo preB.hi)
                                      (Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                        (m' := preB.dim) hout xin.hiAff) }
                              have hbEq : some bout = some b := by
                                simpa [hxin, hpre, hαopt, hout, bout, αt, Cert.castAffineOut] using
                                  hs'
                              have hb : b = bout := (Option.some.inj hbEq).symm

                              let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                                castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                  hinDim.symm x
                              let xLo : AffineVec ℝ xin.inDim preB.dim := by
                                exact Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                  (m' := preB.dim) hout xin.loAff
                              let xHi : AffineVec ℝ xin.inDim preB.dim := by
                                exact Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                  (m' := preB.dim) hout xin.hiAff
                              let lAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xLo
                                  x'
                              let uAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xHi
                                  x'
                              let z : Tensor ℝ (.dim preB.dim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                  vp.v
                              have hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ toVec αt i ∧ toVec αt i ≤
                                (1 : ℝ) := by
                                simpa [αt] using defaultAlphaVec_range (lo := preB.lo) (hi :=
                                  preB.hi)

                              -- Derive `lAff ≤ z ≤ uAff` by casting the parent enclosure.
                              let zXin : Tensor ℝ (.dim xin.outDim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdn.symm vp.v
                              have hzXin :
                                  Theorems.Semantics.encloses (α := ℝ) (boundsEvalAt (α := ℝ) xin
                                    x') zXin := by
                                have : castDimScalar (α := ℝ) hdimB.symm vp.v = zXin := by
                                  exact castDimScalar_proof_irrel (h₁ := hdimB.symm) (h₂ :=
                                    hdn.symm) (t := vp.v)
                                simpa [this, zXin] using hencB
                              have hzCast0 :=
                                sem_encloses_castDim (B := boundsEvalAt (α := ℝ) xin x') (h := hout)
                                  (x := zXin) hzXin
                              have hzCastZ :
                                  castDimScalar (α := ℝ) hout zXin = z := by
                                have htrans : Eq.trans hdn.symm hout = hdimIbp.symm := by
                                  exact Subsingleton.elim _ _
                                -- Cast composition aligns with `z` (up to proof irrelevance).
                                have := (castDimScalar_trans (h₁ := hdn.symm) (h₂ := hout) (t :=
                                  vp.v)).symm
                                simpa [zXin, z, htrans] using this
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo =
                                    lAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi =
                                    uAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              have hzAff :
                                  Theorems.Semantics.encloses (α := ℝ)
                                    { dim := preB.dim
                                      lo := lAff
                                      hi := uAff } z := by
                                have hzCast1 :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      { dim := preB.dim
                                        lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').lo
                                        hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').hi }
                                      z := by
                                  exact sem_encloses_value_eq
                                    (B := { dim := preB.dim
                                            lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                              xin x').lo
                                            hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                              xin x').hi })
                                    (hxy := hzCastZ) hzCast0
                                have hBoxEq :
                                    ({ dim := preB.dim
                                       lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').lo
                                       hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').hi } : FlatBox ℝ)
                                      =
                                    ({ dim := preB.dim, lo := lAff, hi := uAff } : FlatBox ℝ) := by
                                  refine FlatBox.ext' (hDim := rfl) (hLo := ?_) (hHi := ?_)
                                  · exact heq_of_eq hl
                                  · exact heq_of_eq hu
                                exact sem_encloses_of_eq (h := hBoxEq) (x := z) hzCast1

                              have hzIbp : Theorems.Semantics.encloses (α := ℝ) preB z := by
                                simpa [CertSoundness.encloses, z] using hencIbp

                              -- Now prove the ReLU enclosure for `boundsEvalAt bout x'`.
                              rw [hb]
                              refine ⟨hinDim, ?_⟩
                              dsimp [CrownCertSoundness.EnclosesVec]
                              refine ⟨hdimIbp, ?_⟩
                              have hreluCast :
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                      (Activation.reluSpec (α := ℝ) vp.v)
                                    =
                                    Activation.reluSpec (α := ℝ) z := by
                                simpa [z] using
                                  (relu_spec_castDimScalar (h := hdimIbp.symm) (t := vp.v))
                              -- Reduce to showing `Semantics.encloses` for `Activation.relu_spec
                              -- z`.
                              have :
                                  Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x') (Activation.reluSpec (α := ℝ)
                                        z) := by
                                -- Componentwise enclosure.
                                have hzAffI := (encloses_iff_toVec (n := preB.dim) (lo := lAff) (hi
                                  := uAff) (x := z)).1 hzAff
                                have hzIbpI := (encloses_iff_toVec (n := preB.dim) (lo := preB.lo)
                                  (hi := preB.hi) (x := z)).1 hzIbp
                                refine (encloses_iff_toVec (n := preB.dim)
                                  (lo := (boundsEvalAt (α := ℝ) bout x').lo)
                                  (hi := (boundsEvalAt (α := ℝ) bout x').hi)
                                  (x := Activation.reluSpec (α := ℝ) z)).2 ?_
                                intro i
                                have hzLo := (hzAffI i).1
                                have hzHi := (hzAffI i).2
                                have hzIlo := (hzIbpI i).1
                                have hzIhi := (hzIbpI i).2
                                let li := toVec preB.lo i
                                let ui := toVec preB.hi i
                                let zi := toVec z i
                                let ai := toVec αt i
                                have hai0 : (0 : ℝ) ≤ ai := (hαrange i).1
                                have hai1 : ai ≤ (1 : ℝ) := (hαrange i).2
                                have hsLo : 0 ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope :=
                                  alphaRelaxLowerScalar_slope_nonneg (l := li) (u := ui) (a := ai)
                                    hai0
                                have hsHi : 0 ≤ (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α
                                  := ℝ) li ui).slope :=
                                  relax_scalar_slope_nonneg (l := li) (u := ui)

                                -- Rewrite the output bounds at index `i`.
                                have hlo_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i
                                      =
                                      let rp := toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim)
                                        preB.lo preB.hi αt) i
                                      rp.slope * toVec lAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                        preB.hi αt)
                                      (aff := xLo) (x := x') (i := i))
                                have hhi_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').hi i
                                      =
                                      let rp := toVec
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n
                                        := preB.dim) preB.lo preB.hi) i
                                      rp.slope * toVec uAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α :=
                                        ℝ) (n := preB.dim) preB.lo preB.hi)
                                      (aff := xHi) (x := x') (i := i))

                                have hrpLo :
                                    toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                      preB.hi αt) i
                                      =
                                      alphaRelaxLowerScalar (α := ℝ) li ui ai := by
                                  simpa [li, ui, ai] using
                                    (toVec_alphaRelaxLowerVec (lo := preB.lo) (hi := preB.hi) (αv :=
                                      αt) (i := i))
                                have hrpHi :
                                    toVec (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ)
                                      (n := preB.dim) preB.lo preB.hi) i
                                      =
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li ui
                                        := by
                                  simpa [li, ui] using (toVec_runtime_relu_relax_vector (lo :=
                                    preB.lo) (hi := preB.hi) (i := i))

                                -- Lower bound inequality.
                                have hlo1 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec lAff i +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias := by
                                  have hm : (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec
                                    lAff i
                                      ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi := by
                                    exact mul_le_mul_of_nonneg_left hzLo hsLo
                                  have h' :=
                                    add_le_add_right hm (alphaRelaxLowerScalar (α := ℝ) li ui
                                      ai).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hlo2 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  -- Use the scalar α-relaxation soundness on the true
                                  -- pre-activation value.
                                  have := NN.MLTheory.CROWN.Proofs.alphaRelaxLowerScalar_sound
                                    (lower := li) (upper := ui) (alpha := ai) (input := zi)
                                    (lowerBound := hzIlo) (upperBound := hzIhi)
                                    (alphaNonnegative := hai0) (alphaAtMostOne := hai1)
                                  simpa [li, ui, zi, ai] using this
                                have hlo :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  -- Rewrite `lo` and chain inequalities.
                                  simp [hlo_def, hrpLo]
                                  exact le_trans hlo1 hlo2

                                -- Upper bound inequality.
                                have hhi1 :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have := relu_relax_scalar_upper_real_runtime (l := li) (u := ui)
                                    (x := zi)
                                    (hlx := hzIlo) (hxu := hzIhi)
                                  simpa [li, ui, zi] using this
                                have hhi2 :
                                    (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias
                                      ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have hm :
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi
                                        ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i := by
                                    exact mul_le_mul_of_nonneg_left hzHi hsHi
                                  have h' :=
                                    add_le_add_right hm
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hhi :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤ toVec (boundsEvalAt (α
                                      := ℝ) bout x').hi i := by
                                  simp [hhi_def, hrpHi]
                                  exact le_trans hhi1 hhi2

                                -- Combine, using `toVec` of ReLU.
                                have hrelu : toVec (Activation.reluSpec (α := ℝ) z) i =
                                    Activation.Math.reluSpec (α := ℝ) zi := by
                                  simpa [zi] using (toVec_relu_spec (t := z) (i := i))
                                constructor
                                · rw [hrelu]
                                  exact hlo
                                · rw [hrelu]
                                  exact hhi
                              -- Apply `hreluCast` to match the required casted value.
                              have this' :
                                  Theorems.Semantics.encloses (boundsEvalAt (α := ℝ) bout x')
                                    (Activation.reluSpec (α := ℝ) z) := by
                                simpa [x'] using this
                              exact sem_encloses_value_eq
                                (B := boundsEvalAt (α := ℝ) bout x')
                                (hxy := hreluCast.symm) this'
                            ·
                              have : False := by
                                simp [hxin, hpre, hαopt, hout] at hs'
                              exact False.elim this
                        | some αv =>
                            by_cases hout : xin.outDim = preB.dim
                            ·
                              let αt : Tensor ℝ (.dim preB.dim .scalar) :=
                                if hα : αv.n = preB.dim then
                                  castDimScalar (α := ℝ) (n := αv.n) (n' := preB.dim) hα αv.v
                                else
                                  defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                              let bout : FlatAffineBounds ℝ :=
                                { inDim := xin.inDim
                                  outDim := preB.dim
                                  loAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                                        αt)
                                      (Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                        (m' := preB.dim) hout xin.loAff)
                                  hiAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n :=
                                        preB.dim) preB.lo preB.hi)
                                      (Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                        (m' := preB.dim) hout xin.hiAff) }
                              have hbEq : some bout = some b := by
                                simpa [hxin, hpre, hαopt, hout, bout, αt, Cert.castAffineOut] using
                                  hs'
                              have hb : b = bout := (Option.some.inj hbEq).symm
                              have hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ toVec αt i ∧ toVec αt i ≤
                                (1 : ℝ) := by
                                classical
                                by_cases hα : αv.n = preB.dim
                                ·
                                  -- Use `AlphaOK` for the provided α vector.
                                  have hidA : id < alpha.size := by
                                    by_cases hlt : id < alpha.size
                                    · exact hlt
                                    ·
                                      have : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id =
                                        none := by
                                        simp [NN.MLTheory.CROWN.Cert.getAlpha?, hlt]
                                      have : False := by
                                        simp [this] at hαopt
                                      exact False.elim this
                                  have hentry : alpha[id]! = some αv := by
                                    simpa [NN.MLTheory.CROWN.Cert.getAlpha?, hidA] using hαopt
                                  have hrange0 : ∀ i : Fin αv.n, (0 : ℝ) ≤ toVec αv.v i ∧ toVec αv.v
                                    i ≤ (1 : ℝ) := by
                                    simpa [hentry] using halpha id hidA
                                  intro i
                                  have hri := hrange0 (Fin.cast hα.symm i)
                                  simpa [αt, hα, toVec_castDimScalar] using hri
                                ·
                                  -- Fallback: default α is 0/1.
                                  simpa [αt, hα] using defaultAlphaVec_range (lo := preB.lo) (hi :=
                                    preB.hi)

                              -- This branch uses the same enclosure argument as the default-alpha
                              -- case, but with an explicit alpha tensor and the range proof above.
                              -- Keeping the proof local makes the dependent casts visible to Lean.
                              let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                                castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                  hinDim.symm x
                              let xLo : AffineVec ℝ xin.inDim preB.dim := by
                                exact Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                  (m' := preB.dim) hout xin.loAff
                              let xHi : AffineVec ℝ xin.inDim preB.dim := by
                                exact Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                                  (m' := preB.dim) hout xin.hiAff
                              let lAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xLo
                                  x'
                              let uAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xHi
                                  x'
                              let z : Tensor ℝ (.dim preB.dim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                  vp.v

                              let zXin : Tensor ℝ (.dim xin.outDim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdn.symm vp.v
                              have hzXin :
                                  Theorems.Semantics.encloses (α := ℝ) (boundsEvalAt (α := ℝ) xin
                                    x') zXin := by
                                have : castDimScalar (α := ℝ) hdimB.symm vp.v = zXin := by
                                  exact castDimScalar_proof_irrel (h₁ := hdimB.symm) (h₂ :=
                                    hdn.symm) (t := vp.v)
                                simpa [this, zXin] using hencB
                              have hzCast0 :=
                                sem_encloses_castDim (B := boundsEvalAt (α := ℝ) xin x') (h := hout)
                                  (x := zXin) hzXin
                              have hzCastZ :
                                  castDimScalar (α := ℝ) hout zXin = z := by
                                have htrans : Eq.trans hdn.symm hout = hdimIbp.symm := by
                                  exact Subsingleton.elim _ _
                                have := (castDimScalar_trans (h₁ := hdn.symm) (h₂ := hout) (t :=
                                  vp.v)).symm
                                simpa [zXin, z, htrans] using this
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo =
                                    lAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi =
                                    uAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              have hzAff :
                                  Theorems.Semantics.encloses (α := ℝ)
                                    { dim := preB.dim
                                      lo := lAff
                                      hi := uAff } z := by
                                have hzCast1 :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      { dim := preB.dim
                                        lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').lo
                                        hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').hi }
                                      z := by
                                      exact sem_encloses_value_eq
                                        (B := { dim := preB.dim
                                                lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α
                                                  := ℝ) xin x').lo
                                                hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α
                                                  := ℝ) xin x').hi })
                                        (hxy := hzCastZ) hzCast0
                                have hBoxEq :
                                    ({ dim := preB.dim
                                       lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').lo
                                       hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').hi } : FlatBox ℝ)
                                      =
                                    ({ dim := preB.dim, lo := lAff, hi := uAff } : FlatBox ℝ) := by
                                  refine FlatBox.ext' (hDim := rfl) (hLo := ?_) (hHi := ?_)
                                  · exact heq_of_eq hl
                                  · exact heq_of_eq hu
                                exact sem_encloses_of_eq (h := hBoxEq) (x := z) hzCast1

                              have hzIbp : Theorems.Semantics.encloses (α := ℝ) preB z := by
                                simpa [CertSoundness.encloses, z] using hencIbp

                              rw [hb]
                              refine ⟨hinDim, ?_⟩
                              dsimp [CrownCertSoundness.EnclosesVec]
                              refine ⟨hdimIbp, ?_⟩
                              have hreluCast :
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                      (Activation.reluSpec (α := ℝ) vp.v)
                                    =
                                    Activation.reluSpec (α := ℝ) z := by
                                simpa [z] using
                                  (relu_spec_castDimScalar (h := hdimIbp.symm) (t := vp.v))
                              have :
                                  Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x') (Activation.reluSpec (α := ℝ)
                                        z) := by
                                have hzAffI := (encloses_iff_toVec (n := preB.dim) (lo := lAff) (hi
                                  := uAff) (x := z)).1 hzAff
                                have hzIbpI := (encloses_iff_toVec (n := preB.dim) (lo := preB.lo)
                                  (hi := preB.hi) (x := z)).1 hzIbp
                                refine (encloses_iff_toVec (n := preB.dim)
                                  (lo := (boundsEvalAt (α := ℝ) bout x').lo)
                                  (hi := (boundsEvalAt (α := ℝ) bout x').hi)
                                  (x := Activation.reluSpec (α := ℝ) z)).2 ?_
                                intro i
                                have hzLo := (hzAffI i).1
                                have hzHi := (hzAffI i).2
                                have hzIlo := (hzIbpI i).1
                                have hzIhi := (hzIbpI i).2
                                let li := toVec preB.lo i
                                let ui := toVec preB.hi i
                                let zi := toVec z i
                                let ai := toVec αt i
                                have hai0 : (0 : ℝ) ≤ ai := (hαrange i).1
                                have hai1 : ai ≤ (1 : ℝ) := (hαrange i).2
                                have hsLo : 0 ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope :=
                                  alphaRelaxLowerScalar_slope_nonneg (l := li) (u := ui) (a := ai)
                                    hai0
                                have hsHi : 0 ≤ (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α
                                  := ℝ) li ui).slope :=
                                  relax_scalar_slope_nonneg (l := li) (u := ui)
                                have hlo_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i
                                      =
                                      let rp := toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim)
                                        preB.lo preB.hi αt) i
                                      rp.slope * toVec lAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                        preB.hi αt)
                                      (aff := xLo) (x := x') (i := i))
                                have hhi_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').hi i
                                      =
                                      let rp := toVec
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n
                                        := preB.dim) preB.lo preB.hi) i
                                      rp.slope * toVec uAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α :=
                                        ℝ) (n := preB.dim) preB.lo preB.hi)
                                      (aff := xHi) (x := x') (i := i))
                                have hrpLo :
                                    toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                      preB.hi αt) i
                                      =
                                      alphaRelaxLowerScalar (α := ℝ) li ui ai := by
                                  simpa [li, ui, ai] using
                                    (toVec_alphaRelaxLowerVec (lo := preB.lo) (hi := preB.hi) (αv :=
                                      αt) (i := i))
                                have hrpHi :
                                    toVec (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ)
                                      (n := preB.dim) preB.lo preB.hi) i
                                      =
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li ui
                                        := by
                                  simpa [li, ui] using (toVec_runtime_relu_relax_vector (lo :=
                                    preB.lo) (hi := preB.hi) (i := i))
                                have hlo1 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec lAff i +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias := by
                                  have hm : (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec
                                    lAff i
                                      ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi := by
                                    exact mul_le_mul_of_nonneg_left hzLo hsLo
                                  have h' :=
                                    add_le_add_right hm (alphaRelaxLowerScalar (α := ℝ) li ui
                                      ai).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hlo2 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  have := NN.MLTheory.CROWN.Proofs.alphaRelaxLowerScalar_sound
                                    (lower := li) (upper := ui) (alpha := ai) (input := zi)
                                    (lowerBound := hzIlo) (upperBound := hzIhi)
                                    (alphaNonnegative := hai0) (alphaAtMostOne := hai1)
                                  simpa [li, ui, zi, ai] using this
                                have hlo :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  simp [hlo_def, hrpLo]
                                  exact le_trans hlo1 hlo2
                                have hhi1 :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have := relu_relax_scalar_upper_real_runtime (l := li) (u := ui)
                                    (x := zi)
                                    (hlx := hzIlo) (hxu := hzIhi)
                                  simpa [li, ui, zi] using this
                                have hhi2 :
                                    (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias
                                      ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have hm :
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi
                                        ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i := by
                                    exact mul_le_mul_of_nonneg_left hzHi hsHi
                                  have h' :=
                                    add_le_add_right hm
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hhi :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤ toVec (boundsEvalAt (α
                                      := ℝ) bout x').hi i := by
                                  simp [hhi_def, hrpHi]
                                  exact le_trans hhi1 hhi2
                                have hrelu : toVec (Activation.reluSpec (α := ℝ) z) i =
                                    Activation.Math.reluSpec (α := ℝ) zi := by
                                  simpa [zi] using (toVec_relu_spec (t := z) (i := i))
                                constructor
                                · rw [hrelu]
                                  exact hlo
                                · rw [hrelu]
                                  exact hhi
                              have this' :
                                  Theorems.Semantics.encloses (boundsEvalAt (α := ℝ) bout x')
                                    (Activation.reluSpec (α := ℝ) z) := by
                                simpa [x'] using this
                              exact sem_encloses_value_eq
                                (B := boundsEvalAt (α := ℝ) bout x')
                                (hxy := hreluCast.symm) this'
                            ·
                              have : False := by
                                simp [hxin, hpre, hαopt, hout] at hs'
                              exact False.elim this

    | .permute _ | .randUniform _ | .bernoulliMask _ | .add | .sub | .mul_elem | .abs | .sqrt |
      .inv | .reluDeriv
    | .maxElem | .minElem | .maxPool2d .. | .maxPool2dPad .. | .avgPool2d .. | .avgPool2dPad
      ..
    | .broadcastTo .. | .reduceSum _ | .reduceMean _ | .conv2d .. | .batchNorm2dNchwEval _
      | .tanh | .sigmoid | .exp | .log | .sin | .cos
    | .softmax _ | .layernorm _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .mseLoss =>
      by
        -- Unsupported ops: `alphaCrownStepNode?` can only succeed via the IBP-derived constant
        -- enclosure.
        cases hib : ibp[id]!
        · simp [stepAlpha, alphaCrownStepNode?, hk, hib] at hs
        · rename_i B0
          have hEnc : CertSoundness.EnclosesBox B0 v := by
            have hlt : id < vals.size := by simpa [hsem.1] using hid
            have hraw := hibp id hlt
            simpa [hib, hv] using hraw
          have hbEq : some (Cert.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) = some b := by
            simpa [stepAlpha, alphaCrownStepNode?, hk, hib] using hs
          cases hbEq
          refine ⟨rfl, ?_⟩
          dsimp [CrownCertSoundness.EnclosesAtInput]
          simp [Cert.boundsConst]
          rcases hEnc with ⟨hdim, hbox⟩
          refine ⟨hdim, ?_⟩
          have hBoxEq :
              boundsEvalAt (α := ℝ)
                  (Cert.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) x =
                { dim := B0.dim, lo := B0.lo, hi := B0.hi } := by
            simpa using
              (boundsEvalAt_bounds_const (inDim := ctx.inputDim) (outDim := B0.dim) (lo := B0.lo)
                (hi := B0.hi)
                (x := x))
          have hBoxEval :
              boundsEvalAt (α := ℝ)
                  (Cert.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) x = B0 := by
            cases B0
            simpa using hBoxEq
          have hbox' :=
            sem_encloses_of_eq (h := hBoxEval.symm) (x := castDimScalar (α := ℝ) hdim.symm v.v) hbox
          have hcast :
              castDimScalar (α := ℝ) (congrArg FlatBox.dim hBoxEval.symm)
                  (castDimScalar (α := ℝ) hdim.symm v.v) =
                castDimScalar (α := ℝ) hdim.symm v.v := by
            exact castDimScalar_self _ (castDimScalar (α := ℝ) hdim.symm v.v)
          exact sem_encloses_value_eq (hxy := hcast) hbox'
    )


end

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
