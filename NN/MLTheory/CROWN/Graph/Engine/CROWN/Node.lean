/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Structural

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/-!
# CROWN Node Transfer

Dispatch from graph operations to their affine transfer rules.
-/

/--
Propagate a single node’s *affine bounds* (lower/upper) given parent bounds.

This is the CROWN/DeepPoly-style transfer step used by `runCROWN`. For node kinds without a
dedicated rule, we fall back to the IBP enclosure (turned into a constant affine bound).
-/
def propagateCROWNNode
  (nodes : Array Node) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α)))
  (bounds : Array (Option (FlatAffineBounds α)))
  (ctx : AffineCtx) (id : Nat) : Array (Option (FlatAffineBounds α)) :=
  let node := nodes[id]!
  let getB (pid : Nat) := (bounds[pid]!)
  match node.kind with
  | .input =>
    if node.id = ctx.inputId then
      bounds.set! id (some (boundsIdentity (α:=α) ctx.inputDim))
    else bounds
  | .const _ =>
    match ps.constVals[id]? with
    | some v =>
      -- Exact constant bounds.
      bounds.set! id (some (boundsConst (α:=α) ctx.inputDim v.n v.v v.v))
    | none => bounds
  | .detach =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some b => bounds.set! id (some b)
      | none => bounds
    | _ => bounds
  | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .permute _ | .maxElem | .minElem | .sin |
    .cos | .reluDeriv
  | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad ..
  | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
    -- Conservative fallback: use IBP box as a constant affine bound (A = 0).
    match ibp[id]! with
    | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
    | none => bounds
  | .add =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.loAff
                hiAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.hiAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .sub =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.hiAff
                hiAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.loAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .linear =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ps.linearWB[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w p.b xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .matmul =>
    match node.parents with
    | p1 :: p2 :: _ =>
      -- General (batched) matmul: use McCormick relaxations per product term.
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some aAff, some bAff, some aBox, some bBox =>
        match Internal.propagateMatmulBounds (α:=α)
          (sA := nodes[p1]!.outShape) (sB := nodes[p2]!.outShape)
              aBox bBox aAff bAff with
        | some out =>
          bounds.set! id (some out)
        | none =>
          match ibp[id]! with
          | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
            Bout.hi))
          | none => bounds
      | _, _, _, _ => bounds
    | p1 :: _ =>
      match getB p1, ps.matmulW[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let zb := Spec.fill (α:=α) 0 (.dim p.m .scalar)
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w zb xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .relu =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          let out := propagateReluBounds (α:=α) preB xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .exp =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateExpBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .log =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateLogBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .inv =>
    -- Reciprocal has an asymptote at zero. IBP leaves this node unresolved when the input
    -- interval crosses zero, so a constant fallback is available only on a valid domain.
    match ibp[id]! with
    | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo Bout.hi))
    | none => bounds
  | .sigmoid =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateSigmoidBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .tanh =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateTanhBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .mul_elem =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some xB, some yB, some Bx, some By =>
        if hxo : xB.outDim = Bx.dim then
          if hyo : yB.outDim = By.dim then
            match propagateMulElemBounds (α:=α) Bx By xB yB hxo hyo with
            | some out => bounds.set! id (some out)
            | none =>
              match ibp[id]! with
              | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
                Bout.hi))
              | none => bounds
          else bounds
        else bounds
      | _, _, _, _ => bounds
    | _ => bounds
  | .sum =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        let onesRow : Tensor α (.dim 1 (.dim xin.outDim .scalar)) :=
          Spec.fill (α := α) Numbers.one (.dim 1 (.dim xin.outDim .scalar))
        let loAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.loAff.A
            c := Spec.matVecMulSpec onesRow xin.loAff.c }
        let hiAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.hiAff.A
            c := Spec.matVecMulSpec onesRow xin.hiAff.c }
        bounds.set! id (some { inDim := xin.inDim, outDim := 1, loAff := loAff, hiAff := hiAff })
      | none => bounds
    | _ => bounds
  | .reshape _ _ =>
    -- Flattened representation preserves order; treat as identity.
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .flatten _ =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .concat _ =>
    -- Exact concatenation on flattened vectors.
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hin : b1.inDim = b2.inDim then
          let b2Lo : AffineVec α b1.inDim b2.outDim :=
            castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
              b2.loAff
          let b2Hi : AffineVec α b1.inDim b2.outDim :=
            castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
              b2.hiAff
          match b1.loAff.A, b1.hiAff.A, b1.loAff.c, b1.hiAff.c, b2Lo.A, b2Hi.A, b2Lo.c, b2Hi.c with
          | .dim A1L, .dim A1U, .dim c1L, .dim c1U, .dim A2L, .dim A2U, .dim c2L, .dim c2U =>
            let outDim := b1.outDim + b2.outDim
            let ALo : Tensor α (.dim outDim (.dim b1.inDim .scalar)) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => A1L i1) (fun i2 => A2L i2) i)
            let AHi : Tensor α (.dim outDim (.dim b1.inDim .scalar)) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => A1U i1) (fun i2 => A2U i2) i)
            let cLo : Tensor α (.dim outDim .scalar) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => c1L i1) (fun i2 => c2L i2) i)
            let cHi : Tensor α (.dim outDim .scalar) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => c1U i1) (fun i2 => c2U i2) i)
            bounds.set! id
              (some
                { inDim := b1.inDim
                  outDim := outDim
                  loAff := { A := ALo, c := cLo }
                  hiAff := { A := AHi, c := cHi } })
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .swap_first_two =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match nodes[p1]!.outShape with
        | .dim m (.dim n rest) =>
          let sIn : Shape := .dim m (.dim n rest)
          if xin.outDim = sIn.size then
            let restSize := Spec.Shape.size rest
            let outDim := xin.outDim
            if h0 : outDim = 0 then
              -- Empty tensor: permutation is trivial.
              bounds.set! id (some xin)
            else
              haveI : NeZero outDim := ⟨h0⟩
              let block := m * restSize
              let perm : Fin outDim → Fin outDim := fun idx =>
                let t := idx.val
                let j := t / block
                let rem := t % block
                let i := rem / restSize
                let k := rem % restSize
                let tIn := i * (n * restSize) + j * restSize + k
                Fin.ofNat outDim tIn
              let loAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.loAff
              let hiAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.hiAff
              bounds.set! id (some { inDim := xin.inDim, outDim := outDim, loAff := loAff, hiAff :=
                hiAff })
          else
            match ibp[id]! with
            | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
            | none => bounds
        | _ =>
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | none => bounds
    | _ => bounds
  | .transpose3dLastTwo =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match nodes[p1]!.outShape with
        | .dim a (.dim b (.dim c .scalar)) =>
          let sIn : Shape := .dim a (.dim b (.dim c .scalar))
          if xin.outDim = sIn.size then
            let outDim := xin.outDim
            if h0 : outDim = 0 then
              bounds.set! id (some xin)
            else
              haveI : NeZero outDim := ⟨h0⟩
              let block := c * b
              let perm : Fin outDim → Fin outDim := fun idx =>
                let t := idx.val
                let i := t / block
                let rem := t % block
                let k := rem / b
                let j := rem % b
                let tIn := i * (b * c) + j * c + k
                Fin.ofNat outDim tIn
              let loAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.loAff
              let hiAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.hiAff
              bounds.set! id (some { inDim := xin.inDim, outDim := outDim, loAff := loAff, hiAff :=
                hiAff })
          else
            match ibp[id]! with
            | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
            | none => bounds
        | _ =>
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | none => bounds
    | _ => bounds
  | .layernorm axis =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if axis = Spec.Shape.rank node.outShape - 1 then
          if hout : xin.outDim = preB.dim then
            bounds.set! id (some (propagateLayernormBoundsLastAxis (α:=α) node.outShape preB xin
              hout))
          else
            bounds
        else
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | _, _ => bounds
    | _ => bounds
  | .softmax axis =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if axis = Spec.Shape.rank node.outShape - 1 then
          if hout : xin.outDim = preB.dim then
            bounds.set! id (some (propagateSoftmaxBoundsLastAxis (α:=α) node.outShape preB xin
              hout))
          else
            bounds
        else
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | _, _ => bounds
    | _ => bounds
  | .mseLoss =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some yAff, some tAff, some yB, some tB =>
        if hout : yAff.outDim = yB.dim then
          if tAff.outDim = tB.dim then
            if hdim : yB.dim = tB.dim then
              if hout2 : yAff.outDim = tAff.outDim then
                if hin : yAff.inDim = tAff.inDim then
                  let yLo : AffineVec α tAff.inDim tAff.outDim :=
                    castAffineIn (α:=α) (n:=yAff.inDim) (n':=tAff.inDim) (m:=tAff.outDim) hin
                      (castAffineOut (α:=α) (n:=yAff.inDim) (m:=yAff.outDim) (m':=tAff.outDim) hout2
                        yAff.loAff)
                  let yHi : AffineVec α tAff.inDim tAff.outDim :=
                    castAffineIn (α:=α) (n:=yAff.inDim) (n':=tAff.inDim) (m:=tAff.outDim) hin
                      (castAffineOut (α:=α) (n:=yAff.inDim) (m:=yAff.outDim) (m':=tAff.outDim) hout2
                        yAff.hiAff)
                  let tHiVec := castDimScalar (α:=α) (n:=tB.dim) (n':=yB.dim) (h:=hdim.symm) tB.hi
                  let tLoVec := castDimScalar (α:=α) (n:=tB.dim) (n':=yB.dim) (h:=hdim.symm) tB.lo
                  let diffLoVec : Tensor α (.dim yB.dim .scalar) := Tensor.subSpec yB.lo tHiVec
                  let diffHiVec : Tensor α (.dim yB.dim .scalar) := Tensor.subSpec yB.hi tLoVec
                  let n := yB.dim
                  let hOutToN : tAff.outDim = n := Eq.trans (Eq.symm hout2) hout
                  let yLoN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN yLo
                  let yHiN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN yHi
                  let tLoN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN tAff.loAff
                  let tHiN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN tAff.hiAff
                  let diffLoAff' : AffineVec α tAff.inDim n :=
                    affSub (α:=α) (n:=tAff.inDim) (m:=n) yLoN tHiN
                  let diffHiAff' : AffineVec α tAff.inDim n :=
                    affSub (α:=α) (n:=tAff.inDim) (m:=n) yHiN tLoN
                  -- Square relaxation on each component of `diff`.
                  let flo := getDimScalarFn (α := α) diffLoVec
                  let fhi := getDimScalarFn (α := α) diffHiVec
                  let slopes_hi : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u => Tensor.scalar (u + l))
                  let bias_hi : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u => Tensor.scalar (-(u * l)))
                  let slopes_lo : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u =>
                        let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                          Numbers.zero
                        Tensor.scalar (Numbers.two * d))
                  let bias_lo : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u =>
                        let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                          Numbers.zero
                        Tensor.scalar (-(d * d)))
                  let sqLoAff :=
                    affApplyDiagSignedLower (α:=α) (inDim:=tAff.inDim) (outDim:=n) slopes_lo
                      bias_lo diffLoAff' diffHiAff'
                  let sqHiAff :=
                    affApplyDiagSignedUpper (α:=α) (inDim:=tAff.inDim) (outDim:=n) slopes_hi
                      bias_hi diffLoAff' diffHiAff'
                  if n > 0 then
                    let nA : α := (n : Nat)
                    let scale : α := Numbers.one / nA
                    let scaleRow : Tensor α (.dim 1 (.dim n .scalar)) :=
                      Spec.fill (α := α) scale (.dim 1 (.dim n .scalar))
                    let outLo : AffineVec α tAff.inDim 1 :=
                      { A := Spec.matMulSpec scaleRow sqLoAff.A
                        c := Spec.matVecMulSpec scaleRow sqLoAff.c }
                    let outHi : AffineVec α tAff.inDim 1 :=
                      { A := Spec.matMulSpec scaleRow sqHiAff.A
                        c := Spec.matVecMulSpec scaleRow sqHiAff.c }
                    bounds.set! id (some { inDim := tAff.inDim, outDim := 1, loAff := outLo, hiAff
                      := outHi })
                  else
                    let z : Tensor α (.dim 1 .scalar) := Spec.fill (α := α) Numbers.zero (.dim 1
                      .scalar)
                    bounds.set! id (some (boundsConst (α := α) ctx.inputDim 1 z z))
                else bounds
              else bounds
            else bounds
          else bounds
        else bounds
      | _, _, _, _ => bounds
    | _ => bounds
  | .conv2d .. =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match ps.conv2dCfg[id]? with
        | some cfg =>
          let convIn := cfg.inC * cfg.inH * cfg.inW
          if _hs : cfg.stride = 0 then
            bounds
          else if hout : xin.outDim = convIn then
            let outH := Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding
            let outW := Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding
            let convAff := affOfConv2d (α:=α) cfg
            let out := propagateLinearBounds (α:=α) (n:=convIn) (m:=cfg.outC * outH * outW)
              convAff.A convAff.c xin hout
            bounds.set! id (some out)
          else bounds
        | none =>
          match ps.linearWB[id]? with
          | some p =>
            if hout : xin.outDim = p.n then
              let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w p.b xin hout
              bounds.set! id (some out)
            else bounds
          | none => bounds
      | none => bounds
    | _ => bounds
  | .batchNorm2dNchwEval .. =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ps.batchNorm2dNchwEval[id]? with
      | some xin, some cfg =>
        match batchNorm2dNchwEvalLinear? (α := α) nodes[p1]!.outShape cfg with
        | some p =>
          if hout : xin.outDim = p.n then
            let out := propagateLinearBounds (α := α) (n := p.n) (m := p.m) p.w p.b xin hout
            bounds.set! id (some out)
          else
            bounds
        | none => bounds
      | _, _ => bounds
    | _ => bounds


end NN.MLTheory.CROWN.Graph
