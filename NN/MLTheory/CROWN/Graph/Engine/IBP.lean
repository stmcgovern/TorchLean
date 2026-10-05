/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Interval Bound Propagation

This module runs the flat graph IBP pass. It computes one interval box per node from input boxes,
constant tensors, and per-op interval transfer rules. The proof layer states the topological and
shape hypotheses; this executable pass is the checker-facing computation they refer to.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/-- IBP propagation for one node using `ParamStore`.

This executable function expects parents to have already been processed. The proof layer makes that
precondition explicit via `TopoSorted`; callers that execute graphs directly should use graphs whose
parents appear before their children.
-/
def propagateIBPNode (nodes : Array Node) (ps : ParamStore α) (boxes : Array (Option (FlatBox α)))
  (id : Nat) : Array (Option (FlatBox α)) :=
  let node := nodes[id]!
  let get! (pid : Nat) := (boxes[pid]!).get!
  match node.kind with
  | .input =>
    match ps.inputBoxes[id]? with
    | some B => boxes.set! id (some B)
    | none   => boxes
  | .const _ =>
    match ps.constVals[id]? with
    | some v => boxes.set! id (some { dim := v.n, lo := v.v, hi := v.v })
    | none   => boxes
  | .detach =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (get! p1))
    | _ => boxes
  | .randUniform _ | .bernoulliMask _ | .reluDeriv =>
    -- Stochastic nodes are treated as *nondeterministic-but-bounded* for verification.
    -- Sound enclosure: U[0,1) ⊆ [0,1], Bernoulli mask ⊆ [0,1]. The ReLU derivative
    -- `1[x > 0]` is in `{0, 1} ⊆ [0,1]` whatever its input, so the same enclosure holds.
    let d := node.outShape.size
    let lo := Spec.fill (α := α) Numbers.zero (.dim d .scalar)
    let hi := Spec.fill (α := α) Numbers.one (.dim d .scalar)
    boxes.set! id (some { dim := d, lo := lo, hi := hi })
  | .add =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (box_add (get! p1) (get! p2)))
    | _ => boxes
  | .sub =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (box_sub (get! p1) (get! p2)))
    | _ => boxes
  | .abs =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (boxAbs (α := α) (get! p1)))
    | _ => boxes
  | .sqrt =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (boxSqrt (α := α) (get! p1)))
    | _ => boxes
  | .inv =>
    match node.parents with
    | p1 :: _ =>
        match boxInv? (α := α) (get! p1) with
        | some B => boxes.set! id (some B)
        | none => boxes
    | _ => boxes
  | .maxElem =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (boxMaxElem (α := α) (get! p1) (get! p2)))
    | _ => boxes
  | .minElem =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (boxMinElem (α := α) (get! p1) (get! p2)))
    | _ => boxes
  | .maxPool2d kH kW stride =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Spec.Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                  (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .maxPool2dPad kH kW stride padding =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Spec.Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                  (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .avgPool2d kH kW stride =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Spec.Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                  (h1 := hkH) (h2 := hkW) (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                (h1 := hkH) (h2 := hkW) (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .avgPool2dPad kH kW stride padding =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Spec.Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                  (h1 := hkH) (h2 := hkW) (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                (h1 := hkH) (h2 := hkW) (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .broadcastTo s₁ s₂ =>
    match node.parents with
    | p1 :: _ =>
      match ibpBroadcastTo (α := α) s₁ s₂ (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .reduceSum axis =>
    match node.parents with
    | p1 :: _ =>
      let s := nodes[p1]!.outShape
      match ibpReduceSumAxis (α := α) axis (get! p1) s with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .reduceMean axis =>
    match node.parents with
    | p1 :: _ =>
      let s := nodes[p1]!.outShape
      match ibpReduceMeanAxis (α := α) axis (get! p1) s with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .relu =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (box_relu (get! p1)))
    | _ => boxes
  | .linear =>
    match node.parents with
    | p1 :: _ =>
      match ibp_linear (α:=α) id ps (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none    => boxes
    | _ => boxes
  | .matmul =>
    match node.parents with
    | p1 :: p2 :: _ =>
      let A := get! p1
      let B := get! p2
      let sA := nodes[p1]!.outShape
      let sB := nodes[p2]!.outShape
      let dyn2D? : Option (FlatBox α) :=
        match sA, sB with
        | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
          if hk : k = k' then
            match hk with
            | rfl =>
              if hA : A.dim = m * k then
                if hB : B.dim = k * n then
                  let outDim := m * n
                  let loT : Tensor α (.dim outDim .scalar) :=
                    Tensor.dim (fun idx =>
                      let t := idx.val
                      let i := t / n
                      let j := t % n
                      let (sumLo, _sumHi) :=
                        (List.range k).foldl (fun (acc : α × α) kk =>
                          let (accLo, accHi) := acc
                          let aLo := getAtOrZero A.lo [i * k + kk]
                          let aHi := getAtOrZero A.hi [i * k + kk]
                          let bLo := getAtOrZero B.lo [kk * n + j]
                          let bHi := getAtOrZero B.hi [kk * n + j]
                          let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                          (accLo + pLo, accHi + pHi)
                        ) (0, 0)
                      Tensor.scalar sumLo)
                  let hiT : Tensor α (.dim outDim .scalar) :=
                    Tensor.dim (fun idx =>
                      let t := idx.val
                      let i := t / n
                      let j := t % n
                      let (_sumLo, sumHi) :=
                        (List.range k).foldl (fun (acc : α × α) kk =>
                          let (accLo, accHi) := acc
                          let aLo := getAtOrZero A.lo [i * k + kk]
                          let aHi := getAtOrZero A.hi [i * k + kk]
                          let bLo := getAtOrZero B.lo [kk * n + j]
                          let bHi := getAtOrZero B.hi [kk * n + j]
                          let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                          (accLo + pLo, accHi + pHi)
                        ) (0, 0)
                      Tensor.scalar sumHi)
                  some { dim := outDim, lo := loT, hi := hiT }
                else none
              else none
          else none
        | _, _ => none
      let dyn3D? : Option (FlatBox α) :=
        match sA, sB with
        | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
          if hb : b = b' then
            match hb with
            | rfl =>
              if hk : k = k' then
                match hk with
                | rfl =>
                  if hA : A.dim = b * m * k then
                    if hB : B.dim = b * k * n then
                      let outDim := b * m * n
                      let block : Nat := m * n
                      let strideA : Nat := m * k
                      let strideB : Nat := k * n
                      let loT : Tensor α (.dim outDim .scalar) :=
                        Tensor.dim (fun idx =>
                          let t := idx.val
                          let bi := t / block
                          let rem := t % block
                          let i := rem / n
                          let j := rem % n
                          let baseA := bi * strideA
                          let baseB := bi * strideB
                          let (sumLo, _sumHi) :=
                            (List.range k).foldl (fun (acc : α × α) kk =>
                              let (accLo, accHi) := acc
                              let aLo := getAtOrZero A.lo [baseA + i * k + kk]
                              let aHi := getAtOrZero A.hi [baseA + i * k + kk]
                              let bLo := getAtOrZero B.lo [baseB + kk * n + j]
                              let bHi := getAtOrZero B.hi [baseB + kk * n + j]
                              let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                              (accLo + pLo, accHi + pHi)
                            ) (0, 0)
                          Tensor.scalar sumLo)
                      let hiT : Tensor α (.dim outDim .scalar) :=
                        Tensor.dim (fun idx =>
                          let t := idx.val
                          let bi := t / block
                          let rem := t % block
                          let i := rem / n
                          let j := rem % n
                          let baseA := bi * strideA
                          let baseB := bi * strideB
                          let (_sumLo, sumHi) :=
                            (List.range k).foldl (fun (acc : α × α) kk =>
                              let (accLo, accHi) := acc
                              let aLo := getAtOrZero A.lo [baseA + i * k + kk]
                              let aHi := getAtOrZero A.hi [baseA + i * k + kk]
                              let bLo := getAtOrZero B.lo [baseB + kk * n + j]
                              let bHi := getAtOrZero B.hi [baseB + kk * n + j]
                              let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                              (accLo + pLo, accHi + pHi)
                            ) (0, 0)
                          Tensor.scalar sumHi)
                      some { dim := outDim, lo := loT, hi := hiT }
                    else none
                  else none
              else none
          else none
        | _, _ => none
      match dyn2D?, dyn3D? with
      | some yB, _ => boxes.set! id (some yB)
      | none, some yB => boxes.set! id (some yB)
      | none, none => boxes
    | p1 :: _ =>
      match ibp_matmul (α:=α) id ps (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none    => boxes
    | _ => boxes
  | .reshape _ _ =>
    match node.parents with
    | p1 :: _ => boxes.set! id (boxes[p1]!)
    | _ => boxes
  | .flatten _ =>
    match node.parents with
    | p1 :: _ => boxes.set! id (boxes[p1]!)
    | _ => boxes
  | .swap_first_two =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim m (.dim n rest) =>
        let sIn : Shape := .dim m (.dim n rest)
        if hdim : Xin.dim = sIn.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = sIn.size := by
            simp [sFlat, sIn, Spec.Shape.size, hdim]
          let xLo : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
          let xHi : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
          let yLoT : Tensor α (.dim n (.dim m rest)) := Tensor.swapFirstTwoSpec (α:=α) xLo
          let yHiT : Tensor α (.dim n (.dim m rest)) := Tensor.swapFirstTwoSpec (α:=α) xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id (some { dim := (Shape.dim n (Shape.dim m rest)).size, lo := flatLo, hi :=
            flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .transpose3dLastTwo =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim a (.dim b (.dim c .scalar)) =>
        let sIn : Shape := .dim a (.dim b (.dim c .scalar))
        if hdim : Xin.dim = sIn.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = sIn.size := by
            simp [sFlat, sIn, Spec.Shape.size, hdim]
          let xLo : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
          let xHi : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
          let yLoT : Tensor α (.dim a (.dim c (.dim b .scalar))) :=
            Tensor.transpose3DLastTwoSpec (α:=α) xLo
          let yHiT : Tensor α (.dim a (.dim c (.dim b .scalar))) :=
            Tensor.transpose3DLastTwoSpec (α:=α) xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id
            (some { dim := (Shape.dim a (Shape.dim c (Shape.dim b Shape.scalar))).size,
                    lo := flatLo,
                    hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .permute perm =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let sIn := nodes[p1]!.outShape
      if hdim : Xin.dim = sIn.size then
        let sFlat : Shape := .dim Xin.dim .scalar
        have hsize : sFlat.size = sIn.size := by
          simp [sFlat, sIn, Spec.Shape.size, hdim]
        let xLo : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
        let xHi : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
        match permuteDVal? (α := α) (v := ⟨sIn, xLo⟩) perm, permuteDVal? (α := α) (v := ⟨sIn, xHi⟩)
          perm with
        | some yLoV, some yHiV =>
            let sOut := flatDValShape (α := α) yLoV
            if hSame : flatDValShape (α := α) yHiV = sOut then
              if hOut : sOut = node.outShape then
                let yLoSOut : Tensor α sOut := flatDValTensor (α := α) yLoV
                let yHiSOut : Tensor α sOut := hSame ▸ flatDValTensor (α := α) yHiV
                let yLoT : Tensor α node.outShape := hOut ▸ yLoSOut
                let yHiT : Tensor α node.outShape := hOut ▸ yHiSOut
                let flatLo := Tensor.flattenSpec (α:=α) yLoT
                let flatHi := Tensor.flattenSpec (α:=α) yHiT
                boxes.set! id (some { dim := node.outShape.size, lo := flatLo, hi := flatHi })
              else
                boxes
            else
              boxes
        | _, _ => boxes
      else
        boxes
    | _ => boxes
  | .mul_elem =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match box_mul_elem (α:=α) (get! p1) (get! p2) with
      | some prod => boxes.set! id (some prod)
      | none => boxes
    | _ => boxes
  | .sum =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let loVal := Spec.Tensor.sumSpec Xin.lo
      let hiVal := Spec.Tensor.sumSpec Xin.hi
      let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
      let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
      boxes.set! id (some { dim := 1, lo := loT, hi := hiT })
    | _ => boxes
  | .mseLoss =>
    match node.parents with
    | p1 :: p2 :: _ =>
      let Y := get! p1
      let T := get! p2
      if hdim : Y.dim = T.dim then
        let Thi := castDimScalar (α:=α) (n:=T.dim) (n':=Y.dim) (h:=hdim.symm) T.hi
        let Tlo := castDimScalar (α:=α) (n:=T.dim) (n':=Y.dim) (h:=hdim.symm) T.lo
        let diff : FlatBox α :=
          { dim := Y.dim
            lo := Tensor.subSpec Y.lo Thi
            hi := Tensor.subSpec Y.hi Tlo }
        let sq := boxSquare (α:=α) diff
        let n := sq.dim
        if hn : n > 0 then
          let nA : α := (n : Nat)
          let loVal := (Spec.Tensor.sumSpec sq.lo) / nA
          let hiVal := (Spec.Tensor.sumSpec sq.hi) / nA
          let loT := Spec.fill (α:=α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α:=α) hiVal (.dim 1 .scalar)
          boxes.set! id (some { dim := 1, lo := loT, hi := hiT })
        else
          boxes
      else boxes
    | _ => boxes
  | .conv2d .. =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match ibpConv2dNode (α:=α) id ps Xin with
      | some yB => boxes.set! id (some yB)
      | none =>
        -- Fallback: allow callers to inject a flattened linear form when conv params are absent.
        match ibp_linear (α:=α) id ps Xin with
        | some yB => boxes.set! id (some yB)
        | none    => boxes
    | _ => boxes
  | .batchNorm2dNchwEval .. =>
    match node.parents with
    | p1 :: _ =>
      match ps.batchNorm2dNchwEval[id]? with
      | some cfg =>
        match batchNorm2dNchwEvalLinear? (α := α) nodes[p1]!.outShape cfg with
        | some p =>
          let Xin := get! p1
          match ibpLinearParams (α := α) p Xin with
          | some yB => boxes.set! id (some yB)
          | none => boxes
        | none => boxes
      | none => boxes
    | _ => boxes
  | .exp =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      -- exp is monotone increasing: apply to lo and hi
      let lo := Tensor.expSpec Xin.lo
      let hi := Tensor.expSpec Xin.hi
      boxes.set! id (some { dim := Xin.dim, lo := lo, hi := hi })
    | _ => boxes
  | .log =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let flo := getDimScalarFn (α:=α) Xin.lo
      -- Raw IR `log` rejects nonpositive values. An interval crossing that boundary therefore has
      -- no valid transfer result; callers can use the explicitly totalized safe-log operator when
      -- epsilon clamping is intended.
      if (List.finRange Xin.dim).all (fun i =>
          match flo i with
          | .scalar v => decide (Numbers.zero < v)) then
        let lo := Tensor.logSpec Xin.lo
        let hi := Tensor.logSpec Xin.hi
        boxes.set! id (some { dim := Xin.dim, lo := lo, hi := hi })
      else
        boxes
    | _ => boxes
  -- layernorm/concat handled in dedicated cases below
  | .concat _ =>
    -- Concatenate two flattened boxes along the vector dimension
    match node.parents with
    | p1 :: p2 :: _ =>
      let B1 := get! p1; let B2 := get! p2
      match B1, B2 with
      | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
        let f1lo := getDimScalarFn (α:=α) lo1
        let f2lo := getDimScalarFn (α:=α) lo2
        let f1hi := getDimScalarFn (α:=α) hi1
        let f2hi := getDimScalarFn (α:=α) hi2
        let lo :=
          Tensor.dim (fun i =>
            Fin.addCases (fun i1 => f1lo i1) (fun i2 => f2lo i2) i)
        let hi :=
          Tensor.dim (fun i =>
            Fin.addCases (fun i1 => f1hi i1) (fun i2 => f2hi i2) i)
        boxes.set! id (some { dim := n1 + n2, lo := lo, hi := hi })
    | _ => boxes
  | .layernorm axis =>
    -- Last-axis LayerNorm bounds (without affine gamma/beta).
    -- We only implement `axis = rank-1` (the TorchLean usage); other axes are left unsupported.
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let s := node.outShape
      if axis = Spec.Shape.rank s - 1 then
        if hdim : Xin.dim = s.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = s.size := by
            simp [sFlat, Spec.Shape.size, hdim]
          let xLo : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.lo hsize
          let xHi : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.hi hsize
          let (yLoT, yHiT) := ibpLayernormLastTensor (α:=α) (s := s) xLo xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id (some { dim := s.size, lo := flatLo, hi := flatHi })
        else boxes
      else boxes
    | _ => boxes
  | .softmax axis =>
    -- Last-axis softmax bounds. We only implement `axis = rank-1`.
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let s := node.outShape
      if axis = Spec.Shape.rank s - 1 then
        if hdim : Xin.dim = s.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = s.size := by
            simp [sFlat, Spec.Shape.size, hdim]
          let xLo : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.lo hsize
          let xHi : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.hi hsize
          let (yLoT, yHiT) := ibpSoftmaxLastTensor (α:=α) (s := s) xLo xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id (some { dim := s.size, lo := flatLo, hi := flatHi })
        else boxes
      else boxes
    | _ => boxes
  | .tanh =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))
  | .sigmoid =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))
  | .sin =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))
  | .cos =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))

/-- Run an IBP pass over the whole graph. Caller seeds inputs via ParamStore.inputBoxes. -/
def runIBP (g : Graph) (ps : ParamStore α) : Array (Option (FlatBox α)) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl (fun acc i => propagateIBPNode (α:=α) g.nodes ps acc i) init

end NN.MLTheory.CROWN.Graph
