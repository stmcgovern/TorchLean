/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Helpers

/-!
# IR Node Lowering

The exhaustive checked compiler loop from operation-tagged IR nodes to executable SSA nodes.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

namespace IRExec

/--
Compile the IR graph starting at node index `i`, extending the current SSA `State`.

This is the main compiler loop:
- it checks `i < g.nodes.size`,
- compiles node `i` into a `NodeData.forward` closure (rejecting unsupported ops/shapes), and
- `snoc`s the resulting node into the accumulating `GraphData`.

The public entrypoint `execGraphOfIR` handles node 0 and calls `buildFrom` starting at `i = 1`.

Operationally, `buildFrom` is a checked compiler:
- success means every visited node had well-typed parents and a supported lowering case,
- failure returns a concrete error explaining the first unsupported/malformed node.
-/
def buildFrom
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape)
    (i : Nat) (st : State α inShape) : Except String (State α inShape) := do
  let ⟨ss, gd⟩ := st
  if h : i < g.nodes.size then
    let n ← g.getNode i
    let τ : Shape := n.outShape

    -- Helper: build a typed parent index expecting a specific shape.
    let parentIdx (pid : Nat) (s : Shape) : Except String (Idx ([inShape] ++ ss) s) :=
      mkIdx (inShape := inShape) (ss := ss) pid s

    let fwd (forward : TList α ([inShape] ++ ss) → Tensor α τ) : NodeData α Unit
        ([inShape] ++ ss) τ :=
      mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward

    let nodeData : NodeData α Unit ([inShape] ++ ss) τ ←
      match n.kind with
      | .input =>
          throw s!"IRExec: internal error (handled above)"
      | .const s =>
          let t ← NN.IR.Graph.evalConst (α := α) (payload := payload) (id := n.id) (s := s)
          if hOut : s = τ then
            -- Cast so the node is typed at the declared outShape.
            pure <| fwd (fun _ctx => hOut ▸ t)
          else
            throw s!"IRExec: const node {i}: outShape mismatch: kind={repr s}, declared={repr τ}"
      | .permute perm =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let sIn := pNode.outShape
              let ip ← parentIdx pId sIn
              match Spec.Shape.permute? sIn perm with
              | none =>
                  throw s!"IRExec: node {i}: invalid permutation {repr perm} for shape {repr sIn}"
              | some expected =>
                  let swaps ← NN.IR.Graph.swapDepthsForPerm perm (Spec.Shape.rank sIn)
                  let sFinal : Shape := swapShapeBySwaps sIn swaps
                  if hFinal : sFinal = expected then
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let x := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α sFinal := applySwapsTensor (α := α) (s := sIn) (swaps :=
                          swaps) x
                        let yExpected : Tensor α expected := Tensor.castShape y hFinal
                        Tensor.castShape yExpected hOut
                      pure <| fwd forward
                    else
                      throw <|
                        s!"IRExec: node {i}: permute outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ}"
                  else
                    throw <|
                      s!"IRExec: node {i}: permute shape mismatch: computed={repr sFinal}, " ++
                        s!"expected={repr expected} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: permute expects 1 parent ({n.summary})"
      | .detach =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              if hOut : s = τ then
                let forward := fun ctx : TList α ([inShape] ++ ss) =>
                  hOut ▸ (getIdx (α := α) (xs := ctx) ip)
                pure <| fwd forward
              else
                throw s!"IRExec: node {i}: detach expects outShape=parent.outShape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: detach expects 1 parent ({n.summary})"
      | .randUniform seed =>
          match n.parents with
          | [] =>
              let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
              let t : Tensor α τ := Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := τ)
              pure <| fwd (fun _ctx => t)
          | _ => throw s!"IRExec: node {i}: rand_uniform expects 0 parents ({n.summary})"
      | .bernoulliMask seed =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId Shape.scalar
              let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                let kpT := getIdx (α := α) (xs := ctx) ip
                let kp : α :=
                  match kpT with
                  | Tensor.scalar v => v
                Runtime.Autograd.TorchLean.Random.mask (α := α) key kp (s := τ)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: bernoulli_mask expects 1 parent ({n.summary})"
      | .add =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.addSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: add expects 2 parents ({n.summary})"
      | .sub =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.subSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: sub expects 2 parents ({n.summary})"
      | .mul_elem =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.mulSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: mul_elem expects 2 parents ({n.summary})"
      | .abs =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.absSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: abs expects 1 parent ({n.summary})"
      | .sqrt =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.sqrtSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: sqrt expects 1 parent ({n.summary})"
      | .inv =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.invSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: inv expects 1 parent ({n.summary})"
      | .maxElem =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.maxSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: max_elem expects 2 parents ({n.summary})"
      | .minElem =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.minSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: min_elem expects 2 parents ({n.summary})"
      | .maxPool2d kH kW stride =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: max_pool2d requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: max_pool2d requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: max_pool2d requires stride ≠ 0 ({n.summary})"
                  else
                    let _ ← NN.IR.OpContracts.checkWindowFits "max_pool2d" "height" inH kH 0
                    let _ ← NN.IR.OpContracts.checkWindowFits "max_pool2d" "width" inW kW 0
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                    let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                            (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| fwd forward
                    else
                      throw <|
                        s!"IRExec: node {i}: max_pool2d outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: max_pool2d expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: max_pool2d expects 1 parent ({n.summary})"
      | .maxPool2dPad kH kW stride padding =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: max_pool2d_pad requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: max_pool2d_pad requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: max_pool2d_pad requires stride ≠ 0 ({n.summary})"
                  else
                    let _ ← NN.IR.OpContracts.checkWindowFits "max_pool2d_pad" "height" inH kH padding
                    let _ ← NN.IR.OpContracts.checkWindowFits "max_pool2d_pad" "width" inW kW padding
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride
                      padding
                    let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                              padding)
                            (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| fwd forward
                    else
                      throw <|
                        s!"IRExec: node {i}: max_pool2d_pad outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: max_pool2d_pad expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: max_pool2d_pad expects 1 parent ({n.summary})"
      | .avgPool2d kH kW stride =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d requires stride ≠ 0 ({n.summary})"
                  else
                    let _ ← NN.IR.OpContracts.checkWindowFits "avg_pool2d" "height" inH kH 0
                    let _ ← NN.IR.OpContracts.checkWindowFits "avg_pool2d" "width" inW kW 0
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                    let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                            (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| fwd forward
                    else
                      throw <|
                        s!"IRExec: node {i}: avg_pool2d outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: avg_pool2d expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: avg_pool2d expects 1 parent ({n.summary})"
      | .avgPool2dPad kH kW stride padding =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d_pad requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d_pad requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d_pad requires stride ≠ 0 ({n.summary})"
                  else
                    let _ ← NN.IR.OpContracts.checkWindowFits "avg_pool2d_pad" "height" inH kH padding
                    let _ ← NN.IR.OpContracts.checkWindowFits "avg_pool2d_pad" "width" inW kW padding
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride
                      padding
                    let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                              padding)
                            (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| fwd forward
                    else
                      throw <|
                        s!"IRExec: node {i}: avg_pool2d_pad outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: avg_pool2d_pad expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: avg_pool2d_pad expects 1 parent ({n.summary})"
      | .broadcastTo s₁ s₂ =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId s₁
              match NN.IR.OpContracts.mkCanBroadcastTo? s₁ s₂ with
              | none =>
                  throw s!"IRExec: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
              | some cb =>
                  if hOut : s₂ = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      hOut ▸ Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x
                    pure <| fwd forward
                  else
                    throw <|
                      s!"IRExec: node {i}: broadcastTo outShape mismatch: kind={repr s₂}, " ++
                        s!"declared={repr τ}"
          | _ => throw s!"IRExec: node {i}: broadcastTo expects 1 parent ({n.summary})"
      | .reduceSum axis =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              match Spec.Shape.validAxis? (axis := axis) s with
              | none =>
                  throw s!"IRExec: node {i}: reduce_sum invalid axis={axis} for shape {repr s}"
              | some hAxis =>
                  let hRed := Shape.proveReducibleAlong axis s hAxis.down
                  let expected : Shape := Spec.Tensor.shapeAfterSum s axis
                  if hOut : expected = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      let y : Tensor α expected := Tensor.reduceSum (α := α) (s := s) axis x hRed
                      hOut ▸ y
                    pure <| fwd forward
                  else
                    throw <|
                      s!"IRExec: node {i}: reduce_sum outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: reduce_sum expects 1 parent ({n.summary})"
      | .reduceMean axis =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              match Spec.Shape.validAxis? (axis := axis) s with
              | none =>
                  throw s!"IRExec: node {i}: reduce_mean invalid axis={axis} for shape {repr s}"
              | some hAxis =>
                  let hRed := Shape.proveReducibleAlong axis s hAxis.down
                  let expected : Shape := Spec.Tensor.shapeAfterSum s axis
                  if hOut : expected = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      let y : Tensor α expected := Tensor.reduceMean (α := α) (s := s) axis x hRed
                      hOut ▸ y
                    pure <| fwd forward
                  else
                    throw <|
                      s!"IRExec: node {i}: reduce_mean outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: reduce_mean expects 1 parent ({n.summary})"
      | .sum =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              if hOut : Shape.scalar = τ then
                let forward := fun ctx : TList α ([inShape] ++ ss) =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  hOut ▸ Tensor.scalar (Tensor.sumSpec (α := α) x)
                pure <| fwd forward
              else
                throw s!"IRExec: node {i}: sum expects scalar outShape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: sum expects 1 parent ({n.summary})"
      | .matmul =>
          match n.parents with
          | [aId, bId] =>
              let aNode ← g.getNode aId
              let bNode ← g.getNode bId
              match aNode.outShape, bNode.outShape with
              | .dim m (.dim nDim Shape.scalar), .dim n' (.dim p Shape.scalar) =>
                  if _hn : nDim = n' then
                    let ia ← parentIdx aId (.dim m (.dim nDim .scalar))
                    let ib ← parentIdx bId (.dim nDim (.dim p .scalar))
                    let expected : Shape := .dim m (.dim p .scalar)
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let aT := getIdx (α := α) (xs := ctx) ia
                        let bT := getIdx (α := α) (xs := ctx) ib
                        let y : Tensor α expected := Spec.matMulSpec (α := α) (m := m) (n :=
                          nDim) (p := p) aT bT
                        hOut ▸ y
                      pure <| fwd forward
                    else
                      throw <|
                        s!"IRExec: node {i}: matmul outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
                  else
                    throw s!"IRExec: node {i}: matmul inner dims mismatch: {nDim} vs {n'}"
              | .dim batch (.dim m (.dim nDim Shape.scalar)), .dim batch' (.dim n' (.dim p
                Shape.scalar)) =>
                  if _hb : batch = batch' then
                    if _hn : nDim = n' then
                      let ia ← parentIdx aId (.dim batch (.dim m (.dim nDim .scalar)))
                      let ib ← parentIdx bId (.dim batch (.dim nDim (.dim p .scalar)))
                      let expected : Shape := .dim batch (.dim m (.dim p .scalar))
                      if hOut : expected = τ then
                        let forward := fun ctx : TList α ([inShape] ++ ss) =>
                          let aT := getIdx (α := α) (xs := ctx) ia
                          let bT := getIdx (α := α) (xs := ctx) ib
                          let y : Tensor α expected :=
                            Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := nDim) (p :=
                              p) aT bT
                          hOut ▸ y
                        pure <| fwd forward
                      else
                        throw <|
                          s!"IRExec: node {i}: bmm outShape mismatch: " ++
                            s!"expected={repr expected}, declared={repr τ} ({n.summary})"
                    else
                      throw s!"IRExec: node {i}: matmul inner dims mismatch: {nDim} vs {n'}"
                  else
                    throw s!"IRExec: node {i}: matmul batch dims mismatch: {batch} vs {batch'}"
              | _, _ =>
                  throw <|
                    s!"IRExec: node {i}: unsupported matmul shapes: {repr aNode.outShape} · " ++
                      s!"{repr bNode.outShape}"
          | _ => throw s!"IRExec: node {i}: matmul expects 2 parents ({n.summary})"
      | .linear =>
          match n.parents with
          | [xId] =>
              match payload.linear? n.id with
              | none => throw s!"IRExec: missing linear payload for node {n.id}"
              | some p =>
                  let expectedIn : Shape := .dim p.inDim .scalar
                  let expectedOut : Shape := .dim p.outDim .scalar
                  let ix ← parentIdx xId expectedIn
                  if hOut : expectedOut = τ then
                    let W := p.W
                    let b := p.b
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ix
                      let y : Tensor α expectedOut :=
                        Tensor.addSpec (α := α)
                          (Spec.matVecMulSpec (α := α) (m := p.outDim) (n := p.inDim) W x) b
                      hOut ▸ y
                    pure <| fwd forward
                  else
                    throw <|
                      s!"IRExec: linear {n.id}: declared outShape mismatch: {repr τ} vs " ++
                        s!"expected {repr expectedOut}"
          | _ => throw s!"IRExec: node {i}: linear expects 1 parent ({n.summary})"
      | .conv2d inC outC kH kW stride padding =>
          match n.parents with
          | [xId] =>
              match payload.conv2d? n.id with
              | none => throw s!"IRExec: missing conv2d payload for node {n.id}"
              | some cfg =>
                  -- The payload stores the Conv2d dimensions used to rebuild the layer. `parentIdx`
                  -- checks that the parent has exactly that input shape, and `hOut` checks the
                  -- declared output shape below.
                  let expectedIn : Shape := .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))
                  let _ ←
                    OpContracts.inferConv2dCHWOutShape cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride
                      cfg.padding expectedIn
                  let ix ← parentIdx xId expectedIn
                  let outH : Nat := Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding
                  let outW : Nat := Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding
                  let expected : Shape := .dim cfg.outC (.dim outH (.dim outW .scalar))
                  if hOut : expected = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIRValue (α := α) (ctx := ctx) ix
                      let y : Tensor α expected := Spec.conv2dSpec (α := α) (layer := cfg.spec)
                          (input := x)
                      hOut ▸ y
                    pure <| fwd forward
                  else
                    throw <|
                      s!"IRExec: node {i}: conv2d outShape mismatch: expected={repr expected}, " ++
                        s!"declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: conv2d expects 1 parent ({n.summary})"
      | .batchNorm2dNchwEval channels =>
          match n.parents with
          | [xId] =>
              match payload.batchNorm2dNchwEval? n.id with
              | none => throw s!"IRExec: missing batch_norm2d_nchw_eval payload for node {n.id}"
              | some cfg =>
                  match τ with
                  | .dim nBatch (.dim c (.dim h (.dim w .scalar))) =>
                      if _hc : c = cfg.c then
                        let expected : Shape := .dim nBatch (.dim cfg.c (.dim h (.dim w
                          .scalar)))
                        let ix ← parentIdx xId expected
                        if hOut : expected = τ then
                          let forward := fun ctx : TList α ([inShape] ++ ss) =>
                            let x := getIRValue (α := α) (ctx := ctx) ix
                            let y : Tensor α expected :=
                              NN.IR.Graph.batchNorm2dEvalTensor (α := α) cfg x
                            hOut ▸ y
                          pure <| fwd forward
                        else
                          throw <|
                            s!"IRExec: node {i}: batch_norm2d_nchw_eval outShape mismatch: " ++
                              s!"expected={repr expected}, declared={repr τ} ({n.summary})"
                      else
                        throw <|
                          s!"IRExec: node {i}: batch_norm2d_nchw_eval channel mismatch: " ++
                            s!"op={channels}, declared={c}, payload={cfg.c} ({n.summary})"
                  | _ =>
                      throw s!"IRExec: node {i}: batch_norm2d_nchw_eval expects NCHW outShape"
          | _ =>
              throw s!"IRExec: node {i}: batch_norm2d_nchw_eval expects 1 parent ({n.summary})"
      | .relu =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: relu expects 1 parent ({n.summary})"
      | .reluDeriv =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.reluDerivSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: relu_deriv expects 1 parent ({n.summary})"
      | .tanh =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: tanh expects 1 parent ({n.summary})"
      | .sigmoid =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: sigmoid expects 1 parent ({n.summary})"
      | .exp =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.expSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: exp expects 1 parent ({n.summary})"
      | .log =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                let x := getIdx (α := α) (xs := ctx) ip
                -- This compiled forward closure is pure, so it cannot return the eager engine's
                -- `Except` error. A bad raw-log domain reaches a runtime panic; use `safe_log` for
                -- total epsilon protection.
                if Tensor.allSpec (α := α) (s := τ) (fun v => decide (v > (0 : α))) x then
                  Tensor.logSpec (α := α) x
                else
                  panic! "IRExec: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: log expects 1 parent ({n.summary})"
      | .sin =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.sin x) (getIdx (α := α)
                  (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: sin expects 1 parent ({n.summary})"
      | .cos =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.cos x) (getIdx (α := α)
                  (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: cos expects 1 parent ({n.summary})"
      | .softmax axis =>
          match n.parents with
          | [pId] => do
              -- The runtime primitive is last-axis softmax. We keep the compiler disciplined and
              -- reject any request for a non-last axis (callers can insert an explicit `.permute`
              -- node if they want to model a different axis).
              OpContracts.checkLastAxis "softmax" axis τ
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.softmaxSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| fwd forward
          | _ => throw s!"IRExec: node {i}: softmax expects 1 parent ({n.summary})"
      | .layernorm axis =>
          match n.parents with
          | [pId] => do
              let (seqLen, embedDim) ←
                match OpContracts.layerNorm2DParams axis τ with
                | .ok p => pure p
                | .error msg => throw s!"IRExec: node {i}: layernorm: {msg} ({n.summary})"
              let view2D : Shape := .dim seqLen (.dim embedDim .scalar)
              if hNumel : Spec.Shape.size τ = Spec.Shape.size view2D then
                if hSeq : seqLen > 0 then
                  if hEmb : embedDim > 0 then
                    let ip ← parentIdx pId τ
                    let gamma : Tensor α (.dim embedDim .scalar) :=
                      Spec.fill (α := α) 1 (.dim embedDim .scalar)
                    let beta : Tensor α (.dim embedDim .scalar) :=
                      Spec.fill (α := α) 0 (.dim embedDim .scalar)
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x : Tensor α τ := getIdx (α := α) (xs := ctx) ip
                      let x2D : Tensor α view2D :=
                        Tensor.reshapeSpec (α := α) (s₁ := τ) (s₂ := view2D) x hNumel
                      let y2D : Tensor α view2D :=
                        Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                          (x := x2D) (gamma := gamma) (beta := beta)
                          (h_seq_pos := hSeq) (h_embed_pos := hEmb)
                      Tensor.reshapeSpec (α := α) (s₁ := view2D) (s₂ := τ) y2D hNumel.symm
                    pure <| fwd forward
                  else
                    throw s!"IRExec: node {i}: layernorm embedDim must be > 0 (got {embedDim})"
                else
                  throw s!"IRExec: node {i}: layernorm seqLen must be > 0 (got {seqLen})"
              else
                throw <|
                  s!"IRExec: node {i}: layernorm internal error: bad reshape sizes " ++
                    s!"({Spec.Shape.size τ} vs {Spec.Shape.size view2D}) ({n.summary})"
          | _ =>
              throw s!"IRExec: node {i}: layernorm expects 1 parent ({n.summary})"
      | .reshape inS outS =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId inS
              if hNumel : Spec.Shape.size inS = Spec.Shape.size outS then
                if hOut : outS = τ then
                  let forward := fun ctx : TList α ([inShape] ++ ss) =>
                    let x := getIdx (α := α) (xs := ctx) ip
                    hOut ▸ Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) x hNumel
                  pure <| fwd forward
                else
                  throw <|
                    s!"IRExec: node {i}: reshape outShape mismatch: kind={repr outS}, " ++
                      s!"declared={repr τ}"
              else
                throw <|
                  s!"IRExec: node {i}: reshape numel mismatch: {Spec.Shape.size inS} vs " ++
                    s!"{Spec.Shape.size outS}"
          | _ => throw s!"IRExec: node {i}: reshape expects 1 parent ({n.summary})"
      | .flatten s =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId s
              let expected : Shape := .dim (Spec.Shape.size s) .scalar
              if hOut : expected = τ then
                let forward := fun ctx : TList α ([inShape] ++ ss) =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  let y : Tensor α expected := Tensor.flattenSpec (α := α) (s := s) x
                  hOut ▸ y
                pure <| fwd forward
              else
                throw <|
                  s!"IRExec: node {i}: flatten outShape mismatch: " ++
                    s!"expected={repr expected}, declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: flatten expects 1 parent ({n.summary})"
      | .concat axis =>
          let parents := n.parents
          if parents.length < 2 then
            throw s!"IRExec: node {i}: concat expects at least 2 parents"

          let parentShapes : List Shape ← parents.mapM (fun pid => do
            let pNode ← g.getNode pid
            pure pNode.outShape)
          let expected ←
            match OpContracts.inferConcatOutShape axis parentShapes with
            | .ok s => pure s
            | .error msg => throw s!"IRExec: node {i}: {msg} ({n.summary})"
          if expected != τ then
            throw <|
              s!"IRExec: node {i}: concat outShape mismatch: expected={repr expected}, " ++
                s!"declared={repr τ} ({n.summary})"

          if axis = 0 then
            match hτ : τ with
            | .dim nOut rest =>
                -- Precompute typed indices for each parent and check that tails match.
                let infos : List (Sigma fun nP => Idx ([inShape] ++ ss) (.dim nP rest)) ←
                  parents.mapM (fun pid => do
                    let pNode ← g.getNode pid
                    match pNode.outShape with
                    | .dim nP restP =>
                        if hRest : restP = rest then
                          let ip ← parentIdx pid (.dim nP rest)
                          pure ⟨nP, ip⟩
                        else
                          throw <|
                            s!"IRExec: node {i}: concat axis=0 tail mismatch: {repr restP} vs " ++
                              s!"{repr rest}"
                    | _ =>
                        throw <|
                          s!"IRExec: node {i}: concat axis=0 expects rank≥1 parents, got " ++
                            s!"{repr pNode.outShape}")
                let nSum : Nat := infos.foldl (fun acc info => acc + info.1) 0
                if hSum : nSum = nOut then
                  let forward := fun ctx : TList α ([inShape] ++ ss) =>
                    let outSigma : Sigma fun n => Tensor α (.dim n rest) :=
                      concatLeadingAxisFromInfos (α := α) (Γ := [inShape] ++ ss) (rest := rest) ctx infos
                    have houtSigma :
                        outSigma =
                          concatLeadingAxisFromInfos (α := α) (Γ := [inShape] ++ ss) (rest := rest) ctx
                            infos := rfl
                    let nSum' : Nat := outSigma.1
                    let tSum : Tensor α (.dim nSum' rest) := outSigma.2
                    have hn : nSum' = nSum := by
                      -- `nSum'` is the first component of the same fold used to compute `nSum`.
                      change outSigma.1 = nSum
                      rw [houtSigma]
                      simpa [nSum] using
                        (concatLeadingAxisFromInfos_size_eq_sum (α := α) (Γ := [inShape] ++ ss) (rest :=
                          rest) ctx infos)
                    let tSum' : Tensor α (.dim nSum rest) :=
                      Tensor.castShape tSum (by simp [hn])
                    have hOutShape : Shape.dim nSum rest = τ := by
                      have hDim : Shape.dim nSum rest = Shape.dim nOut rest := by
                        simpa using congrArg (fun k => Shape.dim k rest) hSum
                      exact hDim.trans hτ.symm
                    Tensor.castShape tSum' hOutShape
                  pure <| fwd forward
                else
                  throw <|
                    s!"IRExec: node {i}: concat out dim mismatch: declared {nOut}, computed " ++
                      s!"{nSum} ({n.summary})"
            | _ =>
                throw s!"IRExec: node {i}: concat axis=0 expects rank≥1 outShape, got {repr τ}"
          else
            -- General axis concat: permute `axis` to the front (axis 0), concatenate along axis 0,
            -- then permute back.
            let permFront ←
              match OpContracts.permMoveAxisToFront axis τ with
              | .ok perm => pure perm
              | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
            let permBack ←
              match OpContracts.inversePerm permFront with
              | .ok perm => pure perm
              | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
            match Spec.Shape.permute? τ permFront with
            | none =>
                throw <|
                  s!"IRExec: node {i}: concat: invalid permutation {repr permFront} for " ++
                    s!"shape {repr τ}"
            | some outFrontExpected =>
                match hOutFrontExpected : outFrontExpected with
                | .dim nOutFront restFront =>
                    let swapsFront ← NN.IR.Graph.swapDepthsForPerm permFront (Spec.Shape.rank τ)
                    let τFrontFinal : Shape := swapShapeBySwaps τ swapsFront
                    if hOutFrontFinal : τFrontFinal = outFrontExpected then
                      let swapsBack ← NN.IR.Graph.swapDepthsForPerm permBack (Spec.Shape.rank
                        outFrontExpected)
                      let τBackFinal : Shape := swapShapeBySwaps outFrontExpected swapsBack
                      if hOutBackFinal : τBackFinal = τ then
                        let getters :
                            List (Sigma fun nP => TList α ([inShape] ++ ss) → Tensor α (.dim nP
                              restFront)) ←
                          parents.mapM (fun pid => do
                            let pNode ← g.getNode pid
                            let sIn := pNode.outShape
                            let ip ← parentIdx pid sIn
                            match Spec.Shape.permute? sIn permFront with
                            | none =>
                                throw <|
                                  s!"IRExec: node {i}: concat: invalid permutation " ++
                                    s!"{repr permFront} for parent shape {repr sIn}"
                            | some (.dim nP restP) =>
                                let sFrontExpected : Shape := .dim nP restP
                                if hRest : restP = restFront then
                                  let sFrontFinal : Shape := swapShapeBySwaps sIn swapsFront
                                  if hFinal : sFrontFinal = sFrontExpected then
                                    let getT := fun ctx : TList α ([inShape] ++ ss) =>
                                      let x : Tensor α sIn := getIdx (α := α) (xs := ctx) ip
                                      let yFinal : Tensor α sFrontFinal :=
                                        applySwapsTensor (α := α) (s := sIn) (swaps := swapsFront) x
                                      let yExpected : Tensor α sFrontExpected :=
                                        Tensor.castShape yFinal hFinal
                                      -- `yExpected` has tail `restP`; cast to the shared
                                      -- `restFront`.
                                      let yExpected' : Tensor α (.dim nP restP) := by
                                        simpa [sFrontExpected] using yExpected
                                      (by
                                        simpa [hRest] using yExpected' : Tensor α (.dim nP
                                          restFront))
                                    pure ⟨nP, getT⟩
                                  else
                                    throw <|
                                      s!"IRExec: node {i}: concat permute shape mismatch: " ++
                                      s!"computed={repr sFrontFinal}, " ++
                                        s!"expected={repr sFrontExpected} ({n.summary})"
                                else
                                  throw <|
                                    s!"IRExec: node {i}: concat: permuted tail mismatch: " ++
                                      s!"{repr restP} vs {repr restFront} ({n.summary})"
                            | some _ =>
                                throw <|
                                  s!"IRExec: node {i}: concat expects rank≥1 parents, got " ++
                                    s!"{repr sIn}"
                          )

                        let nSum : Nat := getters.foldl (fun acc info => acc + info.1) 0
                        if hSum : nSum = nOutFront then
                          let forward := fun ctx : TList α ([inShape] ++ ss) =>
                            let empty : Tensor α (.dim 0 restFront) :=
                              Spec.fill (α := α) 0 (.dim 0 restFront)
                            let outSigma :
                                Sigma fun n => Tensor α (.dim n restFront) :=
                              getters.foldl
                                (fun acc nxt =>
                                  match acc, nxt with
                                  | ⟨n1, t1⟩, ⟨n2, get2⟩ =>
                                      let t2 : Tensor α (.dim n2 restFront) := get2 ctx
                                      ⟨n1 + n2, Tensor.concatLeadingAxisSpec (α := α) (n := n1) (m := n2)
                                        (s := restFront) t1 t2⟩)
                                ⟨0, empty⟩
                            let tSum : Tensor α (.dim nSum restFront) :=
                              Tensor.castShape outSigma.2 (by
                                -- The fold's nat component is the sum of the input sizes.
                                have hFold :
                                    outSigma.1 =
                                      getters.foldl (fun acc info => acc + info.1) 0 := by
                                  -- General lemma: the `.1` component of this fold is just a nat
                                  -- fold.
                                  have hGen :
                                      ∀ (xs : List (Sigma fun nP => TList α ([inShape] ++ ss) →
                                        Tensor α (.dim nP restFront)))
                                        (n0 : Nat) (t0 : Tensor α (.dim n0 restFront)),
                                        (xs.foldl
                                            (fun acc nxt =>
                                              match acc, nxt with
                                              | ⟨n1, t1⟩, ⟨n2, get2⟩ =>
                                                  let _t2 : Tensor α (.dim n2 restFront) := get2 ctx
                                                  ⟨n1 + n2, Tensor.concatLeadingAxisSpec (α := α) (n :=
                                                    n1) (m := n2) (s := restFront) t1 _t2⟩)
                                            (⟨n0, t0⟩ : Sigma fun n => Tensor α (.dim n
                                              restFront))).1 =
                                          xs.foldl (fun acc info => acc + info.1) n0 := by
                                    intro xs n0 t0
                                    induction xs generalizing n0 t0 with
                                    | nil =>
                                        simp
                                    | cons x xs ih =>
                                        -- Unfold both folds one step and apply the IH to the
                                        -- updated accumulator.
                                        simp [List.foldl] at *
                                        -- After unfolding, the goal is exactly the IH instantiated
                                        -- at `n0 + x.1`.
                                        simpa using
                                          (ih (n0 := n0 + x.1)
                                            (t0 := Tensor.concatLeadingAxisSpec (α := α) (n := n0) (m :=
                                              x.1)
                                              (s := restFront) t0 (x.2 ctx)))
                                  simpa [outSigma] using (hGen getters 0 empty)
                                have hn : outSigma.1 = nSum := by
                                  simpa [nSum] using hFold
                                simp [hn])
                            have hOutFront : Shape.dim nSum restFront = outFrontExpected := by
                              have hDim : Shape.dim nSum restFront = Shape.dim nOutFront restFront
                                := by
                                simpa using congrArg (fun k => Shape.dim k restFront) hSum
                              simpa [hOutFrontExpected] using hDim
                            let tFront : Tensor α outFrontExpected := Tensor.castShape tSum
                              hOutFront
                            let tBack : Tensor α τBackFinal :=
                              applySwapsTensor (α := α) (s := outFrontExpected) (swaps := swapsBack)
                                tFront
                            Tensor.castShape tBack hOutBackFinal
                          pure <| fwd forward
                        else
                          throw <|
                            s!"IRExec: node {i}: concat out dim mismatch: declared {nOutFront}, " ++
                              s!"computed {nSum} ({n.summary})"
                      else
                        throw <|
                          s!"IRExec: node {i}: concat permute-back shape mismatch: " ++
                            s!"computed={repr τBackFinal}, expected={repr τ} ({n.summary})"
                    else
                      throw <|
                        s!"IRExec: node {i}: concat permute-to-front shape mismatch: " ++
                        s!"computed={repr τFrontFinal}, expected={repr outFrontExpected} " ++
                          s!"({n.summary})"
                | _ =>
                    throw s!"IRExec: node {i}: concat expects rank≥1 outShape, got {repr τ}"
        | .swap_first_two =>
            match n.parents with
            | [pId] =>
                match hτ : τ with
                | .dim nDim (.dim m rest) =>
                    let expectedIn : Shape := .dim m (.dim nDim rest)
                    let ip ← parentIdx pId expectedIn
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let y : Tensor α (.dim nDim (.dim m rest)) :=
                        Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest)
                          (getIRValue (α := α) (ctx := ctx) ip)
                      Tensor.castShape y hτ.symm
                    pure <| fwd forward
                | _ =>
                    throw s!"IRExec: node {i}: swap_first_two expects rank≥2 outShape ({n.summary})"
            | _ =>
                throw s!"IRExec: node {i}: swap_first_two expects 1 parent ({n.summary})"
        | .transpose3dLastTwo =>
            match n.parents with
            | [pId] =>
                match hτ : τ with
                | .dim a (.dim c (.dim b .scalar)) =>
                    let expectedIn : Shape := .dim a (.dim b (.dim c .scalar))
                    let ip ← parentIdx pId expectedIn
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let y : Tensor α (.dim a (.dim c (.dim b .scalar))) :=
                        Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
                          (getIRValue (α := α) (ctx := ctx) ip)
                      Tensor.castShape y hτ.symm
                    pure <| fwd forward
                | _ =>
                    throw <|
                      s!"IRExec: node {i}: transpose3d_last_two expects rank=3 with scalar " ++
                        s!"base outShape ({n.summary})"
            | _ =>
                throw s!"IRExec: node {i}: transpose3d_last_two expects 1 parent ({n.summary})"
        | .mseLoss =>
            match n.parents with
            | [yId, tId] =>
                let yNode ← g.getNode yId
                let tNode ← g.getNode tId
                if hShape : yNode.outShape = tNode.outShape then
                  if hOut : Shape.scalar = τ then
                    let s := yNode.outShape
                    let iy ← parentIdx yId s
                    let it ← parentIdx tId s
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let yhat := getIdx (α := α) (xs := ctx) iy
                      let target := getIdx (α := α) (xs := ctx) it
                      let diff := Tensor.subSpec (α := α) yhat target
                      let sq := Tensor.mulSpec (α := α) diff diff
                      let total : α := Tensor.sumSpec (α := α) sq
                      let y0 : Tensor α Shape.scalar :=
                        Tensor.scalar (total / (↑(NN.IR.Graph.meanDenom s) : α))
                      Tensor.castShape y0 hOut
                    pure <| fwd forward
                  else
                    throw s!"IRExec: node {i}: mse_loss expects scalar outShape ({n.summary})"
                else
                  throw <|
                    s!"IRExec: node {i}: mse_loss expects equal shapes, got " ++
                      s!"{repr yNode.outShape} vs {repr tNode.outShape}"
            | _ => throw s!"IRExec: node {i}: mse_loss expects 2 parents ({n.summary})"

    let st' : State α inShape :=
      ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩
    buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st'
  else
    pure st
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

end IRExec
end Compiled
end Autograd
end Runtime
