/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph

/-!
# Operation Contracts

Shared operation contracts for `NN.IR.Graph`.

Several IR passes need to agree on the same small set of “shape contracts”:

- `NN.IR.Infer`: recompute output shapes from op parameters + parent shapes.
- `NN.IR.Check`: expose the documented `Graph.checkShapes` wrapper.
- `NN.IR.Semantics`: evaluate nodes and reject ill-shaped graphs with readable error messages.

The point of this file is to keep shape arithmetic out of individual passes. If an op has nontrivial
shape behavior (concat, matmul, pooling, convolution, LayerNorm flattening, axis moves), define the
contract here first and call it from inference/semantics instead of copying the formula.
-/

@[expose] public section

namespace NN.IR

open Spec

/-!
## Small shape utilities

These helpers are used by multiple IR passes, especially `Infer` and `Semantics`.
-/

namespace ShapeUtil

/-- The output shape of flattening a tensor of shape `s` to a 1D vector. -/
def flattenOutShape (s : Shape) : Shape :=
  .dim (Spec.Shape.size s) .scalar

/--
If `s` has rank ≥ 2, return the shape obtained by swapping its first two axes.

Example: `(a, b, rest)` becomes `(b, a, rest)`.
-/
def swapFirstTwoShape? : Shape → Option Shape
  | .dim a (.dim b rest) => some (.dim b (.dim a rest))
  | _ => none

/--
If `s` has shape `(a, b, c)` (rank=3 with scalar base), return `(a, c, b)`.

This is the common “transpose the last two axes” pattern for batched matrices.
-/
def transpose3dLastTwoShape? : Shape → Option Shape
  | .dim a (.dim b (.dim c .scalar)) => some (.dim a (.dim c (.dim b .scalar)))
  | _ => none

end ShapeUtil

namespace OpContracts

/-!
## Generic contract helpers

These functions live outside any particular pass (`Infer`/`Check`/`Semantics`) so they can be
reused without introducing import cycles.
-/

/-- Check that an `axis` is in-bounds for a given shape. -/
def checkAxisValid (axis : Nat) (s : Shape) : Except String Unit := do
  if axis < Spec.Shape.rank s then
    pure ()
  else
    throw s!"invalid axis {axis} for rank {Spec.Shape.rank s}"

/-- Check that a natural-number op parameter is nonzero. -/
def checkPositive (tag param : String) (n : Nat) : Except String Unit := do
  if n = 0 then
    throw s!"{tag}: {param} must be > 0"
  else
    pure ()

/--
Reconstruct the proof object required by the typed tensor broadcast primitive.

IR nodes store dynamic shapes, so every pass that accepts `.broadcastTo` must rebuild this witness
instead of trusting that the declared input and output shapes are compatible.
-/
def mkCanBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
  | .scalar, s₂ => some (.scalar_to_any s₂)
  | .dim n₁ t₁, .dim n₂ t₂ =>
      if hEq : n₁ = n₂ then
        (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
          hEq ▸ Shape.CanBroadcastTo.dim_eq (n := n₁) (s₁ := t₁) (s₂ := t₂) tail)
      else if h1 : n₁ = 1 then
        (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
          h1 ▸ Shape.CanBroadcastTo.dim_1_to_n (n := n₂) (s₁ := t₁) (s₂ := t₂) tail)
      else
        (mkCanBroadcastTo? (.dim n₁ t₁) t₂).map (fun tail =>
          Shape.CanBroadcastTo.expand_dims (n := n₂) (s₁ := .dim n₁ t₁) (s₂ := t₂) tail)
  | _, _ => none

/--
Compute the `(seqLen, embedDim)` pair used to interpret `layernorm axis`.

TorchLean’s IR stores LayerNorm as an `axis : Nat` instead of a full `normalized_shape` tuple.
We interpret this in the same way the PyTorch exporter does:

`normalized_shape = dims.drop axis`

That is, we normalize over the **suffix** of dimensions starting at `axis`. To reuse the current
spec primitive (`Spec.layerNorm`), we flatten the input shape `s` into a 2D view:

* `seqLen`   = product of dimensions *before* `axis` (`dims.take axis`)
* `embedDim` = product of dimensions *from* `axis` onward (`dims.drop axis`)

Then we run 2D last-axis LayerNorm on a `(seqLen × embedDim)` tensor and reshape back.
-/
def layerNorm2DParams (axis : Nat) (s : Shape) : Except String (Nat × Nat) := do
  checkAxisValid axis s
  let dims := Shape.toList s
  let seqLen : Nat := (dims.take axis).foldl (fun acc d => acc * d) 1
  let embedDim : Nat := (dims.drop axis).foldl (fun acc d => acc * d) 1
  checkPositive "layernorm" "seqLen" seqLen
  checkPositive "layernorm" "embedDim" embedDim
  pure (seqLen, embedDim)

/--
Check that `axis` refers to the **last** axis of `s`.

This is a convenience predicate for passes/backends that restrict an op to last-axis behavior.
For example, some verification bounds are implemented only for last-axis `softmax`/`layernorm` and
use this check to fail fast with a readable error.
-/
def checkLastAxis (tag : String) (axis : Nat) (s : Shape) : Except String Unit := do
  checkAxisValid axis s
  if axis + 1 = Spec.Shape.rank s then
    pure ()
  else
    throw s!"{tag}: only last-axis is supported (axis={axis}, rank={Spec.Shape.rank s})"

/-- Write `some val` at index `j` of `xs`, refusing a slot already written. -/
def setSlot (perm : List Nat) (r : Nat) :
    (xs : List (Option Nat)) → (axis j val : Nat) → Except String (List (Option Nat))
  | [], _, j, _ => throw s!"permute: internal error: index {j} out of range for invLen={r}"
  | none :: rest, _, 0, val => pure (some val :: rest)
  | some _ :: _, axis, 0, _ => throw s!"permute: duplicate axis {axis} in {repr perm}"
  | x :: rest, axis, k + 1, val => do pure (x :: (← setSlot perm r rest axis k val))

/-- `inv[perm[i]] := i` for each `i`, left to right. -/
def fillInv (perm : List Nat) (r : Nat) :
    (ps : List Nat) → (i : Nat) → (inv : List (Option Nat)) → Except String (List (Option Nat))
  | [], _, inv => pure inv
  | p :: ps, i, inv =>
      if p < r then do
        let inv' ← setSlot perm r inv p p i
        fillInv perm r ps (i + 1) inv'
      else throw s!"permute: axis {p} out of range for rank {r} in {repr perm}"

/-- Read every slot back, failing on an unwritten one. -/
def readInv (perm : List Nat) : (inv : List (Option Nat)) → (j : Nat) → Except String (List Nat)
  | [], _ => pure []
  | some idx :: rest, j => do pure (idx :: (← readInv perm rest (j + 1)))
  | none :: _, j => throw s!"permute: missing axis {j} in {repr perm}"

/--
Compute the inverse of a permutation list.

If `perm` is a permutation of `[0,1,...,r-1]` (where `r = perm.length`), then the inverse `inv`
satisfies `inv[perm[i]] = i`.

Written by structural recursion (`fillInv`, `setSlot`, `readInv`) rather than imperative loops so
that proofs can unfold it. It agrees with the earlier loop form on every list of length at most 6
with entries below 7 (checked exhaustively when the rewrite was made), including every error case.
-/
def inversePerm (perm : List Nat) : Except String (List Nat) := do
  let r := perm.length
  let inv ← fillInv perm r perm 0 (List.replicate r none)
  readInv perm inv 0

/--
Permutation (0-based axes) that moves `axis` to the **last** position, preserving the relative
order of the other axes.

Example: rank=4 and `axis=1` yields `[0,2,3,1]`.
-/
def permMoveAxisToLast (axis : Nat) (s : Shape) : Except String (List Nat) := do
  checkAxisValid axis s
  let r := Spec.Shape.rank s
  pure <| (List.range r).erase axis ++ [axis]

/--
Permutation (0-based axes) that moves `axis` to the **first** position, preserving the relative
order of the other axes.

Example: rank=4 and `axis=2` yields `[2,0,1,3]`.
-/
def permMoveAxisToFront (axis : Nat) (s : Shape) : Except String (List Nat) := do
  checkAxisValid axis s
  let r := Spec.Shape.rank s
  pure <| axis :: (List.range r).erase axis

/--
Infer the output shape for `matmul` from the two parent shapes.

Supported cases:
- 2D: `(m×n) · (n×p) → (m×p)`
- limited 3D “batched matmul”: `(b×m×n) · (b×n×p) → (b×m×p)`
-/
def inferMatmulOutShape (a b : Shape) : Except String Shape := do
  match a, b with
  | .dim m (.dim n .scalar), .dim n' (.dim p .scalar) =>
      if _h : n = n' then
        pure (.dim m (.dim p .scalar))
      else
        throw s!"matmul: inner dims mismatch: {n} vs {n'}"
  | .dim batch (.dim m (.dim n .scalar)), .dim batch' (.dim n' (.dim p .scalar)) =>
      if _hb : batch = batch' then
        if _hn : n = n' then
          pure (.dim batch (.dim m (.dim p .scalar)))
        else
          throw s!"matmul: inner dims mismatch: {n} vs {n'}"
      else
        throw s!"matmul: batch dims mismatch: {batch} vs {batch'}"
  | _, _ =>
      throw s!"matmul: unsupported shapes: {repr a} · {repr b}"

/-- Check that every concat parent has the expected rank. -/
def checkConcatRanks (expected : Nat) : List Shape → Except String Unit
  | [] => .ok ()
  | shape :: shapes =>
      if Spec.Shape.rank shape != expected then
        throw <|
          s!"concat: rank mismatch: expected {expected}, got {Spec.Shape.rank shape} ({repr shape})"
      else
        checkConcatRanks expected shapes

/--
Infer leading-axis concat after the first input has fixed the common tail and initial dimension.
-/
def inferConcatLeadingAxis (expectedTail : Shape) : Nat → List Shape → Except String Shape
  | total, [] => .ok (.dim total expectedTail)
  | total, shape :: shapes =>
      match shape with
      | .dim size tail =>
          if tail != expectedTail then
            throw <|
              s!"concat: axis=0 expects matching tail shapes, got {repr expectedTail} and " ++
                s!"{repr tail}"
          else
            inferConcatLeadingAxis expectedTail (total + size) shapes
      | _ =>
          throw s!"concat: axis=0 expects rank≥1 inputs, got {repr shape}"

/--
Infer the output shape for `concat` from the parent shapes.

All parents must:
- have the same rank,
- agree on every dimension except `axis`, and
- have `axis` in bounds.

The output shape matches the parents except at `axis`, where the dimension is the sum of the input
dimensions.

PyTorch analogy: `torch.cat(xs, dim=axis)` for a list `xs` of tensors.
-/
def inferConcatOutShape (axis : Nat) (parents : List Shape) : Except String Shape := do
  let k := parents.length
  if k < 2 then
    throw s!"concat: expected at least 2 inputs, got {k}"
  let s0 ←
    match parents with
    | [] => throw "concat: internal error"
    | s :: _ => pure s
  checkAxisValid axis s0
  let r0 := Spec.Shape.rank s0
  checkConcatRanks r0 parents

  let rec go (axis : Nat) (shs : List Shape) : Except String Shape := do
    match axis, shs with
    | 0, [] =>
        throw "concat: internal error"
    | 0, s :: rest =>
        match s with
        | .dim n0 tail0 =>
            inferConcatLeadingAxis tail0 n0 rest
        | _ =>
            throw s!"concat: axis=0 expects rank≥1 inputs, got {repr s}"
    | Nat.succ _, [] =>
        throw "concat: internal error"
    | Nat.succ a, s :: rest =>
        match s with
        | .dim n0 tail0 =>
            let mut tailsRev : List Shape := [tail0]
            for t in rest do
              match t with
              | .dim n tail =>
                  if n != n0 then
                    throw s!"concat: non-axis dim mismatch: expected {n0}, got {n}"
                  tailsRev := tail :: tailsRev
              | _ =>
                  throw s!"concat: axis out of range for input shape {repr t}"
            let outTail ← go a tailsRev.reverse
            pure (.dim n0 outTail)
        | _ =>
            throw s!"concat: axis out of range for input shape {repr s}"

  go axis parents

/-!
## Pooling/Conv2D shape arithmetic (CHW-only)

These formulas mirror the spec/runtime conventions (CHW tensors, no dilation, symmetric padding).
Centralizing them gives inference, evaluation, verification, and export code a shared convention
for convolution and pooling shapes.
-/

/-- Output length for a 1D sliding-window op without padding: `⌊(in - k)/stride⌋ + 1`. -/
def slideOut (inLen k stride : Nat) : Nat :=
  Shape.slidingWindowOutDim inLen k stride 0

/-- Output length for a 1D sliding-window op with symmetric padding: `⌊(in + 2*pad - k)/stride⌋ +
  1`. -/
def slideOutPad (inLen k stride padding : Nat) : Nat :=
  Shape.slidingWindowOutDim inLen k stride padding

/--
Reject sliding-window shapes where the kernel has no valid placement.

Lean `Nat` subtraction saturates at zero, so `(in + 2*pad - k)` would otherwise turn an invalid
window into a plausible one-element output.
-/
def checkWindowFits (tag axis : String) (inLen k padding : Nat) : Except String Unit := do
  let padded := inLen + 2 * padding
  if padded < k then
    throw s!"{tag}: {axis} window does not fit padded input: input={inLen}, padding={padding}, kernel={k}"
  else
    pure ()

/-- Output shape for CHW pooling without padding. -/
def pool2dCHWOutShape (c inH inW kH kW stride : Nat) : Shape :=
  let outH := slideOut inH kH stride
  let outW := slideOut inW kW stride
  .dim c (.dim outH (.dim outW .scalar))

/-- Output shape for CHW pooling with symmetric padding. -/
def pool2dCHWOutShapePad (c inH inW kH kW stride padding : Nat) : Shape :=
  let outH := slideOutPad inH kH stride padding
  let outW := slideOutPad inW kW stride padding
  .dim c (.dim outH (.dim outW .scalar))

/-- Output shape for CHW conv2d (single-image, no batch dim). -/
def conv2dCHWOutShape (outC inH inW kH kW stride padding : Nat) : Shape :=
  let outH := slideOutPad inH kH stride padding
  let outW := slideOutPad inW kW stride padding
  .dim outC (.dim outH (.dim outW .scalar))

/-- Infer the output shape for CHW pooling without padding, from a parent shape. -/
def inferPool2dCHWOutShape (tag : String) (kH kW stride : Nat) (parent : Shape) : Except String
  Shape := do
  checkPositive tag "kH" kH
  checkPositive tag "kW" kW
  checkPositive tag "stride" stride
  match parent with
  | .dim c (.dim inH (.dim inW .scalar)) =>
      checkWindowFits tag "height" inH kH 0
      checkWindowFits tag "width" inW kW 0
      pure (pool2dCHWOutShape c inH inW kH kW stride)
  | s =>
      throw s!"{tag}: expected input shape (C,H,W), got {repr s}"

/-- Infer the output shape for CHW pooling with padding, from a parent shape. -/
def inferPool2dCHWOutShapePad (tag : String) (kH kW stride padding : Nat) (parent : Shape) :
    Except String Shape := do
  checkPositive tag "kH" kH
  checkPositive tag "kW" kW
  checkPositive tag "stride" stride
  match parent with
  | .dim c (.dim inH (.dim inW .scalar)) =>
      checkWindowFits tag "height" inH kH padding
      checkWindowFits tag "width" inW kW padding
      pure (pool2dCHWOutShapePad c inH inW kH kW stride padding)
  | s =>
      throw s!"{tag}: expected input shape (C,H,W), got {repr s}"

/-- Infer the output shape for CHW Conv2D, checking the declared `inC` against the parent shape. -/
def inferConv2dCHWOutShape (inC outC kH kW stride padding : Nat) (parent : Shape) : Except String
    Shape := do
  checkPositive "conv2d" "inC" inC
  checkPositive "conv2d" "kH" kH
  checkPositive "conv2d" "kW" kW
  checkPositive "conv2d" "stride" stride
  match parent with
  | .dim inC' (.dim inH (.dim inW .scalar)) =>
      if inC != inC' then
        throw s!"conv2d: inC mismatch: op={inC} vs input={inC'}"
      checkWindowFits "conv2d" "height" inH kH padding
      checkWindowFits "conv2d" "width" inW kW padding
      pure (conv2dCHWOutShape outC inH inW kH kW stride padding)
  | s =>
      throw s!"conv2d: expected input shape (inC,inH,inW), got {repr s}"

/-- Output shape for eval-mode BatchNorm2d on NCHW tensors. -/
def inferBatchNorm2dNchwEvalOutShape (channels : Nat) (parent : Shape) : Except String Shape := do
  checkPositive "batch_norm2d_nchw_eval" "channels" channels
  match parent with
  | .dim n (.dim c (.dim h (.dim w .scalar))) =>
      checkPositive "batch_norm2d_nchw_eval" "batch" n
      checkPositive "batch_norm2d_nchw_eval" "height" h
      checkPositive "batch_norm2d_nchw_eval" "width" w
      if channels = c then
        pure parent
      else
        throw s!"batch_norm2d_nchw_eval: channel mismatch: op={channels} vs input={c}"
  | s =>
      throw s!"batch_norm2d_nchw_eval: expected input shape (N,C,H,W), got {repr s}"

end OpContracts

end NN.IR
