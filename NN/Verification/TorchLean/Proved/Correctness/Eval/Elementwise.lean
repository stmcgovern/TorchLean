/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Core
public import NN.IR.Semantics

/-!
# Elementwise IR Evaluation

These lemmas cover the common elementwise operators emitted by the PyTorch and ONNX bridges.  Each
statement is local to one IR node: if the parent values are already present in the evaluator table,
`Graph.evalAt` returns the corresponding spec tensor operation.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Same-shape binary elementwise operations supported by the local evaluator theorem. -/
inductive BinaryElementwiseOp where
  /-- Elementwise addition. -/
  | add
  /-- Elementwise subtraction. -/
  | sub
  /-- Elementwise multiplication. -/
  | mul
  /-- Elementwise maximum. -/
  | max
  /-- Elementwise minimum. -/
  | min

/-- Translate a binary elementwise operation to its IR opcode. -/
def BinaryElementwiseOp.toOpKind : BinaryElementwiseOp → OpKind
  | .add => .add
  | .sub => .sub
  | .mul => .mul_elem
  | .max => .maxElem
  | .min => .minElem

/-- Denotation of a same-shape binary elementwise operation. -/
def BinaryElementwiseOp.denote
    {α : Type} [Context α] {s : Shape} (op : BinaryElementwiseOp)
    (a b : Tensor α s) : Tensor α s :=
  match op with
  | .add => Tensor.addSpec a b
  | .sub => Tensor.subSpec a b
  | .mul => Tensor.mulSpec a b
  | .max => Tensor.maxSpec a b
  | .min => Tensor.minSpec a b

/-- Evaluate any supported same-shape binary elementwise node. -/
theorem evalAt_binaryElementwise_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (op : BinaryElementwiseOp) (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph op.toOpKind s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (op.denote a b)) := by
  cases op <;>
    simp [BinaryElementwiseOp.toOpKind, BinaryElementwiseOp.denote, Graph.evalAt, binaryGraph,
      binaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape, Bind.bind, Except.bind,
      Pure.pure, Except.pure]

/-- Same-shape unary elementwise operations without additional runtime side conditions. -/
inductive UnaryElementwiseOp where
  /-- Elementwise absolute value. -/
  | abs
  /-- Elementwise square root. -/
  | sqrt
  /-- Elementwise reciprocal. -/
  | inv
  /-- Rectified linear unit. -/
  | relu
  /-- Hyperbolic tangent. -/
  | tanh
  /-- Logistic sigmoid. -/
  | sigmoid
  /-- Elementwise exponential. -/
  | exp
  /-- Elementwise sine. -/
  | sin
  /-- Elementwise cosine. -/
  | cos

/-- Translate a unary elementwise operation to its IR opcode. -/
def UnaryElementwiseOp.toOpKind : UnaryElementwiseOp → OpKind
  | .abs => .abs
  | .sqrt => .sqrt
  | .inv => .inv
  | .relu => .relu
  | .tanh => .tanh
  | .sigmoid => .sigmoid
  | .exp => .exp
  | .sin => .sin
  | .cos => .cos

/-- Denotation of a same-shape unary elementwise operation. -/
def UnaryElementwiseOp.denote
    {α : Type} [Context α] {s : Shape} (op : UnaryElementwiseOp)
    (x : Tensor α s) : Tensor α s :=
  match op with
  | .abs => Tensor.absSpec x
  | .sqrt => Tensor.sqrtSpec x
  | .inv => Tensor.invSpec x
  | .relu => Activation.reluSpec x
  | .tanh => Activation.tanhSpec x
  | .sigmoid => Activation.sigmoidSpec x
  | .exp => Tensor.expSpec x
  | .sin => Tensor.mapSpec (fun v => MathFunctions.sin v) x
  | .cos => Tensor.mapSpec (fun v => MathFunctions.cos v) x

/-- Evaluate any supported same-shape unary elementwise node. -/
theorem evalAt_unaryElementwise_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (op : UnaryElementwiseOp) (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph op.toOpKind s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (op.denote x)) := by
  cases op <;>
    simp [UnaryElementwiseOp.toOpKind, UnaryElementwiseOp.denote, Graph.evalAt, unaryGraph,
      unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape, Bind.bind, Except.bind,
      Pure.pure, Except.pure]

set_option maxHeartbeats 1000000 in
/-- Evaluate a binary elementwise node in an arbitrary graph from the compiler's shape invariant. -/
theorem evalAt_binaryElementwise_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape s : Shape} {ss : List Shape}
    (op : BinaryElementwiseOp) (a b : Idx (Ctx inShape ss) s)
    (g : Graph) (payload : Payload α) (input : DVal α) (vals : Array (DVal α)) (i : Nat)
    (n : NN.IR.Node)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : g.getNode i = pure n)
    (hKind : n.kind = op.toOpKind) (hParents : n.parents = [a.id, b.id])
    (hOut : n.outShape = s) :
    Graph.evalAt (α := α) g payload input vals i = (do
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure (DVal.mk (α := α) s (op.denote ta tb))) := by
  have ha : (vals[a.id]!).1 = s := by
    simpa [DVal.shape] using
      shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := a) (s := s)
  have hb : (vals[b.id]!).1 = s := by
    simpa [DVal.shape] using
      shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := b) (s := s)
  let ta : Tensor α s := ha ▸ (vals[a.id]!).snd
  let tb : Tensor α s := hb ▸ (vals[b.id]!).snd
  have hExpectA : Graph.expectShape (α := α) (expected := s) (vals[a.id]!) = .ok ta := by
    simpa [ta] using expectShape_eq_ok (expected := s) (v := vals[a.id]!) ha
  have hExpectB : Graph.expectShape (α := α) (expected := s) (vals[b.id]!) = .ok tb := by
    simpa [tb] using expectShape_eq_ok (expected := s) (v := vals[b.id]!) hb
  have hGetA : getVal (α := α) (inShape := inShape) (ss := ss) vals a = .ok ta := by
    simpa [ta] using getVal_eq_ok_of_hShapes (vals := vals) (expected := s) (idx := a) ha
  have hGetB : getVal (α := α) (inShape := inShape) (ss := ss) vals b = .ok tb := by
    simpa [tb] using getVal_eq_ok_of_hShapes (vals := vals) (expected := s) (idx := b) hb
  cases op <;> unfold Graph.evalAt <;>
    simp [hGetNode, hKind, hParents, BinaryElementwiseOp.toOpKind, DVal.shape, DVal.tensor,
      DVal.mk, throw, throwThe, MonadExceptOf.throw] <;>
    rw [hOut, hExpectA, hExpectB] <;>
    simp [hGetA, hGetB, BinaryElementwiseOp.denote, ta, tb, Bind.bind, Except.bind]

set_option maxHeartbeats 1000000 in
/-- Evaluate a unary elementwise node in an arbitrary graph from the compiler's shape invariant. -/
theorem evalAt_unaryElementwise_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape s : Shape} {ss : List Shape}
    (op : UnaryElementwiseOp) (x : Idx (Ctx inShape ss) s)
    (g : Graph) (payload : Payload α) (input : DVal α) (vals : Array (DVal α)) (i : Nat)
    (n : NN.IR.Node)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : g.getNode i = pure n)
    (hKind : n.kind = op.toOpKind) (hParents : n.parents = [x.id])
    (hOut : n.outShape = s) :
    Graph.evalAt (α := α) g payload input vals i = (do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure (DVal.mk (α := α) s (op.denote tx))) := by
  have hx : (vals[x.id]!).1 = s := by
    simpa [DVal.shape] using
      shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := x) (s := s)
  let tx : Tensor α s := hx ▸ (vals[x.id]!).snd
  have hExpect : Graph.expectShape (α := α) (expected := s) (vals[x.id]!) = .ok tx := by
    simpa [tx] using expectShape_eq_ok (expected := s) (v := vals[x.id]!) hx
  have hGet : getVal (α := α) (inShape := inShape) (ss := ss) vals x = .ok tx := by
    simpa [tx] using getVal_eq_ok_of_hShapes (vals := vals) (expected := s) (idx := x) hx
  cases op <;> unfold Graph.evalAt <;>
    simp [hGetNode, hKind, hParents, UnaryElementwiseOp.toOpKind, DVal.shape, DVal.tensor, DVal.mk,
      throw, throwThe, MonadExceptOf.throw] <;>
    rw [hOut, hExpect] <;>
    simp [hGet, UnaryElementwiseOp.denote, tx]

/-- Local IR semantics for elementwise addition. -/
theorem evalAt_add_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .add s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.addSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .add a b

/-- Local IR semantics for elementwise subtraction. -/
theorem evalAt_sub_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .sub s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.subSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .sub a b

/-- Local IR semantics for elementwise multiplication. -/
theorem evalAt_mul_elem_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .mul_elem s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.mulSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .mul a b

/-- Local IR semantics for elementwise maximum. -/
theorem evalAt_maxElem_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .maxElem s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.maxSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .max a b

/-- Local IR semantics for elementwise minimum. -/
theorem evalAt_minElem_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .minElem s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.minSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .min a b

/-- Local IR semantics for elementwise absolute value. -/
theorem evalAt_abs_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .abs s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.absSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .abs x

/-- Local IR semantics for elementwise square root. -/
theorem evalAt_sqrt_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sqrt s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.sqrtSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .sqrt x

/-- Local IR semantics for elementwise reciprocal. -/
theorem evalAt_inv_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .inv s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.invSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .inv x

/-- Local IR semantics for ReLU. -/
theorem evalAt_relu_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .relu s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.reluSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .relu x

/-- Local IR semantics for the ReLU derivative `1[x > 0]`. Not a `UnaryElementwiseOp`: the
verified compiler does not emit it, so it needs only the evaluator bridge, proved directly. -/
theorem evalAt_reluDeriv_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .reluDeriv s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.reluDerivSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for tanh. -/
theorem evalAt_tanh_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .tanh s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.tanhSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .tanh x

/-- Local IR semantics for sigmoid. -/
theorem evalAt_sigmoid_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sigmoid s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.sigmoidSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .sigmoid x

/-- Local IR semantics for exp. -/
theorem evalAt_exp_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .exp s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.expSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .exp x

/-- Local IR semantics for sin. -/
theorem evalAt_sin_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sin s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.mapSpec (fun v => MathFunctions.sin v) x)) := by
  exact evalAt_unaryElementwise_eq .sin x

/-- Local IR semantics for cos. -/
theorem evalAt_cos_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .cos s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.mapSpec (fun v => MathFunctions.cos v) x)) := by
  exact evalAt_unaryElementwise_eq .cos x

/-- Local IR semantics for log on inputs satisfying the IR positivity side condition. -/
theorem evalAt_log_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s)
    (hpos : Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v)) x = true) :
    Graph.evalAt (α := α) (g := unaryGraph .log s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.logSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape, hpos,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
