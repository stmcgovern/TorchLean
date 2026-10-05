/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Registry
public import NN.IR.Graph

/-!
# Backend Planning for IR Graphs

Adapter from TorchLean's op-tagged IR to backend capsules.

This is intentionally a planning layer, not an evaluator. The graph remains the semantic object;
the planner selects which backend contract should implement each node or node family.
-/

@[expose] public section

namespace NN
namespace Backend
namespace IR

/--
Backend operation tag for an IR op.

These tags are the same vocabulary used by backend capsules and runtime reports.
If an op maps to a tag that has no capsule for the selected profile, planning fails at the backend
boundary instead of silently using a broader bucket.
-/
def op? : NN.IR.OpKind → Option BackendOp
  | .input => none
  | .const .. => none
  | .detach => none
  | .randUniform .. => some .randUniform
  | .bernoulliMask .. => some .bernoulliMask
  | .add => some .add
  | .sub => some .sub
  | .mul_elem => some .mul
  | .abs => some .abs
  | .sqrt => some .sqrt
  | .inv => some .inv
  | .maxElem => some .max
  | .minElem => some .min
  | .relu => some .relu
  | .reluDeriv => some .reluDeriv
  | .tanh => some .tanh
  | .sigmoid => some .sigmoid
  | .exp => some .exp
  | .log => some .log
  | .sin => some .sin
  | .cos => some .cos
  | .mseLoss => some .mseLoss
  | .matmul => some .matmul
  | .linear => some .linear
  | .conv2d .. => some .conv
  | .maxPool2d .. => some .maxPool
  | .maxPool2dPad .. => some .maxPool
  | .avgPool2d .. => some .avgPool
  | .avgPool2dPad .. => some .avgPool
  | .broadcastTo .. => some .broadcast
  | .reduceSum .. => some .reduceSum
  | .reduceMean .. => some .reduceMean
  | .sum => some .reduceSum
  | .softmax .. => some .softmax
  | .layernorm .. => some .layerNorm
  | .reshape .. => some .reshape
  | .flatten .. => some .reshape
  | .concat .. => some .concat
  | .swap_first_two => some .permute
  | .transpose3dLastTwo => some .permute
  | .permute .. => some .permute
  | .batchNorm2dNchwEval .. => some .batchNorm

/-- Backend operation requested by a graph node, if the node needs runtime work. -/
def nodeOp? (n : NN.IR.Node) : Option BackendOp :=
  op? n.kind

/-- Backend choice for one concrete IR node. -/
structure PlannedNodeKernel where
  nodeId : Nat
  kind : NN.IR.OpKind
  op : BackendOp
  capsule : KernelCapsule
  deriving Repr

/-- Graph-aware execution plan that preserves the IR node identity for every backend choice. -/
structure GraphExecutionPlan where
  kernels : List PlannedNodeKernel
  deriving Repr

namespace GraphExecutionPlan

/-- Node ids covered by backend kernels, in graph order. -/
def nodeIds (p : GraphExecutionPlan) : List Nat :=
  p.kernels.map (·.nodeId)

/-- Selected capsule names, in graph order. -/
def capsuleNames (p : GraphExecutionPlan) : List String :=
  p.kernels.map fun k => k.capsule.name

end GraphExecutionPlan

/-- Plan a single IR node when it corresponds to runtime work. -/
def planNode? (cfg : ExecutionConfig) (availability : Availability)
    (registry : List KernelCapsule) (n : NN.IR.Node) :
    Except String (Option PlannedNodeKernel) := do
  match nodeOp? n with
  | none => pure none
  | some op =>
      let k ← planOp cfg (availability.filterCapsules registry) op
      pure <| some
        { nodeId := n.id
          kind := n.kind
          op := k.op
          capsule := k.capsule }

/-- Plan every runtime-relevant node in graph order. -/
def planGraphNodesWithRegistry (cfg : ExecutionConfig) (availability : Availability)
    (registry : List KernelCapsule) (g : NN.IR.Graph) : Except String GraphExecutionPlan := do
  let mut kernels : List PlannedNodeKernel := []
  for n in g.nodes do
    match (← planNode? cfg availability registry n) with
    | none => pure ()
    | some k => kernels := k :: kernels
  pure { kernels := kernels.reverse }

/-- Check graph well-formedness, then plan every runtime-relevant node. -/
def checkedPlanGraphNodesWithRegistry (cfg : ExecutionConfig) (availability : Availability)
    (registry : List KernelCapsule) (g : NN.IR.Graph) : Except String GraphExecutionPlan := do
  g.checkWellFormed
  planGraphNodesWithRegistry cfg availability registry g

end IR
end Backend
end NN
