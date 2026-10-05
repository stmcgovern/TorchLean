/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Runtime.PyTorch.Export.Core

/-!
# IRPyTorch

IR → PyTorch code generation.

This module takes an op-tagged `NN.IR.Graph` plus a `NN.MLTheory.CROWN.Graph.ParamStore` payload and
emits a standalone PyTorch `nn.Module` implementation as a Python source string.

What this is (and isn't):

* This is an extraction/convenience layer used in round-trip examples: run/train in Python, then
  optionally import weights back to Lean.
* This is **not** a formal proof of semantic equivalence between PyTorch execution and the Lean IR
  denotation. The "source of truth" semantics live on the Lean side (`NN.IR.Graph.denote*`, and the
  compiled runtime).

Assumptions:

* Node ids index `g.nodes` consistently; `getNode` checks that invariant.
* The ParamStore contains parameters for the node kinds that require them (linear/conv2d/etc.).
* Only a supported subset of IR node kinds is lowered.

Failure modes (reported as `Except String`):

* missing/malformed ParamStore entries,
* shape mismatches between graph nodes and parameters,
* unsupported node kinds.

PyTorch context (comments only):

PyTorch’s ONNX exporter and `torch.export` can capture graphs for execution in other runtimes.
TorchLean’s emitter here prints readable Python that mirrors the IR, rather than producing an
execution-focused serialized graph artifact.
-/

public section


namespace Export
namespace IRPyTorch

open Spec
open Tensor
open NN.IR
open NN.MLTheory.CROWN.Graph
open Std
open Export.PyTorch

/-- Flatten a tensor and render it as a Python list literal (used for
  `torch.tensor([...]).reshape(...)`). -/
def tensorToPyFlat {s : Shape} (t : Tensor Float s) : String :=
  vectorTensorToPy (n := Spec.Shape.size s) (Tensor.flattenSpec (α := Float) t)

/-- Return `true` iff every element of the list equals the first element (vacuously true on `[]`).
  -/
def allEq (xs : List Float) : Bool :=
  match xs with
  | [] => true
  | x :: rest => rest.all (fun y => y == x)

/--
Detect whether a flattened `(seqLen × embedDim)` tensor is a broadcast of a single row.

If so, return that row. This is used to compress large broadcasted constants into a smaller
learnable vector parameter in the emitted PyTorch code.
-/
def broadcastRow2D? (seqLen embedDim : Nat) (flat : List Float) : Option (List Float) :=
  if _h0 : seqLen = 0 then
    none
  else
    let row0 := flat.take embedDim
    if row0.length != embedDim then
      none
    else
      let ok := (List.range seqLen).all fun i =>
        let start := i * embedDim
        let row := (flat.drop start).take embedDim
        row == row0
      if ok then some row0 else none

/--
Options controlling IR-to-PyTorch emission.

Most knobs here configure example emission, dtype handling, and how to
materialize constant nodes (buffers vs parameters, and whether to compress broadcasted parameters).
-/
structure Options where
  /-- Class name to use in the emitted Python source. -/
  className : String := "ExportedIRModel"
  /-- Python expression used for the tensor dtype (e.g. `"torch.float32"`). -/
  dtypeExpr : String := "torch.float32"
  /-- If true, include a small training skeleton and runtime-check in the emitted script. -/
  includeTrainingSkeleton : Bool := true
  /-- If true, emit IR `const` nodes as `nn.Parameter` when appropriate (learnable). -/
  learnableConsts : Bool := true
  /-- If true, compress broadcasted 2D consts into a smaller learnable vector parameter. -/
  compressBroadcastParams : Bool := true
  deriving Repr

/--
How an IR `const` node is represented in the emitted PyTorch module.

- `bufferFull`: a non-learnable `register_buffer(...)` tensor
- `paramFull`: a learnable `nn.Parameter` with the full tensor shape
- `paramBroadcast2D`: a learnable vector `nn.Parameter` that is expanded at runtime to a
  `(seqLen × embedDim)` tensor (compression for broadcasted 2D constants).
-/
inductive ConstBinding where
  | bufferFull (attr : String)
  | paramFull (attr : String)
  | paramBroadcast2D (attr : String) (seqLen embedDim : Nat)
  deriving Repr

/-- Map from IR node id to how its constant should be referenced in the PyTorch module. -/
abbrev ConstBindings := HashMap Nat ConstBinding

/-- The Python attribute name used to reference a bound constant (`self.<attr>`). -/
def ConstBinding.attr : ConstBinding → String
  | .bufferFull a => a
  | .paramFull a => a
  | .paramBroadcast2D a _ _ => a

/-- Default attribute name for a const node: `self.const_<id>`. -/
def constAttr (id : Nat) : String := s!"const_{id}"

/-- Attribute name for a linear layer weight tensor in the ParamStore (`self.linear_<id>_W`). -/
def linearWAttr (id : Nat) : String := s!"linear_{id}_W"
/-- Attribute name for a linear layer bias tensor in the ParamStore (`self.linear_<id>_b`). -/
def linearBAttr (id : Nat) : String := s!"linear_{id}_b"

/-- Attribute name for a conv2d kernel tensor in the ParamStore (`self.conv2d_<id>_kernel`). -/
def conv2dKernelAttr (id : Nat) : String := s!"conv2d_{id}_kernel"
/-- Attribute name for a conv2d bias tensor in the ParamStore (`self.conv2d_<id>_bias`). -/
def conv2dBiasAttr (id : Nat) : String := s!"conv2d_{id}_bias"

/-- Attribute name for a BatchNorm scale tensor (`self.batchnorm_<id>_gamma`). -/
def batchNormGammaAttr (id : Nat) : String := s!"batchnorm_{id}_gamma"
/-- Attribute name for a BatchNorm bias tensor (`self.batchnorm_<id>_beta`). -/
def batchNormBetaAttr (id : Nat) : String := s!"batchnorm_{id}_beta"
/-- Attribute name for a BatchNorm running mean tensor (`self.batchnorm_<id>_mean`). -/
def batchNormMeanAttr (id : Nat) : String := s!"batchnorm_{id}_mean"
/-- Attribute name for a BatchNorm running variance tensor (`self.batchnorm_<id>_var`). -/
def batchNormVarAttr (id : Nat) : String := s!"batchnorm_{id}_var"

/-- Emit a `torch.tensor([...]).reshape(...)` expression from a flat Python list literal and a
  `Shape`. -/
def pyTensorFromFlat (flatList : String) (shape : Shape) (dtypeVar : String := "dtype") : String :=
  s!"torch.tensor({flatList}, dtype={dtypeVar}).reshape({shapeToPyTupleString shape})"

/--
Retrieve a node from the graph and validate its id invariant.

This yields more actionable error messages than directly indexing into `g.nodes`.
-/
def getNode (g : NN.IR.Graph) (id : Nat) : Except String NN.IR.Node := do
  match g.nodes[id]? with
  | none => throw s!"IR→PyTorch: node id out of bounds: {id}"
  | some n =>
      if n.id != id then
        throw s!"IR→PyTorch: internal error: nodes[{id}].id = {n.id} (expected {id})"
      pure n

/-- Expect a node to have exactly one parent, returning that parent id. -/
private def expectUnary (id : Nat) (parents : List Nat) : Except String Nat := do
  match parents with
  | [p] => pure p
  | _ => throw s!"IR→PyTorch: node {id}: expected 1 parent, got {parents.length}"

/-- Expect a node to have exactly two parents, returning the pair. -/
private def expectBinary (id : Nat) (parents : List Nat) : Except String (Nat × Nat) := do
  match parents with
  | [a, b] => pure (a, b)
  | _ => throw s!"IR→PyTorch: node {id}: expected 2 parents, got {parents.length}"

/-- Return `true` iff `id` refers to a `.const` node in the graph. -/
private def isConstId (g : NN.IR.Graph) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some n =>
      match n.kind with
      | .const _ => true
      | _ => false

/--
Detect constant node ids that correspond to **LayerNorm affine parameters** (gamma/beta) emitted
by the TorchLean→IR compiler.

In the IR backend, `layer_norm(x, gamma, beta)` is lowered into:
1) `layernorm(x)` (pure normalization)
2) `mul_elem(layernorm(x), gammaB)` where `gammaB` is a broadcasted const
3) `add(..., betaB)` where `betaB` is a broadcasted const

Those broadcasted consts should be emitted as `nn.Parameter` in PyTorch, even if they are
uniform (gamma initialized to ones, beta to zeros).
-/
private def detectLayerNormAffineConstIds (g : NN.IR.Graph) : Std.HashSet Nat :=
  Id.run do
    let mut s : Std.HashSet Nat := {}
    for ln in g.nodes do
      match ln.kind with
      | .layernorm _ =>
          for mulN in g.nodes do
            match mulN.kind with
            | .mul_elem =>
                match mulN.parents with
                | [a, b] =>
                    let gammaId? : Option Nat :=
                      if a == ln.id && isConstId g b then some b
                      else if b == ln.id && isConstId g a then some a
                      else none
                    if let some gammaId := gammaId? then
                      if (g.nodes[gammaId]?.map (fun n => n.outShape) == some ln.outShape) then
                        s := s.insert gammaId
                      for addN in g.nodes do
                        match addN.kind with
                        | .add =>
                            match addN.parents with
                            | [p, q] =>
                                let betaId? : Option Nat :=
                                  if p == mulN.id && isConstId g q then some q
                                  else if q == mulN.id && isConstId g p then some p
                                  else none
                                if let some betaId := betaId? then
                                  if (g.nodes[betaId]?.map (fun n => n.outShape) == some
                                    ln.outShape) then
                                    s := s.insert betaId
                            | _ => ()
                        | _ => ()
                | _ => ()
            | _ => ()
      | _ => ()
    pure s

/--
Collect Python attribute bindings for all learnable parameters and constants.

This returns:
- a mapping from const node id to how to reference it (`ConstBindings`), and
- the list of Python `__init__` lines that materialize parameters/buffers.
-/
private def collectBindings (g : NN.IR.Graph) (ps : ParamStore Float) (opts : Options) :
    Except String (ConstBindings × List String) := do
  let mut bindings : ConstBindings := HashMap.emptyWithCapacity
  let mut initLines : List String := []
  let forcedLearnableConsts := detectLayerNormAffineConstIds g

  -- Pass 1: linear/conv2d parameters (stored in ParamStore keyed by node id).
  for n in g.nodes do
    match n.kind with
    | .linear =>
        match ps.linearWB.get? n.id with
        | none => throw s!"IR→PyTorch: missing linear params for node {n.id}"
        | some lp =>
            let wShape : Shape := .dim lp.m (.dim lp.n .scalar)
            let bShape : Shape := .dim lp.m .scalar
            let wFlat := tensorToPyFlat (s := wShape) lp.w
            let bFlat := tensorToPyFlat (s := bShape) lp.b
            initLines := initLines ++
              [ indentFour s!"self.{linearWAttr n.id} = nn.Parameter({pyTensorFromFlat wFlat wShape})"
              , indentFour s!"self.{linearBAttr n.id} = nn.Parameter({pyTensorFromFlat bFlat bShape})"
              ]
    | .conv2d .. =>
        match ps.conv2dCfg.get? n.id with
        | none => throw s!"IR→PyTorch: missing conv2d params for node {n.id}"
        | some cfg =>
            let kShape : Shape := .dim cfg.outC (.dim cfg.inC (.dim cfg.kH (.dim cfg.kW .scalar)))
            let bShape : Shape := .dim cfg.outC .scalar
            let kFlat := tensorToPyFlat (s := kShape) cfg.spec.kernel
            let bFlat := tensorToPyFlat (s := bShape) cfg.spec.bias
            initLines := initLines ++
              [ indentFour
                s!"self.{conv2dKernelAttr n.id} = nn.Parameter({pyTensorFromFlat kFlat kShape})"
              , indentFour
                s!"self.{conv2dBiasAttr n.id} = nn.Parameter({pyTensorFromFlat bFlat bShape})"
              ]
    | .batchNorm2dNchwEval .. =>
        match ps.batchNorm2dNchwEval.get? n.id with
        | none => throw s!"IR→PyTorch: missing batch_norm2d_nchw_eval params for node {n.id}"
        | some p =>
            let sC : Shape := .dim p.c .scalar
            initLines := initLines ++
              [ indentFour s!"self.{batchNormGammaAttr n.id} = nn.Parameter({pyTensorFromFlat (tensorToPyFlat (s := sC) p.gamma) sC})"
              , indentFour s!"self.{batchNormBetaAttr n.id} = nn.Parameter({pyTensorFromFlat (tensorToPyFlat (s := sC) p.beta) sC})"
              , indentFour s!"self.register_buffer(\"{batchNormMeanAttr n.id}\", {pyTensorFromFlat (tensorToPyFlat (s := sC) p.mean) sC})"
              , indentFour s!"self.register_buffer(\"{batchNormVarAttr n.id}\", {pyTensorFromFlat (tensorToPyFlat (s := sC) p.var) sC})"
              ]
    | _ => pure ()

  -- Pass 2: const nodes (stored as flattened values in ParamStore.constVals).
  for n in g.nodes do
    match n.kind with
    | .const s =>
        match ps.constVals.get? n.id with
        | none => throw s!"IR→PyTorch: missing const value for node {n.id}"
        | some fv =>
            let flatListStr := vectorTensorToPy (n := fv.n) fv.v
            let flatVals := vectorTensorToList (n := fv.n) fv.v
            let uniform := allEq flatVals
            let forceLearn :=
              opts.learnableConsts && forcedLearnableConsts.contains n.id
            let shouldLearn := opts.learnableConsts && (forceLearn || !uniform)
            if shouldLearn && opts.compressBroadcastParams then
              match s with
              | .dim seqLen (.dim embedDim .scalar) =>
                  match broadcastRow2D? seqLen embedDim flatVals with
                  | some row =>
                      let attr := s!"{constAttr n.id}_vec"
                      let rowT : Tensor Float (.dim embedDim .scalar) :=
                        Tensor.dim fun i => Tensor.scalar (row.getD i.val 0.0)
                      let rowFlat := tensorToPyFlat (s := .dim embedDim .scalar) rowT
                      let rowParam := pyTensorFromFlat rowFlat (.dim embedDim .scalar)
                      initLines := initLines ++
                        [ indentFour s!"self.{attr} = nn.Parameter({rowParam})" ]
                      bindings := bindings.insert n.id (.paramBroadcast2D attr seqLen embedDim)
                  | none =>
                      let attr := constAttr n.id
                      initLines := initLines ++
                        [ indentFour s!"self.{attr} = nn.Parameter({pyTensorFromFlat flatListStr s})" ]
                      bindings := bindings.insert n.id (.paramFull attr)
              | _ =>
                  let attr := constAttr n.id
                  initLines := initLines ++
                    [ indentFour s!"self.{attr} = nn.Parameter({pyTensorFromFlat flatListStr s})" ]
                  bindings := bindings.insert n.id (.paramFull attr)
            else
              let attr := constAttr n.id
              initLines := initLines ++
                [ indentFour s!"self.register_buffer(\"{attr}\", {pyTensorFromFlat flatListStr s})" ]
              bindings := bindings.insert n.id (.bufferFull attr)
    | _ => pure ()

  pure (bindings, initLines)

/-- Emit the Python expression used to reference a bound constant (`self.<attr>` plus expansions).
  -/
private def constExpr (bindings : ConstBindings) (id : Nat) : Except String String := do
  match bindings.get? id with
  | none => throw s!"IR→PyTorch: missing const binding for node {id}"
  | some b =>
      match b with
      | .bufferFull a => pure s!"self.{a}"
      | .paramFull a => pure s!"self.{a}"
      | .paramBroadcast2D a seqLen _embedDim =>
          pure s!"self.{a}.unsqueeze(0).expand({seqLen}, -1)"

/--
Emit the body of a Python `forward(self, x)` function for a given IR graph.

Each IR node `id` becomes a Python local `v{id}`. We emit nodes in graph order and end by returning
`v{outputId}`.
-/
private def emitForwardBody (g : NN.IR.Graph) (ps : ParamStore Float) (bindings : ConstBindings)
    (inputId outputId : Nat) : Except String (List String) := do
  -- Validate that input and output nodes exist.
  let _ ← getNode g inputId
  let outNode ← getNode g outputId
  let _ := outNode

  let mut lines : List String := []

  for n in g.nodes do
    let id := n.id
    match n.kind with
    | .input =>
        lines := lines ++ [indentFour s!"v{id} = x"]
    | .const _ =>
        let e ← constExpr bindings id
        lines := lines ++ [indentFour s!"v{id} = {e}"]
    | .randUniform seed =>
        let shp := shapeToPyTupleString n.outShape
        lines := lines ++
          [ indentFour s!"_gen{id} = torch.Generator(device=x.device)"
          , indentFour s!"_gen{id}.manual_seed({seed} + {id})"
          , indentFour
            s!"v{id} = torch.rand({shp}, generator=_gen{id}, device=x.device, dtype=x.dtype)"
          ]
    | .bernoulliMask seed =>
        let p ← expectUnary id n.parents
        let shp := shapeToPyTupleString n.outShape
        let rand :=
          s!"torch.rand({shp}, generator=_gen{id}, device=v{p}.device, dtype=v{p}.dtype)"
        lines := lines ++
          [ indentFour s!"_gen{id} = torch.Generator(device=v{p}.device)"
          , indentFour s!"_gen{id}.manual_seed({seed} + {id})"
          , indentFour s!"v{id} = ({rand} < v{p}).to(v{p}.dtype)"
          ]
    | .detach =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{p}.detach()"]
    | .add =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{a} + v{b}"]
    | .sub =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{a} - v{b}"]
    | .mul_elem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{a} * v{b}"]
    | .minElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.minimum(v{a}, v{b})"]
    | .maxElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.maximum(v{a}, v{b})"]
    | .sum =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.sum(v{p})"]
    | .reduceSum axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.sum(v{p}, dim={axis})"]
    | .reduceMean axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.mean(v{p}, dim={axis})"]
    | .broadcastTo _inShape outShape =>
        let p ← expectUnary id n.parents
        let shp := shapeToPyTupleString outShape
        lines := lines ++ [indentFour s!"v{id} = torch.broadcast_to(v{p}, {shp})"]
    | .matmul =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.matmul(v{a}, v{b})"]
    | .linear =>
        let xId ← expectUnary id n.parents
        match ps.linearWB.get? id with
        | none => throw s!"IR→PyTorch: missing linear params for node {id}"
        | some _ =>
            lines := lines ++
              [ indentFour s!"v{id} = F.linear(v{xId}, self.{linearWAttr id}, self.{linearBAttr id})" ]
    | .conv2d inC outC kH kW stride padding =>
        let xId ← expectUnary id n.parents
        -- We rely on the conv2d params stored in the ParamStore (kernel + bias).
        match ps.conv2dCfg.get? id with
        | none => throw s!"IR→PyTorch: missing conv2d params for node {id}"
        | some _ =>
            let wName := conv2dKernelAttr id
            let bName := conv2dBiasAttr id
            let convLine :=
              s!"_y = F.conv2d(_x, self.{wName}, self.{bName}, " ++
                s!"stride={stride}, padding={padding})"
            lines := lines ++
              [ indentFour s!"_x = v{xId}"
              , indentFour "_batched = (_x.dim() == 3)"
              , indentFour "if _batched:"
              , indentEight "_x = _x.unsqueeze(0)"
              , indentFour convLine
              , indentFour "if _batched:"
              , indentEight "_y = _y.squeeze(0)"
              , indentFour s!"v{id} = _y"
              ]
            let _ := (inC, outC, kH, kW) -- keep args for readability; already embedded in kind
    | .batchNorm2dNchwEval channels =>
        let xId ← expectUnary id n.parents
        match ps.batchNorm2dNchwEval.get? id with
        | none => throw s!"IR→PyTorch: missing batch_norm2d_nchw_eval params for node {id}"
        | some p =>
            lines := lines ++
              [ indentFour s!"_x = v{xId}"
              , indentFour "_batched = (_x.dim() == 4)"
              , indentFour "if not _batched:"
              , indentEight "_x = _x.unsqueeze(0)"
              , indentFour
                s!"_y = F.batch_norm(_x, self.{batchNormMeanAttr id}, self.{batchNormVarAttr id}, self.{batchNormGammaAttr id}, self.{batchNormBetaAttr id}, training=False, eps={p.eps})"
              , indentFour "if not _batched:"
              , indentEight "_y = _y.squeeze(0)"
              , indentFour s!"v{id} = _y"
              ]
            let _ := channels
    | .maxPool2d kH kW stride =>
        let xId ← expectUnary id n.parents
        lines := lines ++
          [ indentFour s!"_x = v{xId}"
          , indentFour "_batched = (_x.dim() == 3)"
          , indentFour "if _batched:"
          , indentEight "_x = _x.unsqueeze(0)"
          , indentFour s!"_y = F.max_pool2d(_x, kernel_size=({kH}, {kW}), stride={stride})"
          , indentFour "if _batched:"
          , indentEight "_y = _y.squeeze(0)"
          , indentFour s!"v{id} = _y"
          ]
    | .maxPool2dPad kH kW stride padding =>
        let xId ← expectUnary id n.parents
        lines := lines ++
          [ indentFour s!"_x = v{xId}"
          , indentFour "_batched = (_x.dim() == 3)"
          , indentFour "if _batched:"
          , indentEight "_x = _x.unsqueeze(0)"
          , indentFour
            s!"_y = F.max_pool2d(_x, kernel_size=({kH}, {kW}), stride={stride}, padding={padding})"
          , indentFour "if _batched:"
          , indentEight "_y = _y.squeeze(0)"
          , indentFour s!"v{id} = _y"
          ]
    | .avgPool2d kH kW stride =>
        let xId ← expectUnary id n.parents
        lines := lines ++
          [ indentFour s!"_x = v{xId}"
          , indentFour "_batched = (_x.dim() == 3)"
          , indentFour "if _batched:"
          , indentEight "_x = _x.unsqueeze(0)"
          , indentFour s!"_y = F.avg_pool2d(_x, kernel_size=({kH}, {kW}), stride={stride})"
          , indentFour "if _batched:"
          , indentEight "_y = _y.squeeze(0)"
          , indentFour s!"v{id} = _y"
          ]
    | .avgPool2dPad kH kW stride padding =>
        let xId ← expectUnary id n.parents
        lines := lines ++
          [ indentFour s!"_x = v{xId}"
          , indentFour "_batched = (_x.dim() == 3)"
          , indentFour "if _batched:"
          , indentEight "_x = _x.unsqueeze(0)"
          , indentFour
            s!"_y = F.avg_pool2d(_x, kernel_size=({kH}, {kW}), stride={stride}, padding={padding})"
          , indentFour "if _batched:"
          , indentEight "_y = _y.squeeze(0)"
          , indentFour s!"v{id} = _y"
          ]
    | .relu =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.relu(v{p})"]
    | .reluDeriv =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = (v{p} > 0).to(v{p}.dtype)"]
    | .tanh =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.tanh(v{p})"]
    | .sin =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.sin(v{p})"]
    | .cos =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.cos(v{p})"]
    | .sigmoid =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.sigmoid(v{p})"]
    | .exp =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.exp(v{p})"]
    | .log =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.log(v{p})"]
    | .inv =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.reciprocal(v{p})"]
    | .abs =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.abs(v{p})"]
    | .sqrt =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.sqrt(v{p})"]
    | .softmax axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = torch.softmax(v{p}, dim={axis})"]
    | .layernorm axis =>
        let p ← expectUnary id n.parents
        let dims := shapeDims n.outShape
        let normalized := dims.drop axis
        let normalizedShape :=
          match normalized with
          | [] => "()"
          | [d] => s!"({d},)"
          | _ => "(" ++ ", ".intercalate (normalized.map toString) ++ ")"
        lines := lines ++ [indentFour
          s!"v{id} = F.layer_norm(v{p}, normalized_shape={normalizedShape})"]
    | .reshape _inShape outShape =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{p}.reshape({shapeToPyTupleString outShape})"]
    | .flatten _ =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{p}.reshape({shapeToPyTupleString n.outShape})"]
    | .permute perm =>
        let p ← expectUnary id n.parents
        if perm.isEmpty then
          lines := lines ++ [indentFour s!"v{id} = v{p}"]
        else
          let permStr := ", ".intercalate (perm.map toString)
          lines := lines ++ [indentFour s!"v{id} = v{p}.permute({permStr})"]
    | .concat axis =>
        if n.parents.isEmpty then
          throw s!"IR→PyTorch: node {id}: concat expects ≥1 parent"
        else
          let args := ", ".intercalate (n.parents.map (fun p => s!"v{p}"))
          lines := lines ++ [indentFour s!"v{id} = torch.cat([{args}], dim={axis})"]
    | .swap_first_two =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{p}.transpose(0, 1)"]
    | .transpose3dLastTwo =>
        let p ← expectUnary id n.parents
        lines := lines ++ [indentFour s!"v{id} = v{p}.transpose(-1, -2)"]
    | .mseLoss =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ [indentFour s!"v{id} = F.mse_loss(v{a}, v{b}, reduction='mean')"]

  lines := lines ++ [indentFour s!"return v{outputId}"]
  pure lines

/--
Emit a standalone PyTorch `nn.Module` class for an IR graph.

This is the main entrypoint for IR exporters: it bundles:
- imports,
- a class definition with parameters/buffers materialized from the `ParamStore`,
- a `forward` method implementing the IR, and
- (optionally) a compact training file for export checks.
-/
def emit
    (g : NN.IR.Graph) (ps : ParamStore Float) (inputId outputId : Nat)
    (opts : Options := {}) :
    Except String String := do
  let inNode ← getNode g inputId
  let outNode ← getNode g outputId
  let inputShape := inNode.outShape
  let outputShape := outNode.outShape

  let (bindings, initParamLines) ← collectBindings g ps opts
  let forwardBody ← emitForwardBody g ps bindings inputId outputId

  let imports : List String :=
    [ "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    ]

  let classHeader : List String :=
    [ s!"class {opts.className}(nn.Module):"
    , indentTwo "def __init__(self):"
    , indentFour "super().__init__()"
    , indentFour s!"dtype = {opts.dtypeExpr}"
    ]

  let classForwardHeader : List String :=
    [ ""
    , indentTwo "def forward(self, x):"
    ]

  let classLines : List String :=
    classHeader ++ initParamLines ++ classForwardHeader ++ forwardBody

  let helpers : List String :=
    if !opts.includeTrainingSkeleton then
      []
    else
      let outIsLoss : Bool :=
        match outNode.kind with
        | .mseLoss => true
        | _ => false
      let xTuple := shapeToPyTupleString inputShape
      let yTuple := shapeToPyTupleString outputShape
      let (trainStepDef, mainLines) :=
        if outIsLoss then
          ( [ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, opt=None):"
            , indentTwo "model.train()"
            , indentTwo "if opt is not None:"
            , indentFour "opt.zero_grad(set_to_none=True)"
            , indentTwo "loss = model(x)"
            , indentTwo "if opt is not None:"
            , indentFour "loss.backward()"
            , indentFour "opt.step()"
            , indentTwo "return float(loss.detach().cpu())"
            ]
          , [ ""
            , "if __name__ == '__main__':"
            , indentTwo "torch.manual_seed(0)"
            , indentTwo "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indentTwo s!"model = {opts.className}().to(device)"
            , indentTwo "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indentTwo s!"x = torch.randn({xTuple}, dtype={opts.dtypeExpr}, device=device)"
            , indentTwo "loss = train_step(model, x, opt)"
            , indentTwo "print('loss', loss)"
            ] )
        else
          ( [ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, opt=None):"
            , indentTwo "model.train()"
            , indentTwo "if opt is not None:"
            , indentFour "opt.zero_grad(set_to_none=True)"
            , indentTwo "out = model(x)"
            , indentTwo "loss = F.mse_loss(out, y, reduction='mean')"
            , indentTwo "if opt is not None:"
            , indentFour "loss.backward()"
            , indentFour "opt.step()"
            , indentTwo "return float(loss.detach().cpu())"
            ]
          , [ ""
            , "if __name__ == '__main__':"
            , indentTwo "torch.manual_seed(0)"
            , indentTwo "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indentTwo s!"model = {opts.className}().to(device)"
            , indentTwo "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indentTwo s!"x = torch.randn({xTuple}, dtype={opts.dtypeExpr}, device=device)"
            , indentTwo s!"y = torch.randn({yTuple}, dtype={opts.dtypeExpr}, device=device)"
            , indentTwo "loss = train_step(model, x, y, opt)"
            , indentTwo "print('loss', loss)"
            ] )

      [ ""
      , "def make_optimizer(model: nn.Module, kind: str = 'adam', lr: float = 1e-3):"
      , indentTwo "params = [p for p in model.parameters() if p.requires_grad]"
      , indentTwo "if len(params) == 0:"
      , indentFour "return None"
      , indentTwo "if kind == 'sgd':"
      , indentFour "return torch.optim.SGD(params, lr=lr)"
      , indentTwo "if kind == 'adam':"
      , indentFour "return torch.optim.Adam(params, lr=lr)"
      , indentTwo "raise ValueError(f'unknown optimizer kind: {kind}')"
      ] ++ trainStepDef ++ mainLines

  pure (joinLines (imports ++ classLines ++ helpers))

end IRPyTorch
end Export
