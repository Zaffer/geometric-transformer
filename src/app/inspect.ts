// Forward passes for the views, cached by version. The "view set" is every
// example (validation first, capped) run through the mirror model with a
// full trace: the point clouds, patterns and cells are all read from it.
// The "probe" is the one sequence under the magnifying glass.

import type { ForwardResult } from '../engine/model';
import { packBatch, type Example } from '../engine/tasks';
import { Tensor, withGrad } from '../engine/tensor';
import { crossEntropy } from '../engine/ops';
import type { Session } from './session';
import { rt, ui } from './state';

export const VIEW_CAP = 256;

export interface ViewSet {
  examples: Example[];
  /** true where the example is from the validation split */
  isVal: boolean[];
  B: number;
  T: number;
  x: Int32Array;
  y: Int32Array;
  fwd: ForwardResult;
  /** softmax of the logits, [B*T, V] */
  probs: Float64Array;
}

export interface Probe {
  tokens: Int32Array;
  targets: Int32Array | null;
  fwd: ForwardResult;
  probs: Float64Array;
  /** index in the view set, or -1 for a custom sequence */
  index: number;
}

let session: Session;
let vsKey = '';
let vs: ViewSet | null = null;
let prKey = '';
let pr: Probe | null = null;
let gradKey = '';
let grads: Map<string, Float64Array> | null = null;

export function initInspect(s: Session): void {
  session = s;
}

function softmax(logits: Tensor): Float64Array {
  const V = logits.cols, P = new Float64Array(logits.size);
  for (let i = 0; i < logits.rows; i++) {
    let mx = -Infinity;
    for (let j = 0; j < V; j++) mx = Math.max(mx, logits.data[i * V + j]);
    let s = 0;
    for (let j = 0; j < V; j++) s += P[i * V + j] = Math.exp(logits.data[i * V + j] - mx);
    for (let j = 0; j < V; j++) P[i * V + j] /= s;
  }
  return P;
}

/** The traced view set. Reactive: re-read inside an effect to follow edits. */
export function viewSet(): ViewSet {
  const key = `${rt.modelVersion()}:${rt.paramsVersion()}`;
  if (vs && key === vsKey) return vs;
  const d = session.data;
  const examples = [...d.val, ...d.train].slice(0, VIEW_CAP);
  const isVal = examples.map((_, i) => i < d.val.length);
  const { x, y, B } = packBatch(examples, d.T);
  const fwd = withGrad(false, () => session.model.forward(x, B, true));
  vs = { examples, isVal, B, T: d.T, x, y, fwd, probs: softmax(fwd.logits) };
  vsKey = key;
  return vs;
}

/** The probe sequence, traced. */
export function probe(): Probe {
  const custom = ui.probeTokens();
  const idx = ui.probe();
  const key = `${rt.modelVersion()}:${rt.paramsVersion()}:${idx}:${custom?.join(',') ?? ''}`;
  if (pr && key === prKey) return pr;
  const v = viewSet();
  let tokens: Int32Array, targets: Int32Array | null, index: number;
  if (custom && custom.length) {
    tokens = Int32Array.from(custom.slice(0, session.data.T));
    targets = null;
    index = -1;
  } else {
    index = Math.min(idx, v.examples.length - 1);
    tokens = v.examples[index].x;
    targets = v.examples[index].y;
  }
  const fwd = withGrad(false, () => session.model.forward(tokens, 1, true));
  pr = { tokens, targets, fwd, probs: softmax(fwd.logits), index };
  prKey = key;
  return pr;
}

/** Full-batch gradient of the training loss for every tensor (for the grad heatmaps). */
export function fullGrad(): Map<string, Float64Array> {
  const key = `${rt.modelVersion()}:${rt.paramsVersion()}`;
  if (grads && key === gradKey) return grads;
  const d = session.data, m = session.model;
  const ex = d.train.slice(0, 1024);
  const { x, y, B } = packBatch(ex, d.T);
  m.zeroGrad();
  const loss = withGrad(true, () => crossEntropy(m.forward(x, B).logits, y));
  loss.backward();
  grads = new Map(m.params.map((p) => [p.name, p.t.grad ? p.t.grad.slice() : new Float64Array(p.t.size)]));
  m.zeroGrad();
  gradKey = key;
  return grads;
}

/** Rows [b*T, (b+1)*T) of a traced tensor, for one sequence. */
export function rowsOf(t: Tensor, b: number, T: number): Float64Array {
  return t.data.subarray(b * T * t.cols, (b + 1) * T * t.cols);
}
