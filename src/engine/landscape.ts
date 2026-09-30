// Phase 2: the loss landscape and the cells the activations carve into it.
//
// A slice of parameter space: θ(a) = θ₀ + Σₖ aₖ uₖ, with 1–3 directions uₖ.
// At every point we evaluate the full-dataset loss and accuracy, and the
// activation pattern: for every MLP neuron at every position of every
// example, is it on (pre-activation > 0)? With ReLU and the pattern held
// fixed, every ReLU is a fixed 0/1 mask, so inside one pattern cell the loss
// is a smooth function of the parameters; it can only kink on the cell
// walls. Those cells are the volumes the activation functions carve into
// the loss landscape. Two more carvings: which key each attention query
// weights most (softmax is smooth, but its "decision" flips), and the
// predicted token at every supervised position (the 0–1 loss is constant
// inside each prediction cell).

import { Model } from './model';
import { crossEntropy } from './ops';
import { RNG } from './rng';
import { symEig } from './linalg';
import type { Spec } from './spec';
import { buildDataset, packBatch, type Dataset, type Example } from './tasks';
import { withGrad } from './tensor';

export type DirKind = 'weight' | 'tensor' | 'random' | 'gradient' | 'hessian' | 'pca' | 'init';
export type DirNorm = 'filter' | 'unit' | 'none';

export interface DirSpec {
  kind: DirKind;
  /** weight / tensor: parameter name */
  name: string;
  /** weight: flat index within the tensor */
  index: number;
  /** random: seed */
  seed: number;
  /** hessian: eigenvector rank (0 = largest); pca: component */
  k: number;
}

/** What a carving hash covers. The training loss's true cells are the
 *  overlay of every example's own cells, so they shrink as examples are
 *  added; the coarser carvings make that visible. */
export interface CarveOpts {
  /** 'examples' carving: the first k examples */
  k: number;
  /** 'neuron' / 'layer' carvings */
  layer: number;
  neuron: number;
}

export const CARVES = ['mlp', 'examples', 'layer', 'neuron', 'attn', 'pred'] as const;
export type CarveKey = (typeof CARVES)[number];

/** Packed evaluation of one parameter point. */
export interface PointEval {
  loss: number;
  acc: number;
  tokAcc: number;
  /** 2×32-bit hashes, one per carving */
  h: Record<CarveKey, [number, number]>;
  /** MLP (neuron, position) switches that differ from the center point */
  flips: number;
  /** fraction of MLP (neuron, position) pairs that are on */
  active: number;
}

export const EVAL_STRIDE = 5 + 2 * CARVES.length;

/** examples evaluated per landscape point (the default tasks have far fewer) */
export const EVAL_CAP = 2048;

export function packEval(e: PointEval, out: Float64Array, o: number): void {
  out[o] = e.loss;
  out[o + 1] = e.acc;
  out[o + 2] = e.tokAcc;
  out[o + 3] = e.flips;
  out[o + 4] = e.active;
  CARVES.forEach((c, i) => {
    out[o + 5 + 2 * i] = e.h[c][0];
    out[o + 6 + 2 * i] = e.h[c][1];
  });
}

export function unpackEval(a: Float64Array, o: number): PointEval {
  const h = {} as Record<CarveKey, [number, number]>;
  CARVES.forEach((c, i) => (h[c] = [a[o + 5 + 2 * i], a[o + 6 + 2 * i]]));
  return { loss: a[o], acc: a[o + 1], tokAcc: a[o + 2], flips: a[o + 3], active: a[o + 4], h };
}

/** Two independent FNV-1a hashes over a byte stream. */
class Hasher {
  a = 2166136261;
  b = 2166136261 ^ 0x5bd1e995;
  push(v: number) {
    this.a = Math.imul(this.a ^ (v & 255), 16777619);
    this.b = Math.imul(this.b ^ ((v * 131 + 7) & 255), 16777619);
  }
  get(): [number, number] {
    return [this.a >>> 0, this.b >>> 0];
  }
}

/** A model + dataset ready to evaluate arbitrary parameter vectors. */
export class Evaluator {
  readonly model: Model;
  readonly data: Dataset;
  readonly examples: Example[];
  private x: Int32Array;
  private y: Int32Array;
  private B: number;
  private centerBits: Uint8Array | null = null;
  carve: CarveOpts = { k: 1, layer: 0, neuron: 0 };

  constructor(spec: Spec, split: 'train' | 'val' | 'all' = 'train', cap = EVAL_CAP) {
    this.data = buildDataset(spec.task);
    this.model = new Model(spec.model, this.data.vocab.length, this.data.T, spec.init);
    const ex = split === 'train' ? this.data.train : split === 'val' ? this.data.val : [...this.data.train, ...this.data.val];
    this.examples = (ex.length ? ex : this.data.train).slice(0, cap);
    const p = packBatch(this.examples, this.data.T);
    this.x = p.x;
    this.y = p.y;
    this.B = p.B;
  }

  get numParams(): number {
    return this.model.numParams();
  }

  /** MLP on/off bits for the current weights: all layers, all positions. */
  private mlpBits(trace: Map<string, { data: Float64Array }>): Uint8Array {
    const s = this.model.spec;
    const chunks: Float64Array[] = [];
    for (let l = 0; l < s.nLayer; l++) if (s.layers[l].mlp) chunks.push(trace.get(`L${l}.mlp.pre`)!.data);
    const n = chunks.reduce((a, c) => a + c.length, 0);
    const bits = new Uint8Array(n);
    let o = 0;
    for (const c of chunks) {
      for (let i = 0; i < c.length; i++) bits[o + i] = c[i] > 0 ? 1 : 0;
      o += c.length;
    }
    return bits;
  }

  /** Remember the pattern at θ₀ so later points report their flips from it. */
  setCenter(theta: ArrayLike<number>): void {
    this.model.setFlat(theta);
    const fwd = withGrad(false, () => this.model.forward(this.x, this.B, true));
    this.centerBits = this.mlpBits(fwd.trace);
  }

  evaluate(theta: ArrayLike<number>): PointEval {
    const m = this.model;
    m.setFlat(theta);
    const T = this.data.T, V = m.vocab, s = m.spec;
    return withGrad(false, () => {
      const fwd = m.forward(this.x, this.B, true);
      const probs: { value?: Float64Array } = {};
      const loss = crossEntropy(fwd.logits, this.y, probs).data[0];
      const P = probs.value!;
      // predictions + accuracy
      const hp = new Hasher();
      let tokOk = 0, tokN = 0, seqOk = 0;
      for (let b = 0; b < this.B; b++) {
        let ok = true;
        for (let t = 0; t < T; t++) {
          const i = b * T + t;
          if (this.y[i] < 0) continue;
          let best = 0;
          for (let v = 1; v < V; v++) if (P[i * V + v] > P[i * V + best]) best = v;
          hp.push(best);
          tokN++;
          if (best === this.y[i]) tokOk++;
          else ok = false;
        }
        if (ok) seqOk++;
      }
      // MLP pattern: every (layer, row = example·position, neuron)
      const bits = this.mlpBits(fwd.trace);
      let on = 0, flips = 0;
      for (let i = 0; i < bits.length; i++) {
        on += bits[i];
        if (this.centerBits && this.centerBits.length === bits.length && this.centerBits[i] !== bits[i]) flips++;
      }
      const F = s.dMlp, rows = this.B * T;
      const mlpLayers: number[] = [];
      for (let l = 0; l < s.nLayer; l++) if (s.layers[l].mlp) mlpLayers.push(l);
      const hAll = new Hasher(), hEx = new Hasher(), hLayer = new Hasher(), hNeuron = new Hasher();
      const kEx = Math.max(1, Math.min(this.B, this.carve.k)) * T;
      const selL = mlpLayers.indexOf(this.carve.layer) >= 0 ? mlpLayers.indexOf(this.carve.layer) : 0;
      const selN = Math.max(0, Math.min(F - 1, this.carve.neuron));
      mlpLayers.forEach((_, li) => {
        const base = li * rows * F;
        for (let r = 0; r < rows; r++) {
          let byte = 0, nb = 0;
          for (let j = 0; j < F; j++) {
            const bit = bits[base + r * F + j];
            byte |= bit << nb;
            if (++nb === 8 || j === F - 1) {
              hAll.push(byte);
              if (r < kEx) hEx.push(byte);
              if (li === selL) hLayer.push(byte);
              byte = 0;
              nb = 0;
            }
          }
          hNeuron.push(li === selL ? bits[base + r * F + selN] : 2);
        }
      });
      // attention argmax pattern
      const ha = new Hasher();
      for (let l = 0; l < s.nLayer; l++) {
        if (!s.layers[l].attn) continue;
        for (let h = 0; h < s.nHead; h++) {
          const pr = fwd.trace.get(`L${l}.attn.probs.h${h}`)!.data;
          for (let r = 0; r < pr.length / T; r++) {
            let best = 0;
            for (let j = 1; j < T; j++) if (pr[r * T + j] > pr[r * T + best]) best = j;
            ha.push(best);
          }
        }
      }
      return {
        loss,
        acc: seqOk / Math.max(1, this.B),
        tokAcc: tokOk / Math.max(1, tokN),
        h: { mlp: hAll.get(), examples: hEx.get(), layer: hLayer.get(), neuron: hNeuron.get(), attn: ha.get(), pred: hp.get() },
        flips,
        active: bits.length ? on / bits.length : 0,
      };
    });
  }

  /** Full-batch loss gradient at θ. */
  gradient(theta: ArrayLike<number>): { loss: number; grad: Float64Array } {
    const m = this.model;
    m.setFlat(theta);
    m.zeroGrad();
    const loss = withGrad(true, () => crossEntropy(m.forward(this.x, this.B).logits, this.y));
    loss.backward();
    const g = m.flatGrad();
    m.zeroGrad();
    return { loss: loss.data[0], grad: g };
  }

  /** Hessian-vector product by central differences of the gradient. */
  hvp(theta: ArrayLike<number>, v: Float64Array, eps = 1e-4): Float64Array {
    const n = v.length;
    const vn = Math.sqrt(dot(v, v)) || 1;
    const h = eps / vn;
    const tp = new Float64Array(n), tm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      tp[i] = theta[i] + h * v[i];
      tm[i] = theta[i] - h * v[i];
    }
    const gp = this.gradient(tp).grad, gm = this.gradient(tm).grad;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = (gp[i] - gm[i]) / (2 * h);
    return out;
  }

  /**
   * Extreme eigenpairs of the Hessian at θ (Lanczos with full
   * reorthogonalization). Returns Ritz values descending and their vectors.
   */
  hessianEigen(theta: ArrayLike<number>, iters = 24, seed = 1, mask?: Uint8Array): { values: number[]; vectors: Float64Array[] } {
    const n = theta.length;
    const m = Math.min(iters, n);
    const rng = new RNG(seed);
    const Q: Float64Array[] = [];
    const alpha: number[] = [], beta: number[] = [];
    let q: Float64Array = new Float64Array(n);
    for (let i = 0; i < n; i++) q[i] = mask && !mask[i] ? 0 : rng.normal();
    scaleTo(q, 1);
    let prev: Float64Array | null = null;
    let b = 0;
    for (let j = 0; j < m; j++) {
      Q.push(q);
      const w: Float64Array = this.hvp(theta, q);
      if (mask) for (let i = 0; i < n; i++) if (!mask[i]) w[i] = 0;
      const a = dot(w, q);
      alpha.push(a);
      for (let i = 0; i < n; i++) w[i] -= a * q[i] + (prev ? b * prev[i] : 0);
      // full reorthogonalization (twice is enough)
      for (let r = 0; r < 2; r++) for (const qq of Q) axpy(w, qq, -dot(w, qq));
      b = Math.sqrt(dot(w, w));
      if (b < 1e-10 || j === m - 1) break;
      beta.push(b);
      prev = q;
      q = w;
      for (let i = 0; i < n; i++) q[i] /= b;
    }
    const k = alpha.length;
    const Tm = new Float64Array(k * k);
    for (let i = 0; i < k; i++) {
      Tm[i * k + i] = alpha[i];
      if (i + 1 < k) Tm[i * k + i + 1] = Tm[(i + 1) * k + i] = beta[i];
    }
    const { values, vectors } = symEig(Tm, k);
    const vecs = vectors.map((y) => {
      const v = new Float64Array(n);
      for (let j = 0; j < k; j++) axpy(v, Q[j], y[j]);
      return v;
    });
    return { values, vectors: vecs };
  }
}

// ---- directions ----

/** A random direction, filter-normalized (Li et al. 2018): each filter (one
 *  neuron's incoming weights: a column of an x @ W matrix, a row of an
 *  embedding table) is scaled to the norm of the same filter of θ; biases
 *  and norm gains get no component. */
export function randomDirection(model: Model, theta: ArrayLike<number>, seed: number, norm: DirNorm): Float64Array {
  const rng = new RNG(seed * 7919 + 17);
  const d = new Float64Array(model.numParams());
  let o = 0;
  for (const p of model.params) {
    const R = p.t.rows, C = p.t.cols;
    const matrix = p.kind === 'weight' || p.kind === 'embed';
    for (let i = 0; i < p.t.size; i++) d[o + i] = norm === 'filter' && !matrix ? 0 : rng.normal();
    if (norm === 'filter' && matrix) {
      // index of element i of filter f
      const byCol = p.kind === 'weight';
      const nF = byCol ? C : R, len = byCol ? R : C;
      const at = (f: number, i: number) => o + (byCol ? i * C + f : f * C + i);
      for (let f = 0; f < nF; f++) {
        let dn = 0, tn = 0;
        for (let i = 0; i < len; i++) {
          dn += d[at(f, i)] ** 2;
          tn += Number(theta[at(f, i)]) ** 2;
        }
        const sc = dn > 0 ? Math.sqrt(tn / dn) : 0;
        for (let i = 0; i < len; i++) d[at(f, i)] *= sc;
      }
    }
    o += p.t.size;
  }
  if (norm === 'unit') scaleTo(d, 1);
  return d;
}

export function unitDirection(model: Model, name: string, index: number): Float64Array {
  const d = new Float64Array(model.numParams());
  const off = model.offsets().get(name);
  if (off === undefined) throw new Error(`no param ${name}`);
  d[off + Math.max(0, Math.min(index, model.byName.get(name)!.t.size - 1))] = 1;
  return d;
}

/** Scale one whole tensor: u = θ restricted to that tensor (a = −1 zeroes it, a = 1 doubles it). */
export function tensorDirection(model: Model, theta: ArrayLike<number>, name: string): Float64Array {
  const d = new Float64Array(model.numParams());
  const off = model.offsets().get(name);
  const p = model.byName.get(name);
  if (off === undefined || !p) throw new Error(`no param ${name}`);
  for (let i = 0; i < p.t.size; i++) d[off + i] = Number(theta[off + i]);
  return d;
}

/** PCA of a trajectory (rows = checkpoints) about its last point. */
export function trajectoryPCA(points: Float64Array[], center: ArrayLike<number>, k: number): { dirs: Float64Array[]; explained: number[] } {
  const n = points.length, P = center.length;
  if (n < 2) return { dirs: [], explained: [] };
  // Gram matrix trick: n is small (≤ 400), P may be larger
  const X = points.map((p) => {
    const r = new Float64Array(P);
    for (let i = 0; i < P; i++) r[i] = p[i] - Number(center[i]);
    return r;
  });
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) G[i * n + j] = G[j * n + i] = dot(X[i], X[j]);
  const { values, vectors } = symEig(G, n);
  const tot = values.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  const dirs: Float64Array[] = [];
  const explained: number[] = [];
  for (let c = 0; c < Math.min(k, n); c++) {
    const d = new Float64Array(P);
    for (let i = 0; i < n; i++) axpy(d, X[i], vectors[c][i]);
    const nn = Math.sqrt(dot(d, d));
    if (nn > 0) for (let i = 0; i < P; i++) d[i] /= nn;
    // scale so that a = ±1 spans the trajectory's extent along this axis
    let ext = 0;
    for (const x of X) ext = Math.max(ext, Math.abs(dot(x, d)));
    for (let i = 0; i < P; i++) d[i] *= ext || 1;
    dirs.push(d);
    explained.push(Math.max(0, values[c]) / tot);
  }
  return { dirs, explained };
}

/** Gram–Schmidt: make each direction orthogonal to the earlier ones, keeping its length. */
export function orthogonalize(dirs: Float64Array[]): Float64Array[] {
  const out: Float64Array[] = [];
  for (const d of dirs) {
    const len = Math.sqrt(dot(d, d));
    const v = d.slice();
    for (const u of out) {
      const uu = dot(u, u);
      if (uu > 0) axpy(v, u, -dot(v, u) / uu);
    }
    const l2 = Math.sqrt(dot(v, v));
    if (l2 > 1e-12 && len > 0) for (let i = 0; i < v.length; i++) v[i] *= len / l2;
    out.push(v);
  }
  return out;
}

/** Least-squares coordinates of θ in the slice: a = argmin |θ − θ₀ − U a|, plus the off-slice distance. */
export function projectToSlice(theta: ArrayLike<number>, center: ArrayLike<number>, dirs: Float64Array[]): { a: number[]; off: number } {
  const k = dirs.length, n = center.length;
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) r[i] = Number(theta[i]) - Number(center[i]);
  const G = new Float64Array(k * k), bb = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    bb[i] = dot(dirs[i], r);
    for (let j = 0; j < k; j++) G[i * k + j] = dot(dirs[i], dirs[j]);
  }
  const a = solveSmall(G, bb, k);
  const res = r.slice();
  for (let i = 0; i < k; i++) axpy(res, dirs[i], -a[i]);
  return { a, off: Math.sqrt(dot(res, res)) };
}

function solveSmall(A: Float64Array, b: Float64Array, k: number): number[] {
  const M = Array.from({ length: k }, (_, i) => [...Array.from(A.subarray(i * k, i * k + k)), b[i]]);
  for (let c = 0; c < k; c++) {
    let piv = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    if (Math.abs(M[c][c]) < 1e-18) continue;
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let j = c; j <= k; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row, i) => (Math.abs(row[i]) < 1e-18 ? 0 : row[k] / row[i]));
}

export function thetaAt(center: ArrayLike<number>, dirs: Float64Array[], a: ArrayLike<number>, out?: Float64Array): Float64Array {
  const n = center.length;
  const t = out ?? new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let v = Number(center[i]);
    for (let k = 0; k < dirs.length; k++) v += a[k] * dirs[k][i];
    t[i] = v;
  }
  return t;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function axpy(y: Float64Array, x: ArrayLike<number>, a: number) {
  for (let i = 0; i < y.length; i++) y[i] += a * x[i];
}

function scaleTo(v: Float64Array, len: number) {
  const n = Math.sqrt(dot(v, v)) || 1;
  for (let i = 0; i < v.length; i++) v[i] *= len / n;
}

/** Coarse-to-fine visiting order for a grid of n points per axis, k axes:
 *  the first few hundred points already cover the whole slice. */
export function coarseToFine(n: number, k: number): number[] {
  const total = Math.pow(n, k);
  const seen = new Uint8Array(total);
  const order: number[] = [];
  let stride = 1;
  while (stride * 2 < n) stride *= 2;
  for (; stride >= 1; stride = Math.floor(stride / 2)) {
    for (let idx = 0; idx < total; idx++) {
      if (seen[idx]) continue;
      let ok = true, r = idx;
      for (let d = 0; d < k; d++) {
        if ((r % n) % stride !== 0) ok = false;
        r = Math.floor(r / n);
      }
      if (ok) {
        seen[idx] = 1;
        order.push(idx);
      }
    }
    if (stride === 1) break;
  }
  return order;
}
