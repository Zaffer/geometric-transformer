// The whole transformer. One layer, one attention head, one ReLU MLP, no norm.
// The residual stream is 2-dimensional, so every vector the model ever holds is
// a point on one plane.
//
//   x_i   = E[token_i] + P[i]                       a point
//   s_ij  = (WQ x_i) . (WK x_j)      j <= i         a score per earlier point
//   a_i   = softmax_j(s_ij)
//   r1_i  = x_i + WO sum_j a_ij (WV x_j)            pulled toward what it attends to
//   r2_i  = r1_i + W2 relu(W1 r1_i + b1)            folded along each neuron's line
//   logit = U r2_i + bU                             one plane per token; highest wins
//
// All parameters live in one flat vector so that "any single parameter" is an index.

export const D = 2;

export interface Config {
  V: number; // vocabulary size
  T: number; // context length
  H: number; // MLP neurons
}

export interface Block {
  name: BlockName;
  off: number;
  rows: number;
  cols: number;
}

export const BLOCK_NAMES = ['E', 'P', 'WQ', 'WK', 'WV', 'WO', 'W1', 'b1', 'W2', 'U', 'bU'] as const;
export type BlockName = (typeof BLOCK_NAMES)[number];

export interface Layout {
  blocks: Record<BlockName, Block>;
  size: number;
}

export function layout(c: Config): Layout {
  const shapes: Record<BlockName, [number, number]> = {
    E: [c.V, D], P: [c.T, D],
    WQ: [D, D], WK: [D, D], WV: [D, D], WO: [D, D],
    W1: [c.H, D], b1: [c.H, 1], W2: [D, c.H],
    U: [c.V, D], bU: [c.V, 1],
  };
  const blocks = {} as Record<BlockName, Block>;
  let off = 0;
  for (const name of BLOCK_NAMES) {
    const [rows, cols] = shapes[name];
    blocks[name] = { name, off, rows, cols };
    off += rows * cols;
  }
  return { blocks, size: off };
}

/** Index of element (r, c) of a block in the flat parameter vector. */
export const at = (b: Block, r: number, c = 0) => b.off + r * b.cols + c;

/** Which block and element a flat index is. */
export function locate(L: Layout, i: number): { block: Block; r: number; c: number } {
  for (const name of BLOCK_NAMES) {
    const b = L.blocks[name];
    if (i < b.off + b.rows * b.cols) {
      const k = i - b.off;
      return { block: b, r: Math.floor(k / b.cols), c: k % b.cols };
    }
  }
  throw new Error(`parameter index ${i} out of range`);
}

// Small deterministic RNG (mulberry32) with a Gaussian on top.
export function rng(seed: number) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal, int: (n: number) => Math.floor(uniform() * n) };
}

export function init(c: Config, seed: number): Float64Array {
  const L = layout(c);
  const th = new Float64Array(L.size);
  const r = rng(seed);
  const std: Record<BlockName, number> = {
    E: 1, P: 0.3, WQ: 0.7, WK: 0.7, WV: 0.7, WO: 0.7, W1: 1, b1: 0.3, W2: 0.5, U: 1, bU: 0,
  };
  for (const name of BLOCK_NAMES) {
    const b = L.blocks[name];
    for (let i = 0; i < b.rows * b.cols; i++) th[b.off + i] = std[name] * r.normal();
  }
  return th;
}

/** Everything the forward pass computes for one sequence, kept for the views and for backward. */
export interface Trace {
  n: number;
  tokens: number[];
  x: Float64Array; // n x 2   embedded points
  q: Float64Array; // n x 2
  k: Float64Array; // n x 2
  v: Float64Array; // n x 2
  s: Float64Array; // n x n   scores (j <= i used)
  a: Float64Array; // n x n   attention weights
  z: Float64Array; // n x 2   attended average of values
  r1: Float64Array; // n x 2  after attention
  pre: Float64Array; // n x H pre-activations
  r2: Float64Array; // n x 2  after MLP
  logits: Float64Array; // n x V
  p: Float64Array; // n x V   softmax(logits)
}

export function forward(c: Config, L: Layout, th: Float64Array, tokens: number[]): Trace {
  const n = tokens.length, { V, H } = c, B = L.blocks;
  const t: Trace = {
    n, tokens,
    x: new Float64Array(n * 2), q: new Float64Array(n * 2), k: new Float64Array(n * 2), v: new Float64Array(n * 2),
    s: new Float64Array(n * n), a: new Float64Array(n * n), z: new Float64Array(n * 2),
    r1: new Float64Array(n * 2), pre: new Float64Array(n * H), r2: new Float64Array(n * 2),
    logits: new Float64Array(n * V), p: new Float64Array(n * V),
  };
  const mat = (b: Block, src: Float64Array, i: number, dst: Float64Array) => {
    const x0 = src[2 * i], x1 = src[2 * i + 1];
    dst[2 * i] = th[b.off] * x0 + th[b.off + 1] * x1;
    dst[2 * i + 1] = th[b.off + 2] * x0 + th[b.off + 3] * x1;
  };
  for (let i = 0; i < n; i++) {
    t.x[2 * i] = th[at(B.E, tokens[i], 0)] + th[at(B.P, i, 0)];
    t.x[2 * i + 1] = th[at(B.E, tokens[i], 1)] + th[at(B.P, i, 1)];
    mat(B.WQ, t.x, i, t.q); mat(B.WK, t.x, i, t.k); mat(B.WV, t.x, i, t.v);
  }
  for (let i = 0; i < n; i++) {
    let max = -Infinity;
    for (let j = 0; j <= i; j++) {
      const s = t.q[2 * i] * t.k[2 * j] + t.q[2 * i + 1] * t.k[2 * j + 1];
      t.s[i * n + j] = s;
      if (s > max) max = s;
    }
    let sum = 0;
    for (let j = 0; j <= i; j++) sum += t.a[i * n + j] = Math.exp(t.s[i * n + j] - max);
    for (let j = 0; j <= i; j++) {
      const a = (t.a[i * n + j] /= sum);
      t.z[2 * i] += a * t.v[2 * j];
      t.z[2 * i + 1] += a * t.v[2 * j + 1];
    }
    tail(c, L, th, t.x[2 * i], t.x[2 * i + 1], t.z[2 * i], t.z[2 * i + 1], t, i);
  }
  return t;
}

/** Everything after the attention average: WO, add, MLP, add, unembed, softmax. Writes row i of the trace. */
function tail(c: Config, L: Layout, th: Float64Array, x0: number, x1: number, z0: number, z1: number, t: Trace, i: number) {
  const { V, H } = c, B = L.blocks;
  const o = B.WO.off;
  const r10 = x0 + th[o] * z0 + th[o + 1] * z1;
  const r11 = x1 + th[o + 2] * z0 + th[o + 3] * z1;
  t.r1[2 * i] = r10; t.r1[2 * i + 1] = r11;
  let m0 = 0, m1 = 0;
  for (let h = 0; h < H; h++) {
    const pre = th[at(B.W1, h, 0)] * r10 + th[at(B.W1, h, 1)] * r11 + th[at(B.b1, h)];
    t.pre[i * H + h] = pre;
    if (pre > 0) { m0 += th[at(B.W2, 0, h)] * pre; m1 += th[at(B.W2, 1, h)] * pre; }
  }
  const r20 = r10 + m0, r21 = r11 + m1;
  t.r2[2 * i] = r20; t.r2[2 * i + 1] = r21;
  let max = -Infinity;
  for (let w = 0; w < V; w++) {
    const l = th[at(B.U, w, 0)] * r20 + th[at(B.U, w, 1)] * r21 + th[at(B.bU, w)];
    t.logits[i * V + w] = l;
    if (l > max) max = l;
  }
  let sum = 0;
  for (let w = 0; w < V; w++) sum += t.p[i * V + w] = Math.exp(t.logits[i * V + w] - max);
  for (let w = 0; w < V; w++) t.p[i * V + w] /= sum;
}

/**
 * Cross-entropy of one sequence against its targets (target < 0 means "not scored"),
 * weighted by `scale`. Adds the gradient into `g`. Returns the weighted loss.
 */
export function backward(c: Config, L: Layout, th: Float64Array, t: Trace, targets: number[], scale: number, g: Float64Array): number {
  const { n } = t, { V, H } = c, B = L.blocks;
  const dx = new Float64Array(n * 2), dq = new Float64Array(n * 2), dk = new Float64Array(n * 2), dv = new Float64Array(n * 2);
  const da = new Float64Array(n);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const y = targets[i];
    if (y < 0) continue;
    loss -= scale * Math.log(Math.max(t.p[i * V + y], 1e-300));
    // unembed
    const r20 = t.r2[2 * i], r21 = t.r2[2 * i + 1];
    let d0 = 0, d1 = 0; // d loss / d r2
    for (let w = 0; w < V; w++) {
      const dl = scale * (t.p[i * V + w] - (w === y ? 1 : 0));
      g[at(B.U, w, 0)] += dl * r20; g[at(B.U, w, 1)] += dl * r21; g[at(B.bU, w)] += dl;
      d0 += th[at(B.U, w, 0)] * dl; d1 += th[at(B.U, w, 1)] * dl;
    }
    // MLP (residual: r2 = r1 + mlp(r1))
    const r10 = t.r1[2 * i], r11 = t.r1[2 * i + 1];
    let e0 = d0, e1 = d1; // d loss / d r1
    for (let h = 0; h < H; h++) {
      const pre = t.pre[i * H + h];
      if (pre <= 0) continue;
      g[at(B.W2, 0, h)] += d0 * pre; g[at(B.W2, 1, h)] += d1 * pre;
      const dpre = th[at(B.W2, 0, h)] * d0 + th[at(B.W2, 1, h)] * d1;
      g[at(B.W1, h, 0)] += dpre * r10; g[at(B.W1, h, 1)] += dpre * r11; g[at(B.b1, h)] += dpre;
      e0 += th[at(B.W1, h, 0)] * dpre; e1 += th[at(B.W1, h, 1)] * dpre;
    }
    // attention (residual: r1 = x + WO z)
    dx[2 * i] += e0; dx[2 * i + 1] += e1;
    const o = B.WO.off, z0 = t.z[2 * i], z1 = t.z[2 * i + 1];
    g[o] += e0 * z0; g[o + 1] += e0 * z1; g[o + 2] += e1 * z0; g[o + 3] += e1 * z1;
    const dz0 = th[o] * e0 + th[o + 2] * e1, dz1 = th[o + 1] * e0 + th[o + 3] * e1;
    let mean = 0;
    for (let j = 0; j <= i; j++) {
      const a = t.a[i * n + j];
      da[j] = dz0 * t.v[2 * j] + dz1 * t.v[2 * j + 1];
      mean += a * da[j];
      dv[2 * j] += a * dz0; dv[2 * j + 1] += a * dz1;
    }
    for (let j = 0; j <= i; j++) {
      const ds = t.a[i * n + j] * (da[j] - mean);
      dq[2 * i] += ds * t.k[2 * j]; dq[2 * i + 1] += ds * t.k[2 * j + 1];
      dk[2 * j] += ds * t.q[2 * i]; dk[2 * j + 1] += ds * t.q[2 * i + 1];
    }
  }
  const back = (b: Block, d: Float64Array, i: number) => {
    const x0 = t.x[2 * i], x1 = t.x[2 * i + 1], d0 = d[2 * i], d1 = d[2 * i + 1];
    g[b.off] += d0 * x0; g[b.off + 1] += d0 * x1; g[b.off + 2] += d1 * x0; g[b.off + 3] += d1 * x1;
    dx[2 * i] += th[b.off] * d0 + th[b.off + 2] * d1;
    dx[2 * i + 1] += th[b.off + 1] * d0 + th[b.off + 3] * d1;
  };
  for (let i = 0; i < n; i++) {
    back(B.WQ, dq, i); back(B.WK, dk, i); back(B.WV, dv, i);
    g[at(B.E, t.tokens[i], 0)] += dx[2 * i]; g[at(B.E, t.tokens[i], 1)] += dx[2 * i + 1];
    g[at(B.P, i, 0)] += dx[2 * i]; g[at(B.P, i, 1)] += dx[2 * i + 1];
  }
  return loss;
}

/**
 * The model's output at one position as a function of where that position's point is.
 * The earlier positions are fixed (attention is causal, so they never see this one);
 * `x` roams the plane. This is what the landscape samples.
 */
export interface Probe {
  /** Fill `out` (a one-row Trace) for the point (x0, x1) at position `pos`. */
  eval(x0: number, x1: number): Trace;
  /** Scores of each key (earlier points, then the roaming point itself). */
  nKeys: number;
}

export function probe(c: Config, L: Layout, th: Float64Array, ctx: Trace, pos: number): Probe {
  const B = L.blocks, nKeys = pos + 1;
  const out: Trace = {
    n: 1, tokens: [-1],
    x: new Float64Array(2), q: new Float64Array(2), k: new Float64Array(2), v: new Float64Array(2),
    s: new Float64Array(nKeys), a: new Float64Array(nKeys), z: new Float64Array(2),
    r1: new Float64Array(2), pre: new Float64Array(c.H), r2: new Float64Array(2),
    logits: new Float64Array(c.V), p: new Float64Array(c.V),
  };
  const q = B.WQ.off, k = B.WK.off, v = B.WV.off;
  return {
    nKeys,
    eval(x0, x1) {
      out.x[0] = x0; out.x[1] = x1;
      const q0 = th[q] * x0 + th[q + 1] * x1, q1 = th[q + 2] * x0 + th[q + 3] * x1;
      const k0 = th[k] * x0 + th[k + 1] * x1, k1 = th[k + 2] * x0 + th[k + 3] * x1;
      const v0 = th[v] * x0 + th[v + 1] * x1, v1 = th[v + 2] * x0 + th[v + 3] * x1;
      out.q[0] = q0; out.q[1] = q1; out.k[0] = k0; out.k[1] = k1; out.v[0] = v0; out.v[1] = v1;
      let max = -Infinity;
      for (let j = 0; j < nKeys; j++) {
        const s = j < pos ? q0 * ctx.k[2 * j] + q1 * ctx.k[2 * j + 1] : q0 * k0 + q1 * k1;
        out.s[j] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let j = 0; j < nKeys; j++) sum += out.a[j] = Math.exp(out.s[j] - max);
      let z0 = 0, z1 = 0;
      for (let j = 0; j < nKeys; j++) {
        const a = (out.a[j] /= sum);
        z0 += a * (j < pos ? ctx.v[2 * j] : v0);
        z1 += a * (j < pos ? ctx.v[2 * j + 1] : v1);
      }
      out.z[0] = z0; out.z[1] = z1;
      tail(c, L, th, x0, x1, z0, z1, out, 0);
      return out;
    },
  };
}
