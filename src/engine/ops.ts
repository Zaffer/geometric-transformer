// The complete op set of the transformer. Each op: forward, then (if the
// graph is tracking) a backward closure. Nothing else in the engine does
// calculus. tests/gradcheck.test.ts checks every op against finite
// differences.

import { Tensor, link, tracks } from './tensor';

export type Activation =
  | 'relu'
  | 'leaky_relu'
  | 'gelu'
  | 'silu'
  | 'tanh'
  | 'sigmoid'
  | 'softplus'
  | 'identity';

export const ACTIVATIONS: Activation[] = [
  'relu',
  'leaky_relu',
  'gelu',
  'silu',
  'tanh',
  'sigmoid',
  'softplus',
  'identity',
];

/** Activations that are piecewise linear: they cut space into exact polytopes. */
export const PIECEWISE_LINEAR: ReadonlySet<Activation> = new Set(['relu', 'leaky_relu', 'identity']);

/** a[n,k] @ b[k,m] */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.cols !== b.rows) throw new Error(`matmul ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
  const n = a.rows, k = a.cols, m = b.cols;
  const out = new Tensor(n, m);
  const A = a.data, B = b.data, O = out.data;
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const av = A[i * k + p];
      if (av === 0) continue;
      const bo = p * m, oo = i * m;
      for (let j = 0; j < m; j++) O[oo + j] += av * B[bo + j];
    }
  }
  if (!tracks(a, b)) return out;
  return link(out, [a, b], () => {
    const G = out.grad!;
    if (a.requiresGrad || a.backFn) {
      const GA = a.g();
      for (let i = 0; i < n; i++)
        for (let p = 0; p < k; p++) {
          let s = 0;
          for (let j = 0; j < m; j++) s += G[i * m + j] * B[p * m + j];
          GA[i * k + p] += s;
        }
    }
    if (b.requiresGrad || b.backFn) {
      const GB = b.g();
      for (let i = 0; i < n; i++)
        for (let p = 0; p < k; p++) {
          const av = A[i * k + p];
          if (av === 0) continue;
          for (let j = 0; j < m; j++) GB[p * m + j] += av * G[i * m + j];
        }
    }
  });
}

/** a[n,k] @ b[m,k]^T, used by the tied unembedding (logits = x @ wte^T) */
export function matmulBT(a: Tensor, b: Tensor): Tensor {
  if (a.cols !== b.cols) throw new Error(`matmulBT ${a.rows}x${a.cols} @ (${b.rows}x${b.cols})^T`);
  const n = a.rows, k = a.cols, m = b.rows;
  const out = new Tensor(n, m);
  const A = a.data, B = b.data, O = out.data;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i * k + p] * B[j * k + p];
      O[i * m + j] = s;
    }
  if (!tracks(a, b)) return out;
  return link(out, [a, b], () => {
    const G = out.grad!;
    const GA = a.requiresGrad || a.backFn ? a.g() : null;
    const GB = b.requiresGrad || b.backFn ? b.g() : null;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) {
        const gv = G[i * m + j];
        if (gv === 0) continue;
        for (let p = 0; p < k; p++) {
          if (GA) GA[i * k + p] += gv * B[j * k + p];
          if (GB) GB[j * k + p] += gv * A[i * k + p];
        }
      }
  });
}

/** x[n,m] + b[1,m] broadcast over rows */
export function addBias(x: Tensor, b: Tensor): Tensor {
  if (b.rows !== 1 || b.cols !== x.cols) throw new Error('addBias shape');
  const n = x.rows, m = x.cols;
  const out = new Tensor(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out.data[i * m + j] = x.data[i * m + j] + b.data[j];
  if (!tracks(x, b)) return out;
  return link(out, [x, b], () => {
    const G = out.grad!;
    if (x.requiresGrad || x.backFn) {
      const GX = x.g();
      for (let i = 0; i < G.length; i++) GX[i] += G[i];
    }
    if (b.requiresGrad || b.backFn) {
      const GB = b.g();
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) GB[j] += G[i * m + j];
    }
  });
}

/** elementwise a + b, same shape */
export function add(a: Tensor, b: Tensor): Tensor {
  if (a.rows !== b.rows || a.cols !== b.cols) throw new Error('add shape');
  const out = new Tensor(a.rows, a.cols);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] + b.data[i];
  if (!tracks(a, b)) return out;
  return link(out, [a, b], () => {
    const G = out.grad!;
    for (const t of [a, b]) {
      if (!(t.requiresGrad || t.backFn)) continue;
      const GT = t.g();
      for (let i = 0; i < G.length; i++) GT[i] += G[i];
    }
  });
}

// ---- activations ----

const GELU_C = Math.sqrt(2 / Math.PI);
const GELU_A = 0.044715;

function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

/** f(x) for one activation, scalar form (also used by the views) */
export function actScalar(kind: Activation, x: number, slope = 0.01): number {
  switch (kind) {
    case 'relu':
      return x > 0 ? x : 0;
    case 'leaky_relu':
      return x > 0 ? x : slope * x;
    case 'gelu':
      return 0.5 * x * (1 + Math.tanh(GELU_C * (x + GELU_A * x * x * x)));
    case 'silu':
      return x * sigmoid(x);
    case 'tanh':
      return Math.tanh(x);
    case 'sigmoid':
      return sigmoid(x);
    case 'softplus':
      return x > 20 ? x : Math.log1p(Math.exp(x));
    case 'identity':
      return x;
  }
}

/** f'(x) */
export function actDeriv(kind: Activation, x: number, slope = 0.01): number {
  switch (kind) {
    case 'relu':
      return x > 0 ? 1 : 0;
    case 'leaky_relu':
      return x > 0 ? 1 : slope;
    case 'gelu': {
      const u = GELU_C * (x + GELU_A * x * x * x);
      const th = Math.tanh(u);
      return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * GELU_C * (1 + 3 * GELU_A * x * x);
    }
    case 'silu': {
      const s = sigmoid(x);
      return s * (1 + x * (1 - s));
    }
    case 'tanh': {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
    case 'sigmoid': {
      const s = sigmoid(x);
      return s * (1 - s);
    }
    case 'softplus':
      return sigmoid(x);
    case 'identity':
      return 1;
  }
}

export function activate(x: Tensor, kind: Activation, slope = 0.01): Tensor {
  const out = new Tensor(x.rows, x.cols);
  for (let i = 0; i < x.size; i++) out.data[i] = actScalar(kind, x.data[i], slope);
  if (!tracks(x)) return out;
  return link(out, [x], () => {
    const G = out.grad!, GX = x.g();
    for (let i = 0; i < G.length; i++) GX[i] += G[i] * actDeriv(kind, x.data[i], slope);
  });
}

// ---- normalization ----

export type NormKind = 'layernorm' | 'rmsnorm' | 'none';

export interface NormAux {
  /** per-row 1/std (layernorm) or 1/rms (rmsnorm) */
  rstd: Float64Array;
  /** normalized rows before gain and bias */
  xhat: Float64Array;
}

/**
 * Row-wise normalization. layernorm: (x - mean) / std; rmsnorm: x / rms.
 * Then optional gain g[1,m] and bias b[1,m].
 */
export function norm(
  x: Tensor,
  kind: 'layernorm' | 'rmsnorm',
  g: Tensor | null,
  b: Tensor | null,
  eps: number,
  aux?: { value?: NormAux },
): Tensor {
  const n = x.rows, m = x.cols;
  const xhat = new Float64Array(n * m);
  const rstd = new Float64Array(n);
  const out = new Tensor(n, m);
  for (let i = 0; i < n; i++) {
    const o = i * m;
    let mean = 0;
    if (kind === 'layernorm') {
      for (let j = 0; j < m; j++) mean += x.data[o + j];
      mean /= m;
    }
    let v = 0;
    for (let j = 0; j < m; j++) {
      const d = x.data[o + j] - mean;
      v += d * d;
    }
    v /= m;
    const r = 1 / Math.sqrt(v + eps);
    rstd[i] = r;
    for (let j = 0; j < m; j++) {
      const h = (x.data[o + j] - mean) * r;
      xhat[o + j] = h;
      out.data[o + j] = h * (g ? g.data[j] : 1) + (b ? b.data[j] : 0);
    }
  }
  if (aux) aux.value = { rstd, xhat };
  const parents = [x, ...(g ? [g] : []), ...(b ? [b] : [])];
  if (!tracks(...parents)) return out;
  return link(out, parents, () => {
    const G = out.grad!;
    if (g && (g.requiresGrad || g.backFn)) {
      const GG = g.g();
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) GG[j] += G[i * m + j] * xhat[i * m + j];
    }
    if (b && (b.requiresGrad || b.backFn)) {
      const GB = b.g();
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) GB[j] += G[i * m + j];
    }
    if (x.requiresGrad || x.backFn) {
      const GX = x.g();
      for (let i = 0; i < n; i++) {
        const o = i * m;
        // dxhat = G * g
        let sumD = 0, sumDX = 0;
        for (let j = 0; j < m; j++) {
          const d = G[o + j] * (g ? g.data[j] : 1);
          sumD += d;
          sumDX += d * xhat[o + j];
        }
        const r = rstd[i];
        for (let j = 0; j < m; j++) {
          const d = G[o + j] * (g ? g.data[j] : 1);
          if (kind === 'layernorm') GX[o + j] += r * (d - sumD / m - (xhat[o + j] * sumDX) / m);
          else GX[o + j] += r * (d - (xhat[o + j] * sumDX) / m);
        }
      }
    }
  });
}

// ---- embeddings ----

/** rows of W[V,D] picked by ids → [ids.length, D] */
export function gather(W: Tensor, ids: Int32Array): Tensor {
  const D = W.cols;
  const out = new Tensor(ids.length, D);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    if (id < 0 || id >= W.rows) throw new Error(`gather id ${id} out of range ${W.rows}`);
    out.data.set(W.data.subarray(id * D, id * D + D), i * D);
  }
  if (!tracks(W)) return out;
  return link(out, [W], () => {
    const G = out.grad!, GW = W.g();
    for (let i = 0; i < ids.length; i++) {
      const o = ids[i] * D;
      for (let j = 0; j < D; j++) GW[o + j] += G[i * D + j];
    }
  });
}

/** columns [start, start+len) of x */
export function sliceCols(x: Tensor, start: number, len: number): Tensor {
  const n = x.rows, m = x.cols;
  const out = new Tensor(n, len);
  for (let i = 0; i < n; i++) for (let j = 0; j < len; j++) out.data[i * len + j] = x.data[i * m + start + j];
  if (!tracks(x)) return out;
  return link(out, [x], () => {
    const G = out.grad!, GX = x.g();
    for (let i = 0; i < n; i++) for (let j = 0; j < len; j++) GX[i * m + start + j] += G[i * len + j];
  });
}

/** [x0 | x1 | ...] column concatenation */
export function concatCols(xs: Tensor[]): Tensor {
  const n = xs[0].rows;
  const m = xs.reduce((s, t) => s + t.cols, 0);
  const out = new Tensor(n, m);
  let off = 0;
  for (const t of xs) {
    for (let i = 0; i < n; i++) for (let j = 0; j < t.cols; j++) out.data[i * m + off + j] = t.data[i * t.cols + j];
    off += t.cols;
  }
  if (!tracks(...xs)) return out;
  return link(out, xs, () => {
    const G = out.grad!;
    let o = 0;
    for (const t of xs) {
      if (t.requiresGrad || t.backFn) {
        const GT = t.g();
        for (let i = 0; i < n; i++) for (let j = 0; j < t.cols; j++) GT[i * t.cols + j] += G[i * m + o + j];
      }
      o += t.cols;
    }
  });
}

// ---- attention ----
// Rows are (sequence b, position t) flattened as b*T + t. Scores and
// probabilities are [B*T, T]: row (b, i) holds query i's scores against
// keys 0..T-1 of the same sequence. Masked entries hold -Infinity.

/** scale * q_i . k_j within each sequence; causal masks j > i */
export function attnScores(q: Tensor, k: Tensor, B: number, T: number, scale: number, causal: boolean): Tensor {
  const dh = q.cols;
  const out = new Tensor(B * T, T);
  for (let b = 0; b < B; b++)
    for (let i = 0; i < T; i++) {
      const qo = (b * T + i) * dh, oo = (b * T + i) * T;
      for (let j = 0; j < T; j++) {
        if (causal && j > i) {
          out.data[oo + j] = -Infinity;
          continue;
        }
        const ko = (b * T + j) * dh;
        let s = 0;
        for (let p = 0; p < dh; p++) s += q.data[qo + p] * k.data[ko + p];
        out.data[oo + j] = s * scale;
      }
    }
  if (!tracks(q, k)) return out;
  return link(out, [q, k], () => {
    const G = out.grad!;
    const GQ = q.requiresGrad || q.backFn ? q.g() : null;
    const GK = k.requiresGrad || k.backFn ? k.g() : null;
    for (let b = 0; b < B; b++)
      for (let i = 0; i < T; i++) {
        const qo = (b * T + i) * dh, oo = (b * T + i) * T;
        for (let j = 0; j < T; j++) {
          if (causal && j > i) continue;
          const gv = G[oo + j] * scale;
          if (gv === 0) continue;
          const ko = (b * T + j) * dh;
          for (let p = 0; p < dh; p++) {
            if (GQ) GQ[qo + p] += gv * k.data[ko + p];
            if (GK) GK[ko + p] += gv * q.data[qo + p];
          }
        }
      }
  });
}

/** row-wise softmax; -Infinity entries get probability 0 */
export function softmaxRows(x: Tensor): Tensor {
  const n = x.rows, m = x.cols;
  const out = new Tensor(n, m);
  for (let i = 0; i < n; i++) {
    const o = i * m;
    let mx = -Infinity;
    for (let j = 0; j < m; j++) if (x.data[o + j] > mx) mx = x.data[o + j];
    let s = 0;
    for (let j = 0; j < m; j++) {
      const e = x.data[o + j] === -Infinity ? 0 : Math.exp(x.data[o + j] - mx);
      out.data[o + j] = e;
      s += e;
    }
    for (let j = 0; j < m; j++) out.data[o + j] /= s;
  }
  if (!tracks(x)) return out;
  return link(out, [x], () => {
    const G = out.grad!, GX = x.g(), P = out.data;
    for (let i = 0; i < n; i++) {
      const o = i * m;
      let dot = 0;
      for (let j = 0; j < m; j++) dot += P[o + j] * G[o + j];
      for (let j = 0; j < m; j++) GX[o + j] += P[o + j] * (G[o + j] - dot);
    }
  });
}

/** out_(b,i) = sum_j p[(b,i), j] * v_(b,j): a weighted average of value vectors */
export function attnMix(p: Tensor, v: Tensor, B: number, T: number): Tensor {
  const dh = v.cols;
  const out = new Tensor(B * T, dh);
  for (let b = 0; b < B; b++)
    for (let i = 0; i < T; i++) {
      const po = (b * T + i) * T, oo = (b * T + i) * dh;
      for (let j = 0; j < T; j++) {
        const pv = p.data[po + j];
        if (pv === 0) continue;
        const vo = (b * T + j) * dh;
        for (let c = 0; c < dh; c++) out.data[oo + c] += pv * v.data[vo + c];
      }
    }
  if (!tracks(p, v)) return out;
  return link(out, [p, v], () => {
    const G = out.grad!;
    const GP = p.requiresGrad || p.backFn ? p.g() : null;
    const GV = v.requiresGrad || v.backFn ? v.g() : null;
    for (let b = 0; b < B; b++)
      for (let i = 0; i < T; i++) {
        const po = (b * T + i) * T, oo = (b * T + i) * dh;
        for (let j = 0; j < T; j++) {
          const vo = (b * T + j) * dh;
          if (GP) {
            let s = 0;
            for (let c = 0; c < dh; c++) s += G[oo + c] * v.data[vo + c];
            GP[po + j] += s;
          }
          if (GV) {
            const pv = p.data[po + j];
            if (pv !== 0) for (let c = 0; c < dh; c++) GV[vo + c] += pv * G[oo + c];
          }
        }
      }
  });
}

// ---- loss ----

/**
 * Mean cross-entropy over rows whose target is >= 0 (target -1 = ignore).
 * Returns a 1x1 tensor. The softmax probabilities land in probsOut if given.
 */
export function crossEntropy(logits: Tensor, targets: Int32Array, probsOut?: { value?: Float64Array }): Tensor {
  const n = logits.rows, V = logits.cols;
  const P = new Float64Array(n * V);
  let loss = 0, count = 0;
  for (let i = 0; i < n; i++) {
    const o = i * V;
    let mx = -Infinity;
    for (let j = 0; j < V; j++) if (logits.data[o + j] > mx) mx = logits.data[o + j];
    let s = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp(logits.data[o + j] - mx);
      P[o + j] = e;
      s += e;
    }
    for (let j = 0; j < V; j++) P[o + j] /= s;
    const t = targets[i];
    if (t >= 0) {
      loss -= Math.log(Math.max(P[o + t], 1e-300));
      count++;
    }
  }
  if (probsOut) probsOut.value = P;
  const out = new Tensor(1, 1);
  out.data[0] = count ? loss / count : 0;
  if (!tracks(logits) || count === 0) return out;
  return link(out, [logits], () => {
    const g = out.grad![0] / count, GL = logits.g();
    for (let i = 0; i < n; i++) {
      const t = targets[i];
      if (t < 0) continue;
      const o = i * V;
      for (let j = 0; j < V; j++) GL[o + j] += g * (P[o + j] - (j === t ? 1 : 0));
    }
  });
}
