// A complete GPT-2 style transformer with hand-written forward and backward
// passes on plain Float32Arrays. Every parameter is a scalar in a named
// Tensor, so the visualization can address, read, and write each one.

import type { ModelConfig } from './config';
import { RNG } from './rng';

export interface Tensor {
  name: string;
  rows: number;
  cols: number;
  data: Float32Array;
  grad: Float32Array;
  m: Float32Array; // AdamW first moment
  v: Float32Array; // AdamW second moment
  decay: boolean; // weight decay applies (2D matmul weights only)
}

export interface LNCache {
  xhat: Float32Array; // T x D normalized input
  rstd: Float32Array; // T reciprocal std
  out: Float32Array; // T x D output
}

export interface BlockCache {
  ln1: LNCache;
  qkv: Float32Array; // T x 3D
  att: Float32Array; // H x T x T attention probabilities
  atty: Float32Array; // T x D concatenated head outputs, before wo
  attnOut: Float32Array; // T x D
  resid1: Float32Array; // T x D after attention residual
  ln2: LNCache;
  h: Float32Array; // T x F pre-GELU
  hact: Float32Array; // T x F post-GELU
  mlpOut: Float32Array; // T x D
  resid2: Float32Array; // T x D after MLP residual
}

export interface ForwardCache {
  T: number;
  tokens: Int32Array;
  x0: Float32Array; // T x D token + position embedding
  blocks: BlockCache[];
  lnf: LNCache;
  logits: Float32Array; // T x V
  probs: Float32Array; // T x V
}

const GELU_C = Math.sqrt(2 / Math.PI);
const GELU_A = 0.044715;

function geluForward(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const u = GELU_C * (v + GELU_A * v * v * v);
    out[i] = 0.5 * v * (1 + Math.tanh(u));
  }
  return out;
}

function geluBackward(dy: Float32Array, x: Float32Array): Float32Array {
  const dx = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const u = GELU_C * (v + GELU_A * v * v * v);
    const th = Math.tanh(u);
    const sech2 = 1 - th * th;
    const du = GELU_C * (1 + 3 * GELU_A * v * v);
    dx[i] = dy[i] * (0.5 * (1 + th) + 0.5 * v * sech2 * du);
  }
  return dx;
}

// y[T x OUT] = x[T x IN] @ W[IN x OUT] + b[1 x OUT]
function linearForward(
  x: Float32Array,
  W: Float32Array,
  b: Float32Array,
  T: number,
  IN: number,
  OUT: number,
): Float32Array {
  const y = new Float32Array(T * OUT);
  for (let t = 0; t < T; t++) {
    const xo = t * IN;
    const yo = t * OUT;
    for (let o = 0; o < OUT; o++) y[yo + o] = b[o];
    for (let i = 0; i < IN; i++) {
      const xv = x[xo + i];
      if (xv === 0) continue;
      const wo = i * OUT;
      for (let o = 0; o < OUT; o++) y[yo + o] += xv * W[wo + o];
    }
  }
  return y;
}

// Accumulates dW, db; returns dx.
function linearBackward(
  dy: Float32Array,
  x: Float32Array,
  W: Float32Array,
  dW: Float32Array,
  db: Float32Array,
  T: number,
  IN: number,
  OUT: number,
): Float32Array {
  const dx = new Float32Array(T * IN);
  for (let t = 0; t < T; t++) {
    const xo = t * IN;
    const yo = t * OUT;
    for (let o = 0; o < OUT; o++) db[o] += dy[yo + o];
    for (let i = 0; i < IN; i++) {
      const wo = i * OUT;
      const xv = x[xo + i];
      let acc = 0;
      for (let o = 0; o < OUT; o++) {
        const d = dy[yo + o];
        acc += d * W[wo + o];
        dW[wo + o] += xv * d;
      }
      dx[xo + i] = acc;
    }
  }
  return dx;
}

const LN_EPS = 1e-5;

function layerNormForward(
  x: Float32Array,
  g: Float32Array,
  b: Float32Array,
  T: number,
  D: number,
): LNCache {
  const out = new Float32Array(T * D);
  const xhat = new Float32Array(T * D);
  const rstd = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let mean = 0;
    for (let d = 0; d < D; d++) mean += x[o + d];
    mean /= D;
    let variance = 0;
    for (let d = 0; d < D; d++) {
      const c = x[o + d] - mean;
      variance += c * c;
    }
    variance /= D;
    const r = 1 / Math.sqrt(variance + LN_EPS);
    rstd[t] = r;
    for (let d = 0; d < D; d++) {
      const xh = (x[o + d] - mean) * r;
      xhat[o + d] = xh;
      out[o + d] = g[d] * xh + b[d];
    }
  }
  return { out, xhat, rstd };
}

// Accumulates dg, db; returns dx.
function layerNormBackward(
  dy: Float32Array,
  cache: LNCache,
  g: Float32Array,
  dg: Float32Array,
  db: Float32Array,
  T: number,
  D: number,
): Float32Array {
  const { xhat, rstd } = cache;
  const dx = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let meanDxhat = 0;
    let meanDxhatXhat = 0;
    for (let d = 0; d < D; d++) {
      const dyv = dy[o + d];
      const xh = xhat[o + d];
      dg[d] += dyv * xh;
      db[d] += dyv;
      const dxh = dyv * g[d];
      meanDxhat += dxh;
      meanDxhatXhat += dxh * xh;
    }
    meanDxhat /= D;
    meanDxhatXhat /= D;
    const r = rstd[t];
    for (let d = 0; d < D; d++) {
      const dxh = dy[o + d] * g[d];
      dx[o + d] = r * (dxh - meanDxhat - xhat[o + d] * meanDxhatXhat);
    }
  }
  return dx;
}

function addInto(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export class Transformer {
  readonly cfg: ModelConfig;
  readonly params: Tensor[] = [];
  private byName = new Map<string, Tensor>();

  constructor(cfg: ModelConfig, seed = 1337) {
    this.cfg = cfg;
    const rng = new RNG(seed);
    const D = cfg.dModel;
    const V = cfg.vocab;
    const C = cfg.nCtx;
    const F = cfg.dModel * cfg.mlpRatio;
    const std = 0.02;
    // GPT-2 scales residual-branch projections by 1/sqrt(2 * nLayer).
    const resStd = std / Math.sqrt(2 * cfg.nLayer);

    const add = (
      name: string,
      rows: number,
      cols: number,
      init: 'randn' | 'zeros' | 'ones',
      s = std,
      decay = false,
    ): Tensor => {
      const n = rows * cols;
      const data = new Float32Array(n);
      if (init === 'randn') {
        for (let i = 0; i < n; i++) data[i] = rng.normal() * s;
      } else if (init === 'ones') {
        data.fill(1);
      }
      const t: Tensor = {
        name,
        rows,
        cols,
        data,
        grad: new Float32Array(n),
        m: new Float32Array(n),
        v: new Float32Array(n),
        decay,
      };
      this.params.push(t);
      this.byName.set(name, t);
      return t;
    };

    add('wte', V, D, 'randn', std, true);
    add('wpe', C, D, 'randn', std, true);
    for (let b = 0; b < cfg.nLayer; b++) {
      add(`b${b}.ln1g`, 1, D, 'ones');
      add(`b${b}.ln1b`, 1, D, 'zeros');
      add(`b${b}.wqkv`, D, 3 * D, 'randn', std, true);
      add(`b${b}.bqkv`, 1, 3 * D, 'zeros');
      add(`b${b}.wo`, D, D, 'randn', resStd, true);
      add(`b${b}.bo`, 1, D, 'zeros');
      add(`b${b}.ln2g`, 1, D, 'ones');
      add(`b${b}.ln2b`, 1, D, 'zeros');
      add(`b${b}.wfc`, D, F, 'randn', std, true);
      add(`b${b}.bfc`, 1, F, 'zeros');
      add(`b${b}.wproj`, F, D, 'randn', resStd, true);
      add(`b${b}.bproj`, 1, D, 'zeros');
    }
    add('lnfg', 1, D, 'ones');
    add('lnfb', 1, D, 'zeros');
    if (!cfg.tieWeights) add('wun', D, V, 'randn', std, true);
  }

  get(name: string): Tensor {
    const t = this.byName.get(name);
    if (!t) throw new Error(`unknown tensor: ${name}`);
    return t;
  }

  has(name: string): boolean {
    return this.byName.has(name);
  }

  paramCount(): number {
    let n = 0;
    for (const p of this.params) n += p.data.length;
    return n;
  }

  zeroGrads(): void {
    for (const p of this.params) p.grad.fill(0);
  }

  forward(tokens: Int32Array): ForwardCache {
    const { dModel: D, nHead: H, vocab: V } = this.cfg;
    const F = D * this.cfg.mlpRatio;
    const T = tokens.length;
    if (T > this.cfg.nCtx) throw new Error(`sequence too long: ${T} > ${this.cfg.nCtx}`);
    const dh = D / H;
    const scale = 1 / Math.sqrt(dh);

    const wte = this.get('wte').data;
    const wpe = this.get('wpe').data;

    const x0 = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const tok = tokens[t];
      for (let d = 0; d < D; d++) x0[t * D + d] = wte[tok * D + d] + wpe[t * D + d];
    }

    let x: Float32Array = x0;
    const blocks: BlockCache[] = [];
    const scratch = new Float32Array(T);

    for (let b = 0; b < this.cfg.nLayer; b++) {
      const ln1 = layerNormForward(x, this.get(`b${b}.ln1g`).data, this.get(`b${b}.ln1b`).data, T, D);
      const qkv = linearForward(ln1.out, this.get(`b${b}.wqkv`).data, this.get(`b${b}.bqkv`).data, T, D, 3 * D);

      const att = new Float32Array(H * T * T);
      const atty = new Float32Array(T * D);
      for (let h = 0; h < H; h++) {
        const qo = h * dh;
        const ko = D + h * dh;
        const vo = 2 * D + h * dh;
        for (let i = 0; i < T; i++) {
          let maxv = -Infinity;
          for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let d = 0; d < dh; d++) s += qkv[i * 3 * D + qo + d] * qkv[j * 3 * D + ko + d];
            s *= scale;
            scratch[j] = s;
            if (s > maxv) maxv = s;
          }
          let sum = 0;
          for (let j = 0; j <= i; j++) {
            const e = Math.exp(scratch[j] - maxv);
            scratch[j] = e;
            sum += e;
          }
          for (let j = 0; j <= i; j++) {
            const p = scratch[j] / sum;
            att[h * T * T + i * T + j] = p;
            for (let d = 0; d < dh; d++) {
              atty[i * D + h * dh + d] += p * qkv[j * 3 * D + vo + d];
            }
          }
        }
      }

      const attnOut = linearForward(atty, this.get(`b${b}.wo`).data, this.get(`b${b}.bo`).data, T, D, D);
      const resid1 = addInto(x, attnOut);
      const ln2 = layerNormForward(resid1, this.get(`b${b}.ln2g`).data, this.get(`b${b}.ln2b`).data, T, D);
      const h = linearForward(ln2.out, this.get(`b${b}.wfc`).data, this.get(`b${b}.bfc`).data, T, D, F);
      const hact = geluForward(h);
      const mlpOut = linearForward(hact, this.get(`b${b}.wproj`).data, this.get(`b${b}.bproj`).data, T, F, D);
      const resid2 = addInto(resid1, mlpOut);

      blocks.push({ ln1, qkv, att, atty, attnOut, resid1, ln2, h, hact, mlpOut, resid2 });
      x = resid2;
    }

    const lnf = layerNormForward(x, this.get('lnfg').data, this.get('lnfb').data, T, D);
    const logits = new Float32Array(T * V);
    if (this.cfg.tieWeights) {
      for (let t = 0; t < T; t++) {
        for (let v = 0; v < V; v++) {
          let s = 0;
          for (let d = 0; d < D; d++) s += lnf.out[t * D + d] * wte[v * D + d];
          logits[t * V + v] = s;
        }
      }
    } else {
      const wun = this.get('wun').data;
      for (let t = 0; t < T; t++) {
        for (let v = 0; v < V; v++) {
          let s = 0;
          for (let d = 0; d < D; d++) s += lnf.out[t * D + d] * wun[d * V + v];
          logits[t * V + v] = s;
        }
      }
    }

    const probs = new Float32Array(T * V);
    for (let t = 0; t < T; t++) {
      let maxv = -Infinity;
      for (let v = 0; v < V; v++) if (logits[t * V + v] > maxv) maxv = logits[t * V + v];
      let sum = 0;
      for (let v = 0; v < V; v++) {
        const e = Math.exp(logits[t * V + v] - maxv);
        probs[t * V + v] = e;
        sum += e;
      }
      for (let v = 0; v < V; v++) probs[t * V + v] /= sum;
    }

    return { T, tokens: Int32Array.from(tokens), x0, blocks, lnf, logits, probs };
  }

  // Cross-entropy loss over positions with target >= 0.
  // Accumulates gradients into every tensor's grad array. Returns the loss.
  lossBackward(cache: ForwardCache, targets: Int32Array): number {
    const { dModel: D, nHead: H, vocab: V } = this.cfg;
    const F = D * this.cfg.mlpRatio;
    const T = cache.T;
    const dh = D / H;
    const scale = 1 / Math.sqrt(dh);

    let count = 0;
    for (let t = 0; t < T; t++) if (targets[t] >= 0) count++;
    if (count === 0) return 0;

    let loss = 0;
    const dlogits = new Float32Array(T * V);
    for (let t = 0; t < T; t++) {
      const tgt = targets[t];
      if (tgt < 0) continue;
      loss += -Math.log(Math.max(cache.probs[t * V + tgt], 1e-12));
      for (let v = 0; v < V; v++) {
        dlogits[t * V + v] = (cache.probs[t * V + v] - (v === tgt ? 1 : 0)) / count;
      }
    }
    loss /= count;

    // Unembedding backward.
    const dnormedF = new Float32Array(T * D);
    const wte = this.get('wte');
    if (this.cfg.tieWeights) {
      for (let t = 0; t < T; t++) {
        for (let v = 0; v < V; v++) {
          const dl = dlogits[t * V + v];
          if (dl === 0) continue;
          for (let d = 0; d < D; d++) {
            dnormedF[t * D + d] += dl * wte.data[v * D + d];
            wte.grad[v * D + d] += dl * cache.lnf.out[t * D + d];
          }
        }
      }
    } else {
      const wun = this.get('wun');
      for (let t = 0; t < T; t++) {
        for (let v = 0; v < V; v++) {
          const dl = dlogits[t * V + v];
          if (dl === 0) continue;
          for (let d = 0; d < D; d++) {
            dnormedF[t * D + d] += dl * wun.data[d * V + v];
            wun.grad[d * V + v] += dl * cache.lnf.out[t * D + d];
          }
        }
      }
    }

    const lnfg = this.get('lnfg');
    let dx = layerNormBackward(dnormedF, cache.lnf, lnfg.data, lnfg.grad, this.get('lnfb').grad, T, D);

    for (let b = this.cfg.nLayer - 1; b >= 0; b--) {
      const blk = cache.blocks[b];

      // resid2 = resid1 + mlpOut
      const dmlpOut = dx;
      const wproj = this.get(`b${b}.wproj`);
      const dhact = linearBackward(dmlpOut, blk.hact, wproj.data, wproj.grad, this.get(`b${b}.bproj`).grad, T, F, D);
      const dhpre = geluBackward(dhact, blk.h);
      const wfc = this.get(`b${b}.wfc`);
      const dnormed2 = linearBackward(dhpre, blk.ln2.out, wfc.data, wfc.grad, this.get(`b${b}.bfc`).grad, T, D, F);
      const ln2g = this.get(`b${b}.ln2g`);
      const dresidFromLn2 = layerNormBackward(dnormed2, blk.ln2, ln2g.data, ln2g.grad, this.get(`b${b}.ln2b`).grad, T, D);
      const dresid1 = addInto(dx, dresidFromLn2);

      // resid1 = x + attnOut
      const dattnOut = dresid1;
      const wo = this.get(`b${b}.wo`);
      const datty = linearBackward(dattnOut, blk.atty, wo.data, wo.grad, this.get(`b${b}.bo`).grad, T, D, D);

      const dqkv = new Float32Array(T * 3 * D);
      const dP = new Float32Array(T);
      for (let h = 0; h < H; h++) {
        const qo = h * dh;
        const ko = D + h * dh;
        const vo = 2 * D + h * dh;
        for (let i = 0; i < T; i++) {
          // dP and dV
          for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let d = 0; d < dh; d++) s += datty[i * D + h * dh + d] * blk.qkv[j * 3 * D + vo + d];
            dP[j] = s;
            const p = blk.att[h * T * T + i * T + j];
            if (p !== 0) {
              for (let d = 0; d < dh; d++) {
                dqkv[j * 3 * D + vo + d] += p * datty[i * D + h * dh + d];
              }
            }
          }
          // softmax backward
          let dot = 0;
          for (let j = 0; j <= i; j++) dot += dP[j] * blk.att[h * T * T + i * T + j];
          for (let j = 0; j <= i; j++) {
            const dS = blk.att[h * T * T + i * T + j] * (dP[j] - dot) * scale;
            if (dS === 0) continue;
            for (let d = 0; d < dh; d++) {
              dqkv[i * 3 * D + qo + d] += dS * blk.qkv[j * 3 * D + ko + d];
              dqkv[j * 3 * D + ko + d] += dS * blk.qkv[i * 3 * D + qo + d];
            }
          }
        }
      }

      const wqkv = this.get(`b${b}.wqkv`);
      const dnormed1 = linearBackward(dqkv, blk.ln1.out, wqkv.data, wqkv.grad, this.get(`b${b}.bqkv`).grad, T, D, 3 * D);
      const ln1g = this.get(`b${b}.ln1g`);
      const dxFromLn1 = layerNormBackward(dnormed1, blk.ln1, ln1g.data, ln1g.grad, this.get(`b${b}.ln1b`).grad, T, D);
      dx = addInto(dresid1, dxFromLn1);
    }

    // Embedding backward.
    const wpe = this.get('wpe');
    for (let t = 0; t < T; t++) {
      const tok = cache.tokens[t];
      for (let d = 0; d < D; d++) {
        wte.grad[tok * D + d] += dx[t * D + d];
        wpe.grad[t * D + d] += dx[t * D + d];
      }
    }

    return loss;
  }
}
