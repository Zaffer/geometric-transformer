// Every op's backward, and the whole model's backward under every
// architecture toggle, against central finite differences in float64.

import { describe, expect, it } from 'vitest';
import {
  activate,
  add,
  addBias,
  ACTIVATIONS,
  attnMix,
  attnScores,
  concatCols,
  crossEntropy,
  gather,
  matmul,
  matmulBT,
  norm,
  sliceCols,
  softmaxRows,
} from '../src/engine/ops';
import { Model } from '../src/engine/model';
import { RNG } from '../src/engine/rng';
import { defaultSpec, normalizeSpec, type ModelSpec } from '../src/engine/spec';
import { Tensor, withGrad } from '../src/engine/tensor';

const rng = new RNG(42);
function rand(r: number, c: number, s = 1): Tensor {
  const t = new Tensor(r, c, undefined, true);
  for (let i = 0; i < t.size; i++) t.data[i] = rng.normal() * s;
  return t;
}

/** A fixed random projection turns any output into a scalar loss. */
function scalarize(out: Tensor, w: Float64Array): Tensor {
  // loss = sum_i out_i * w_i (masked -Infinity entries skipped)
  let s = 0;
  for (let i = 0; i < out.size; i++) {
    const v = out.data[i];
    if (Number.isFinite(v)) s += v * w[i];
  }
  const r = new Tensor(1, 1);
  r.data[0] = s;
  r.parents = [out];
  r.backFn = () => {
    const G = out.g();
    for (let i = 0; i < out.size; i++) if (Number.isFinite(out.data[i])) G[i] += r.grad![0] * w[i];
  };
  return r;
}

function checkOp(name: string, inputs: Tensor[], f: () => Tensor, tol = 1e-6) {
  it(name, () => {
    const out0 = withGrad(false, f);
    const w = new Float64Array(out0.size);
    for (let i = 0; i < w.length; i++) w[i] = rng.normal();
    for (const t of inputs) t.grad = null;
    const L = withGrad(true, () => scalarize(f(), w));
    L.backward();
    const h = 1e-6;
    const val = () => withGrad(false, () => scalarize(f(), w).data[0]);
    for (const t of inputs) {
      for (let i = 0; i < t.size; i++) {
        const o = t.data[i];
        t.data[i] = o + h;
        const lp = val();
        t.data[i] = o - h;
        const lm = val();
        t.data[i] = o;
        const num = (lp - lm) / (2 * h);
        const ana = t.grad ? t.grad[i] : 0;
        const err = Math.abs(num - ana) / Math.max(1, Math.abs(num), Math.abs(ana));
        expect(err, `${name} ${t.name || 'input'}[${i}] num=${num} ana=${ana}`).toBeLessThan(tol);
      }
    }
  });
}

describe('ops', () => {
  const a = rand(3, 4), b = rand(4, 5), c = rand(5, 4), bias = rand(1, 5), a2 = rand(3, 4);
  checkOp('matmul', [a, b], () => matmul(a, b));
  checkOp('matmulBT', [a, c], () => matmulBT(a, c));
  checkOp('addBias', [b, bias], () => addBias(b, bias));
  checkOp('add', [a, a2], () => add(a, a2));
  for (const k of ACTIVATIONS) {
    const x = rand(4, 5, 2);
    checkOp(`activate ${k}`, [x], () => activate(x, k, 0.1), k === 'relu' || k === 'leaky_relu' ? 1e-4 : 1e-6);
  }
  for (const kind of ['layernorm', 'rmsnorm'] as const) {
    const x = rand(4, 3), g = rand(1, 3), bb = rand(1, 3);
    checkOp(`norm ${kind} affine`, [x, g, bb], () => norm(x, kind, g, bb, 1e-5));
    const x2 = rand(4, 5);
    checkOp(`norm ${kind} plain`, [x2], () => norm(x2, kind, null, null, 1e-5));
  }
  const W = rand(5, 3);
  const ids = Int32Array.from([0, 4, 2, 4, 1]);
  checkOp('gather', [W], () => gather(W, ids));
  const s = rand(3, 6);
  checkOp('sliceCols', [s], () => sliceCols(s, 2, 3));
  const c1 = rand(3, 2), c2 = rand(3, 3);
  checkOp('concatCols', [c1, c2], () => concatCols([c1, c2]));
  const B = 2, T = 4, dh = 3;
  const q = rand(B * T, dh), k = rand(B * T, dh), v = rand(B * T, dh);
  checkOp('attnScores causal', [q, k], () => attnScores(q, k, B, T, 0.7, true));
  checkOp('attnScores full', [q, k], () => attnScores(q, k, B, T, 0.7, false));
  const sc = rand(B * T, T);
  checkOp('softmaxRows', [sc], () => softmaxRows(sc));
  checkOp('softmaxRows masked', [q, k], () => softmaxRows(attnScores(q, k, B, T, 1, true)));
  const p = rand(B * T, T);
  checkOp('attnMix', [p, v], () => attnMix(p, v, B, T));
  const lg = rand(5, 4);
  const tg = Int32Array.from([0, -1, 3, 2, 1]);
  checkOp('crossEntropy', [lg], () => crossEntropy(lg, tg));
});

// ---- whole-model gradient check across toggles ----

function modelSpec(over: Partial<ModelSpec>): ModelSpec {
  const s = defaultSpec();
  Object.assign(s.model, { activation: 'gelu' }, over);
  normalizeSpec(s);
  return s.model;
}

function checkModel(label: string, over: Partial<ModelSpec>) {
  it(label, () => {
    const spec = defaultSpec();
    const ms = modelSpec(over);
    const model = new Model(ms, 4, 5, { ...spec.init, embedStd: 0.8, gain: 1.3 });
    const r = new RNG(9);
    // nudge biases and gains off their init so their grads are generic
    for (const p of model.params)
      if (p.kind === 'bias' || p.kind === 'gain') for (let i = 0; i < p.t.size; i++) p.t.data[i] += r.normal() * 0.3;
    const B = 2, T = 5;
    const x = Int32Array.from({ length: B * T }, () => r.int(4));
    const y = Int32Array.from({ length: B * T }, (_, i) => (i % 3 === 0 ? -1 : r.int(4)));
    model.zeroGrad();
    const { loss } = withGrad(true, () => model.loss(x, y, B));
    loss.backward();
    const f = () => withGrad(false, () => model.loss(x, y, B).loss.data[0]);
    const h = 1e-6;
    let worst = 0, where = '';
    for (const p of model.params) {
      for (let i = 0; i < p.t.size; i++) {
        const o = p.t.data[i];
        p.t.data[i] = o + h;
        const lp = f();
        p.t.data[i] = o - h;
        const lm = f();
        p.t.data[i] = o;
        const num = (lp - lm) / (2 * h);
        const ana = p.t.grad ? p.t.grad[i] : 0;
        const err = Math.abs(num - ana) / Math.max(1e-3, Math.abs(num) + Math.abs(ana));
        if (err > worst) {
          worst = err;
          where = `${p.name}[${i}] num=${num} ana=${ana}`;
        }
      }
    }
    expect(worst, where).toBeLessThan(1e-5);
  });
}

describe('model backward', () => {
  checkModel('default (pre-LN, learned pos, untied, biases)', {});
  checkModel('tied, no biases', { tie: true, biasQKV: false, biasAttnOut: false, biasMlp: false, biasUnembed: false });
  checkModel('rmsnorm', { norm: 'rmsnorm' });
  checkModel('no norm', { norm: 'none' });
  checkModel('post-norm', { normPlacement: 'post', finalNorm: false });
  checkModel('norm without affine', { normGain: false, normBias: false });
  checkModel('sinusoidal pos', { pos: 'sinusoidal' });
  checkModel('no pos, non-causal', { pos: 'none', causal: false });
  checkModel('attention-only', { layers: [{ attn: true, mlp: false }, { attn: true, mlp: false }] });
  checkModel('mlp-only', { layers: [{ attn: false, mlp: true }, { attn: false, mlp: true }] });
  checkModel('one head, wide head', { nHead: 1, dHead: 5 });
  checkModel('three layers, silu, scaled attention', { nLayer: 3, activation: 'silu', attnScaleMult: 2.5 });
  checkModel('tanh, wider residual', { activation: 'tanh', dModel: 5 });
  checkModel('zero layers (bigram)', { nLayer: 0 });
});
