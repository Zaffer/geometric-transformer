// Optimizers: SGD, SGD + momentum, Adam, AdamW. One class, one step().
// Respects each parameter's lrMult, frozen flag and per-element freeze mask,
// so any single weight can be pinned while the rest train.

import type { Model, Param } from './model';
import type { OptimSpec } from './spec';

interface State {
  m: Float64Array;
  v: Float64Array;
}

export interface StepStats {
  lr: number;
  gradNorm: number;
  /** per-param: gradient norm, update norm, value norm */
  perParam: { name: string; grad: number; update: number; value: number }[];
}

export function lrAt(o: OptimSpec, step: number): number {
  let f = 1;
  if (o.warmup > 0 && step < o.warmup) f = (step + 1) / o.warmup;
  else if (o.schedule !== 'constant') {
    const t = Math.min(1, (step - o.warmup) / Math.max(1, o.totalSteps - o.warmup));
    f = o.schedule === 'cosine' ? 0.5 * (1 + Math.cos(Math.PI * t)) : 1 - t;
  }
  return o.lr * f;
}

export function decays(p: Param, o: OptimSpec): boolean {
  return o.decayAll || p.kind === 'weight' || p.kind === 'embed';
}

export class Optimizer {
  spec: OptimSpec;
  private state = new Map<string, State>();
  t = 0;

  constructor(spec: OptimSpec) {
    this.spec = spec;
  }

  /** Carry over moments for every param whose name and size still match (architecture edits). */
  adoptState(other: Optimizer, model: Model): void {
    this.t = other.t;
    for (const p of model.params) {
      const st = other.state.get(p.name);
      if (st && st.m.length === p.t.size) this.state.set(p.name, { m: st.m.slice(), v: st.v.slice() });
    }
  }

  reset(): void {
    this.state.clear();
    this.t = 0;
  }

  private st(p: Param): State {
    let s = this.state.get(p.name);
    if (!s || s.m.length !== p.t.size) {
      s = { m: new Float64Array(p.t.size), v: new Float64Array(p.t.size) };
      this.state.set(p.name, s);
    }
    return s;
  }

  /** One update from the gradients currently in the model's params. */
  step(model: Model, stats = false): StepStats {
    const o = this.spec;
    const lr = lrAt(o, this.t);
    this.t++;

    // global gradient norm over the parameters that can move
    let g2 = 0;
    for (const p of model.params) {
      if (p.frozen || p.lrMult === 0 || !p.t.grad) continue;
      const G = p.t.grad, M = p.mask;
      for (let i = 0; i < G.length; i++) if (!M || !M[i]) g2 += G[i] * G[i];
    }
    const gradNorm = Math.sqrt(g2);
    const clipF = o.clip > 0 && gradNorm > o.clip ? o.clip / gradNorm : 1;

    const perParam: StepStats['perParam'] = [];
    const b1 = o.beta1, b2 = o.beta2;
    const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
    for (const p of model.params) {
      const G = p.t.grad;
      const W = p.t.data;
      if (p.frozen || p.lrMult === 0 || !G) {
        if (stats) perParam.push({ name: p.name, grad: G ? norm2(G) : 0, update: 0, value: norm2(W) });
        continue;
      }
      const a = lr * p.lrMult;
      const wd = decays(p, o) ? o.weightDecay : 0;
      const s = this.st(p);
      const M = p.mask;
      let u2 = 0;
      for (let i = 0; i < W.length; i++) {
        if (M && M[i]) continue;
        let g = G[i] * clipF;
        let du: number;
        switch (o.kind) {
          case 'sgd':
            du = -a * (g + wd * W[i]);
            break;
          case 'momentum':
            g += wd * W[i];
            s.m[i] = o.momentum * s.m[i] + g;
            du = -a * s.m[i];
            break;
          case 'adam':
          case 'adamw': {
            if (o.kind === 'adam') g += wd * W[i];
            s.m[i] = b1 * s.m[i] + (1 - b1) * g;
            s.v[i] = b2 * s.v[i] + (1 - b2) * g * g;
            du = (-a * (s.m[i] / bc1)) / (Math.sqrt(s.v[i] / bc2) + o.eps);
            if (o.kind === 'adamw') du -= a * wd * W[i];
            break;
          }
        }
        W[i] += du;
        u2 += du * du;
      }
      if (stats) perParam.push({ name: p.name, grad: norm2(G), update: Math.sqrt(u2), value: norm2(W) });
    }
    return { lr, gradNorm, perParam };
  }
}

function norm2(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}
