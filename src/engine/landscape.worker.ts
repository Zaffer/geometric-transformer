// Landscape worker: evaluates parameter points of a slice (loss, accuracy,
// carving hashes) and computes Hessian eigenpairs. Many run in parallel.

import { EVAL_STRIDE, Evaluator, packEval, thetaAt, type CarveOpts } from './landscape';
import type { Spec } from './spec';

export type LandIn =
  | { type: 'setup'; job: number; key: string; spec: Spec; split: 'train' | 'val' | 'all'; center: Float64Array; dirs: Float64Array[]; carve: CarveOpts }
  | { type: 'points'; job: number; start: number; idx: Int32Array; coords: Float64Array }
  | { type: 'eigen'; job: number; key: string; spec: Spec; split: 'train' | 'val' | 'all'; theta: Float64Array; iters: number; seed: number }
  | { type: 'grad'; job: number; key: string; spec: Spec; split: 'train' | 'val' | 'all'; theta: Float64Array }
  | { type: 'thetas'; job: number; key: string; spec: Spec; split: 'train' | 'val' | 'all'; center: Float64Array; thetas: Float64Array[]; carve: CarveOpts };

export type LandOut =
  | { type: 'points'; job: number; idx: Int32Array; out: Float64Array }
  | { type: 'eigen'; job: number; values: number[]; vectors: Float64Array[] }
  | { type: 'grad'; job: number; loss: number; grad: Float64Array }
  | { type: 'thetas'; job: number; out: Float64Array }
  | { type: 'error'; job: number; message: string };

let ev: Evaluator | null = null;
let evKey = '';
let job = -1;
let center: Float64Array | null = null;
let dirs: Float64Array[] = [];

const post = (m: LandOut, t: Transferable[] = []) => (self as unknown as Worker).postMessage(m, t);

function evaluator(key: string, spec: Spec, split: 'train' | 'val' | 'all'): Evaluator {
  if (!ev || key !== evKey) {
    ev = new Evaluator(spec, split);
    evKey = key;
  }
  return ev;
}

self.onmessage = (e: MessageEvent<LandIn>) => {
  const m = e.data;
  try {
    if (m.type === 'setup') {
      job = m.job;
      const E = evaluator(m.key, m.spec, m.split);
      center = m.center;
      dirs = m.dirs;
      E.carve = m.carve;
      E.setCenter(center);
    } else if (m.type === 'points') {
      if (m.job !== job || !ev || !center) return;
      const k = dirs.length, n = m.idx.length;
      const out = new Float64Array(n * EVAL_STRIDE);
      const th = new Float64Array(center.length);
      for (let i = 0; i < n; i++) {
        thetaAt(center, dirs, m.coords.subarray(i * k, i * k + k), th);
        packEval(ev.evaluate(th), out, i * EVAL_STRIDE);
      }
      post({ type: 'points', job: m.job, idx: m.idx, out }, [out.buffer]);
    } else if (m.type === 'eigen') {
      const E = evaluator(m.key, m.spec, m.split);
      const r = E.hessianEigen(m.theta, m.iters, m.seed);
      post({ type: 'eigen', job: m.job, values: r.values, vectors: r.vectors });
    } else if (m.type === 'thetas') {
      const E = evaluator(m.key, m.spec, m.split);
      const keep = E.carve;
      E.carve = m.carve;
      E.setCenter(m.center);
      const out = new Float64Array(m.thetas.length * EVAL_STRIDE);
      m.thetas.forEach((th, i) => packEval(E.evaluate(th), out, i * EVAL_STRIDE));
      post({ type: 'thetas', job: m.job, out }, [out.buffer]);
      // put back the running slice's center (flips are measured from it)
      E.carve = keep;
      if (center && ev === E) E.setCenter(center);
    } else if (m.type === 'grad') {
      const E = evaluator(m.key, m.spec, m.split);
      const r = E.gradient(m.theta);
      post({ type: 'grad', job: m.job, loss: r.loss, grad: r.grad });
    }
  } catch (err) {
    post({ type: 'error', job: m.job, message: String((err as Error)?.stack ?? err) });
  }
};
