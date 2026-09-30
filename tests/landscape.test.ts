import { describe, expect, it } from 'vitest';
import {
  Evaluator,
  coarseToFine,
  dot,
  orthogonalize,
  projectToSlice,
  randomDirection,
  thetaAt,
  trajectoryPCA,
  unitDirection,
} from '../src/engine/landscape';
import { defaultSpec } from '../src/engine/spec';
import { Trainer, evaluate } from '../src/engine/trainer';

function trained(steps = 300, mut?: (s: ReturnType<typeof defaultSpec>) => void) {
  const s = defaultSpec();
  mut?.(s);
  const tr = new Trainer(s);
  for (let i = 0; i < steps; i++) tr.trainStep();
  return tr;
}

describe('landscape', () => {
  it('evaluates the same loss as the trainer', () => {
    const tr = trained(200);
    const ev = new Evaluator(tr.spec, 'train');
    const e = ev.evaluate(tr.model.flat());
    const ref = evaluate(tr.model, tr.data.train, tr.data.T);
    expect(e.loss).toBeCloseTo(ref.loss, 10);
    expect(e.acc).toBeCloseTo(ref.seqAcc, 10);
  });

  it('Hessian-vector products are symmetric', () => {
    const tr = trained(100, (s) => (s.model.activation = 'gelu'));
    const ev = new Evaluator(tr.spec, 'train', 64);
    const th = tr.model.flat();
    const u = randomDirection(tr.model, th, 1, 'unit');
    const v = randomDirection(tr.model, th, 2, 'unit');
    const a = dot(u, ev.hvp(th, v)), b = dot(v, ev.hvp(th, u));
    expect(Math.abs(a - b)).toBeLessThan(1e-4 * Math.max(1, Math.abs(a)));
  });

  it('Lanczos finds an eigenpair of the Hessian', () => {
    const tr = trained(100, (s) => (s.model.activation = 'gelu'));
    const ev = new Evaluator(tr.spec, 'train', 64);
    const th = tr.model.flat();
    const { values, vectors } = ev.hessianEigen(th, 30);
    const v = vectors[0];
    const Hv = ev.hvp(th, v);
    const lam = values[0];
    const res = Math.sqrt(dot(Hv.map((x, i) => x - lam * v[i]), Hv.map((x, i) => x - lam * v[i])));
    expect(lam).toBeGreaterThan(0);
    expect(res / Math.abs(lam)).toBeLessThan(0.05);
    // the top value bounds random Rayleigh quotients
    const r = randomDirection(tr.model, th, 5, 'unit');
    expect(dot(r, ev.hvp(th, r))).toBeLessThanOrEqual(lam + 1e-6);
  });

  it('cell walls sit exactly where a first-layer neuron switches', () => {
    // no norm: the layer-0 MLP input z does not depend on W1, so moving W1[i,j]
    // by a changes neuron j's pre-activation at every position by a·z_i:
    // it switches at a* = −pre/z_i, and the MLP pattern hash must change there.
    const tr = trained(150, (s) => {
      s.model.norm = 'none';
      s.model.finalNorm = false;
    });
    const ev = new Evaluator(tr.spec, 'train', 32);
    const th = tr.model.flat();
    const F = tr.model.spec.dMlp, i = 1, j = 4;
    const dir = unitDirection(tr.model, 'L0.mlp.W1', i * F + j);
    const fwd = tr.model.forward(new Int32Array(ev.examples.flatMap((e) => Array.from(e.x))), ev.examples.length, true);
    const z = fwd.trace.get('L0.ln2')!.data, pre = fwd.trace.get('L0.mlp.pre')!.data;
    const D = tr.model.spec.dModel;
    const stars: number[] = [];
    for (let r = 0; r < z.length / D; r++) {
      const zi = z[r * D + i];
      if (Math.abs(zi) > 1e-6) stars.push(-pre[r * F + j] / zi);
    }
    const inRange = stars.filter((a) => Math.abs(a) < 2).sort((a, b) => a - b);
    expect(inRange.length).toBeGreaterThan(0);
    let checked = 0;
    for (let k = 0; k < inRange.length && checked < 6; k++) {
      const a = inRange[k];
      const gap = Math.min(k > 0 ? a - inRange[k - 1] : 1, k + 1 < inRange.length ? inRange[k + 1] - a : 1);
      if (gap < 1e-4) continue;
      const d = gap / 4;
      const lo = ev.evaluate(thetaAt(th, [dir], [a - d])).h.mlp;
      const hi = ev.evaluate(thetaAt(th, [dir], [a + d])).h.mlp;
      expect(lo).not.toEqual(hi);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it('coarser carvings have fewer walls: one example ⊂ all examples', () => {
    const tr = trained(300);
    const ev = new Evaluator(tr.spec, 'train', 64);
    const th = tr.model.flat();
    const dir = randomDirection(tr.model, th, 1, 'filter');
    ev.carve = { k: 1, layer: 0, neuron: 0 };
    const walls = { mlp: 0, examples: 0, neuron: 0 };
    let prev = ev.evaluate(th);
    for (let i = 1; i <= 120; i++) {
      const e = ev.evaluate(thetaAt(th, [dir], [(i / 120) * 0.3]));
      for (const k of ['mlp', 'examples', 'neuron'] as const) if (e.h[k][0] !== prev.h[k][0] || e.h[k][1] !== prev.h[k][1]) walls[k]++;
      prev = e;
    }
    expect(walls.mlp).toBeGreaterThan(walls.examples);
    expect(walls.mlp).toBeGreaterThanOrEqual(walls.neuron);
    expect(walls.examples).toBeGreaterThan(0);
  });

  it('flips count switches from the center pattern', () => {
    const tr = trained(100);
    const ev = new Evaluator(tr.spec, 'train', 32);
    const th = tr.model.flat();
    ev.setCenter(th);
    expect(ev.evaluate(th).flips).toBe(0);
    const far = thetaAt(th, [randomDirection(tr.model, th, 3, 'filter')], [1]);
    expect(ev.evaluate(far).flips).toBeGreaterThan(0);
  });

  it('coarse-to-fine visits every grid point once', () => {
    for (const [n, k] of [[9, 2], [16, 2], [7, 3], [30, 1]]) {
      const o = coarseToFine(n, k);
      expect(o.length).toBe(n ** k);
      expect(new Set(o).size).toBe(n ** k);
    }
  });

  it('projects points back into the slice', () => {
    const tr = trained(10);
    const th = tr.model.flat();
    const dirs = orthogonalize([randomDirection(tr.model, th, 1, 'filter'), randomDirection(tr.model, th, 2, 'filter')]);
    const p = thetaAt(th, dirs, [0.3, -0.7]);
    const { a, off } = projectToSlice(p, th, dirs);
    expect(a[0]).toBeCloseTo(0.3, 8);
    expect(a[1]).toBeCloseTo(-0.7, 8);
    expect(off).toBeLessThan(1e-9);
  });

  it('trajectory PCA spans the path', () => {
    const tr = trained(400);
    const pts = tr.history.map((c) => c.params);
    const { dirs, explained } = trajectoryPCA(pts, tr.model.flat(), 2);
    expect(dirs.length).toBe(2);
    expect(explained[0]).toBeGreaterThan(explained[1]);
    const first = projectToSlice(pts[0], tr.model.flat(), dirs);
    expect(Math.abs(first.a[0])).toBeGreaterThan(0.5);
  });
});
