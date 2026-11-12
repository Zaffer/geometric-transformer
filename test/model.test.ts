import { expect, test } from 'bun:test';
import { type Config, backward, forward, init, layout, locate, probe } from '../src/model';
import { TASKS, evaluate, trainer } from '../src/train';

const c: Config = { V: 4, T: 5, H: 6 };
const L = layout(c);

test('hand-written gradient matches finite differences for every parameter', () => {
  const th = init(c, 7);
  const tokens = [2, 0, 3, 1, 2], targets = TASKS.max.targets(tokens);
  const g = new Float64Array(L.size);
  backward(c, L, th, forward(c, L, th, tokens), targets, 1, g);
  const loss = (p: Float64Array) => backward(c, L, p, forward(c, L, p, tokens), targets, 1, new Float64Array(L.size));
  const h = 1e-6;
  for (let i = 0; i < L.size; i++) {
    const up = th.slice(), dn = th.slice();
    up[i] += h; dn[i] -= h;
    const fd = (loss(up) - loss(dn)) / (2 * h);
    const where = locate(L, i);
    expect(Math.abs(fd - g[i]), `${where.block.name}[${where.r},${where.c}]`).toBeLessThan(1e-6 * Math.max(1, Math.abs(fd)));
  }
});

test('the probe at a token\'s own point reproduces the forward pass', () => {
  const th = init(c, 3);
  const tokens = [1, 3, 0, 2, 1], t = forward(c, L, th, tokens);
  for (let pos = 0; pos < c.T; pos++) {
    const out = probe(c, L, th, t, pos).eval(t.x[2 * pos], t.x[2 * pos + 1]);
    for (let w = 0; w < c.V; w++) expect(out.logits[w]).toBeCloseTo(t.logits[pos * c.V + w], 10);
    for (let j = 0; j <= pos; j++) expect(out.a[j]).toBeCloseTo(t.a[pos * c.T + j], 10);
  }
});

test('the default model learns "largest so far"', () => {
  const th = init(c, 1), tr = trainer(c, L, TASKS.max, 1);
  for (let s = 0; s < 3000; s++) tr.step(th, 32, 0.01);
  expect(evaluate(c, L, th, TASKS.max).acc).toBeGreaterThan(0.95);
});
