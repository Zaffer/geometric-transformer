// Numeric gradient check. This proves the hand-written backward pass against
// central finite differences, for every tensor kind, tied and untied.

import { describe, expect, it } from 'vitest';
import { makeConfig } from '../src/model/config';
import { Transformer } from '../src/model/transformer';
import { makeSample } from '../src/model/sortTask';
import { RNG } from '../src/model/rng';

function computeLoss(model: Transformer, tokens: Int32Array, targets: Int32Array): number {
  const cache = model.forward(tokens);
  const V = model.cfg.vocab;
  let loss = 0;
  let count = 0;
  for (let t = 0; t < cache.T; t++) {
    const tgt = targets[t];
    if (tgt < 0) continue;
    loss += -Math.log(Math.max(cache.probs[t * V + tgt], 1e-12));
    count++;
  }
  return loss / count;
}

function checkAllTensors(tieWeights: boolean) {
  const cfg = makeConfig({ nLayer: 2, nHead: 2, dModel: 8, seqLen: 3, tieWeights });
  const model = new Transformer(cfg, 7);
  const rng = new RNG(123);
  const { tokens, targets } = makeSample(rng, cfg.seqLen, cfg.vocab);

  model.zeroGrads();
  const cache = model.forward(tokens);
  model.lossBackward(cache, targets);

  // Per-coordinate central differences. The loss has strong curvature, so
  // eps must be small; the tolerance covers the float32 noise floor.
  const eps = 1e-3;
  const pick = new RNG(999);
  for (const p of model.params) {
    for (let k = 0; k < 4; k++) {
      const i = pick.int(p.data.length);
      const orig = p.data[i];
      p.data[i] = orig + eps;
      const lp = computeLoss(model, tokens, targets);
      p.data[i] = orig - eps;
      const lm = computeLoss(model, tokens, targets);
      p.data[i] = orig;
      const numeric = (lp - lm) / (2 * eps);
      const analytic = p.grad[i];
      const tol = 5e-4 + 0.08 * Math.max(Math.abs(numeric), Math.abs(analytic));
      expect(
        Math.abs(numeric - analytic),
        `${p.name}[${i}] numeric=${numeric} analytic=${analytic}`,
      ).toBeLessThanOrEqual(tol);
    }
  }

  // Directional derivative across ALL parameters at once. This is the strong
  // whole-graph check: d/dt L(theta + t*g) at t=0 must equal ||g||^2.
  let gnorm2 = 0;
  for (const q of model.params) for (let i = 0; i < q.grad.length; i++) gnorm2 += q.grad[i] * q.grad[i];
  const delta = 1e-3 / Math.sqrt(gnorm2);
  const saved = model.params.map((q) => q.data.slice());
  for (const q of model.params) for (let i = 0; i < q.data.length; i++) q.data[i] += delta * q.grad[i];
  const lp = computeLoss(model, tokens, targets);
  model.params.forEach((q, idx) => q.data.set(saved[idx]));
  for (const q of model.params) for (let i = 0; i < q.data.length; i++) q.data[i] -= delta * q.grad[i];
  const lm = computeLoss(model, tokens, targets);
  model.params.forEach((q, idx) => q.data.set(saved[idx]));
  const directional = (lp - lm) / (2 * delta);
  expect(Math.abs(directional - gnorm2) / gnorm2).toBeLessThan(0.01);
}

describe('backward pass', () => {
  it('matches numeric gradients with tied weights', () => {
    checkAllTensors(true);
  });

  it('matches numeric gradients with untied weights', () => {
    checkAllTensors(false);
  });

  it('produces normalized probabilities', () => {
    const cfg = makeConfig({ nLayer: 1, nHead: 2, dModel: 8, seqLen: 4, tieWeights: true });
    const model = new Transformer(cfg, 3);
    const rng = new RNG(5);
    const { tokens } = makeSample(rng, cfg.seqLen, cfg.vocab);
    const cache = model.forward(tokens);
    for (let t = 0; t < cache.T; t++) {
      let sum = 0;
      for (let v = 0; v < cfg.vocab; v++) sum += cache.probs[t * cfg.vocab + v];
      expect(Math.abs(sum - 1)).toBeLessThan(1e-4);
    }
  });
});
