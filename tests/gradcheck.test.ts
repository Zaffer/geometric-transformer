// Numeric gradient check. This proves the hand-written backward pass against
// central finite differences, for every tensor kind and every toggle.

import { describe, expect, it } from 'vitest';
import { makeConfig, type ConfigInput } from '../src/model/config';
import { Transformer } from '../src/model/transformer';
import { makeSample } from '../src/model/sortTask';
import { RNG } from '../src/model/rng';

// Loss plus the ReLU sign pattern of every MLP pre-activation. A finite
// difference is only valid when both probe points share the pattern; at a
// kink the analytic subgradient and the numeric slope legitimately differ.
function computeLoss(model: Transformer, tokens: Int32Array, targets: Int32Array): { loss: number; pattern: string } {
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
  let pattern = '';
  if (model.cfg.activation === 'relu') {
    for (const blk of cache.blocks) {
      if (blk.h) for (let i = 0; i < blk.h.length; i++) pattern += blk.h[i] > 0 ? '1' : '0';
    }
  }
  return { loss: loss / count, pattern };
}

function checkAllTensors(input: Partial<ConfigInput>) {
  const cfg = makeConfig({ nLayer: 2, nHead: 2, dModel: 8, seqLen: 3, tieWeights: true, ...input });
  const model = new Transformer(cfg, 7);
  const rng = new RNG(123);
  const { tokens, targets } = makeSample(rng, cfg.seqLen, cfg.vocab);

  model.zeroGrads();
  const cache = model.forward(tokens);
  model.lossBackward(cache, targets);

  // Per-coordinate central differences. The loss has strong curvature, so
  // eps must be small; the tolerance covers the float32 noise floor.
  // ReLU gets a smaller step, so fewer probes cross a kink.
  const eps = cfg.activation === 'relu' ? 2e-4 : 1e-3;
  const pick = new RNG(999);
  let checked = 0;
  let skipped = 0;
  for (const p of model.params) {
    for (let k = 0; k < 4; k++) {
      const i = pick.int(p.data.length);
      const orig = p.data[i];
      p.data[i] = orig + eps;
      const lp = computeLoss(model, tokens, targets);
      p.data[i] = orig - eps;
      const lm = computeLoss(model, tokens, targets);
      p.data[i] = orig;
      if (lp.pattern !== lm.pattern) {
        skipped++;
        continue;
      }
      checked++;
      const numeric = (lp.loss - lm.loss) / (2 * eps);
      const analytic = p.grad[i];
      const tol = 5e-4 + 0.08 * Math.max(Math.abs(numeric), Math.abs(analytic));
      expect(
        Math.abs(numeric - analytic),
        `${p.name}[${i}] numeric=${numeric} analytic=${analytic}`,
      ).toBeLessThanOrEqual(tol);
    }
  }
  // Kink skips must stay rare, or the check proves nothing.
  expect(skipped).toBeLessThanOrEqual(Math.floor(0.2 * (checked + skipped)));

  // Directional derivative across ALL parameters at once. This is the strong
  // whole-graph check: d/dt L(theta + t*g) at t=0 must equal ||g||^2.
  // For ReLU the step shrinks until both probes share one sign pattern.
  let gnorm2 = 0;
  for (const q of model.params) for (let i = 0; i < q.grad.length; i++) gnorm2 += q.grad[i] * q.grad[i];
  const saved = model.params.map((q) => q.data.slice());
  let delta = 1e-3 / Math.sqrt(gnorm2);
  let directional = Number.NaN;
  for (let attempt = 0; attempt < 4; attempt++) {
    for (const q of model.params) for (let i = 0; i < q.data.length; i++) q.data[i] += delta * q.grad[i];
    const lp = computeLoss(model, tokens, targets);
    model.params.forEach((q, idx) => q.data.set(saved[idx]));
    for (const q of model.params) for (let i = 0; i < q.data.length; i++) q.data[i] -= delta * q.grad[i];
    const lm = computeLoss(model, tokens, targets);
    model.params.forEach((q, idx) => q.data.set(saved[idx]));
    if (lp.pattern === lm.pattern) {
      directional = (lp.loss - lm.loss) / (2 * delta);
      break;
    }
    delta /= 4;
  }
  expect(Number.isFinite(directional), 'directional check found no kink-free step').toBe(true);
  expect(Math.abs(directional - gnorm2) / gnorm2).toBeLessThan(0.02);
}

const VARIANTS: Array<[string, Partial<ConfigInput>]> = [
  ['pure GPT-2, tied weights', {}],
  ['pure GPT-2, untied weights', { tieWeights: false }],
  ['relu activation', { activation: 'relu' }],
  ['rmsnorm', { norm: 'rmsnorm' }],
  ['no norm', { norm: 'none' }],
  ['no mlp', { mlp: false }],
  ['attention only (no norm, no mlp)', { norm: 'none', mlp: false }],
  ['relu + rmsnorm + untied', { activation: 'relu', norm: 'rmsnorm', tieWeights: false }],
];

describe('backward pass', () => {
  for (const [name, input] of VARIANTS) {
    it(`matches numeric gradients: ${name}`, () => {
      checkAllTensors(input);
    });
  }

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

  it('has the expected parameter counts per toggle', () => {
    const base = { nLayer: 1, nHead: 2, dModel: 8, seqLen: 3, tieWeights: true };
    const full = new Transformer(makeConfig(base)).paramCount();
    const noMlp = new Transformer(makeConfig({ ...base, mlp: false })).paramCount();
    const rms = new Transformer(makeConfig({ ...base, norm: 'rmsnorm' })).paramCount();
    const none = new Transformer(makeConfig({ ...base, norm: 'none' })).paramCount();
    // MLP: ln2 (16) + wfc (8*32) + bfc (32) + wproj (32*8) + bproj (8)
    expect(full - noMlp).toBe(16 + 256 + 32 + 256 + 8);
    // RMSNorm drops the three norm biases (ln1b, ln2b, lnfb), 8 each.
    expect(full - rms).toBe(3 * 8);
    // No norm drops gains and biases, 3 x 16.
    expect(full - none).toBe(3 * 16);
  });
});
