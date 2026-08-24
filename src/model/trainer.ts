// Batch training driver. Browser-independent so tests can run it in node.

import type { Transformer } from './transformer';
import { AdamW, clipGradNorm } from './adamw';
import { makeSample } from './sortTask';
import { RNG } from './rng';

export class Trainer {
  readonly model: Transformer;
  readonly opt: AdamW;
  readonly rng: RNG;
  stepCount = 0;

  constructor(model: Transformer, lr: number, seed = 42) {
    this.model = model;
    this.opt = new AdamW(lr);
    this.rng = new RNG(seed);
  }

  // One optimizer step over a fresh random batch. Returns the mean loss.
  step(batchSize: number): number {
    const m = this.model;
    m.zeroGrads();
    let loss = 0;
    for (let s = 0; s < batchSize; s++) {
      const { tokens, targets } = makeSample(this.rng, m.cfg.seqLen, m.cfg.vocab);
      const cache = m.forward(tokens);
      loss += m.lossBackward(cache, targets);
    }
    loss /= batchSize;
    // Mean gradient over the batch.
    for (const p of m.params) {
      for (let i = 0; i < p.grad.length; i++) p.grad[i] /= batchSize;
    }
    clipGradNorm(m.params, 1.0);
    this.opt.step(m.params);
    this.stepCount++;
    return loss;
  }
}
