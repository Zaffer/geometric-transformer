// Training smoke test: the micro model must learn the sort task noticeably
// within a few hundred steps.

import { describe, expect, it } from 'vitest';
import { MICRO, makeConfig } from '../src/model/config';
import { Transformer } from '../src/model/transformer';
import { Trainer } from '../src/model/trainer';
import { evalAccuracy } from '../src/model/sortTask';
import { RNG } from '../src/model/rng';

describe('training', () => {
  it('reduces the loss and sorts better than chance', { timeout: 120_000 }, () => {
    const cfg = makeConfig(MICRO);
    const model = new Transformer(cfg, 1337);
    const trainer = new Trainer(model, 0.01, 42);

    const firstLoss = trainer.step(8);
    let loss = firstLoss;
    for (let s = 0; s < 300; s++) loss = trainer.step(8);

    expect(loss).toBeLessThan(0.7);
    expect(loss).toBeLessThan(firstLoss);

    const acc = evalAccuracy(model, new RNG(9), 50);
    expect(acc).toBeGreaterThan(0.3);
  });
});
