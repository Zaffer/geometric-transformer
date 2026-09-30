// Headless training runs for choosing defaults (see sweep-run.ts).
import { Trainer } from '../src/engine/trainer';
import { defaultSpec, type Spec } from '../src/engine/spec';

export function run(label: string, mut: (s: Spec) => void, steps = 3000): void {
  const s = defaultSpec();
  mut(s);
  const tr = new Trainer(s);
  const t0 = performance.now();
  let out = '';
  for (let i = 1; i <= steps; i++) {
    tr.trainStep();
    if (i % Math.max(1, Math.floor(steps / 4)) === 0) {
      const e = tr.evalAll();
      out += ` | ${i}: loss ${e.train.loss.toFixed(3)} acc ${e.train.seqAcc.toFixed(2)} val ${e.val.seqAcc.toFixed(2)}`;
    }
  }
  console.log(`${label.padEnd(28)} ${tr.model.numParams()} params ${((performance.now() - t0) / steps).toFixed(2)} ms/step${out}`);
}
