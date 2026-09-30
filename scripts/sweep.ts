// Quick headless sweep: does a configuration learn? (dev tool)
import { Trainer } from '../src/engine/trainer';
import { defaultSpec } from '../src/engine/spec';

const configs: [string, (s: ReturnType<typeof defaultSpec>) => void][] = JSON.parse(process.env.CFG ?? 'null') ?? [];
export function run(label: string, mut: (s: ReturnType<typeof defaultSpec>) => void, steps = 3000) {
  const s = defaultSpec();
  mut(s);
  const tr = new Trainer(s);
  const t0 = performance.now();
  let out = '';
  for (let i = 1; i <= steps; i++) {
    tr.trainStep();
    if (i % (steps / 4) === 0) {
      const e = tr.evalAll();
      out += ` | ${i}: tr ${e.train.loss.toFixed(3)} acc ${e.train.seqAcc.toFixed(2)} val ${e.val.seqAcc.toFixed(2)}`;
    }
  }
  console.log(`${label.padEnd(40)} params ${tr.model.numParams()} ${((performance.now() - t0) / steps).toFixed(2)}ms/step${out}`);
}
void configs;
