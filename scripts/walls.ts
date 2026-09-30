// How finely do the activations carve the loss landscape?
// Walks a random filter-normalized line through the trained default model
// and counts how often the MLP on/off pattern changes, for several ranges
// and several granularities: every example (the true cells of the training
// loss), layer 0 only, one neuron, one example.   npm run walls
import { Trainer } from '../src/engine/trainer';
import { defaultSpec } from '../src/engine/spec';
import { Evaluator, randomDirection, thetaAt } from '../src/engine/landscape';
import { withGrad } from '../src/engine/tensor';

const tr = new Trainer(defaultSpec());
for (let i = 0; i < 1500; i++) tr.trainStep();
const ev = new Evaluator(tr.spec, 'train');
const th = tr.model.flat();
const dir = randomDirection(tr.model, th, 1, 'filter');
const m = ev.model;
const pack = (ex: typeof ev.examples) => { const x = new Int32Array(ex.length * ev.data.T); ex.forEach((e, i) => x.set(e.x, i * ev.data.T)); return x; };
const xAll = pack(ev.examples), x1 = pack(ev.examples.slice(0, 1));
function key(theta: Float64Array, which: string): string {
  m.setFlat(theta);
  const x = which === 'ex' ? x1 : xAll;
  const B = x.length / ev.data.T;
  const f = withGrad(false, () => m.forward(x, B, true));
  let s = '';
  for (const l of [0, 1]) {
    if (which === 'L0' && l !== 0) continue;
    const pre = f.trace.get(`L${l}.mlp.pre`)!.data;
    const F = 12;
    for (let i = 0; i < pre.length; i++) {
      if (which === 'n0' && (l !== 0 || i % F !== 0)) continue;
      s += pre[i] > 0 ? '1' : '0';
    }
  }
  return s;
}
for (const R of [1, 0.3, 0.1, 0.03, 0.01]) {
  const row: string[] = [];
  for (const which of ['all', 'L0', 'n0', 'ex']) {
    let walls = 0, prev = '';
    const N = 400;
    for (let i = 0; i < N; i++) {
      const a = -R + (2 * R * i) / (N - 1);
      const k = key(thetaAt(th, [dir], [a]), which);
      if (i && k !== prev) walls++;
      prev = k;
    }
    row.push(`${which}:${walls}`);
  }
  console.log(`range ±${R}: walls along 400 points: ${row.join('  ')}`);
}
