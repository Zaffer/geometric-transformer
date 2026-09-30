// The training loop: sample a batch → forward → loss → backward → optimizer.
// Also exact evaluation on the whole train/val sets and a checkpoint
// history that thins itself so it always spans the whole run.

import { Model } from './model';
import { Optimizer, type StepStats } from './optim';
import { RNG } from './rng';
import { normalizeSpec, type Spec } from './spec';
import { buildDataset, packBatch, type Dataset, type Example } from './tasks';
import { withGrad } from './tensor';

export interface EvalResult {
  loss: number;
  /** fraction of supervised tokens predicted correctly (argmax) */
  tokenAcc: number;
  /** fraction of examples with every supervised token correct */
  seqAcc: number;
  n: number;
}

export interface Metrics {
  step: number;
  loss: number;
  lr: number;
  gradNorm: number;
  train?: EvalResult;
  val?: EvalResult;
}

export interface Checkpoint {
  step: number;
  params: Float64Array;
  loss: number;
}

export function evaluate(model: Model, data: Example[], T: number, cap = 4096): EvalResult {
  const ex = data.length > cap ? data.slice(0, cap) : data;
  if (!ex.length) return { loss: NaN, tokenAcc: NaN, seqAcc: NaN, n: 0 };
  return withGrad(false, () => {
    let lossSum = 0, tokOk = 0, tokN = 0, seqOk = 0;
    const CH = 256;
    for (let s = 0; s < ex.length; s += CH) {
      const chunk = ex.slice(s, s + CH);
      const { x, y, B } = packBatch(chunk, T);
      const { loss, probs } = model.loss(x, y, B);
      let n = 0;
      for (let i = 0; i < y.length; i++) if (y[i] >= 0) n++;
      lossSum += loss.data[0] * n;
      const V = model.vocab;
      for (let b = 0; b < B; b++) {
        let ok = true;
        for (let t = 0; t < T; t++) {
          const i = b * T + t;
          if (y[i] < 0) continue;
          let best = 0;
          for (let v = 1; v < V; v++) if (probs[i * V + v] > probs[i * V + best]) best = v;
          tokN++;
          if (best === y[i]) tokOk++;
          else ok = false;
        }
        if (ok) seqOk++;
      }
    }
    return { loss: lossSum / Math.max(1, tokN), tokenAcc: tokOk / Math.max(1, tokN), seqAcc: seqOk / ex.length, n: ex.length };
  });
}

export class Trainer {
  spec: Spec;
  data: Dataset;
  model: Model;
  opt: Optimizer;
  step = 0;
  private rng: RNG;
  history: Checkpoint[] = [];
  private every = 1;
  readonly maxCheckpoints = 400;

  constructor(spec: Spec, keep?: Model) {
    this.spec = normalizeSpec(spec);
    this.data = buildDataset(spec.task);
    this.model = new Model(spec.model, this.data.vocab.length, this.data.T, spec.init);
    if (keep) this.model.adoptFrom(keep);
    this.opt = new Optimizer(spec.optim);
    this.rng = new RNG(spec.optim.seed);
    this.checkpoint(NaN);
  }

  /** Change optimizer settings without touching the weights. */
  setOptim(o: Spec['optim']): void {
    const seedChanged = o.seed !== this.spec.optim.seed;
    this.spec.optim = o;
    this.opt.spec = o;
    if (seedChanged) this.rng = new RNG(o.seed + this.step);
  }

  sampleBatch(): Example[] {
    const tr = this.data.train;
    const B = this.spec.optim.batchSize;
    const out: Example[] = [];
    for (let i = 0; i < B; i++) out.push(tr[this.rng.int(tr.length)]);
    return out;
  }

  trainStep(stats = false): { loss: number; stats: StepStats } {
    const batch = this.sampleBatch();
    const { x, y, B } = packBatch(batch, this.data.T);
    this.model.zeroGrad();
    const { loss } = withGrad(true, () => this.model.loss(x, y, B));
    loss.backward();
    const st = this.opt.step(this.model, stats);
    this.step++;
    const l = loss.data[0];
    if (this.step % this.every === 0) this.checkpoint(l);
    return { loss: l, stats: st };
  }

  checkpoint(loss: number): void {
    this.history.push({ step: this.step, params: this.model.flat(), loss });
    if (this.history.length > this.maxCheckpoints) {
      // keep every other checkpoint, and record half as often from now on
      this.history = this.history.filter((_, i) => i % 2 === 0 || i === this.history.length - 1);
      this.every *= 2;
    }
  }

  /** Forget the future: training resumes from the current weights. */
  truncateHistory(step: number): void {
    this.history = this.history.filter((c) => c.step <= step);
  }

  evalAll(): { train: EvalResult; val: EvalResult } {
    return {
      train: evaluate(this.model, this.data.train, this.data.T),
      val: evaluate(this.model, this.data.val, this.data.T),
    };
  }
}
