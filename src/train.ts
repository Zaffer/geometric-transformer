// Tasks, and Adam. Each task scores every position, so the model is trained on all
// prefixes of every sequence at once.

import { type Config, type Layout, backward, forward, rng } from './model';

export interface Task {
  name: string;
  about: string;
  targets(tokens: number[]): number[];
}

export const TASKS: Record<string, Task> = {
  max: {
    name: 'largest so far',
    about: 'Say the largest token seen so far. It must look back: the answer is usually an earlier token.',
    targets: (t) => t.map((_, i) => Math.max(...t.slice(0, i + 1))),
  },
  first: {
    name: 'first token',
    about: 'Say the first token of the sequence. It must find position 0, so it needs the position embedding.',
    targets: (t) => t.map(() => t[0]),
  },
  same: {
    name: 'echo',
    about: 'Say the current token. No attention needed: a baseline.',
    targets: (t) => t.slice(),
  },
};

export interface Trainer {
  step(th: Float64Array, batch: number, lr: number): number;
  reset(): void;
  steps: number;
}

export function trainer(c: Config, L: Layout, task: Task, seed: number): Trainer {
  let r = rng(seed);
  const m = new Float64Array(L.size), v = new Float64Array(L.size), g = new Float64Array(L.size);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const self: Trainer = {
    steps: 0,
    reset() { r = rng(seed); m.fill(0); v.fill(0); self.steps = 0; },
    step(th, batch, lr) {
      g.fill(0);
      let loss = 0;
      for (let b = 0; b < batch; b++) {
        const tokens = Array.from({ length: c.T }, () => r.int(c.V));
        loss += backward(c, L, th, forward(c, L, th, tokens), task.targets(tokens), 1 / (batch * c.T), g);
      }
      const k = ++self.steps;
      const fix = Math.sqrt(1 - b2 ** k) / (1 - b1 ** k);
      for (let i = 0; i < L.size; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        th[i] -= lr * fix * m[i] / (Math.sqrt(v[i]) + eps);
      }
      return loss;
    },
  };
  return self;
}

/** Exact loss and accuracy over every possible sequence. */
export function evaluate(c: Config, L: Layout, th: Float64Array, task: Task): { loss: number; acc: number } {
  const total = c.V ** c.T;
  let loss = 0, right = 0;
  for (let id = 0; id < total; id++) {
    const tokens: number[] = [];
    for (let i = 0, x = id; i < c.T; i++, x = Math.floor(x / c.V)) tokens.push(x % c.V);
    const t = forward(c, L, th, tokens), y = task.targets(tokens);
    for (let i = 0; i < c.T; i++) {
      loss -= Math.log(Math.max(t.p[i * c.V + y[i]], 1e-300));
      let best = 0;
      for (let w = 1; w < c.V; w++) if (t.p[i * c.V + w] > t.p[i * c.V + best]) best = w;
      if (best === y[i]) right++;
    }
  }
  return { loss: loss / (total * c.T), acc: right / (total * c.T) };
}
