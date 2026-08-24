// The minGPT sort task. The model reads seqLen random tokens from {A, B, C}
// and must emit them in sorted order. The training sequence is
// cat = [x..., sorted(x)...]; input = cat[:-1]; target = cat[1:], with the
// first seqLen - 1 targets masked (-1).

import type { Transformer } from './transformer';
import { RNG } from './rng';

export interface Sample {
  tokens: Int32Array; // length 2n - 1
  targets: Int32Array; // length 2n - 1, -1 = masked
}

export function makeSample(rng: RNG, seqLen: number, vocab: number): Sample {
  const n = seqLen;
  const x = new Int32Array(n);
  for (let i = 0; i < n; i++) x[i] = rng.int(vocab);
  const sorted = Int32Array.from(x).sort();
  const cat = new Int32Array(2 * n);
  cat.set(x, 0);
  cat.set(sorted, n);
  const tokens = cat.slice(0, 2 * n - 1);
  const targets = new Int32Array(2 * n - 1).fill(-1);
  for (let t = n - 1; t < 2 * n - 1; t++) targets[t] = cat[t + 1];
  return { tokens, targets };
}

// Greedy autoregressive generation: read n prompt tokens, emit n tokens.
export function generateSorted(model: Transformer, prompt: Int32Array): Int32Array {
  const n = prompt.length;
  const V = model.cfg.vocab;
  const out = new Int32Array(n);
  let seq = Int32Array.from(prompt);
  for (let k = 0; k < n; k++) {
    const cache = model.forward(seq);
    const last = (cache.T - 1) * V;
    let best = 0;
    for (let v = 1; v < V; v++) {
      if (cache.probs[last + v] > cache.probs[last + best]) best = v;
    }
    out[k] = best;
    if (k < n - 1) {
      const next = new Int32Array(seq.length + 1);
      next.set(seq);
      next[seq.length] = best;
      seq = next;
    }
  }
  return out;
}

// Fraction of prompts where the full sorted output is correct.
export function evalAccuracy(model: Transformer, rng: RNG, samples: number): number {
  const n = model.cfg.seqLen;
  const V = model.cfg.vocab;
  let correct = 0;
  for (let s = 0; s < samples; s++) {
    const x = new Int32Array(n);
    for (let i = 0; i < n; i++) x[i] = rng.int(V);
    const sorted = Int32Array.from(x).sort();
    const out = generateSorted(model, x);
    let ok = true;
    for (let i = 0; i < n; i++) {
      if (out[i] !== sorted[i]) {
        ok = false;
        break;
      }
    }
    if (ok) correct++;
  }
  return correct / samples;
}
