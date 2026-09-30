// Tasks: small, finite datasets for next-token prediction.
//
// Every task is a list of examples. An example is a token sequence x of
// length T and targets y of length T, y[t] = the token that should follow
// x[0..t], or -1 where no loss is taken. This is exactly how an LLM is
// trained; the algorithmic tasks just mask the positions whose next token
// is random (the input half), so the loss measures only what is learnable.
//
// The datasets are small enough to enumerate. Exact train/val losses and
// the landscape (phase 2) use the whole dataset, not a sample.

import { RNG } from './rng';
import type { TaskSpec } from './spec';

export interface Example {
  x: Int32Array;
  y: Int32Array;
}

export interface Dataset {
  vocab: string[];
  /** context length: every example has exactly this many tokens */
  T: number;
  train: Example[];
  val: Example[];
  /** a short human description */
  about: string;
  /** index of the first supervised position (the "answer" start), for display */
  answerStart: number;
  /** a token string for display */
  fmt(ids: ArrayLike<number>): string;
}

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

/** all sequences of `len` symbols from an alphabet of n, capped */
function enumerate(n: number, len: number, cap: number, rng: RNG): number[][] {
  const total = Math.pow(n, len);
  const out: number[][] = [];
  if (total <= cap) {
    for (let k = 0; k < total; k++) {
      const s: number[] = [];
      let r = k;
      for (let i = 0; i < len; i++) {
        s.push(r % n);
        r = Math.floor(r / n);
      }
      out.push(s.reverse());
    }
  } else {
    const seen = new Set<string>();
    while (out.length < cap) {
      const s = Array.from({ length: len }, () => rng.int(n));
      const key = s.join(',');
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(s);
    }
  }
  return out;
}

function split(all: Example[], frac: number, rng: RNG): { train: Example[]; val: Example[] } {
  rng.shuffle(all);
  const nTrain = Math.max(1, Math.min(all.length, Math.round(all.length * frac)));
  return { train: all.slice(0, nTrain), val: all.slice(nTrain) };
}

/** Sequence-to-sequence tasks in the minGPT style: [input | output], no separator. */
function seqTask(spec: TaskSpec, rng: RNG, transform: (s: number[]) => number[], name: string): Dataset {
  const n = spec.nSymbols, L = spec.length;
  const vocab = LETTERS.slice(0, n).split('');
  const T = 2 * L - 1;
  const ex: Example[] = enumerate(n, L, 20000, rng).map((inp) => {
    const full = [...inp, ...transform(inp)];
    const x = Int32Array.from(full.slice(0, T));
    const y = Int32Array.from(full.slice(1, T + 1));
    if (spec.lossOn === 'answer') for (let t = 0; t < L - 1; t++) y[t] = -1;
    return { x, y };
  });
  return {
    vocab,
    T,
    ...split(ex, spec.trainFrac, rng),
    about: `${name}: read ${L} symbols from {${vocab.join(',')}}, then write them ${name === 'sort' ? 'in sorted order' : name === 'reverse' ? 'reversed' : 'again'}`,
    answerStart: L - 1,
    fmt: (ids) => Array.from(ids, (i) => vocab[i] ?? '?').join(''),
  };
}

function modadd(spec: TaskSpec, rng: RNG): Dataset {
  const p = spec.modulus;
  const vocab = [...Array.from({ length: p }, (_, i) => String(i)), '='];
  const EQ = p;
  const ex: Example[] = [];
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++) {
      const x = Int32Array.from([a, b, EQ]);
      const y = Int32Array.from([spec.lossOn === 'all' ? b : -1, spec.lossOn === 'all' ? EQ : -1, (a + b) % p]);
      ex.push({ x, y });
    }
  return {
    vocab,
    T: 3,
    ...split(ex, spec.trainFrac, rng),
    about: `modular addition: a b = → (a + b) mod ${p}`,
    answerStart: 2,
    fmt: (ids) => Array.from(ids, (i) => vocab[i] ?? '?').join(' '),
  };
}

function text(spec: TaskSpec, rng: RNG): Dataset {
  const corpus = spec.text.length >= 2 ? spec.text : 'ab';
  const chars = [...new Set(corpus.split(''))].sort();
  const idx = new Map(chars.map((c, i) => [c, i]));
  const ids = Int32Array.from(corpus.split(''), (c) => idx.get(c)!);
  const T = Math.max(1, Math.min(spec.ctx, ids.length - 1));
  const ex: Example[] = [];
  for (let s = 0; s + T < ids.length; s++) {
    ex.push({ x: ids.slice(s, s + T), y: ids.slice(s + 1, s + T + 1) });
  }
  const shown = chars.map((c) => (c === '\n' ? '⏎' : c === ' ' ? '␣' : c));
  return {
    vocab: shown,
    T,
    ...split(ex, spec.trainFrac, rng),
    about: `character-level language model: ${chars.length} characters, ${ids.length} tokens of text`,
    answerStart: 0,
    fmt: (a) => Array.from(a, (i) => shown[i] ?? '?').join(''),
  };
}

export function buildDataset(spec: TaskSpec): Dataset {
  const rng = new RNG(spec.seed);
  switch (spec.kind) {
    case 'sort':
      return seqTask(spec, rng, (s) => [...s].sort((a, b) => a - b), 'sort');
    case 'reverse':
      return seqTask(spec, rng, (s) => [...s].reverse(), 'reverse');
    case 'copy':
      return seqTask(spec, rng, (s) => [...s], 'copy');
    case 'modadd':
      return modadd(spec, rng);
    case 'text':
      return text(spec, rng);
  }
}

/** Pack examples into flat [B*T] token and target arrays. */
export function packBatch(examples: Example[], T: number): { x: Int32Array; y: Int32Array; B: number } {
  const B = examples.length;
  const x = new Int32Array(B * T), y = new Int32Array(B * T);
  examples.forEach((e, b) => {
    x.set(e.x, b * T);
    y.set(e.y, b * T);
  });
  return { x, y, B };
}
