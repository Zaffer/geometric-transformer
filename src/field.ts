// The landscape: the model's output at one position, sampled over a square of the plane.
// Every grid point is "what if this position's point were here?" with the earlier
// positions held fixed.

import { type Config, type Layout, type Trace, probe } from './model';

export interface Field {
  G: number; // grid points per side
  cx: number; cy: number; R: number; // the square: centre and half-width
  pos: number;
  nKeys: number;
  r1: Float32Array; // G*G*2  where the point is after attention
  r2: Float32Array; // G*G*2  where it is after the MLP
  score: Float32Array; // G*G*nKeys  attention score of each key
  pre: Float32Array; // G*G*H  each neuron's pre-activation
  logit: Float32Array; // G*G*V
  topKey: Uint8Array; // which key gets the most attention
  topTok: Uint8Array; // which token is predicted
  pattern: Uint32Array; // bitmask of which neurons are on: the polytope id
}

export function sample(
  c: Config, L: Layout, th: Float64Array, ctx: Trace, pos: number,
  G: number, cx: number, cy: number, R: number, reuse?: Field,
): Field {
  const N = G * G, nKeys = pos + 1;
  const f: Field = reuse && reuse.G === G && reuse.nKeys === nKeys ? reuse : {
    G, cx, cy, R, pos, nKeys,
    r1: new Float32Array(N * 2), r2: new Float32Array(N * 2),
    score: new Float32Array(N * nKeys), pre: new Float32Array(N * c.H), logit: new Float32Array(N * c.V),
    topKey: new Uint8Array(N), topTok: new Uint8Array(N), pattern: new Uint32Array(N),
  };
  f.cx = cx; f.cy = cy; f.R = R; f.pos = pos;
  const p = probe(c, L, th, ctx, pos);
  for (let j = 0, n = 0; j < G; j++) {
    const y = cy + R * ((2 * j) / (G - 1) - 1);
    for (let i = 0; i < G; i++, n++) {
      const o = p.eval(cx + R * ((2 * i) / (G - 1) - 1), y);
      f.r1[2 * n] = o.r1[0]; f.r1[2 * n + 1] = o.r1[1];
      f.r2[2 * n] = o.r2[0]; f.r2[2 * n + 1] = o.r2[1];
      let best = 0;
      for (let k = 0; k < nKeys; k++) {
        f.score[n * nKeys + k] = o.s[k];
        if (o.s[k] > o.s[best]) best = k;
      }
      f.topKey[n] = best;
      let mask = 0;
      for (let h = 0; h < c.H; h++) {
        f.pre[n * c.H + h] = o.pre[h];
        if (o.pre[h] > 0) mask |= 1 << h;
      }
      f.pattern[n] = mask;
      best = 0;
      for (let w = 0; w < c.V; w++) {
        f.logit[n * c.V + w] = o.logits[w];
        if (o.logits[w] > o.logits[best]) best = w;
      }
      f.topTok[n] = best;
    }
  }
  return f;
}
