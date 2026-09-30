// The transformer. Complete but minimal:
//
//   tokens → token embedding (+ position embedding) = residual stream
//   N × [ norm → multi-head causal self-attention → add,
//         norm → MLP (Linear → activation → Linear) → add ]
//   → final norm → unembedding → softmax → next-token distribution
//
// Every parameter is a named Tensor. Every intermediate of a forward pass
// is kept (by name) in a Trace, so every view can read any activation.

import {
  activate,
  add,
  addBias,
  attnMix,
  attnScores,
  concatCols,
  crossEntropy,
  gather,
  matmul,
  matmulBT,
  norm,
  sliceCols,
  softmaxRows,
} from './ops';
import { RNG } from './rng';
import type { ModelSpec, InitSpec } from './spec';
import { Tensor, withGrad } from './tensor';

export type ParamKind = 'embed' | 'weight' | 'bias' | 'gain';

export interface Param {
  name: string;
  t: Tensor;
  kind: ParamKind;
  /** the architecture-map node this belongs to, e.g. 'L0.attn' */
  node: string;
  layer: number;
  /** learning-rate multiplier (0 = frozen) */
  lrMult: number;
  frozen: boolean;
  /** per-element freeze: 1 = this scalar never moves */
  mask: Uint8Array | null;
  about: string;
  rowLabel: string;
  colLabel: string;
  /** init std for this tensor under the current init spec */
  initStd: number;
  /** output projection (GPT-2 residual scaling applies) */
  residOut?: boolean;
}

export interface Stage {
  /** trace key of the residual stream at this point */
  id: string;
  label: string;
  layer: number;
}

export type Trace = Map<string, Tensor>;

export interface ForwardResult {
  logits: Tensor;
  trace: Trace;
  B: number;
  T: number;
}

function hashName(s: string, seed: number): number {
  let h = 2166136261 ^ seed;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

export class Model {
  readonly spec: ModelSpec;
  readonly vocab: number;
  readonly ctx: number;
  readonly params: Param[] = [];
  readonly byName = new Map<string, Param>();
  /** fixed sinusoidal position table (not a parameter) */
  readonly sinPos: Tensor | null = null;

  constructor(spec: ModelSpec, vocab: number, ctx: number, init: InitSpec) {
    this.spec = spec;
    this.vocab = vocab;
    this.ctx = ctx;
    const D = spec.dModel, H = spec.nHead, dh = spec.dHead, F = spec.dMlp, V = vocab;
    const hasNorm = spec.norm !== 'none';

    this.add('wte', V, D, 'embed', 'embed', -1, 'token embedding: one row per token, a point in the residual space', 'token', 'dim');
    if (spec.pos === 'learned') {
      this.add('wpe', ctx, D, 'embed', 'pos', -1, 'position embedding: one row per position, added to the token row', 'position', 'dim');
    } else if (spec.pos === 'sinusoidal') {
      const pe = new Tensor(ctx, D);
      for (let t = 0; t < ctx; t++)
        for (let j = 0; j < D; j++) {
          const f = Math.pow(10000, -(2 * Math.floor(j / 2)) / D);
          pe.data[t * D + j] = j % 2 === 0 ? Math.sin(t * f) : Math.cos(t * f);
        }
      this.sinPos = pe;
    }

    const normParams = (node: string, layer: number, what: string) => {
      if (!hasNorm) return;
      if (spec.normGain) this.add(`${node}.g`, 1, D, 'gain', node, layer, `${what} gain: stretches each normalized axis`, '', 'dim');
      if (spec.normBias) this.add(`${node}.b`, 1, D, 'bias', node, layer, `${what} bias: shifts the normalized point`, '', 'dim');
    };

    for (let l = 0; l < spec.nLayer; l++) {
      const L = `L${l}`;
      const tog = spec.layers[l];
      if (tog.attn) {
        normParams(`${L}.ln1`, l, 'pre-attention norm');
        const a = `${L}.attn`;
        for (const [w, what] of [
          ['q', 'query'],
          ['k', 'key'],
          ['v', 'value'],
        ] as const) {
          this.add(`${a}.W${w}`, D, H * dh, 'weight', a, l, `${what} projection: residual → ${what} space, all heads side by side`, 'dim', 'head·dim');
          if (spec.biasQKV) this.add(`${a}.b${w}`, 1, H * dh, 'bias', a, l, `${what} bias`, '', 'head·dim');
        }
        this.add(`${a}.Wo`, H * dh, D, 'weight', a, l, 'output projection: concatenated head outputs → residual', 'head·dim', 'dim').residOut = true;
        if (spec.biasAttnOut) this.add(`${a}.bo`, 1, D, 'bias', a, l, 'attention output bias', '', 'dim');
      }
      if (tog.mlp) {
        normParams(`${L}.ln2`, l, 'pre-MLP norm');
        const m = `${L}.mlp`;
        this.add(`${m}.W1`, D, F, 'weight', m, l, 'MLP input weights: column j is neuron j\'s normal vector', 'dim', 'neuron');
        if (spec.biasMlp) this.add(`${m}.b1`, 1, F, 'bias', m, l, 'MLP neuron biases: offset of each neuron\'s hyperplane', '', 'neuron');
        this.add(`${m}.W2`, F, D, 'weight', m, l, 'MLP output weights: row j is what neuron j writes to the residual', 'neuron', 'dim').residOut = true;
        if (spec.biasMlp) this.add(`${m}.b2`, 1, D, 'bias', m, l, 'MLP output bias', '', 'dim');
      }
    }
    if (spec.finalNorm) normParams('lnf', -1, 'final norm');
    if (!spec.tie) this.add('Wu', D, V, 'weight', 'unembed', -1, 'unembedding: column k is token k\'s readout direction', 'dim', 'token');
    if (spec.biasUnembed) this.add('bu', 1, V, 'bias', 'unembed', -1, 'unembedding bias: per-token logit offset', '', 'token');

    this.init(init);
  }

  private add(
    name: string,
    rows: number,
    cols: number,
    kind: ParamKind,
    node: string,
    layer: number,
    about: string,
    rowLabel: string,
    colLabel: string,
  ): Param {
    const t = new Tensor(rows, cols, undefined, true);
    t.name = name;
    const p: Param = { name, t, kind, node, layer, lrMult: 1, frozen: false, mask: null, about, rowLabel, colLabel, initStd: 0 };
    this.params.push(p);
    this.byName.set(name, p);
    return p;
  }

  p(name: string): Tensor {
    const p = this.byName.get(name);
    if (!p) throw new Error(`no param ${name}`);
    return p.t;
  }

  has(name: string): boolean {
    return this.byName.has(name);
  }

  /** init std for one param under an init spec */
  initStd(p: Param, init: InitSpec): number {
    if (p.kind === 'embed') return init.embedStd;
    if (p.kind !== 'weight') return 0;
    const fin = p.t.rows, fout = p.t.cols;
    let s: number;
    switch (init.scheme) {
      case 'normal':
        s = init.std;
        break;
      case 'xavier':
      case 'uniform':
        s = Math.sqrt(2 / (fin + fout));
        break;
      case 'kaiming':
        s = Math.sqrt(2 / fin);
        break;
      case 'lecun':
        s = Math.sqrt(1 / fin);
        break;
    }
    s *= init.gain;
    if (init.residScale && p.residOut && this.spec.nLayer > 0) s /= Math.sqrt(2 * this.spec.nLayer);
    return s;
  }

  /** (Re)initialize one parameter. Each tensor has its own RNG stream, so
   *  toggling one part of the model never changes another part's init. */
  initParam(p: Param, init: InitSpec): void {
    const std = this.initStd(p, init);
    p.initStd = std;
    const d = p.t.data;
    if (p.kind === 'gain') {
      d.fill(1);
      return;
    }
    if (p.kind === 'bias') {
      d.fill(0);
      return;
    }
    const rng = new RNG(hashName(p.name, init.seed));
    if (init.scheme === 'uniform' && p.kind === 'weight') {
      const a = std * Math.sqrt(3);
      for (let i = 0; i < d.length; i++) d[i] = (2 * rng.next() - 1) * a;
    } else {
      for (let i = 0; i < d.length; i++) d[i] = rng.normal() * std;
    }
  }

  init(init: InitSpec): void {
    for (const p of this.params) this.initParam(p, init);
  }

  numParams(): number {
    return this.params.reduce((s, p) => s + p.t.size, 0);
  }

  /** all parameters as one flat vector (parameter order) */
  flat(): Float64Array {
    const out = new Float64Array(this.numParams());
    let o = 0;
    for (const p of this.params) {
      out.set(p.t.data, o);
      o += p.t.size;
    }
    return out;
  }

  setFlat(v: ArrayLike<number>): void {
    let o = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.t.size; i++) p.t.data[i] = v[o + i];
      o += p.t.size;
    }
  }

  /** flat gradient (after backward) */
  flatGrad(): Float64Array {
    const out = new Float64Array(this.numParams());
    let o = 0;
    for (const p of this.params) {
      if (p.t.grad) out.set(p.t.grad, o);
      o += p.t.size;
    }
    return out;
  }

  /** offset of each param in the flat vector */
  offsets(): Map<string, number> {
    const m = new Map<string, number>();
    let o = 0;
    for (const p of this.params) {
      m.set(p.name, o);
      o += p.t.size;
    }
    return m;
  }

  /** Copy values (and per-param settings) from another model wherever a
   *  param of the same name and shape exists. Returns the number copied. */
  adoptFrom(other: Model): number {
    let n = 0;
    for (const p of this.params) {
      const q = other.byName.get(p.name);
      if (q && q.t.rows === p.t.rows && q.t.cols === p.t.cols) {
        p.t.data.set(q.t.data);
        p.lrMult = q.lrMult;
        p.frozen = q.frozen;
        p.mask = q.mask ? q.mask.slice() : null;
        n++;
      }
    }
    return n;
  }

  zeroGrad(): void {
    for (const p of this.params) p.t.zeroGrad();
  }

  /** The residual-stream checkpoints, in order, for the geometry views. */
  stages(): Stage[] {
    const s = this.spec;
    const out: Stage[] = [{ id: 'resid.embed', label: 'embed', layer: -1 }];
    for (let l = 0; l < s.nLayer; l++) {
      if (s.layers[l].attn) out.push({ id: `L${l}.resid.mid`, label: `L${l} +attn`, layer: l });
      if (s.layers[l].mlp) out.push({ id: `L${l}.resid.out`, label: `L${l} +mlp`, layer: l });
    }
    if (s.finalNorm && s.norm !== 'none') out.push({ id: 'final', label: 'final norm', layer: s.nLayer });
    return out;
  }

  private normOp(x: Tensor, node: string): Tensor {
    const s = this.spec;
    if (s.norm === 'none') return x;
    const g = this.byName.get(`${node}.g`)?.t ?? null;
    const b = this.byName.get(`${node}.b`)?.t ?? null;
    return norm(x, s.norm, g, b, s.normEps);
  }

  private lin(x: Tensor, W: string, b: string): Tensor {
    const y = matmul(x, this.p(W));
    const bp = this.byName.get(b);
    return bp ? addBias(y, bp.t) : y;
  }

  /**
   * Forward pass over B sequences packed as [B*T] token ids.
   * With trace on, every intermediate is kept by name.
   */
  forward(ids: Int32Array, B: number, trace = false): ForwardResult {
    const s = this.spec;
    const T = ids.length / B;
    if (T > this.ctx) throw new Error(`sequence length ${T} > context ${this.ctx}`);
    const tr: Trace = new Map();
    const keep = (k: string, t: Tensor) => {
      if (trace) tr.set(k, t);
      return t;
    };

    let h = keep('embed.tok', gather(this.p('wte'), ids));
    if (s.pos !== 'none') {
      const pos = new Int32Array(B * T);
      for (let i = 0; i < pos.length; i++) pos[i] = i % T;
      const pe = s.pos === 'learned' ? gather(this.p('wpe'), pos) : gather(this.sinPos!, pos);
      keep('embed.pos', pe);
      h = add(h, pe);
    }
    keep('resid.embed', h);

    const pre = s.normPlacement === 'pre';
    const dh = s.dHead, H = s.nHead;
    const scale = s.attnScaleMult / Math.sqrt(dh);

    for (let l = 0; l < s.nLayer; l++) {
      const L = `L${l}`;
      if (s.layers[l].attn) {
        const a = `${L}.attn`;
        const inp = keep(`${L}.ln1`, pre ? this.normOp(h, `${L}.ln1`) : h);
        const q = keep(`${a}.q`, this.lin(inp, `${a}.Wq`, `${a}.bq`));
        const k = keep(`${a}.k`, this.lin(inp, `${a}.Wk`, `${a}.bk`));
        const v = keep(`${a}.v`, this.lin(inp, `${a}.Wv`, `${a}.bv`));
        const zs: Tensor[] = [];
        for (let hh = 0; hh < H; hh++) {
          const qh = H === 1 ? q : sliceCols(q, hh * dh, dh);
          const kh = H === 1 ? k : sliceCols(k, hh * dh, dh);
          const vh = H === 1 ? v : sliceCols(v, hh * dh, dh);
          const sc = keep(`${a}.scores.h${hh}`, attnScores(qh, kh, B, T, scale, s.causal));
          const pr = keep(`${a}.probs.h${hh}`, softmaxRows(sc));
          zs.push(keep(`${a}.z.h${hh}`, attnMix(pr, vh, B, T)));
        }
        const z = keep(`${a}.z`, H === 1 ? zs[0] : concatCols(zs));
        const out = keep(`${a}.out`, this.lin(z, `${a}.Wo`, `${a}.bo`));
        h = add(h, out);
        if (!pre) h = this.normOp(h, `${L}.ln1`);
        keep(`${L}.resid.mid`, h);
      }
      if (s.layers[l].mlp) {
        const m = `${L}.mlp`;
        const inp = keep(`${L}.ln2`, pre ? this.normOp(h, `${L}.ln2`) : h);
        const pa = keep(`${m}.pre`, this.lin(inp, `${m}.W1`, `${m}.b1`));
        const ac = keep(`${m}.act`, activate(pa, s.activation, s.leakySlope));
        const out = keep(`${m}.out`, this.lin(ac, `${m}.W2`, `${m}.b2`));
        h = add(h, out);
        if (!pre) h = this.normOp(h, `${L}.ln2`);
        keep(`${L}.resid.out`, h);
      }
    }
    if (s.finalNorm && s.norm !== 'none') h = keep('final', this.normOp(h, 'lnf'));
    keep('unembed.in', h);
    let logits = s.tie ? matmulBT(h, this.p('wte')) : matmul(h, this.p('Wu'));
    const bu = this.byName.get('bu');
    if (bu) logits = addBias(logits, bu.t);
    keep('logits', logits);
    return { logits, trace: tr, B, T };
  }

  /** Mean next-token loss; with grad on, call .backward() on the result. */
  loss(ids: Int32Array, targets: Int32Array, B: number): { loss: Tensor; probs: Float64Array } {
    const { logits } = this.forward(ids, B);
    const probs: { value?: Float64Array } = {};
    const loss = crossEntropy(logits, targets, probs);
    return { loss, probs: probs.value! };
  }

  /** Readout for a single residual vector: logits = unembed(finalNorm(x)).
   *  This is the "logit lens" when applied to an intermediate stage. */
  readout(x: Float64Array, applyFinalNorm = true): Float64Array {
    return withGrad(false, () => {
      const D = this.spec.dModel;
      let h = new Tensor(x.length / D, D, Float64Array.from(x));
      if (applyFinalNorm && this.spec.finalNorm && this.spec.norm !== 'none') h = this.normOp(h, 'lnf');
      let logits = this.spec.tie ? matmulBT(h, this.p('wte')) : matmul(h, this.p('Wu'));
      const bu = this.byName.get('bu');
      if (bu) logits = addBias(logits, bu.t);
      return logits.data;
    });
  }

  /** Apply one named norm to rows of x (for views). */
  applyNorm(x: Float64Array, node: string): Float64Array {
    return withGrad(false, () => this.normOp(new Tensor(x.length / this.spec.dModel, this.spec.dModel, Float64Array.from(x)), node).data);
  }
}
