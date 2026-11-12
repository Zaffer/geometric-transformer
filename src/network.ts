// The network itself, for the focused position: every unit a circle, every parameter exactly
// one line. Click a line to select that parameter; scroll over it to nudge it.

import { type Block, type Config, type Layout, type Trace, at, locate } from './model';
import { type Mode, PALETTE } from './scene';

export interface NetHandlers {
  select(param: number): void;
  nudge(param: number, by: number): void;
  neuron(h: number): void;
  token(w: number): void;
  position(p: number): void;
}

const NS = 'http://www.w3.org/2000/svg';
const W = 440, HEIGHT = 560, MID = 240;
const Y = { in: 34, x: 96, qkv: 160, att: 218, z: 270, r1: 330, hid: 398, r2: 464, out: 532 };
const WARM = [214, 96, 77], COOL = [67, 147, 195];

const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  parent?.appendChild(e);
  return e;
};
const row = (n: number, mid: number, gap: number) => Array.from({ length: n }, (_, i) => mid + (i - (n - 1) / 2) * gap);
const signed = (v: number, alpha = 1) => {
  const c = v >= 0 ? WARM : COOL;
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
};
/** White at zero, warm when positive, cool when negative. */
const shade = (v: number) => {
  const u = Math.tanh(Math.abs(v) / 1.5), c = v >= 0 ? WARM : COOL;
  return `rgb(${c.map((x) => Math.round(255 + u * (x - 255))).join(',')})`;
};

interface Node { x: number; y: number; circle: SVGCircleElement }
interface Synapse { param: number; line: SVGLineElement; hit: SVGLineElement }

export class Network {
  private syn: Synapse[] = [];
  private n: Record<string, Node[]> = {};
  private attBars: SVGRectElement[] = [];
  private attLabels: SVGTextElement[] = [];
  private outLabels: SVGTextElement[] = [];

  constructor(svg: SVGSVGElement, private c: Config, private L: Layout, private on: NetHandlers) {
    svg.setAttribute('viewBox', `0 0 ${W} ${HEIGHT}`);
    const B = L.blocks;
    const back = el('g', {}, svg), wires = el('g', {}, svg), front = el('g', {}, svg);

    const nodes = (name: string, xs: number[], y: number, r = 9) =>
      (this.n[name] = xs.map((x) => ({ x, y, circle: el('circle', { cx: x, cy: y, r, class: 'node', fill: '#fff' }, front) })));
    const caption = (text: string, y: number, x = 8, anchor = 'start') => {
      const t = el('text', { x, y: y + 4, class: 'cap', 'text-anchor': anchor }, back);
      t.textContent = text;
      return t;
    };

    nodes('tok', row(c.V, 150, 27), Y.in);
    nodes('pos', row(c.T, 335, 27), Y.in);
    nodes('x', row(2, MID, 44), Y.x);
    nodes('q', row(2, 130, 34), Y.qkv);
    nodes('k', row(2, MID, 34), Y.qkv);
    nodes('v', row(2, 350, 34), Y.qkv);
    nodes('z', row(2, 300, 40), Y.z);
    nodes('r1', row(2, MID, 44), Y.r1);
    nodes('hid', row(c.H, MID, Math.min(48, 220 / c.H)), Y.hid, 11);
    nodes('r2', row(2, MID, 44), Y.r2);
    nodes('out', row(c.V, MID, 44), Y.out, 12);
    nodes('one1', [404], Y.hid - 34, 8);
    nodes('oneU', [404], Y.out - 34, 8);

    caption('token', Y.in - 20, 150, 'middle'); caption('position', Y.in - 20, 335, 'middle');
    caption('point', Y.x); caption('query', Y.qkv - 22, 96, 'end'); caption('key', Y.qkv + 22, MID, 'middle');
    caption('value', Y.qkv - 22, 384, 'start'); caption('attention', Y.att); caption('+ attended', Y.r1);
    caption('neurons', Y.hid); caption('+ MLP', Y.r2); caption('next token', Y.out);
    for (const o of [this.n.one1[0], this.n.oneU[0]]) {
      const t = el('text', { x: o.x, y: o.y + 4, class: 'glyph' }, front);
      t.textContent = '1';
    }
    this.n.tok.forEach((nd, w) => this.glyph(front, nd, String(w)));
    this.n.pos.forEach((nd, p) => this.glyph(front, nd, String(p)));
    this.outLabels = this.n.out.map((nd, w) => this.glyph(front, nd, String(w)));

    // one line per parameter
    const connect = (b: Block, from: Node[], to: Node[], index: (f: number, t: number) => number) => {
      from.forEach((a, f) => to.forEach((z, t) => {
        const xy = { x1: a.x, y1: a.y, x2: z.x, y2: z.y };
        const param = index(f, t);
        const line = el('line', { ...xy, class: 'syn' }, wires);
        const hit = el('line', { ...xy, class: 'hit' }, wires);
        hit.addEventListener('click', () => on.select(param));
        hit.addEventListener('wheel', (e) => { e.preventDefault(); on.nudge(param, e.deltaY < 0 ? 0.05 : -0.05); }, { passive: false });
        this.syn.push({ param, line, hit });
      }));
    };
    connect(B.E, this.n.tok, this.n.x, (w, d) => at(B.E, w, d));
    connect(B.P, this.n.pos, this.n.x, (p, d) => at(B.P, p, d));
    connect(B.WQ, this.n.x, this.n.q, (d, r) => at(B.WQ, r, d));
    connect(B.WK, this.n.x, this.n.k, (d, r) => at(B.WK, r, d));
    connect(B.WV, this.n.x, this.n.v, (d, r) => at(B.WV, r, d));
    connect(B.WO, this.n.z, this.n.r1, (d, r) => at(B.WO, r, d));
    connect(B.W1, this.n.r1, this.n.hid, (d, h) => at(B.W1, h, d));
    connect(B.b1, this.n.one1, this.n.hid, (_, h) => at(B.b1, h));
    connect(B.W2, this.n.hid, this.n.r2, (h, r) => at(B.W2, r, h));
    connect(B.U, this.n.r2, this.n.out, (d, w) => at(B.U, w, d));
    connect(B.bU, this.n.oneU, this.n.out, (_, w) => at(B.bU, w));

    // what is computed rather than learned: the attention weights, the average, the two additions
    const flow = (d: string) => el('path', { d, class: 'flow' }, back);
    const attX = row(c.T, 185, 26);
    for (const q of this.n.q) flow(`M${q.x} ${q.y} L${185 - 30} ${Y.att - 14}`);
    for (const k of this.n.k) flow(`M${k.x} ${k.y} L${185 + 30} ${Y.att - 14}`);
    el('rect', { x: attX[0] - 16, y: Y.att - 14, width: attX[c.T - 1] - attX[0] + 32, height: 30, rx: 5, class: 'box' }, back);
    for (const z of this.n.z) {
      flow(`M${attX[c.T - 1] + 16} ${Y.att + 4} L${z.x} ${z.y}`);
      for (const v of this.n.v) flow(`M${v.x} ${v.y} L${z.x} ${z.y}`);
    }
    attX.forEach((x) => {
      this.attBars.push(el('rect', { x: x - 9, y: Y.att - 10, width: 18, height: 22, rx: 3 }, front));
      const t = el('text', { x, y: Y.att + 5, class: 'glyph' }, front);
      this.attLabels.push(t);
    });
    const skip = (a: Node[], b: Node[], side: number) => {
      const x = MID - side;
      flow(`M${a[0].x} ${a[0].y} C${x} ${a[0].y + 20} ${x} ${b[0].y - 20} ${b[0].x} ${b[0].y}`);
    };
    skip(this.n.x, this.n.r1, 175);
    skip(this.n.r1, this.n.r2, 160);

    this.n.hid.forEach((nd, h) => nd.circle.addEventListener('click', () => on.neuron(h)));
    this.n.tok.forEach((nd, w) => nd.circle.addEventListener('click', () => on.token(w)));
    this.n.pos.forEach((nd, p) => nd.circle.addEventListener('click', () => on.position(p)));
    for (const nd of [...this.n.hid, ...this.n.tok, ...this.n.pos]) nd.circle.classList.add('pick');
  }

  private glyph(parent: Element, nd: Node, text: string) {
    const t = el('text', { x: nd.x, y: nd.y + 4, class: 'glyph' }, parent);
    t.textContent = text;
    return t;
  }

  update(th: Float64Array, t: Trace, tokens: number[], pos: number, target: number, selected: number | null, neuron: number, mode: Mode) {
    const { c } = this;
    for (const s of this.syn) {
      const w = th[s.param], m = Math.tanh(Math.abs(w) / 1.5);
      s.line.setAttribute('stroke', signed(w, 0.25 + 0.75 * m));
      s.line.setAttribute('stroke-width', (0.5 + 3.2 * m).toFixed(2));
      s.hit.classList.toggle('sel', s.param === selected);
    }
    const fill = (name: string, vals: ArrayLike<number>, off: number) =>
      this.n[name].forEach((nd, i) => nd.circle.setAttribute('fill', shade(vals[off + i])));
    fill('x', t.x, 2 * pos); fill('q', t.q, 2 * pos); fill('k', t.k, 2 * pos); fill('v', t.v, 2 * pos);
    fill('z', t.z, 2 * pos); fill('r1', t.r1, 2 * pos); fill('r2', t.r2, 2 * pos);
    this.n.hid.forEach((nd, h) => {
      const pre = t.pre[pos * c.H + h];
      nd.circle.setAttribute('fill', pre > 0 ? shade(pre) : '#e4e1d8');
      nd.circle.classList.toggle('sel', mode === 'mlp' && h === neuron);
    });
    this.n.tok.forEach((nd, w) => nd.circle.setAttribute('fill', w === tokens[pos] ? PALETTE[w % PALETTE.length] : '#fff'));
    this.n.pos.forEach((nd, p) => nd.circle.setAttribute('fill', p === pos ? '#c9c4b6' : '#fff'));
    this.n.out.forEach((nd, w) => {
      const p = t.p[pos * c.V + w];
      nd.circle.setAttribute('fill', PALETTE[w % PALETTE.length]);
      nd.circle.setAttribute('fill-opacity', (0.08 + 0.92 * p).toFixed(3));
      nd.circle.classList.toggle('sel', w === target);
      this.outLabels[w].textContent = String(w);
    });
    this.attBars.forEach((bar, j) => {
      const on = j <= pos;
      bar.setAttribute('visibility', on ? 'visible' : 'hidden');
      this.attLabels[j].setAttribute('visibility', on ? 'visible' : 'hidden');
      if (!on) return;
      const a = t.a[pos * t.n + j];
      bar.setAttribute('fill', `rgba(28,27,25,${(0.06 + 0.8 * a).toFixed(3)})`);
      this.attLabels[j].textContent = String(tokens[j]);
      this.attLabels[j].setAttribute('fill', a > 0.55 ? '#fff' : '#1c1b19');
    });
  }
}

/** What a parameter is, in words, and which picture shows it best. */
export function describe(L: Layout, i: number): { name: string; what: string; mode: Mode; neuron?: number; flow?: number } {
  const { block: b, r, c } = locate(L, i), ax = 'xy'[c], row = 'xy'[r];
  switch (b.name) {
    case 'E': return { name: `E[token ${r}].${ax}`, mode: 'output', what: `Where token ${r} sits on the plane, along ${ax}. Moves its dot.` };
    case 'P': return { name: `P[position ${r}].${ax}`, mode: 'attention', what: `The shift along ${ax} given to whatever token is at position ${r}. Moves every dot at that position together.` };
    case 'WQ': return { name: `W_Q[${r},${c}]`, mode: 'attention', what: 'Query: turns the focused point into what it is looking for. Tilts every score sheet at once.' };
    case 'WK': return { name: `W_K[${r},${c}]`, mode: 'attention', what: 'Key: turns each earlier point into what it offers. Tilts the score sheet of every earlier point.' };
    case 'WV': return { name: `W_V[${r},${c}]`, mode: 'attention', flow: 1, what: 'Value: what an attended point hands over. Changes where attention moves the sheet (flow = 1).' };
    case 'WO': return { name: `W_O[${r},${c}]`, mode: 'attention', flow: 1, what: `Output: how the handed-over value moves the point along ${row}. Changes where attention moves the sheet (flow = 1).` };
    case 'W1': return { name: `W1[neuron ${r}].${ax}`, mode: 'mlp', neuron: r, what: `Tilt of neuron ${r}'s sheet along ${ax}. Turns its fold line.` };
    case 'b1': return { name: `b1[neuron ${r}]`, mode: 'mlp', neuron: r, what: `Neuron ${r}'s cutoff: raises or lowers its sheet through zero, sweeping its fold line across the plane.` };
    case 'W2': return { name: `W2[${row}, neuron ${c}]`, mode: 'mlp', neuron: c, flow: 2, what: `How far neuron ${c} pushes points on its on-side along ${row}. Sets how sharply the sheet folds (flow = 2).` };
    case 'U': return { name: `U[token ${r}].${ax}`, mode: 'output', what: `Tilt of token ${r}'s logit sheet along ${ax}. Moves the borders of its cell.` };
    case 'bU': return { name: `bU[token ${r}]`, mode: 'output', what: `Height of token ${r}'s logit sheet. Grows or shrinks its cell.` };
  }
}
