// The network as neurons and synapses, unrolled over token positions.
//
// Every unit of the forward pass is a sphere: token / position one-hots, the
// residual stream, norms, q / k / v, attention heads, the MLP's hidden
// neurons, the residual adds and the output probabilities. Every scalar
// parameter is one synapse: each weight, each bias (from a constant-1 bias
// node) and each norm gain (the one-to-one edges into a norm).
//
// Each token position is its own sheet of the same network, stacked in
// depth. The weights are shared by every sheet, so synapses are drawn once,
// on the focused position. Attention is the only thing that links sheets:
// its links are synapses whose strength p is computed from the data on the fly.

import * as THREE from 'three';
import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Session } from '../app/session';
import { fullGrad, probe } from '../app/inspect';
import { rt, ui } from '../app/state';
import { catColor, css, diverging, sequential, token, type RGB } from '../ui/palette';
import { hideTip, showTip } from '../ui/heatmap';
import { fmtNum } from '../ui/controls';
import { nodeTitle } from '../ui/inspector';
import { Stage3D, colorOf, disposeTree } from './stage3d';
import type { View } from './view';
import type { Trace } from '../engine/model';

const XSTEP = 1.25;
const YSTEP = 2.4;
const UNIT_R = 0.14;
const EDGE_R = 0.03;
const MAX_COL_H = 5;

type ColKind = 'onehot' | 'value' | 'prob' | 'bias' | 'head';

interface Ctx {
  tr: Trace;
  tokens: Int32Array;
  T: number;
  probs: Float64Array;
}

interface Col {
  id: string;
  label: string;
  /** architecture-map node (for focus) */
  node: string;
  layer: number;
  kind: ColKind;
  n: number;
  x: number;
  zc: number;
  sp: number;
  /** [T × n] values for the probe sequence */
  read: (c: Ctx) => Float64Array;
  unit: (i: number) => string;
  /** drawn on the focused sheet only (bias nodes) */
  focusOnly?: boolean;
  /** k / v: kept on every sheet even in "focused position" mode, so attention links land */
  kv?: boolean;
  /** extra tooltip line */
  extra?: (c: Ctx, t: number, i: number) => string;
}

interface Syn {
  param: string;
  kind: 'w' | 'b' | 'g';
  from: Col;
  fi: number;
  to: Col;
  tj: number;
  index: number;
}

interface Link {
  head: number;
  key: number;
  p: number;
  from: Col;
  fi: number;
  to: Col;
  tj: number;
}

type EdgeMode = 'weight' | 'grad' | 'flow';

export class NetworkView implements View {
  readonly id = 'network';
  readonly title = 'network';
  readonly el = document.createElement('div');
  private session: Session;
  private stage: Stage3D;
  private root = new THREE.Group();
  private cols: Col[] = [];
  private syns: Syn[] = [];
  /** identity paths (no parameter); skip = the residual stream bypassing a sublayer */
  private struct: [Col, number, Col, number, boolean][] = [];
  private attn: { layer: number; q: Col; k: Col; v: Col; heads: Col; z: Col }[] = [];
  private sheets = 1;
  // units: instance k ↔ (column, sheet, index)
  private uCol = new Int32Array(0);
  private uT = new Int32Array(0);
  private uI = new Int32Array(0);
  private uPos = new Float32Array(0);
  private uShown = new Uint8Array(0);
  private units: THREE.InstancedMesh | null = null;
  private synMesh: THREE.InstancedMesh | null = null;
  private synA = new Float32Array(0);
  private synB = new Float32Array(0);
  private synShown = new Uint8Array(0);
  private synVal = new Float64Array(0);
  private linkMesh: THREE.InstancedMesh | null = null;
  private links: Link[] = [];
  private linkA = new Float32Array(0);
  private linkB = new Float32Array(0);
  private structLines: THREE.LineSegments | null = null;
  private selMesh: THREE.Mesh | null = null;
  private sheetFrame: THREE.LineLoop | null = null;
  private vals = new Map<Col, Float64Array>();
  private ctx: Ctx | null = null;
  private focusT = 0;
  private labelKey = '';
  private sel: { param: string; index: number } | null = null;
  private hover: { kind: 'unit' | 'syn' | 'link'; k: number } | null = null;
  private opts = {
    mode: 'weight' as EdgeMode,
    positions: 'all' as 'all' | 'focus',
    scale: 'tensor' as 'tensor' | 'global',
    threshold: 0.05,
    thickness: 1,
    biases: true,
    skips: true,
    labels: true,
    zoom: 'all',
  };
  private active = false;
  private needBuild = true;
  private needRefresh = true;
  private fitted = false;
  private gradT = 0;
  private gradTimer = 0;
  private hud!: HTMLDivElement;
  private editor!: HTMLDivElement;
  private caption!: HTMLDivElement;
  private posLabel!: HTMLSpanElement;
  private zoomSel!: HTMLSelectElement;
  private down = { x: 0, y: 0 };

  constructor(session: Session) {
    this.session = session;
    this.stage = new Stage3D(this.el);
    // a long, flat network: a narrower lens keeps the far sheets from shrinking away
    this.stage.camera.fov = 28;
    this.stage.scene.add(this.root);
    this.buildHud();
    const cv = this.stage.renderer.domElement;
    cv.addEventListener('pointerdown', (e) => (this.down = { x: e.clientX, y: e.clientY }));
    cv.addEventListener('pointermove', (e) => this.onHover(e));
    cv.addEventListener('pointerleave', () => {
      this.hover = null;
      hideTip();
    });
    cv.addEventListener('click', (e) => {
      if (Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) < 5) this.onClick(e);
    });
    // shift + wheel over a synapse nudges it (captured before the orbit zoom sees it)
    this.el.addEventListener('wheel', (e) => this.onWheel(e), { capture: true, passive: false });

    effect(() => {
      rt.modelVersion();
      this.needBuild = true;
      if (untracked(() => this.active)) untracked(() => this.build());
    });
    effect(() => {
      rt.paramsVersion();
      ui.probe();
      ui.probeTokens();
      ui.probePos();
      ui.neuron();
      ui.focus();
      this.needRefresh = true;
      if (untracked(() => this.active)) untracked(() => this.refresh());
    });
  }

  activate(): void {
    this.active = true;
    this.stage.start();
    if (this.needBuild) this.build();
    else if (this.needRefresh) this.refresh();
    if (!this.fitted) {
      this.fitted = true;
      this.fitZoom();
    }
  }

  deactivate(): void {
    this.active = false;
    this.stage.stop();
    hideTip();
  }

  themeChanged(): void {
    this.stage.themeChanged();
    this.labelKey = '';
    this.needRefresh = true;
    if (this.active) this.refresh();
  }

  // ---------------------------------------------------------------- HUD

  private buildHud() {
    const hud = document.createElement('div');
    hud.className = 'hud';
    Object.assign(hud.style, { left: '8px', top: '8px', maxWidth: '250px' });
    this.hud = hud;
    const body = document.createElement('div');
    const fold = document.createElement('button');
    fold.textContent = 'controls ▾';
    fold.title = 'show / hide the controls';
    fold.onclick = () => {
      body.hidden = !body.hidden;
      fold.textContent = body.hidden ? 'controls ▸' : 'controls ▾';
    };
    hud.append(fold, body);

    const sel = <K extends 'mode' | 'positions' | 'scale' | 'zoom'>(label: string, key: K, options: [string, string][], title: string, after?: () => void) => {
      const l = document.createElement('label');
      l.title = title;
      l.append(label);
      const s = document.createElement('select');
      for (const [v, t] of options) s.append(new Option(t, v));
      s.value = String(this.opts[key]);
      s.dataset.role = key;
      s.onchange = () => {
        (this.opts as Record<string, unknown>)[key] = s.value;
        if (after) after();
        else this.refresh();
      };
      l.append(s);
      body.append(l);
      return s;
    };
    sel('synapses', 'mode', [
      ['weight', 'weight'],
      ['grad', '−gradient'],
      ['flow', 'flow  x·w'],
    ], 'weight: the parameter value. −gradient: where full-batch training pushes it. flow: what it carries for the focused position right now (input activation × weight).');
    sel('sheets', 'positions', [
      ['all', 'every position'],
      ['focus', 'focused position'],
    ], 'every position is its own copy of the network; the weights are shared');

    const pos = document.createElement('div');
    Object.assign(pos.style, { display: 'flex', gap: '4px', alignItems: 'center', margin: '2px 0' });
    const prev = document.createElement('button');
    prev.textContent = '◀';
    prev.title = 'focus the previous position';
    prev.onclick = () => ui.probePos(Math.max(0, this.focusT - 1));
    const next = document.createElement('button');
    next.textContent = '▶';
    next.title = 'focus the next position';
    next.onclick = () => ui.probePos(Math.min(this.sheets - 1, this.focusT + 1));
    this.posLabel = document.createElement('span');
    pos.append('position', prev, next, this.posLabel);
    body.append(pos);

    const range = (label: string, key: 'threshold' | 'thickness', min: number, max: number, step: number, title: string) => {
      const l = document.createElement('label');
      l.title = title;
      l.append(label);
      const r = document.createElement('input');
      r.type = 'range';
      r.min = String(min);
      r.max = String(max);
      r.step = String(step);
      r.value = String(this.opts[key]);
      r.oninput = () => {
        this.opts[key] = Number(r.value);
        this.refresh();
      };
      l.append(r);
      body.append(l);
    };
    range('hide below', 'threshold', 0, 0.9, 0.01, 'hide synapses weaker than this fraction of the strongest (per tensor, or in the whole network)');
    range('thickness', 'thickness', 0.3, 3, 0.1, 'synapse thickness');
    sel('relative to', 'scale', [
      ['tensor', 'its tensor'],
      ['global', 'whole network'],
    ], 'synapse strength is scaled to the strongest synapse in its own tensor, or in the whole network');

    const tog = (label: string, key: 'biases' | 'skips' | 'labels', title: string) => {
      const l = document.createElement('label');
      l.title = title;
      const c = document.createElement('input');
      c.type = 'checkbox';
      c.checked = this.opts[key];
      c.onchange = () => {
        this.opts[key] = c.checked;
        this.labelKey = '';
        this.refresh();
      };
      l.append(c, label);
      body.append(l);
    };
    tog('biases and norm gains', 'biases', 'bias synapses (from a constant-1 node) and the one-to-one norm gains');
    tog('residual skips', 'skips', 'the identity paths: every add receives the stream unchanged plus the sublayer output');
    tog('labels', 'labels', 'column, token and position labels');

    const zr = document.createElement('div');
    Object.assign(zr.style, { display: 'flex', gap: '4px', alignItems: 'center', marginTop: '2px' });
    this.zoomSel = document.createElement('select');
    this.zoomSel.title = 'point the camera at part of the network';
    this.zoomSel.onchange = () => {
      this.opts.zoom = this.zoomSel.value;
      this.fitZoom();
    };
    const fit = document.createElement('button');
    fit.textContent = 'fit';
    fit.onclick = () => this.fitZoom();
    zr.append('view', this.zoomSel, fit);
    body.append(zr);
    this.stage.overlay.append(hud);

    this.editor = document.createElement('div');
    this.editor.className = 'hud';
    Object.assign(this.editor.style, { right: '8px', top: '8px', width: '250px' });
    this.editor.hidden = true;
    this.stage.overlay.append(this.editor);

    this.caption = document.createElement('div');
    this.caption.className = 'caption';
    this.stage.overlay.append(this.caption);
  }

  // ---------------------------------------------------------------- structure

  private build() {
    this.needBuild = false;
    const m = this.session.model, s = m.spec;
    const D = s.dModel, H = s.nHead, dh = s.dHead, HD = H * dh, F = s.dMlp, V = m.vocab, T = m.ctx;
    const vocab = this.session.data.vocab;
    const hasNorm = s.norm !== 'none';
    const pre = s.normPlacement === 'pre';
    const cols: Col[] = [];
    const syns: Syn[] = [];
    const struct: [Col, number, Col, number, boolean][] = [];
    this.attn = [];
    const spOf = (n: number) => Math.min(0.42, MAX_COL_H / Math.max(1, n));
    const dimL = (i: number) => `d${i}`;
    const hdL = (i: number) => `h${Math.floor(i / dh)}.${i % dh}`;
    const tr = (key: string) => (c: Ctx) => c.tr.get(key)!.data;
    let slot = 0;
    const col = (o: Omit<Col, 'x' | 'sp' | 'zc'> & { zc?: number; slot?: number }): Col => {
      const c: Col = { ...o, x: (o.slot ?? slot) * XSTEP, sp: spOf(o.n), zc: o.zc ?? 0 };
      cols.push(c);
      return c;
    };
    const heightOf = (n: number) => (n - 1) * spOf(n);
    const has = (name: string) => m.has(name);
    const dense = (param: string, from: Col, to: Col, index: (i: number, j: number) => number) => {
      if (!has(param)) return;
      for (let i = 0; i < from.n; i++) for (let j = 0; j < to.n; j++) syns.push({ param, kind: 'w', from, fi: i, to, tj: j, index: index(i, j) });
    };
    const bias = (param: string, to: Col, node: string, layer: number) => {
      if (!has(param)) return;
      const b = col({
        id: `${param}.node`, label: '1', node, layer, kind: 'bias', n: 1, focusOnly: true,
        read: (c) => new Float64Array(c.T).fill(1), unit: () => 'bias input (constant 1)',
        zc: to.zc - heightOf(to.n) / 2 - 0.45, slot: to.x / XSTEP - 0.45,
      });
      for (let j = 0; j < to.n; j++) syns.push({ param, kind: 'b', from: b, fi: 0, to, tj: j, index: j });
    };
    const oneToOne = (from: Col, to: Col, skip = false) => {
      for (let i = 0; i < Math.min(from.n, to.n); i++) struct.push([from, i, to, i, skip]);
    };
    // a norm column: gains are one-to-one synapses (the shared ÷rms is not a weight)
    const normCol = (id: string, label: string, node: string, layer: number, from: Col, read: Col['read']): Col => {
      slot++;
      const c = col({ id, label, node, layer, kind: 'value', n: D, read, unit: dimL });
      if (has(`${node}.g`)) for (let i = 0; i < D; i++) syns.push({ param: `${node}.g`, kind: 'g', from, fi: i, to: c, tj: i, index: i });
      else oneToOne(from, c);
      bias(`${node}.b`, c, node, layer);
      return c;
    };

    // ---- embedding ----
    const posOn = s.pos !== 'none';
    const tokH = heightOf(V), posN = s.pos === 'learned' ? T : D, posH = heightOf(posN);
    const tok = col({
      id: 'tok', label: 'token', node: 'tokens', layer: -1, kind: 'onehot', n: V,
      zc: posOn ? tokH / 2 + 0.5 : 0,
      read: (c) => {
        const o = new Float64Array(c.T * V);
        for (let t = 0; t < c.T; t++) o[t * V + c.tokens[t]] = 1;
        return o;
      },
      unit: (i) => `token ${vocab[i] ?? i}`,
    });
    let pos: Col | null = null;
    if (s.pos === 'learned') {
      pos = col({
        id: 'pos', label: 'position', node: 'pos', layer: -1, kind: 'onehot', n: T, zc: -(posH / 2 + 0.5),
        read: (c) => {
          const o = new Float64Array(c.T * T);
          for (let t = 0; t < c.T; t++) o[t * T + t] = 1;
          return o;
        },
        unit: (i) => `position ${i}`,
      });
    } else if (s.pos === 'sinusoidal') {
      pos = col({
        id: 'pos', label: 'sin pos', node: 'pos', layer: -1, kind: 'value', n: D, zc: -(posH / 2 + 0.5),
        read: (c) => m.sinPos!.data.slice(0, c.T * D), unit: dimL,
      });
    }
    slot++;
    const embed = col({ id: 'resid.embed', label: 'embed', node: 'embed', layer: -1, kind: 'value', n: D, read: tr('resid.embed'), unit: dimL });
    dense('wte', tok, embed, (i, j) => i * D + j);
    if (s.pos === 'learned' && pos) dense('wpe', pos, embed, (i, j) => i * D + j);
    else if (pos) oneToOne(pos, embed);
    let cur = embed;

    // a residual add (pre-norm), or add then norm (post-norm)
    const addCol = (L: string, l: number, sub: string, key: string, normNode: string, out: Col): Col => {
      slot++;
      if (pre || !hasNorm) {
        const a = col({ id: key, label: '+', node: `${L}.${sub}`, layer: l, kind: 'value', n: D, read: tr(key), unit: dimL });
        oneToOne(cur, a, true);
        oneToOne(out, a);
        return a;
      }
      const prev = cur;
      const sum = col({
        id: `${key}.sum`, label: '+', node: `${L}.${sub}`, layer: l, kind: 'value', n: D, unit: dimL,
        read: (c) => {
          const a = this.vals.get(prev) ?? prev.read(c), b = this.vals.get(out) ?? out.read(c);
          return a.map((v, i) => v + b[i]);
        },
      });
      oneToOne(prev, sum, true);
      oneToOne(out, sum);
      return normCol(key, s.norm === 'rmsnorm' ? 'rms' : 'ln', normNode, l, sum, tr(key));
    };

    for (let l = 0; l < s.nLayer; l++) {
      const L = `L${l}`;
      if (s.layers[l].attn) {
        const a = `${L}.attn`;
        let inp = cur;
        if (pre && hasNorm) inp = normCol(`${L}.ln1`, s.norm === 'rmsnorm' ? 'rms' : 'ln', `${L}.ln1`, l, cur, tr(`${L}.ln1`));
        slot++;
        const hh = heightOf(HD), gap = 0.9;
        const q = col({ id: `${a}.q`, label: 'q', node: a, layer: l, kind: 'value', n: HD, zc: hh + gap, read: tr(`${a}.q`), unit: hdL });
        const k = col({ id: `${a}.k`, label: 'k', node: a, layer: l, kind: 'value', n: HD, zc: 0, read: tr(`${a}.k`), unit: hdL, kv: true });
        const v = col({ id: `${a}.v`, label: 'v', node: a, layer: l, kind: 'value', n: HD, zc: -(hh + gap), read: tr(`${a}.v`), unit: hdL, kv: true });
        for (const [w, c] of [['q', q], ['k', k], ['v', v]] as const) {
          dense(`${a}.W${w}`, inp, c, (i, j) => i * HD + j);
          bias(`${a}.b${w}`, c, a, l);
        }
        slot++;
        const heads = col({
          id: `${a}.heads`, label: 'heads', node: a, layer: l, kind: 'head', n: H, zc: hh / 2 + gap / 2,
          read: (c) => {
            const o = new Float64Array(c.T * H);
            for (let h = 0; h < H; h++) {
              const P = c.tr.get(`${a}.probs.h${h}`)!.data;
              for (let t = 0; t < c.T; t++) {
                let mx = 0;
                for (let j = 0; j < c.T; j++) mx = Math.max(mx, P[t * c.T + j]);
                o[t * H + h] = mx;
              }
            }
            return o;
          },
          unit: (h) => `head ${h}`,
          extra: (c, t, h) => {
            const P = c.tr.get(`${a}.probs.h${h}`)!.data;
            let best = 0;
            for (let j = 0; j < c.T; j++) if (P[t * c.T + j] > P[t * c.T + best]) best = j;
            return `attends most to p${best}:${vocab[c.tokens[best]]} (p = ${fmtNum(P[t * c.T + best])})\nshown value: its largest attention weight`;
          },
        });
        for (let h = 0; h < H; h++) for (let d = 0; d < dh; d++) struct.push([q, h * dh + d, heads, h, false]);
        slot++;
        const z = col({ id: `${a}.z`, label: 'Σ p·v', node: a, layer: l, kind: 'value', n: HD, read: tr(`${a}.z`), unit: hdL });
        slot++;
        const out = col({ id: `${a}.out`, label: 'out', node: a, layer: l, kind: 'value', n: D, read: tr(`${a}.out`), unit: dimL });
        dense(`${a}.Wo`, z, out, (i, j) => i * D + j);
        bias(`${a}.bo`, out, a, l);
        this.attn.push({ layer: l, q, k, v, heads, z });
        cur = addCol(L, l, 'attn', `${L}.resid.mid`, `${L}.ln1`, out);
      }
      if (s.layers[l].mlp) {
        const mm = `${L}.mlp`;
        let inp = cur;
        if (pre && hasNorm) inp = normCol(`${L}.ln2`, s.norm === 'rmsnorm' ? 'rms' : 'ln', `${L}.ln2`, l, cur, tr(`${L}.ln2`));
        slot++;
        const hid = col({
          id: `${mm}.act`, label: s.activation, node: mm, layer: l, kind: 'value', n: F, read: tr(`${mm}.act`), unit: (i) => `n${i}`,
          extra: (c, t, i) => `pre-activation ${fmtNum(c.tr.get(`${mm}.pre`)!.data[t * F + i])}`,
        });
        dense(`${mm}.W1`, inp, hid, (i, j) => i * F + j);
        bias(`${mm}.b1`, hid, mm, l);
        slot++;
        const out = col({ id: `${mm}.out`, label: 'out', node: mm, layer: l, kind: 'value', n: D, read: tr(`${mm}.out`), unit: dimL });
        dense(`${mm}.W2`, hid, out, (i, j) => i * D + j);
        bias(`${mm}.b2`, out, mm, l);
        cur = addCol(L, l, 'mlp', `${L}.resid.out`, `${L}.ln2`, out);
      }
    }

    // ---- readout ----
    if (s.finalNorm && hasNorm) cur = normCol('final', s.norm === 'rmsnorm' ? 'rms' : 'ln', 'lnf', s.nLayer, cur, tr('final'));
    slot++;
    const logits = col({
      id: 'logits', label: 'p(next)', node: 'unembed', layer: s.nLayer, kind: 'prob', n: V, read: (c) => c.probs,
      unit: (i) => `token ${vocab[i] ?? i}`,
      extra: (c, t, i) => `logit ${fmtNum(c.tr.get('logits')!.data[t * V + i])}`,
    });
    if (s.tie) dense('wte', cur, logits, (i, j) => j * D + i);
    else dense('Wu', cur, logits, (i, j) => i * V + j);
    bias('bu', logits, 'unembed', s.nLayer);

    this.cols = cols;
    this.syns = syns;
    this.struct = struct;
    this.sheets = T;
    if (this.sel && !(m.byName.get(this.sel.param) && this.sel.index < m.byName.get(this.sel.param)!.t.size)) this.select(null);

    // ---- meshes ----
    for (const c of [...this.root.children]) disposeTree(c);
    let nu = 0;
    for (const c of cols) nu += c.n * T;
    this.uCol = new Int32Array(nu);
    this.uT = new Int32Array(nu);
    this.uI = new Int32Array(nu);
    this.uPos = new Float32Array(nu * 3);
    this.uShown = new Uint8Array(nu);
    let k = 0;
    cols.forEach((c, ci) => {
      for (let t = 0; t < T; t++)
        for (let i = 0; i < c.n; i++) {
          this.uCol[k] = ci;
          this.uT[k] = t;
          this.uI[k] = i;
          const p = this.unitPos(c, i, t);
          this.uPos.set([p.x, p.y, p.z], k * 3);
          k++;
        }
    });
    this.units = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), new THREE.MeshLambertMaterial(), nu);
    this.synMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 6, 1, true), new THREE.MeshBasicMaterial(), Math.max(1, syns.length));
    this.synA = new Float32Array(syns.length * 3);
    this.synB = new Float32Array(syns.length * 3);
    this.synShown = new Uint8Array(syns.length);
    this.synVal = new Float64Array(syns.length);
    let nLinks = 0;
    for (const a of this.attn) nLinks += a.k.n * T + a.v.n * T;
    this.linkMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 6, 1, true), new THREE.MeshBasicMaterial(), Math.max(1, nLinks));
    for (const mesh of [this.units, this.synMesh, this.linkMesh]) {
      mesh.frustumCulled = false;
      for (let i = 0; i < mesh.count; i++) mesh.setColorAt(i, new THREE.Color());
    }
    this.structLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ transparent: true, opacity: 0.55 }));
    this.selMesh = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 8, 1, true), new THREE.MeshBasicMaterial());
    this.selMesh.visible = false;
    this.sheetFrame = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ transparent: true, opacity: 0.5 }));
    this.root.add(this.units, this.synMesh, this.linkMesh, this.structLines, this.selMesh, this.sheetFrame);

    // zoom targets
    const prevZoom = this.opts.zoom;
    this.zoomSel.replaceChildren(new Option('whole network', 'all'), new Option('embedding', 'embed'));
    for (let l = 0; l < s.nLayer; l++) this.zoomSel.append(new Option(`layer ${l}`, `L${l}`));
    this.zoomSel.append(new Option('readout', 'readout'));
    this.zoomSel.value = [...this.zoomSel.options].some((o) => o.value === prevZoom) ? prevZoom : 'all';
    this.opts.zoom = this.zoomSel.value;
    this.labelKey = '';
    this.refresh(true);
  }

  private unitPos(c: Col, i: number, t: number): THREE.Vector3 {
    return new THREE.Vector3(c.x, -t * YSTEP, c.zc + ((c.n - 1) / 2 - i) * c.sp);
  }

  // ---------------------------------------------------------------- data

  private refresh(force = false) {
    if (this.needBuild) return this.build();
    this.needRefresh = false;
    const m = this.session.model;
    if (!this.units || !this.synMesh || !this.linkMesh) return;

    // the full-batch gradient is costly: throttle it while training
    let grads: Map<string, Float64Array> | null = null;
    if (this.opts.mode === 'grad') {
      const now = performance.now();
      if (!force && rt.running() && now - this.gradT < 300) {
        clearTimeout(this.gradTimer);
        this.gradTimer = window.setTimeout(() => this.active && this.refresh(), 300);
        return;
      }
      this.gradT = now;
      grads = fullGrad();
    }

    const pr = probe();
    const Tp = pr.tokens.length;
    const ctx: Ctx = { tr: pr.fwd.trace, tokens: pr.tokens, T: Tp, probs: pr.probs };
    this.ctx = ctx;
    this.vals.clear();
    for (const c of this.cols) this.vals.set(c, c.read(ctx));
    const f = ui.probePos() >= 0 && ui.probePos() < Tp ? ui.probePos() : Tp - 1;
    this.focusT = f;
    const vocab = this.session.data.vocab;
    this.posLabel.textContent = `p${f}: ${vocab[pr.tokens[f]] ?? '?'}`;

    const bg = token('panel');
    const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    const tmp = new THREE.Color();
    const setColor = (mesh: THREE.InstancedMesh, k: number, c: RGB) => mesh.setColorAt(k, tmp.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace));
    const m4 = new THREE.Matrix4();

    // ---- units ----
    const colMax = new Map<Col, number>();
    for (const c of this.cols) {
      const v = this.vals.get(c)!;
      let mx = 0;
      for (let i = 0; i < v.length; i++) if (Number.isFinite(v[i])) mx = Math.max(mx, Math.abs(v[i]));
      colMax.set(c, mx || 1);
    }
    const focusNode = ui.focus();
    const neuronL = focusNode.match(/^L(\d+)\.mlp$/);
    const nSel = ui.neuron();
    for (let k = 0; k < this.uCol.length; k++) {
      const c = this.cols[this.uCol[k]], t = this.uT[k], i = this.uI[k];
      const onFocus = t === f;
      let shown = t < Tp && (onFocus || (!c.focusOnly && (this.opts.positions === 'all' || c.kv)));
      if (c.kind === 'bias' && !this.opts.biases) shown = false;
      this.uShown[k] = shown ? 1 : 0;
      if (!shown) {
        m4.makeScale(0, 0, 0);
        this.units.setMatrixAt(k, m4);
        continue;
      }
      const v = this.vals.get(c)![t * c.n + i];
      let col: RGB;
      if (c.kind === 'onehot') col = v > 0 ? (c.id === 'tok' ? catColor(i) : token('accent')) : mix(bg, token('muted'), 0.35);
      else if (c.kind === 'prob') col = sequential(v);
      else if (c.kind === 'bias') col = token('muted');
      else if (c.kind === 'head') col = mix(token('zero'), token('sel'), v);
      else col = diverging(v / colMax.get(c)!);
      let r = UNIT_R * (c.kind === 'bias' ? 0.7 : c.kind === 'head' ? 1.5 : 1);
      if (!onFocus) {
        col = mix(bg, col, 0.55);
        r *= 0.65;
      }
      if (onFocus && neuronL && c.id === `${focusNode}.act` && i === nSel) r *= 1.8;
      setColor(this.units, k, col);
      m4.makeScale(r, r, r).setPosition(this.uPos[k * 3], this.uPos[k * 3 + 1], this.uPos[k * 3 + 2]);
      this.units.setMatrixAt(k, m4);
    }
    this.units.instanceMatrix.needsUpdate = true;
    this.units.instanceColor!.needsUpdate = true;

    // ---- synapses (on the focused sheet) ----
    const tensorMax = new Map<string, number>();
    const fy = -f * YSTEP;
    for (let e = 0; e < this.syns.length; e++) {
      const sy = this.syns[e];
      const p = m.byName.get(sy.param)!;
      const w = p.t.data[sy.index];
      let v: number;
      if (this.opts.mode === 'weight') v = w;
      else if (this.opts.mode === 'grad') v = -(grads!.get(sy.param)?.[sy.index] ?? 0);
      else if (sy.kind === 'w') v = this.vals.get(sy.from)![f * sy.from.n + sy.fi] * w;
      else if (sy.kind === 'b') v = w;
      else {
        // gain: what it writes is the norm's output (minus the norm bias)
        const b = m.byName.get(sy.param.replace(/\.g$/, '.b'))?.t.data[sy.index] ?? 0;
        v = this.vals.get(sy.to)![f * sy.to.n + sy.tj] - b;
      }
      this.synVal[e] = v;
      const key = this.opts.scale === 'global' ? '' : sy.param;
      if (Number.isFinite(v)) tensorMax.set(key, Math.max(tensorMax.get(key) ?? 0, Math.abs(v)));
    }
    const pa = new THREE.Vector3(), pb = new THREE.Vector3();
    for (let e = 0; e < this.syns.length; e++) {
      const sy = this.syns[e];
      const a = this.unitPos(sy.from, sy.fi, f), b = this.unitPos(sy.to, sy.tj, f);
      this.synA.set([a.x, a.y, a.z], e * 3);
      this.synB.set([b.x, b.y, b.z], e * 3);
      const v = this.synVal[e];
      const mx = tensorMax.get(this.opts.scale === 'global' ? '' : sy.param) || 1;
      const t = Math.min(1, Math.abs(v) / mx);
      const shown = Number.isFinite(v) && t >= this.opts.threshold && t > 1e-6 && (sy.kind === 'w' || this.opts.biases);
      this.synShown[e] = shown ? 1 : 0;
      setSeg(this.synMesh, e, a, b, shown ? EDGE_R * this.opts.thickness * (0.2 + 0.8 * Math.sqrt(t)) : 0);
      if (shown) setColor(this.synMesh, e, mix(bg, diverging(Math.sign(v)), 0.2 + 0.8 * t));
    }
    this.synMesh.instanceMatrix.needsUpdate = true;
    this.synMesh.instanceColor!.needsUpdate = true;

    // ---- attention links: synapses computed from the data (focused query only) ----
    const links: Link[] = [];
    for (const at of this.attn) {
      const s = m.spec, dh = s.dHead;
      for (let h = 0; h < s.nHead; h++) {
        const P = ctx.tr.get(`L${at.layer}.attn.probs.h${h}`)!.data;
        for (let j = 0; j < Tp; j++) {
          const p = P[f * Tp + j];
          if (p < 1e-3) continue;
          for (let d = 0; d < dh; d++) {
            links.push({ head: h, key: j, p, from: at.k, fi: h * dh + d, to: at.heads, tj: h });
            links.push({ head: h, key: j, p, from: at.v, fi: h * dh + d, to: at.z, tj: h * dh + d });
          }
        }
      }
    }
    this.links = links.slice(0, this.linkMesh.count);
    this.linkA = new Float32Array(this.links.length * 3);
    this.linkB = new Float32Array(this.links.length * 3);
    const sel = token('sel');
    for (let e = 0; e < this.linkMesh.count; e++) {
      const L = this.links[e];
      if (!L) {
        m4.makeScale(0, 0, 0);
        this.linkMesh.setMatrixAt(e, m4);
        continue;
      }
      pa.copy(this.unitPos(L.from, L.fi, L.key));
      pb.copy(this.unitPos(L.to, L.tj, f));
      this.linkA.set([pa.x, pa.y, pa.z], e * 3);
      this.linkB.set([pb.x, pb.y, pb.z], e * 3);
      setSeg(this.linkMesh, e, pa, pb, EDGE_R * this.opts.thickness * (0.2 + 0.8 * Math.sqrt(L.p)));
      setColor(this.linkMesh, e, mix(bg, sel, 0.15 + 0.85 * L.p));
    }
    this.linkMesh.instanceMatrix.needsUpdate = true;
    this.linkMesh.instanceColor!.needsUpdate = true;

    // ---- identity paths on the focused sheet ----
    const pts: number[] = [];
    for (const [a, i, b, j, skip] of this.struct) {
      if (skip && !this.opts.skips) continue;
      const A = this.unitPos(a, i, f), B = this.unitPos(b, j, f);
      pts.push(A.x, A.y, A.z, B.x, B.y, B.z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    this.structLines!.geometry.dispose();
    this.structLines!.geometry = g;
    (this.structLines!.material as THREE.LineBasicMaterial).color = colorOf(token('muted'));

    // ---- focused-sheet frame ----
    const box = this.sheetBox();
    const fr = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(box.min.x, fy, box.min.z),
      new THREE.Vector3(box.max.x, fy, box.min.z),
      new THREE.Vector3(box.max.x, fy, box.max.z),
      new THREE.Vector3(box.min.x, fy, box.max.z),
    ]);
    this.sheetFrame!.geometry.dispose();
    this.sheetFrame!.geometry = fr;
    (this.sheetFrame!.material as THREE.LineBasicMaterial).color = colorOf(token('sel'));

    this.updateSelection();
    this.updateLabels(Tp);
    this.updateCaption();
    this.stage.invalidate();
  }

  /** bounding box of one sheet, padded */
  private sheetBox(): THREE.Box3 {
    const box = new THREE.Box3();
    for (const c of this.cols) {
      box.expandByPoint(this.unitPos(c, 0, 0));
      box.expandByPoint(this.unitPos(c, c.n - 1, 0));
    }
    box.min.x -= 0.6;
    box.max.x += 0.6;
    box.min.z -= 0.5;
    box.max.z += 0.9;
    return box;
  }

  private updateLabels(Tp: number) {
    const key = `${this.focusT}:${Tp}:${this.opts.labels}:${this.opts.positions}:${rt.modelVersion()}:${ui.focus()}`;
    if (key === this.labelKey) return;
    this.labelKey = key;
    this.stage.clearLabels();
    if (!this.opts.labels) return;
    const f = this.focusT, vocab = this.session.data.vocab, toks = this.ctx!.tokens;
    const focus = ui.focus();
    const accent = css(token('accent'));
    for (const c of this.cols) {
      if (c.kind === 'bias' || c.label === '+') continue;
      // the position column hangs under the token column: its label goes underneath
      const at = c.id === 'pos' ? this.unitPos(c, c.n - 1, f).add(new THREE.Vector3(0, 0, -0.4)) : this.unitPos(c, 0, f).add(new THREE.Vector3(0, 0, 0.35));
      this.stage.label(c.label, at, c.node === focus ? accent : undefined);
      if ((c.id === 'tok' || c.id === 'logits') && c.n <= 16)
        for (let i = 0; i < c.n; i++) {
          const p = this.unitPos(c, i, f).add(new THREE.Vector3(c.id === 'tok' ? -0.35 : 0.35, 0, 0));
          this.stage.label(vocab[i] ?? String(i), p, css(catColor(i)));
        }
    }
    // layer titles
    const box = this.sheetBox();
    for (let l = 0; l < this.session.model.spec.nLayer; l++) {
      const cs = this.cols.filter((c) => c.layer === l && c.kind !== 'bias');
      if (!cs.length) continue;
      const x = (Math.min(...cs.map((c) => c.x)) + Math.max(...cs.map((c) => c.x))) / 2;
      this.stage.label(`layer ${l}`, new THREE.Vector3(x, -f * YSTEP, box.max.z + 0.5), css(token('muted')));
    }
    // sheet titles
    for (let t = 0; t < Tp; t++) {
      if (this.opts.positions === 'focus' && t !== f) continue;
      const p = new THREE.Vector3(box.min.x - 1.8, -t * YSTEP, 0);
      this.stage.label(`p${t}: ${vocab[toks[t]] ?? '?'}`, p, t === f ? css(token('sel')) : css(token('muted')));
    }
  }

  private updateCaption() {
    const mode = this.opts.mode;
    const what = mode === 'weight'
      ? 'Synapses: every weight, bias and gain. Red +, blue −; brighter and thicker = larger |w|.'
      : mode === 'grad'
        ? 'Synapses: −gradient of the full-batch training loss: the direction training pushes each weight. Red: grow, blue: shrink.'
        : `Synapses: flow at position ${this.focusT}: input activation × weight, what each synapse adds to its target right now. Only the current token's embedding row carries flow.`;
    this.caption.textContent = `${what} Spheres: activations for the probe sequence. Yellow links: attention, synapses computed from the data (p), from the earlier positions' k and v to the focused query. Click a synapse to edit it; shift+wheel nudges it.`;
  }

  // ---------------------------------------------------------------- selection + editing

  private select(s: { param: string; index: number } | null) {
    this.sel = s;
    ui.cell(s ? { name: s.param, index: s.index } : null);
    if (s) {
      const p = this.session.model.byName.get(s.param);
      if (p && ui.focus() !== p.node) ui.focus(p.node);
    }
    this.renderEditor();
    this.updateSelection();
    this.stage.invalidate();
  }

  private updateSelection() {
    if (!this.selMesh) return;
    const s = this.sel;
    const e = s ? this.syns.findIndex((y) => y.param === s.param && y.index === s.index) : -1;
    if (e < 0) {
      this.selMesh.visible = false;
      return;
    }
    const sy = this.syns[e];
    setSeg(this.selMesh, -1, this.unitPos(sy.from, sy.fi, this.focusT), this.unitPos(sy.to, sy.tj, this.focusT), EDGE_R * this.opts.thickness * 1.6);
    (this.selMesh.material as THREE.MeshBasicMaterial).color = colorOf(token('sel'));
    this.selMesh.visible = true;
    this.updateEditorValues();
  }

  private editorInputs: { num: HTMLInputElement; range: HTMLInputElement; info: HTMLDivElement; freeze: HTMLButtonElement } | null = null;

  private renderEditor() {
    const s = this.sel;
    this.editor.replaceChildren();
    this.editorInputs = null;
    if (!s) {
      this.editor.hidden = true;
      return;
    }
    const m = this.session.model;
    const p = m.byName.get(s.param)!;
    const sy = this.syns.find((y) => y.param === s.param && y.index === s.index);
    this.editor.hidden = false;
    const h = document.createElement('div');
    h.style.fontWeight = 'bold';
    const r = Math.floor(s.index / p.t.cols), c = s.index % p.t.cols;
    h.textContent = p.t.rows === 1 ? `${p.name}[${c}]` : `${p.name}[${r}, ${c}]`;
    const sub = document.createElement('div');
    sub.className = 'muted';
    sub.style.whiteSpace = 'pre-wrap';
    sub.textContent = `${nodeTitle(p.node)}${sy ? `\n${sy.from.label}·${sy.from.unit(sy.fi)} → ${sy.to.label}·${sy.to.unit(sy.tj)}` : ''}`;
    let mx = 0;
    for (const v of p.t.data) mx = Math.max(mx, Math.abs(v));
    const R = Math.max(1, 1.5 * mx);
    const num = document.createElement('input');
    num.type = 'number';
    num.step = 'any';
    num.style.width = '90px';
    num.onchange = () => this.session.edit(s.param, s.index, Number(num.value));
    const range = document.createElement('input');
    range.type = 'range';
    range.min = String(-R);
    range.max = String(R);
    range.step = String(R / 500);
    range.style.width = '100%';
    range.oninput = () => this.session.edit(s.param, s.index, Number(range.value));
    const row = document.createElement('div');
    Object.assign(row.style, { display: 'flex', gap: '4px', alignItems: 'center', flexWrap: 'wrap', marginTop: '3px' });
    const btn = (label: string, title: string, f: () => void) => {
      const b = document.createElement('button');
      b.textContent = label;
      b.title = title;
      b.onclick = f;
      row.append(b);
      return b;
    };
    btn('0', 'set to zero', () => this.session.edit(s.param, s.index, 0));
    btn('−', 'negate', () => this.session.edit(s.param, s.index, -p.t.data[s.index]));
    const freeze = btn('freeze', 'freeze / unfreeze this one scalar (training will not move it)', () => {
      const mask = p.mask ? p.mask.slice() : new Uint8Array(p.t.size);
      mask[s.index] = mask[s.index] ? 0 : 1;
      this.session.setMeta(p.name, { mask: mask.some((x) => x) ? mask : null });
      this.updateEditorValues();
    });
    for (const [axis, label] of [[0, '→x'], [1, '→y']] as const)
      btn(label, `make this weight landscape axis ${label.slice(1)}`, () => {
        ui.landscapeAxis({ axis, name: s.param, index: s.index, seq: Math.random() });
        ui.tab('landscape');
      });
    btn('×', 'deselect', () => this.select(null));
    const info = document.createElement('div');
    info.className = 'muted';
    info.style.whiteSpace = 'pre-wrap';
    info.style.marginTop = '3px';
    const vrow = document.createElement('div');
    Object.assign(vrow.style, { display: 'flex', gap: '4px', alignItems: 'center', marginTop: '3px' });
    vrow.append('value', num);
    this.editor.append(h, sub, vrow, range, row, info);
    this.editorInputs = { num, range, info, freeze };
    this.updateEditorValues();
  }

  private updateEditorValues() {
    const s = this.sel, E = this.editorInputs;
    if (!s || !E) return;
    const p = this.session.model.byName.get(s.param);
    if (!p) return;
    const w = p.t.data[s.index];
    if (document.activeElement !== E.num) E.num.value = String(Number(w.toPrecision(6)));
    if (document.activeElement !== E.range) E.range.value = String(w);
    const frozen = p.frozen || !!p.mask?.[s.index];
    E.freeze.classList.toggle('on', !!p.mask?.[s.index]);
    const e = this.syns.findIndex((y) => y.param === s.param && y.index === s.index);
    const lines: string[] = [];
    if (e >= 0 && this.opts.mode !== 'weight') lines.push(`${this.opts.mode === 'grad' ? '−gradient' : `flow at p${this.focusT}`}: ${fmtNum(this.synVal[e])}`);
    lines.push(`lr × ${p.lrMult}${frozen ? ' · frozen' : ''}`);
    E.info.textContent = lines.join('\n');
    // the full-batch gradient costs a backward pass: only once edits settle, and not while training
    clearTimeout(this.gradLineTimer);
    if (rt.running()) return;
    this.gradLineTimer = window.setTimeout(() => {
      if (this.sel !== s || this.editorInputs !== E) return;
      E.info.textContent = `${lines.join('\n')}\n−gradient (full batch): ${fmtNum(-(fullGrad().get(s.param)?.[s.index] ?? 0))}`;
    }, 250);
  }

  private gradLineTimer = 0;

  // ---------------------------------------------------------------- picking

  private pick(e: MouseEvent): { kind: 'unit' | 'syn' | 'link'; k: number } | null {
    const rect = this.stage.renderer.domElement.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const W = rect.width, H = rect.height;
    const cam = this.stage.camera;
    const v = new THREE.Vector3();
    const proj = (arr: Float32Array, k: number): [number, number, number] => {
      v.set(arr[k * 3], arr[k * 3 + 1], arr[k * 3 + 2]).project(cam);
      return [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H, v.z];
    };
    // pixels per world unit at a point (for the sphere radius)
    const scale = (arr: Float32Array, k: number): number => {
      const d = cam.position.distanceTo(v.set(arr[k * 3], arr[k * 3 + 1], arr[k * 3 + 2]));
      return H / (2 * d * Math.tan((cam.fov * Math.PI) / 360));
    };
    // units under the cursor: the front-most one wins (sheets overlap on screen)
    let best: { kind: 'unit' | 'syn' | 'link'; k: number } | null = null, bestD = Infinity;
    for (let k = 0; k < this.uShown.length; k++) {
      if (!this.uShown[k]) continue;
      const [x, y, z] = proj(this.uPos, k);
      if (z > 1) continue;
      const d = Math.hypot(x - mx, y - my);
      const r = Math.max(5, UNIT_R * 1.3 * scale(this.uPos, k));
      if (d < r && z < bestD) {
        bestD = z;
        best = { kind: 'unit', k };
      }
    }
    if (best) return best;
    const seg = (A: Float32Array, B: Float32Array, k: number): number => {
      const [ax, ay, az] = proj(A, k);
      const [bx, by, bz] = proj(B, k);
      if (az > 1 || bz > 1) return Infinity;
      const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
      const t = L2 > 0 ? Math.max(0, Math.min(1, ((mx - ax) * dx + (my - ay) * dy) / L2)) : 0;
      return Math.hypot(ax + t * dx - mx, ay + t * dy - my);
    };
    bestD = 5;
    for (let k = 0; k < this.syns.length; k++) {
      if (!this.synShown[k]) continue;
      const d = seg(this.synA, this.synB, k);
      if (d < bestD) {
        bestD = d;
        best = { kind: 'syn', k };
      }
    }
    for (let k = 0; k < this.links.length; k++) {
      const d = seg(this.linkA, this.linkB, k);
      if (d < bestD) {
        bestD = d;
        best = { kind: 'link', k };
      }
    }
    return best;
  }

  private onHover(e: PointerEvent) {
    if (e.buttons) return;
    const h = this.pick(e);
    this.hover = h;
    if (!h || !this.ctx) return hideTip();
    const vocab = this.session.data.vocab, toks = this.ctx.tokens;
    if (h.kind === 'unit') {
      const c = this.cols[this.uCol[h.k]], t = this.uT[h.k], i = this.uI[h.k];
      if (c.kind === 'bias') return showTip(`bias node: a constant 1, so its synapses are the biases\n(${c.node})`, e);
      const v = this.vals.get(c)![t * c.n + i];
      const extra = c.extra ? `\n${c.extra(this.ctx, t, i)}` : '';
      const act = c.id.endsWith('.act') ? '\nclick: select this neuron (MLP cells view)' : t !== this.focusT ? '\nclick: focus this position' : '\nclick: focus this part in the inspector';
      return showTip(`${c.label} · ${c.unit(i)} · p${t}:${vocab[toks[t]] ?? '?'}\n= ${fmtNum(v)}${extra}${act}`, e);
    }
    if (h.kind === 'syn') {
      const sy = this.syns[h.k];
      const p = this.session.model.byName.get(sy.param)!;
      const r = Math.floor(sy.index / p.t.cols), c = sy.index % p.t.cols;
      const nm = p.t.rows === 1 ? `${p.name}[${c}]` : `${p.name}[${r}, ${c}]`;
      const w = p.t.data[sy.index];
      const shown = this.opts.mode === 'weight' ? '' : `\n${this.opts.mode === 'grad' ? '−gradient' : `flow at p${this.focusT}`} = ${fmtNum(this.synVal[h.k])}`;
      const frozen = p.frozen || p.mask?.[sy.index] ? '\nfrozen' : '';
      return showTip(`${nm} = ${fmtNum(w)}${shown}${frozen}\n${sy.from.label}·${sy.from.unit(sy.fi)} → ${sy.to.label}·${sy.to.unit(sy.tj)}\nclick: select and edit · shift+wheel: nudge`, e);
    }
    const L = this.links[h.k];
    const at = this.attn.find((a) => a.k === L.from || a.v === L.from);
    return showTip(`layer ${at?.layer} head ${L.head}: query p${this.focusT} → key p${L.key}:${vocab[toks[L.key]] ?? '?'}\np = ${fmtNum(L.p)} (computed from q·k, not a parameter)\n${L.from.label === 'k' ? 'k feeds the score' : 'v is mixed into Σ p·v with weight p'}`, e);
  }

  private onClick(e: MouseEvent) {
    const h = this.pick(e);
    if (!h) return;
    if (h.kind === 'syn') {
      const sy = this.syns[h.k];
      this.select({ param: sy.param, index: sy.index });
      return;
    }
    if (h.kind === 'link') {
      const L = this.links[h.k];
      const at = this.attn.find((a) => a.k === L.from || a.v === L.from);
      ui.head(L.head);
      if (at) ui.focus(`L${at.layer}.attn`);
      return;
    }
    const c = this.cols[this.uCol[h.k]], t = this.uT[h.k], i = this.uI[h.k];
    if (t !== this.focusT) {
      ui.probePos(t);
      return;
    }
    if (c.node !== 'tokens') ui.focus(c.node);
    if (c.id.endsWith('.act')) ui.neuron(i);
    if (c.kind === 'head') ui.head(i);
  }

  private onWheel(e: WheelEvent) {
    if (!e.shiftKey) return;
    const h = this.hover;
    if (!h || h.kind !== 'syn') return;
    e.preventDefault();
    e.stopPropagation();
    const sy = this.syns[h.k];
    const p = this.session.model.byName.get(sy.param)!;
    let mx = 0;
    for (const v of p.t.data) mx = Math.max(mx, Math.abs(v));
    const dy = e.deltaY || e.deltaX;
    if (!dy) return;
    const step = (mx || 1) * 0.02;
    this.session.edit(sy.param, sy.index, p.t.data[sy.index] - Math.sign(dy) * step);
    if (!this.sel || this.sel.param !== sy.param || this.sel.index !== sy.index) this.select({ param: sy.param, index: sy.index });
    showTip(`${sy.param}[${sy.index}] = ${fmtNum(p.t.data[sy.index])}`, e);
  }

  // ---------------------------------------------------------------- camera

  private fitZoom() {
    if (!this.cols.length) return;
    const z = this.opts.zoom;
    const s = this.session.model.spec;
    const pick = (c: Col) =>
      z === 'all' ? true : z === 'embed' ? c.layer === -1 : z === 'readout' ? c.layer === s.nLayer : `L${c.layer}` === z;
    // the overview takes in every sheet (seen from above, so they fan out); a part, only the focused one
    const whole = z === 'all' && this.opts.positions === 'all';
    const Tp = this.ctx?.tokens.length ?? this.sheets;
    const box = new THREE.Box3();
    for (const c of this.cols.filter(pick))
      for (const t of whole ? [0, Tp - 1] : [this.focusT]) {
        box.expandByPoint(this.unitPos(c, 0, t));
        box.expandByPoint(this.unitPos(c, c.n - 1, t));
      }
    if (box.isEmpty()) return;
    box.expandByVector(new THREE.Vector3(0.6, 0, 0.6));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const cam = this.stage.camera;
    const el = this.stage.el;
    // frame the part of the canvas the controls do not cover
    const hudW = this.hud.offsetWidth ? this.hud.offsetWidth + 16 : 0;
    const W = el.clientWidth || 800, H = el.clientHeight || 500;
    const free = Math.max(0.4, (W - hudW) / H);
    // look from the front (later positions nearer), a little above
    cam.position.copy(center).add(new THREE.Vector3(0, -1, whole ? 0.6 : 0.3));
    this.stage.controls.target.copy(center);
    const r = Math.max(size.z * 0.55, (size.x * (whole ? 0.62 : 0.5)) / free, whole ? size.y * 0.3 : 0, 1.2);
    this.stage.fit(center, r);
    // shift the picture right by half the controls' width
    const dist = cam.position.distanceTo(center);
    const perPx = (2 * dist * Math.tan((cam.fov * Math.PI) / 360)) / H;
    // (the camera looks along +y, so screen-right is +x)
    const shift = new THREE.Vector3(-(hudW / 2) * perPx, 0, 0);
    cam.position.add(shift);
    this.stage.controls.target.add(shift);
    this.stage.controls.update();
    this.stage.invalidate();
  }
}

const UP = new THREE.Vector3(0, 1, 0);
const _dir = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();
const _m = new THREE.Matrix4();

/** Place a unit cylinder along a → b with radius r (instance k, or the mesh itself when k < 0). */
function setSeg(mesh: THREE.InstancedMesh | THREE.Mesh, k: number, a: THREE.Vector3, b: THREE.Vector3, r: number): void {
  _dir.subVectors(b, a);
  const len = _dir.length();
  if (len < 1e-9 || r <= 0) _m.makeScale(0, 0, 0);
  else {
    _q.setFromUnitVectors(UP, _dir.divideScalar(len));
    _m.compose(_mid.addVectors(a, b).multiplyScalar(0.5), _q, _s.set(r, len, r));
  }
  if (k >= 0) (mesh as THREE.InstancedMesh).setMatrixAt(k, _m);
  else {
    _m.decompose(mesh.position, mesh.quaternion, mesh.scale);
  }
}
