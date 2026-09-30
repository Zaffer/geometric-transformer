// The MLP, seen as geometry (the polytope lens, inside a transformer).
//
// Neuron j is the hyperplane W1[:,j]·z + b1[j] = 0 in the MLP's input
// space; it is on (for ReLU: nonzero) on one side. The on/off pattern of
// all neurons cuts the input space into convex cells; inside one cell the
// MLP is a single affine map. After a norm, inputs live on the norm's
// manifold, so what matters is how the planes cut that sphere / circle.
//
// Left: the input space in 3D: data points, neuron planes, the manifold
// painted by cell, exact 3D cells (optional, exploded).
// Right: the manifold unrolled flat (sphere → map, circle → strip, no
// norm → a slice), the activation function with the selected neuron's
// pre-activations, and the neuron table.

import * as THREE from 'three';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Session } from '../app/session';
import { viewSet } from '../app/inspect';
import { rt, ui } from '../app/state';
import { actScalar, PIECEWISE_LINEAR } from '../engine/ops';
import type { Model } from '../engine/model';
import { catColor, css, cssVar, sequential, token, type RGB } from '../ui/palette';
import { hideTip, showTip } from '../ui/heatmap';
import { fmtNum } from '../ui/controls';
import { Stage3D, colorOf, disposeTree } from './stage3d';
import { focusLayer } from './attention';
import type { View } from './view';

type Paint = 'pattern' | 'count' | 'neuron';

/** hue from a pattern: arbitrary but stable identity color for a cell */
export function patternColor(bits: Uint8Array | number[], F: number): RGB {
  let h = 2166136261;
  for (let j = 0; j < F; j++) h = Math.imul(h ^ (bits[j] ? 0x9e3779b1 + j : j), 16777619);
  const hue = ((h >>> 0) % 3600) / 3600;
  let on = 0;
  for (let j = 0; j < F; j++) on += bits[j] ? 1 : 0;
  const l = 0.42 + 0.22 * (on / Math.max(1, F));
  return hsl(hue, 0.55, l);
}

function hsl(h: number, s: number, l: number): RGB {
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

/** The norm whose output the MLP of layer L reads (null = raw residual). */
export function mlpInputNorm(model: Model, L: number): string | null {
  const s = model.spec;
  if (s.norm === 'none') return null;
  if (s.normPlacement === 'pre') return `L${L}.ln2`;
  // post-norm: the stream was last normalized by the previous sublayer's norm
  if (s.layers[L].attn) return `L${L}.ln1`;
  for (let l = L - 1; l >= 0; l--) {
    if (s.layers[l].mlp) return `L${l}.ln2`;
    if (s.layers[l].attn) return `L${l}.ln1`;
  }
  return null;
}

interface Neuron {
  w: Float64Array;
  b: number;
}

export class MlpView implements View {
  readonly id = 'mlp';
  readonly title = 'MLP cells';
  readonly el = document.createElement('div');
  private session: Session;
  private stage: Stage3D;
  private g3 = new THREE.Group();
  private side = document.createElement('div');
  private mapCv = document.createElement('canvas');
  private actCv = document.createElement('canvas');
  private table = document.createElement('div');
  private hud!: HTMLDivElement;
  private cap!: HTMLDivElement;
  private active = false;
  private dirty = true;
  private fitted = false;
  private opts = { paint: 'pattern' as Paint, planes: true, cells: false, explode: 0.25, points: true, manifold: true, sliceAxis: 2, slice: 0 };
  private cache: { L: number; neurons: Neuron[]; F: number; D: number; norm: string | null; g: number[]; bb: number[]; R: number; z: Float64Array; y: Int32Array; x: Int32Array; n: number } | null = null;

  constructor(session: Session) {
    this.session = session;
    this.el.classList.add('flexview');
    const left = document.createElement('div');
    Object.assign(left.style, { position: 'relative', flex: '1.35', minWidth: '0' });
    Object.assign(this.side.style, { flex: '1', minWidth: '260px', overflow: 'auto', padding: '8px', borderLeft: '1px solid var(--border)' });
    this.el.append(left, this.side);
    this.stage = new Stage3D(left);
    this.stage.scene.add(this.g3);
    this.buildHud();
    const mapT = document.createElement('div');
    mapT.className = 'note';
    mapT.dataset.role = 'maptitle';
    this.mapCv.style.width = '100%';
    this.mapCv.style.display = 'block';
    this.mapCv.style.cursor = 'crosshair';
    const actT = document.createElement('div');
    actT.className = 'note';
    actT.dataset.role = 'acttitle';
    this.actCv.style.width = '100%';
    this.actCv.style.height = '130px';
    this.actCv.style.display = 'block';
    this.side.append(mapT, this.mapCv, actT, this.actCv, this.table);
    this.mapCv.addEventListener('pointermove', (e) => this.mapHover(e));
    this.mapCv.addEventListener('pointerleave', () => hideTip());

    effect(() => {
      rt.modelVersion();
      rt.paramsVersion();
      ui.focus();
      ui.neuron();
      this.dirty = true;
      if (untracked(() => this.active)) untracked(() => this.render());
    });
    new ResizeObserver(() => this.active && this.drawSide()).observe(this.side);
  }

  activate(): void {
    this.active = true;
    this.stage.start();
    if (this.dirty) this.render();
  }

  deactivate(): void {
    this.active = false;
    this.stage.stop();
    hideTip();
  }

  themeChanged(): void {
    this.stage.themeChanged();
    this.dirty = true;
    if (this.active) this.render();
  }

  private buildHud() {
    const hud = document.createElement('div');
    hud.className = 'hud';
    hud.style.left = '8px';
    hud.style.top = '8px';
    const sel = document.createElement('label');
    sel.append('paint cells by');
    const s = document.createElement('select');
    for (const [v, t] of [['pattern', 'activation pattern'], ['count', 'neurons on'], ['neuron', 'selected neuron']]) s.append(new Option(t, v));
    s.onchange = () => {
      this.opts.paint = s.value as Paint;
      this.render();
    };
    sel.append(s);
    hud.append(sel);
    const tog = (label: string, key: 'planes' | 'cells' | 'points' | 'manifold', title: string) => {
      const l = document.createElement('label');
      const c = document.createElement('input');
      c.type = 'checkbox';
      c.checked = this.opts[key];
      c.onchange = () => {
        this.opts[key] = c.checked;
        this.render();
      };
      l.append(c, label);
      l.title = title;
      hud.append(l);
    };
    tog('neuron planes', 'planes', 'W1[:,j]·z + b1[j] = 0 for every neuron j');
    tog('painted manifold / slice', 'manifold', 'the sphere, circle or slice plane, colored by cell');
    tog('exact 3D cells', 'cells', 'every convex region of the plane arrangement (ReLU-family), pulled apart');
    tog('data points', 'points', 'the MLP inputs of every example, colored by target');
    const ex = document.createElement('label');
    ex.append('explode');
    const r = document.createElement('input');
    r.type = 'range';
    r.min = '0';
    r.max = '1';
    r.step = '0.05';
    r.value = String(this.opts.explode);
    r.oninput = () => {
      this.opts.explode = Number(r.value);
      if (this.opts.cells) this.render();
    };
    ex.append(r);
    hud.append(ex);
    const sl = document.createElement('label');
    sl.dataset.role = 'slice';
    sl.append('slice');
    const ax = document.createElement('select');
    for (const [v, t] of [['2', 'd2 ='], ['1', 'd1 ='], ['0', 'd0 =']]) ax.append(new Option(t, v));
    ax.onchange = () => {
      this.opts.sliceAxis = Number(ax.value);
      this.render();
    };
    const sr = document.createElement('input');
    sr.type = 'range';
    sr.min = '-1';
    sr.max = '1';
    sr.step = '0.01';
    sr.value = '0';
    sr.oninput = () => {
      this.opts.slice = Number(sr.value);
      this.render();
    };
    sl.append(ax, sr);
    hud.append(sl);
    this.hud = hud;
    this.stage.overlay.append(hud);
    this.cap = document.createElement('div');
    this.cap.className = 'caption';
    this.stage.overlay.append(this.cap);
  }

  private gather(): boolean {
    const model = this.session.model, s = model.spec;
    const L = focusLayer(this.session, 'mlp');
    if (L < 0) {
      this.cache = null;
      return false;
    }
    const D = s.dModel, F = s.dMlp;
    const W1 = model.p(`L${L}.mlp.W1`).data;
    const b1 = model.byName.get(`L${L}.mlp.b1`)?.t.data;
    const neurons: Neuron[] = [];
    for (let j = 0; j < F; j++) {
      const w = new Float64Array(D);
      for (let i = 0; i < D; i++) w[i] = W1[i * F + j];
      neurons.push({ w, b: b1 ? b1[j] : 0 });
    }
    const norm = mlpInputNorm(model, L);
    const g = norm ? Array.from(model.byName.get(`${norm}.g`)?.t.data ?? new Array(D).fill(1)) : [];
    const bb = norm ? Array.from(model.byName.get(`${norm}.b`)?.t.data ?? new Array(D).fill(0)) : [];
    const vs = viewSet();
    const zt = vs.fwd.trace.get(`L${L}.ln2`)!;
    let R = 0;
    for (let i = 0; i < zt.rows; i++) {
      let r = 0;
      for (let j = 0; j < D; j++) r += zt.data[i * D + j] ** 2;
      R = Math.max(R, Math.sqrt(r));
    }
    this.cache = { L, neurons, F, D, norm, g, bb, R: Math.max(0.5, R * 1.15), z: zt.data, y: vs.y, x: vs.x, n: zt.rows };
    return true;
  }

  private pattern(z: ArrayLike<number>, off: number, out: Uint8Array): number {
    const c = this.cache!;
    let on = 0;
    for (let j = 0; j < c.F; j++) {
      const nr = c.neurons[j];
      let a = nr.b;
      for (let i = 0; i < c.D; i++) a += nr.w[i] * z[off + i];
      out[j] = a > 0 ? 1 : 0;
      on += out[j];
    }
    return on;
  }

  private paintColor(bits: Uint8Array, on: number): RGB {
    const c = this.cache!;
    if (this.opts.paint === 'count') return sequential(0.15 + (0.85 * on) / Math.max(1, c.F));
    if (this.opts.paint === 'neuron') {
      const n = ui.neuron();
      if (n < 0 || n >= c.F) return token('zero');
      return bits[n] ? token('pos') : token('zero');
    }
    return patternColor(bits, c.F);
  }

  private render() {
    this.dirty = false;
    for (const ch of [...this.g3.children]) disposeTree(ch);
    this.stage.clearLabels();
    if (!this.gather()) {
      this.cap.textContent = 'No MLP in this model (every layer has its MLP switched off).';
      this.side.querySelector<HTMLElement>('[data-role=maptitle]')!.textContent = '';
      this.table.replaceChildren();
      this.stage.invalidate();
      return;
    }
    const c = this.cache!, model = this.session.model, s = model.spec;
    const exact3 = c.D === 3;
    const R = c.R;
    (this.hud.querySelector('[data-role=slice]') as HTMLElement).hidden = !(exact3 && !c.norm);
    const pl = PIECEWISE_LINEAR.has(s.activation);
    this.cap.textContent = exact3
      ? `layer ${c.L} MLP input space (exact 3D). ${c.F} neurons = ${c.F} planes. ${c.norm ? `Inputs lie on the ${s.norm === 'rmsnorm' ? 'sphere' : 'circle'} of ${c.norm}.` : 'No norm: inputs fill space.'} ${pl ? 'Inside every cell the MLP is exactly one affine map.' : `${s.activation} is smooth: the planes mark where each neuron's soft switch sits.`}`
      : `d_model = ${c.D}: the planes and cells live in ${c.D}-d; the 3D picture needs d_model = 3. The side panels still work on the data.`;

    const sel = ui.neuron();
    const selC = new THREE.Color(cssVar('sel'));
    const muted = new THREE.Color(cssVar('muted'));

    if (exact3) {
      // data points
      if (this.opts.points) {
        const geo = new THREE.IcosahedronGeometry(R * 0.012, 1);
        const mesh = new THREE.InstancedMesh(geo, new THREE.MeshLambertMaterial({ color: 0xffffff }), c.n);
        const m = new THREE.Matrix4();
        let k = 0;
        for (let i = 0; i < c.n; i++) {
          if (c.y[i] < 0) continue;
          m.makeTranslation(c.z[i * 3], c.z[i * 3 + 1], c.z[i * 3 + 2]);
          mesh.setMatrixAt(k, m);
          mesh.setColorAt(k, colorOf(catColor(c.y[i])));
          k++;
        }
        mesh.count = k;
        this.g3.add(mesh);
      }
      // planes
      if (this.opts.planes) {
        c.neurons.forEach((nr, j) => {
          const poly = planeBox(nr.w, nr.b, R);
          if (poly.length < 3) return;
          const isSel = j === sel;
          const geo = new THREE.BufferGeometry().setFromPoints(poly);
          const idx: number[] = [];
          for (let i = 1; i + 1 < poly.length; i++) idx.push(0, i, i + 1);
          geo.setIndex(idx);
          this.g3.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: isSel ? selC : muted, transparent: true, opacity: isSel ? 0.22 : 0.05, side: THREE.DoubleSide, depthWrite: false })));
          this.g3.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(poly), new THREE.LineBasicMaterial({ color: isSel ? selC : muted, transparent: true, opacity: isSel ? 0.9 : 0.3 })));
          const ctr = poly.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(1 / poly.length);
          if (isSel) {
            const n = new THREE.Vector3(nr.w[0], nr.w[1], nr.w[2]).normalize();
            this.g3.add(new THREE.ArrowHelper(n, ctr, R * 0.3, selC, R * 0.06, R * 0.03));
          }
          if (isSel) this.stage.label(`n${j}`, poly[0], css(token('sel')));
        });
      }
      // painted manifold / slice
      if (this.opts.manifold) this.paintManifold3D();
      if (this.opts.cells) this.drawCells();
      this.axes(R);
    }
    if (!this.fitted && exact3) {
      this.fitted = true;
      this.stage.fit(new THREE.Vector3(), R * 1.35);
    }
    this.stage.invalidate();
    this.drawSide();
  }

  private paintManifold3D() {
    const c = this.cache!, s = this.session.model.spec;
    const bits = new Uint8Array(c.F);
    const colorAt = (p: ArrayLike<number>) => {
      const on = this.pattern(p, 0, bits);
      return colorOf(this.paintColor(bits, on));
    };
    if (c.norm && s.norm === 'rmsnorm') {
      const geo = new THREE.SphereGeometry(Math.sqrt(3), 96, 48);
      const pa = geo.attributes.position as THREE.BufferAttribute;
      const cols = new Float32Array(pa.count * 3);
      const p = [0, 0, 0];
      for (let i = 0; i < pa.count; i++) {
        p[0] = pa.getX(i) * c.g[0] + c.bb[0];
        p[1] = pa.getY(i) * c.g[1] + c.bb[1];
        p[2] = pa.getZ(i) * c.g[2] + c.bb[2];
        pa.setXYZ(i, p[0], p[1], p[2]);
        const col = colorAt(p);
        cols.set([col.r, col.g, col.b], i * 3);
      }
      geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      this.g3.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8, side: THREE.DoubleSide })));
      // exact cell walls: where neuron j's plane cuts the sphere, a circle
      // d·n = c on the unit directions d, mapped through z = g ⊙ √3 d + b
      const sel = ui.neuron();
      c.neurons.forEach((nr, j) => {
        const wg = [0, 1, 2].map((i) => nr.w[i] * c.g[i] * Math.sqrt(3));
        const len = Math.hypot(...wg);
        if (len < 1e-9) return;
        const off = nr.b + nr.w[0] * c.bb[0] + nr.w[1] * c.bb[1] + nr.w[2] * c.bb[2];
        const cc = -off / len;
        if (Math.abs(cc) >= 1) return;
        const n = new THREE.Vector3(wg[0], wg[1], wg[2]).normalize();
        const e1 = new THREE.Vector3(1, 0, 0).cross(n);
        if (e1.lengthSq() < 1e-6) e1.set(0, 1, 0).cross(n);
        e1.normalize();
        const e2 = n.clone().cross(e1);
        const rr = Math.sqrt(1 - cc * cc);
        const pts: THREE.Vector3[] = [];
        for (let k = 0; k <= 128; k++) {
          const a = (k / 128) * Math.PI * 2;
          const d = n.clone().multiplyScalar(cc).addScaledVector(e1, rr * Math.cos(a)).addScaledVector(e2, rr * Math.sin(a));
          pts.push(new THREE.Vector3(d.x * Math.sqrt(3) * c.g[0] + c.bb[0], d.y * Math.sqrt(3) * c.g[1] + c.bb[1], d.z * Math.sqrt(3) * c.g[2] + c.bb[2]).multiplyScalar(1.003));
        }
        const isSel = j === sel;
        this.g3.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: isSel ? new THREE.Color(cssVar('sel')) : new THREE.Color(cssVar('text')), transparent: true, opacity: isSel ? 1 : 0.35 })));
      });
    } else if (c.norm && s.norm === 'layernorm') {
      const u = new THREE.Vector3(1, -1, 0).normalize(), w = new THREE.Vector3(1, 1, -2).normalize();
      const n = 720, R = Math.sqrt(3);
      const pos: number[] = [], col: number[] = [];
      const at = (i: number) => {
        const a = (i / n) * Math.PI * 2;
        const q = u.clone().multiplyScalar(R * Math.cos(a)).addScaledVector(w, R * Math.sin(a));
        return [q.x * c.g[0] + c.bb[0], q.y * c.g[1] + c.bb[1], q.z * c.g[2] + c.bb[2]];
      };
      // a ribbon so the ring is visible
      const nrm = new THREE.Vector3(1, 1, 1).normalize();
      const wdt = c.R * 0.03;
      for (let i = 0; i < n; i++) {
        const p0 = at(i), p1 = at(i + 1);
        const cc = colorAt(p0);
        const a0 = new THREE.Vector3(...p0), a1 = new THREE.Vector3(...p1);
        const quad = [a0.clone().addScaledVector(nrm, -wdt), a0.clone().addScaledVector(nrm, wdt), a1.clone().addScaledVector(nrm, wdt), a0.clone().addScaledVector(nrm, -wdt), a1.clone().addScaledVector(nrm, wdt), a1.clone().addScaledVector(nrm, -wdt)];
        for (const q of quad) {
          pos.push(q.x, q.y, q.z);
          col.push(cc.r, cc.g, cc.b);
        }
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
      this.g3.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })));
    } else {
      // no norm: a slice plane through the input space
      const ax = this.opts.sliceAxis, R = c.R;
      const val = this.opts.slice * R;
      const N = 160;
      const geo = new THREE.PlaneGeometry(2 * R, 2 * R, N, N);
      const pa = geo.attributes.position as THREE.BufferAttribute;
      const cols = new Float32Array(pa.count * 3);
      const p = [0, 0, 0];
      for (let i = 0; i < pa.count; i++) {
        const a = pa.getX(i), b = pa.getY(i);
        const others = [0, 1, 2].filter((k) => k !== ax);
        p[others[0]] = a;
        p[others[1]] = b;
        p[ax] = val;
        pa.setXYZ(i, p[0], p[1], p[2]);
        const cc = colorAt(p);
        cols.set([cc.r, cc.g, cc.b], i * 3);
      }
      geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      this.g3.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75, side: THREE.DoubleSide, depthWrite: false })));
    }
  }

  /** Exact convex cells of the plane arrangement inside the box, found by
   *  sampling patterns, then brute-force vertex enumeration per pattern. */
  private drawCells() {
    const c = this.cache!, R = c.R, F = c.F;
    const bits = new Uint8Array(F);
    const seen = new Map<string, Uint8Array>();
    const G = 18;
    const p = [0, 0, 0];
    for (let a = 0; a < G; a++)
      for (let b = 0; b < G; b++)
        for (let k = 0; k < G; k++) {
          p[0] = -R + ((a + 0.5) / G) * 2 * R;
          p[1] = -R + ((b + 0.5) / G) * 2 * R;
          p[2] = -R + ((k + 0.5) / G) * 2 * R;
          this.pattern(p, 0, bits);
          const key = bits.join('');
          if (!seen.has(key)) seen.set(key, bits.slice());
        }
    for (let i = 0; i < c.n; i++) {
      this.pattern(c.z, i * 3, bits);
      const key = bits.join('');
      if (!seen.has(key)) seen.set(key, bits.slice());
    }
    // half-spaces: s_j (w_j·x + b_j) >= 0, plus the box
    const planes: { n: number[]; d: number }[] = c.neurons.map((nr) => ({ n: Array.from(nr.w), d: nr.b }));
    for (let i = 0; i < 3; i++) {
      const e = [0, 0, 0];
      e[i] = 1;
      planes.push({ n: e, d: R }); // x_i + R >= 0
      planes.push({ n: e.map((v) => -v), d: R }); // -x_i + R >= 0
    }
    const P = planes.length;
    let drawn = 0;
    for (const [, pat] of seen) {
      if (drawn > 400) break;
      const sign = (j: number) => (j < F ? (pat[j] ? 1 : -1) : 1);
      const verts: THREE.Vector3[] = [];
      for (let i = 0; i < P; i++)
        for (let j = i + 1; j < P; j++)
          for (let k = j + 1; k < P; k++) {
            const x = solve3(planes[i], planes[j], planes[k]);
            if (!x) continue;
            let ok = true;
            for (let m = 0; m < P && ok; m++) {
              const v = planes[m].n[0] * x[0] + planes[m].n[1] * x[1] + planes[m].n[2] * x[2] + planes[m].d;
              if (sign(m) * v < -1e-7) ok = false;
            }
            if (ok) verts.push(new THREE.Vector3(x[0], x[1], x[2]));
          }
      if (verts.length < 4) continue;
      try {
        const geo = new ConvexGeometry(verts);
        const ctr = verts.reduce((a, v) => a.add(v), new THREE.Vector3()).multiplyScalar(1 / verts.length);
        const on = pat.reduce((a, b) => a + b, 0);
        const col = colorOf(this.paintColor(pat, on));
        const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
        const off = ctr.clone().multiplyScalar(this.opts.explode);
        mesh.position.copy(off);
        this.g3.add(mesh);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.9 }));
        edges.position.copy(off);
        this.g3.add(edges);
        drawn++;
      } catch {
        /* flat cell */
      }
    }
    this.cap.textContent += `  ${drawn} cells in the box.`;
  }

  private axes(R: number) {
    const pos: number[] = [];
    for (let i = 0; i < 3; i++) {
      const a = [0, 0, 0], b = [0, 0, 0];
      a[i] = -R;
      b[i] = R;
      pos.push(...a, ...b);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.g3.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('border')) })));
    ['d0', 'd1', 'd2'].forEach((n, i) => {
      const p = new THREE.Vector3();
      p.setComponent(i, R * 1.05);
      this.stage.label(n, p, css(token('muted')));
    });
  }

  // ---- side panel ----

  /** The 2D map: pixel → input point z (or null outside). */
  private mapGeometry(): { W: number; H: number; toZ: (u: number, v: number) => number[] | null; fromZ: (z: ArrayLike<number>, o: number) => [number, number] | null; title: string } | null {
    const c = this.cache!, s = this.session.model.spec;
    if (c.D !== 3) return null;
    const W = Math.max(200, this.side.clientWidth - 20);
    if (c.norm && s.norm === 'rmsnorm') {
      const H = Math.round(W / 2);
      return {
        W, H,
        title: 'the sphere unrolled (longitude × latitude). Every MLP input is on it; each color is one cell; dots = inputs colored by target.',
        toZ: (u, v) => {
          const lon = (u / W) * 2 * Math.PI - Math.PI, lat = Math.PI / 2 - (v / H) * Math.PI;
          const d = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
          return d.map((x, i) => Math.sqrt(3) * x * c.g[i] + c.bb[i]);
        },
        fromZ: (z, o) => {
          const x = [0, 1, 2].map((i) => (c.g[i] !== 0 ? (z[o + i] - c.bb[i]) / c.g[i] : 0));
          const r = Math.hypot(x[0], x[1], x[2]);
          if (r < 1e-9) return null;
          const lat = Math.asin(Math.max(-1, Math.min(1, x[2] / r))), lon = Math.atan2(x[1], x[0]);
          return [((lon + Math.PI) / (2 * Math.PI)) * W, ((Math.PI / 2 - lat) / Math.PI) * H];
        },
      };
    }
    if (c.norm && s.norm === 'layernorm') {
      const H = 34;
      const u = [1 / Math.SQRT2, -1 / Math.SQRT2, 0], w = [1 / Math.sqrt(6), 1 / Math.sqrt(6), -2 / Math.sqrt(6)];
      return {
        W, H,
        title: 'the circle unrolled (angle 0 → 360°): at d=3 LayerNorm leaves one degree of freedom. Rows below: each neuron on (color) / off.',
        toZ: (x) => {
          const a = (x / W) * 2 * Math.PI;
          return [0, 1, 2].map((i) => Math.sqrt(3) * (Math.cos(a) * u[i] + Math.sin(a) * w[i]) * c.g[i] + c.bb[i]);
        },
        fromZ: (z, o) => {
          const x = [0, 1, 2].map((i) => (c.g[i] !== 0 ? (z[o + i] - c.bb[i]) / c.g[i] : 0));
          const a = Math.atan2(x[0] * w[0] + x[1] * w[1] + x[2] * w[2], x[0] * u[0] + x[1] * u[1] + x[2] * u[2]);
          return [(((a + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI)) * W, H / 2];
        },
      };
    }
    const H = W, R = c.R, ax = this.opts.sliceAxis;
    const others = [0, 1, 2].filter((k) => k !== ax);
    const val = this.opts.slice * R;
    return {
      W, H,
      title: `slice d${ax} = ${fmtNum(val)} through the input space (no norm: inputs fill 3D). Dots: inputs near the slice.`,
      toZ: (a, b) => {
        const p = [0, 0, 0];
        p[others[0]] = -R + (a / W) * 2 * R;
        p[others[1]] = R - (b / H) * 2 * R;
        p[ax] = val;
        return p;
      },
      fromZ: (z, o) => {
        if (Math.abs(z[o + ax] - val) > R * 0.15) return null;
        return [((z[o + others[0]] + R) / (2 * R)) * W, ((R - z[o + others[1]]) / (2 * R)) * H];
      },
    };
  }

  private drawSide() {
    if (!this.cache) return;
    const c = this.cache, s = this.session.model.spec;
    const mg = this.mapGeometry();
    const mapTitle = this.side.querySelector<HTMLElement>('[data-role=maptitle]')!;
    const dpr = window.devicePixelRatio || 1;
    if (mg) {
      const isLN = c.norm && s.norm === 'layernorm';
      const rowsH = isLN ? c.F * 7 + 14 : 0;
      const Ht = mg.H + rowsH;
      mapTitle.textContent = mg.title;
      this.mapCv.width = Math.round(mg.W * dpr);
      this.mapCv.height = Math.round(Ht * dpr);
      this.mapCv.style.height = `${Ht}px`;
      const ctx = this.mapCv.getContext('2d')!;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const img = ctx.createImageData(this.mapCv.width, Math.round(mg.H * dpr));
      const bits = new Uint8Array(c.F);
      const step = 2; // compute every other device pixel, fill 2x2
      const gw = Math.ceil(img.width / step);
      // two 32-bit hashes of the pattern: exact for any d_mlp (a float key loses bits past 53)
      const keys = new Uint32Array(gw * Math.ceil(img.height / step));
      const keys2 = new Uint32Array(keys.length);
      for (let py = 0; py < img.height; py += step)
        for (let px = 0; px < img.width; px += step) {
          const z = mg.toZ(px / dpr, py / dpr);
          if (!z) continue;
          const on = this.pattern(z, 0, bits);
          let key = 2166136261, key2 = 2166136261 ^ 0x5bd1e995;
          for (let j = 0; j < c.F; j++) {
            key = Math.imul(key ^ (bits[j] + 2 * j), 16777619);
            key2 = Math.imul(key2 ^ (bits[j] * 131 + j), 16777619);
          }
          key >>>= 0;
          key2 >>>= 0;
          const gi = (py / step) * gw + px / step;
          keys[gi] = key;
          keys2[gi] = key2;
          let col = this.paintColor(bits, on);
          const differs = (k: number) => keys[k] !== key || keys2[k] !== key2;
          const edge = (px > 0 && differs(gi - 1)) || (py > 0 && differs(gi - gw));
          if (edge) col = [col[0] * 0.35, col[1] * 0.35, col[2] * 0.35];
          for (let dy = 0; dy < step; dy++)
            for (let dx = 0; dx < step; dx++) {
              const k = ((py + dy) * img.width + px + dx) * 4;
              if (k >= img.data.length) continue;
              img.data[k] = col[0] * 255;
              img.data[k + 1] = col[1] * 255;
              img.data[k + 2] = col[2] * 255;
              img.data[k + 3] = 255;
            }
        }
      ctx.putImageData(img, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // data points
      for (let i = 0; i < c.n; i++) {
        if (c.y[i] < 0) continue;
        const q = mg.fromZ(c.z, i * 3);
        if (!q) continue;
        ctx.fillStyle = css(catColor(c.y[i]));
        ctx.strokeStyle = css(token('panel'));
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(q[0], isLN ? mg.H / 2 : q[1], 2.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      if (isLN) {
        // per-neuron on/off rows along the angle
        const y0 = mg.H + 8;
        for (let j = 0; j < c.F; j++) {
          for (let x = 0; x < mg.W; x++) {
            const z = mg.toZ(x, 0)!;
            let a = c.neurons[j].b;
            for (let i = 0; i < 3; i++) a += c.neurons[j].w[i] * z[i];
            ctx.fillStyle = a > 0 ? (j === ui.neuron() ? css(token('sel')) : css(token('pos'))) : css(token('zero'));
            ctx.fillRect(x, y0 + j * 7, 1, 6);
          }
        }
        ctx.fillStyle = css(token('muted'));
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillText('n0', 1, y0 - 1);
      }
    } else {
      mapTitle.textContent = 'The unrolled map needs d_model = 3.';
      this.mapCv.height = 0;
      this.mapCv.style.height = '0px';
    }
    this.drawActivation();
    this.drawTable();
  }

  private mapHover(e: PointerEvent) {
    const mg = this.cache && this.mapGeometry();
    if (!mg) return;
    const r = this.mapCv.getBoundingClientRect();
    const u = e.clientX - r.left, v = e.clientY - r.top;
    if (v > mg.H) return hideTip();
    const z = mg.toZ(u, v);
    if (!z) return hideTip();
    const bits = new Uint8Array(this.cache!.F);
    const on = this.pattern(z, 0, bits);
    const act = Array.from(bits).map((b, j) => (b ? `n${j}` : '')).filter(Boolean);
    showTip(`z = (${z.map((x) => fmtNum(x)).join(', ')})\n${on} of ${this.cache!.F} neurons on: ${act.join(' ') || 'none'}`, e);
  }

  private drawActivation() {
    const c = this.cache!, s = this.session.model.spec;
    const t = this.side.querySelector<HTMLElement>('[data-role=acttitle]')!;
    const sel = ui.neuron();
    const vs = viewSet();
    const pre = vs.fwd.trace.get(`L${c.L}.mlp.pre`)!;
    const F = c.F;
    // pre-activation values of the selected neuron (or all neurons)
    const vals: number[] = [];
    for (let i = 0; i < pre.rows; i++) {
      if (c.y[i] < 0) continue;
      if (sel >= 0 && sel < F) vals.push(pre.data[i * F + sel]);
      else for (let j = 0; j < F; j++) vals.push(pre.data[i * F + j]);
    }
    let lo = -3, hi = 3;
    for (const v of vals) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    lo = -m;
    hi = m;
    t.textContent = `${s.activation}(x) with the pre-activations of ${sel >= 0 && sel < F ? `neuron n${sel}` : 'every neuron'} on the data (histogram). Mass left of 0 is switched off.`;
    const dpr = window.devicePixelRatio || 1;
    const W = this.actCv.clientWidth || 240, H = 130;
    this.actCv.width = Math.round(W * dpr);
    this.actCv.height = Math.round(H * dpr);
    const ctx = this.actCv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const X = (x: number) => ((x - lo) / (hi - lo)) * (W - 8) + 4;
    // histogram
    const nb = 48, hist = new Array(nb).fill(0);
    for (const v of vals) hist[Math.min(nb - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * nb)))]++;
    const hm = Math.max(1, ...hist);
    ctx.fillStyle = css(token('neg'), 0.45);
    for (let b = 0; b < nb; b++) {
      const h = (hist[b] / hm) * (H * 0.45);
      ctx.fillRect(X(lo + (b / nb) * (hi - lo)), H - 12 - h, (W - 8) / nb - 1, h);
    }
    // axes
    ctx.strokeStyle = css(token('border'));
    ctx.beginPath();
    ctx.moveTo(4, H - 12);
    ctx.lineTo(W - 4, H - 12);
    ctx.moveTo(X(0), 4);
    ctx.lineTo(X(0), H - 12);
    ctx.stroke();
    // f(x)
    let fmin = Infinity, fmax = -Infinity;
    const N = 200;
    const fs: number[] = [];
    for (let i = 0; i <= N; i++) {
      const x = lo + (i / N) * (hi - lo);
      const f = actScalar(s.activation, x, s.leakySlope);
      fs.push(f);
      fmin = Math.min(fmin, f);
      fmax = Math.max(fmax, f);
    }
    const Y = (f: number) => H - 12 - ((f - Math.min(0, fmin)) / Math.max(1e-9, fmax - Math.min(0, fmin))) * (H - 20);
    ctx.strokeStyle = css(token('text'));
    ctx.lineWidth = 2;
    ctx.beginPath();
    fs.forEach((f, i) => {
      const x = X(lo + (i / N) * (hi - lo));
      if (i) ctx.lineTo(x, Y(f));
      else ctx.moveTo(x, Y(f));
    });
    ctx.stroke();
    ctx.fillStyle = css(token('muted'));
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(fmtNum(lo), 4, H - 1);
    ctx.fillText('0', X(0) - 3, H - 1);
    ctx.fillText(fmtNum(hi), W - 30, H - 1);
  }

  private drawTable() {
    const c = this.cache!;
    const vs = viewSet();
    const pre = vs.fwd.trace.get(`L${c.L}.mlp.pre`)!;
    const F = c.F;
    const act = new Array(F).fill(0);
    let n = 0;
    for (let i = 0; i < pre.rows; i++) {
      if (c.y[i] < 0) continue;
      n++;
      for (let j = 0; j < F; j++) if (pre.data[i * F + j] > 0) act[j]++;
    }
    const sel = ui.neuron();
    const rows = c.neurons.map((nr, j) => {
      const frac = n ? act[j] / n : 0;
      const wn = Math.hypot(...nr.w);
      const status = frac === 0 ? 'dead' : frac === 1 ? 'always on' : '';
      return `<tr data-n="${j}" style="cursor:pointer;${j === sel ? 'background:var(--panel2);outline:1px solid var(--sel)' : ''}"><td>n${j}</td><td>${(frac * 100).toFixed(0)}%</td><td>${fmtNum(wn)}</td><td>${fmtNum(nr.b)}</td><td>${fmtNum(wn > 0 ? -nr.b / wn : 0)}</td><td class="muted">${status}</td></tr>`;
    });
    this.table.innerHTML = `<div class="note" style="margin-top:8px">neurons of layer ${c.L} (click to select; answer positions only). offset = signed distance of the plane from the origin. "dead" = never on for any input: it gets no gradient.</div>
      <table style="border-collapse:collapse;font-size:11px;width:100%"><tr class="muted"><th>n</th><th>on</th><th>|w|</th><th>b</th><th>offset</th><th></th></tr>${rows.join('')}</table>`;
    for (const tr of this.table.querySelectorAll<HTMLElement>('tr[data-n]')) {
      tr.onclick = () => ui.neuron(ui.neuron() === Number(tr.dataset.n) ? -1 : Number(tr.dataset.n));
    }
  }
}

/** Polygon where the plane w·x + b = 0 cuts the cube [-R, R]^3. */
function planeBox(w: ArrayLike<number>, b: number, R: number): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  const corners = [];
  for (let i = 0; i < 8; i++) corners.push([(i & 1 ? 1 : -1) * R, (i & 2 ? 1 : -1) * R, (i & 4 ? 1 : -1) * R]);
  const f = (p: number[]) => w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + b;
  for (let i = 0; i < 8; i++)
    for (let j = i + 1; j < 8; j++) {
      const d = i ^ j;
      if (d !== 1 && d !== 2 && d !== 4) continue; // edges differ in one bit
      const a = corners[i], c = corners[j];
      const fa = f(a), fc = f(c);
      if (fa * fc > 0 || fa === fc) continue;
      const t = fa / (fa - fc);
      pts.push(new THREE.Vector3(a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t, a[2] + (c[2] - a[2]) * t));
    }
  if (pts.length < 3) return [];
  const ctr = pts.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(1 / pts.length);
  const n = new THREE.Vector3(w[0], w[1], w[2]).normalize();
  const e1 = pts[0].clone().sub(ctr).normalize();
  const e2 = n.clone().cross(e1);
  pts.sort((p, q) => Math.atan2(p.clone().sub(ctr).dot(e2), p.clone().sub(ctr).dot(e1)) - Math.atan2(q.clone().sub(ctr).dot(e2), q.clone().sub(ctr).dot(e1)));
  return pts;
}

/** Intersection of three planes n·x + d = 0 (Cramer), or null if degenerate. */
function solve3(a: { n: number[]; d: number }, b: { n: number[]; d: number }, c: { n: number[]; d: number }): number[] | null {
  const [a1, a2, a3] = a.n, [b1, b2, b3] = b.n, [c1, c2, c3] = c.n;
  const det = a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1);
  if (Math.abs(det) < 1e-12) return null;
  const r = [-a.d, -b.d, -c.d];
  const x = (r[0] * (b2 * c3 - b3 * c2) - a2 * (r[1] * c3 - b3 * r[2]) + a3 * (r[1] * c2 - b2 * r[2])) / det;
  const y = (a1 * (r[1] * c3 - b3 * r[2]) - r[0] * (b1 * c3 - b3 * c1) + a3 * (b1 * r[2] - r[1] * c1)) / det;
  const z = (a1 * (b2 * r[2] - r[1] * c2) - a2 * (b1 * r[2] - r[1] * c1) + r[0] * (b1 * c2 - b2 * c1)) / det;
  return [x, y, z];
}
