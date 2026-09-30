// The residual space. Every (example, position) is a point in the d_model
// space of the residual stream. The stage slider moves every point through
// the network: embedding → each sublayer's normalized input → each add →
// final norm. At d_model = 3 nothing is projected.
//
// Overlays: the norm's manifold (sphere / circle / ellipsoid), the
// unembedding's token regions painted on the final manifold, the logit-lens
// regions on a sphere for intermediate stages, embedding / unembedding
// vectors, the probe sequence's trajectories.

import * as THREE from 'three';
import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Session } from '../app/session';
import { probe, viewSet } from '../app/inspect';
import { rt, ui } from '../app/state';
import { catColor, css, cssVar, sequential, token, NUM_CAT, type RGB } from '../ui/palette';
import { hideTip, showTip } from '../ui/heatmap';
import { fmtNum } from '../ui/controls';
import { Stage3D, colorOf, disposeTree } from './stage3d';
import { geoStages, Projector, type GeoStage } from './geom';
import type { View } from './view';
import type { Model } from '../engine/model';

type ColorBy = 'target' | 'token' | 'prediction' | 'position' | 'correct';

function stageCaption(st: GeoStage, model: Model): string {
  const s = model.spec;
  if (st.key === 'resid.embed') return `Each point is one token at one position: wte[token]${s.pos !== 'none' ? ' + pos[position]' : ''}. Only ${model.vocab}${s.pos !== 'none' ? ` × ${model.ctx}` : ''} distinct points exist here.`;
  if (st.kind === 'norm') {
    const what = s.norm === 'rmsnorm' ? 'RMSNorm puts every point on a sphere (radius √d), then gain/bias stretch and shift it' : 'LayerNorm removes the mean (the (1,1,1) direction) and fixes the length: at d=3 every point lands on a circle; only its angle is left';
    return `${what}. This is what the ${st.key === 'final' ? 'unembedding' : st.key.endsWith('ln1') ? 'attention' : 'MLP'} reads.${st.key === 'final' ? ' Colors on the manifold: which token the unembedding picks there.' : ''}`;
  }
  if (st.key.endsWith('mid')) return 'After attention: each point moved by a weighted average of (projected) value vectors from earlier positions: the only step where positions talk to each other.';
  return 'After the MLP: each point moved by an amount that depends only on where it was (the MLP acts on each position alone).';
}

export class ResidualView implements View {
  readonly id = 'residual';
  readonly title = 'residual space';
  readonly el = document.createElement('div');
  private session: Session;
  private stage: Stage3D;
  private proj: Projector | null = null;
  private stages: GeoStage[] = [];
  /** projected positions per stage: stages × N × 3 */
  private P: Float32Array[] = [];
  private probeP: Float32Array[] = [];
  private N = 0;
  private visible: Uint8Array = new Uint8Array(0);
  /** instance k draws point order[k] (only visible points get instances) */
  private order: Int32Array = new Int32Array(0);
  private cloud: THREE.InstancedMesh | null = null;
  private probeMesh: THREE.InstancedMesh | null = null;
  private trails: THREE.LineSegments | null = null;
  private overlays = new THREE.Group();
  private axes = new THREE.Group();
  private s = 0; // continuous stage position
  private playing = false;
  private fitted = false;
  private radius = 3;
  private hud!: HTMLDivElement;
  private caption!: HTMLDivElement;
  private banner!: HTMLDivElement;
  private legend!: HTMLDivElement;
  private stageSlider!: HTMLInputElement;
  private stageLabel!: HTMLSpanElement;
  private opts = {
    colorBy: 'target' as ColorBy,
    positions: 'supervised',
    manifold: true,
    regions: true,
    lens: false,
    vectors: true,
    trails: true,
    axes: true,
    size: 1,
  };
  private probeLabels: { t: number; l: { el: HTMLDivElement; pos: THREE.Vector3 } }[] = [];
  private active = false;
  private needRebuild = true;

  constructor(session: Session) {
    this.session = session;
    this.stage = new Stage3D(this.el);
    this.stage.scene.add(this.overlays, this.axes);
    this.buildHud();
    this.stage.onFrame = (dt) => this.frame(dt);
    const cv = this.stage.renderer.domElement;
    cv.addEventListener('pointermove', (e) => this.hover(e));
    cv.addEventListener('pointerleave', () => hideTip());
    cv.addEventListener('click', (e) => this.click(e));

    effect(() => {
      rt.modelVersion();
      rt.paramsVersion();
      ui.probe();
      ui.probeTokens();
      this.needRebuild = true;
      if (untracked(() => this.active)) untracked(() => this.rebuild());
    });
    effect(() => {
      const k = ui.stage();
      untracked(() => {
        this.stageSlider.value = String(k);
        if (!this.playing && Math.abs(this.s - k) > 1.5) this.s = k;
        this.stage.invalidate();
      });
    });
    effect(() => {
      ui.probePos();
      untracked(() => this.active && this.rebuild());
    });
  }

  activate(): void {
    this.active = true;
    this.stage.start();
    if (this.needRebuild) this.rebuild();
    if (!this.fitted) {
      this.fitted = true;
      this.fitCamera();
    }
  }

  deactivate(): void {
    this.active = false;
    this.stage.stop();
    hideTip();
  }

  themeChanged(): void {
    this.stage.themeChanged();
    this.needRebuild = true;
    if (this.active) this.rebuild();
  }

  private buildHud() {
    const hud = document.createElement('div');
    hud.className = 'hud';
    hud.style.left = '8px';
    hud.style.top = '8px';
    hud.style.maxWidth = '270px';
    this.hud = hud;

    const top = document.createElement('div');
    top.style.display = 'flex';
    top.style.gap = '4px';
    top.style.alignItems = 'center';
    const prev = document.createElement('button');
    prev.textContent = '◀';
    prev.title = 'previous stage ([)';
    prev.onclick = () => ui.stage(Math.max(0, ui.stage() - 1));
    const next = document.createElement('button');
    next.textContent = '▶';
    next.title = 'next stage (])';
    next.onclick = () => ui.stage(Math.min(this.stages.length - 1, ui.stage() + 1));
    const play = document.createElement('button');
    play.textContent = 'flow';
    play.title = 'animate every point through the whole network';
    play.onclick = () => {
      this.playing = !this.playing;
      play.classList.toggle('on', this.playing);
      if (this.playing && this.s >= this.stages.length - 1.01) this.s = 0;
    };
    const fit = document.createElement('button');
    fit.textContent = 'fit';
    fit.title = 'fit the camera to the points';
    fit.onclick = () => this.fitCamera();
    this.stageLabel = document.createElement('span');
    this.stageLabel.style.fontWeight = 'bold';
    top.append(prev, next, play, fit);
    this.stageSlider = document.createElement('input');
    this.stageSlider.type = 'range';
    this.stageSlider.min = '0';
    this.stageSlider.step = '1';
    this.stageSlider.style.width = '100%';
    this.stageSlider.oninput = () => {
      this.playing = false;
      play.classList.remove('on');
      ui.stage(Number(this.stageSlider.value));
    };
    hud.append(top, this.stageSlider, this.stageLabel);

    const sel = (label: string, key: 'colorBy' | 'positions', options: [string, string][]) => {
      const l = document.createElement('label');
      l.append(label);
      const s = document.createElement('select');
      for (const [v, t] of options) s.append(new Option(t, v));
      s.value = String(this.opts[key]);
      s.onchange = () => {
        (this.opts as Record<string, unknown>)[key] = s.value;
        this.rebuild();
      };
      l.append(s);
      hud.append(l);
      return s;
    };
    sel('color', 'colorBy', [
      ['target', 'target (next token)'],
      ['prediction', 'prediction'],
      ['token', 'current token'],
      ['position', 'position'],
      ['correct', 'right / wrong'],
    ]);
    const posSel = sel('show', 'positions', [['supervised', 'answer positions'], ['all', 'all positions']]);
    posSel.dataset.role = 'positions';
    const tog = (label: string, key: keyof ResidualView['opts'], title: string) => {
      const l = document.createElement('label');
      const c = document.createElement('input');
      c.type = 'checkbox';
      c.checked = Boolean(this.opts[key]);
      c.onchange = () => {
        (this.opts as Record<string, unknown>)[key] = c.checked;
        this.rebuild();
      };
      l.append(c, label);
      l.title = title;
      hud.append(l);
    };
    tog('norm manifold', 'manifold', 'the surface a norm puts every point on');
    tog('token regions (final)', 'regions', 'paint the final manifold by which token the unembedding picks there');
    tog('logit lens sphere', 'lens', 'for any stage: which token would win if we read out right here (final norm + unembed)');
    tog('embedding / unembedding vectors', 'vectors', 'wte rows at the start; unembedding directions at the end');
    tog('probe trajectories', 'trails', 'the probe sequence\'s path through every stage');
    tog('axes', 'axes', 'd0, d1, d2');
    const sz = document.createElement('label');
    sz.append('size');
    const r = document.createElement('input');
    r.type = 'range';
    r.min = '0.3';
    r.max = '3';
    r.step = '0.1';
    r.value = '1';
    r.oninput = () => {
      this.opts.size = Number(r.value);
      this.stage.invalidate();
      this.updateInstances();
    };
    sz.append(r);
    hud.append(sz);
    this.stage.overlay.append(hud);

    this.caption = document.createElement('div');
    this.caption.className = 'caption';
    this.stage.overlay.append(this.caption);
    this.banner = document.createElement('div');
    this.banner.className = 'banner';
    this.banner.hidden = true;
    this.stage.overlay.append(this.banner);
    this.legend = document.createElement('div');
    this.legend.className = 'hud';
    this.legend.style.right = '8px';
    this.legend.style.top = '8px';
    this.stage.overlay.append(this.legend);
  }

  private fitCamera() {
    this.stage.fit(new THREE.Vector3(0, 0, 0), this.radius * 1.1);
  }

  private colorFor(b: number, t: number, T: number, x: Int32Array, y: Int32Array, probs: Float64Array, V: number): RGB {
    const i = b * T + t;
    switch (this.opts.colorBy) {
      case 'token':
        return catColor(x[i]);
      case 'target':
        return y[i] >= 0 ? catColor(y[i]) : token('muted');
      case 'prediction': {
        let best = 0;
        for (let v = 1; v < V; v++) if (probs[i * V + v] > probs[i * V + best]) best = v;
        return catColor(best);
      }
      case 'position':
        return sequential(0.25 + (0.75 * t) / Math.max(1, T - 1));
      case 'correct': {
        if (y[i] < 0) return token('muted');
        let best = 0;
        for (let v = 1; v < V; v++) if (probs[i * V + v] > probs[i * V + best]) best = v;
        return best === y[i] ? token('good') : token('bad');
      }
    }
  }

  private rebuild() {
    this.needRebuild = false;
    const model = this.session.model;
    const vs = viewSet();
    const pr = probe();
    const stages = geoStages(model);
    this.stages = stages;
    const D = model.spec.dModel;
    if (!this.proj || this.proj.d !== D) this.proj = new Projector(D);
    this.proj.fit(vs.fwd.trace, stages);
    const proj = this.proj;
    this.stageSlider.max = String(stages.length - 1);
    if (ui.stage() > stages.length - 1) ui.stage(stages.length - 1);

    // project every stage
    const N = vs.B * vs.T;
    this.N = N;
    const v3 = new THREE.Vector3();
    this.P = stages.map((st) => {
      const t = vs.fwd.trace.get(st.key)!;
      const out = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        proj.apply(t.data, i * D, v3);
        out[i * 3] = v3.x;
        out[i * 3 + 1] = v3.y;
        out[i * 3 + 2] = v3.z;
      }
      return out;
    });
    const PT = pr.tokens.length;
    this.probeP = stages.map((st) => {
      const t = pr.fwd.trace.get(st.key)!;
      const out = new Float32Array(PT * 3);
      for (let i = 0; i < PT; i++) {
        proj.apply(t.data, i * D, v3);
        out.set([v3.x, v3.y, v3.z], i * 3);
      }
      return out;
    });
    // radius of the scene
    let r = 0;
    for (const P of this.P) for (let i = 0; i < P.length; i += 3) r = Math.max(r, Math.hypot(P[i], P[i + 1], P[i + 2]));
    this.radius = Math.max(0.5, Math.min(r, 50));

    // visibility filter
    const T = vs.T;
    this.visible = new Uint8Array(N);
    for (let i = 0; i < N; i++) this.visible[i] = this.opts.positions === 'all' || vs.y[i] >= 0 ? 1 : 0;

    // cloud mesh
    if (this.cloud) disposeTree(this.cloud);
    const geo = new THREE.IcosahedronGeometry(1, 1);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const order: number[] = [];
    for (let i = 0; i < N; i++) if (this.visible[i]) order.push(i);
    this.order = Int32Array.from(order);
    this.cloud = new THREE.InstancedMesh(geo, mat, Math.max(1, order.length));
    this.cloud.count = order.length;
    const V = model.vocab;
    order.forEach((i, k) => this.cloud!.setColorAt(k, colorOf(this.colorFor(Math.floor(i / T), i % T, T, vs.x, vs.y, vs.probs, V))));
    this.stage.scene.add(this.cloud);

    // probe mesh (larger, outlined by a ring color)
    if (this.probeMesh) disposeTree(this.probeMesh);
    this.probeMesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), new THREE.MeshLambertMaterial({ color: 0xffffff }), PT);
    const py = pr.targets ?? new Int32Array(PT).fill(-1);
    for (let t = 0; t < PT; t++) {
      const c = this.colorFor(0, t, PT, pr.tokens, py, pr.probs, V);
      this.probeMesh.setColorAt(t, colorOf(c));
    }
    this.stage.scene.add(this.probeMesh);

    this.buildTrails(PT);
    this.buildAxes();
    this.buildProbeLabels(pr.tokens);
    this.buildOverlays();
    this.buildLegend();
    this.banner.hidden = proj.exact;
    this.banner.textContent = proj.exact ? '' : `d_model = ${D}: projected`;
    this.updateInstances();
    this.stage.invalidate();
  }

  private buildTrails(PT: number) {
    if (this.trails) disposeTree(this.trails);
    this.trails = null;
    const focus = ui.probePos();
    if (!this.opts.trails || this.stages.length < 2 || focus < 0) return;
    const pos: number[] = [], col: number[] = [];
    const c = new THREE.Color(cssVar('sel') || '#ffd166');
    const m = new THREE.Color(cssVar('muted') || '#888');
    for (let t = 0; t < PT; t++) {
      if (focus >= 0 && t !== focus) continue;
      for (let k = 0; k + 1 < this.probeP.length; k++) {
        const a = this.probeP[k], b = this.probeP[k + 1];
        pos.push(a[t * 3], a[t * 3 + 1], a[t * 3 + 2], b[t * 3], b[t * 3 + 1], b[t * 3 + 2]);
        const cc = focus >= 0 ? c : m;
        col.push(cc.r, cc.g, cc.b, cc.r, cc.g, cc.b);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    this.trails = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 }));
    this.stage.scene.add(this.trails);
  }

  private buildAxes() {
    for (const c of [...this.axes.children]) disposeTree(c);
    if (!this.opts.axes) return;
    const L = this.radius * 1.15;
    const col = new THREE.Color(cssVar('border') || '#444');
    const pos: number[] = [];
    for (let i = 0; i < 3; i++) {
      const a = [0, 0, 0], b = [0, 0, 0];
      a[i] = -L;
      b[i] = L;
      pos.push(...a, ...b);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.axes.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: col })));
  }

  private buildProbeLabels(tokens: Int32Array) {
    this.stage.clearLabels();
    this.probeLabels = [];
    const vocab = this.session.data.vocab;
    const focus = ui.probePos();
    for (let t = 0; t < tokens.length; t++) {
      if (focus >= 0 && t !== focus) continue;
      const l = this.stage.label(`${t}:${vocab[tokens[t]]}`, new THREE.Vector3(), undefined);
      l.el.style.transform = 'translate(6px, -14px)';
      l.el.style.fontWeight = 'bold';
      this.probeLabels.push({ t, l });
    }
    if (this.opts.axes) {
      const L = this.radius * 1.2;
      const names = this.proj!.exact ? ['d0', 'd1', 'd2'] : ['pc1', 'pc2', 'pc3'];
      [new THREE.Vector3(L, 0, 0), new THREE.Vector3(0, L, 0), new THREE.Vector3(0, 0, L)].forEach((p, i) => this.stage.label(names[i], p, css(token('muted'))));
    }
  }

  private buildLegend() {
    const vocab = this.session.data.vocab;
    const mode = this.opts.colorBy;
    this.legend.replaceChildren();
    const title = document.createElement('div');
    title.className = 'muted';
    title.textContent = { target: 'target token', prediction: 'predicted token', token: 'current token', position: 'position', correct: 'prediction' }[mode];
    this.legend.append(title);
    const item = (c: RGB, text: string) => {
      const d = document.createElement('div');
      d.innerHTML = `<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${css(c)};margin-right:5px"></span>${text}`;
      this.legend.append(d);
    };
    if (mode === 'correct') {
      item(token('good'), 'right');
      item(token('bad'), 'wrong');
      item(token('muted'), 'no target');
    } else if (mode === 'position') {
      item(sequential(0.25), 'first');
      item(sequential(1), 'last');
    } else {
      vocab.slice(0, NUM_CAT).forEach((v, i) => item(catColor(i), v));
      if (vocab.length > NUM_CAT) item(token('muted'), `${vocab.length - NUM_CAT} more (gray)`);
      if (mode === 'target') item(token('muted'), 'no target');
    }
  }

  /** norm manifold, token regions, logit lens, vectors: for the current integer stage */
  private buildOverlays() {
    for (const c of [...this.overlays.children]) disposeTree(c);
    const model = this.session.model;
    const k = Math.round(this.s);
    const st = this.stages[Math.min(k, this.stages.length - 1)];
    if (!st) return;
    const D = model.spec.dModel;
    this.stageLabel.textContent = `${k + 1}/${this.stages.length} · ${st.label}`;
    this.caption.textContent = `${stageCaption(st, model)}  ·  ${this.proj!.describe()}`;
    const exact = D === 3;
    const muted = new THREE.Color(cssVar('muted') || '#888');

    // norm manifold
    if (this.opts.manifold && st.manifold && exact) {
      const g = model.byName.get(`${st.manifold}.g`)?.t.data ?? [1, 1, 1];
      const b = model.byName.get(`${st.manifold}.b`)?.t.data ?? [0, 0, 0];
      const paint = st.key === 'final' && this.opts.regions;
      // readout colors for many points at once: token regions of the unembedding
      const paintAll = (pts: Float64Array): THREE.Color[] => {
        const n = pts.length / 3;
        if (!paint) return Array.from({ length: n }, () => muted);
        const lg = model.readout(pts, false);
        const V = model.vocab, bg = new THREE.Color(cssVar('panel'));
        const out: THREE.Color[] = [];
        for (let i = 0; i < n; i++) {
          let best = 0;
          for (let v = 1; v < V; v++) if (lg[i * V + v] > lg[i * V + best]) best = v;
          let second = -Infinity;
          for (let v = 0; v < V; v++) if (v !== best) second = Math.max(second, lg[i * V + v]);
          const c = colorOf(catColor(best));
          out.push(lg[i * V + best] - second < 0.3 ? c.lerp(bg, 0.55) : c);
        }
        return out;
      };
      if (model.spec.norm === 'rmsnorm') {
        const geo = new THREE.SphereGeometry(Math.sqrt(3), 64, 32);
        const pa = geo.attributes.position as THREE.BufferAttribute;
        const cols = new Float32Array(pa.count * 3);
        const pts = new Float64Array(pa.count * 3);
        for (let i = 0; i < pa.count; i++) {
          pts.set([pa.getX(i) * g[0] + b[0], pa.getY(i) * g[1] + b[1], pa.getZ(i) * g[2] + b[2]], i * 3);
          pa.setXYZ(i, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
        }
        paintAll(pts).forEach((c, i) => cols.set([c.r, c.g, c.b], i * 3));
        geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
        geo.computeVertexNormals();
        const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: paint ? 0.35 : 0.1, side: THREE.DoubleSide, depthWrite: false }));
        this.overlays.add(mesh);
        const wire = new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(Math.sqrt(3), 24, 12)), new THREE.LineBasicMaterial({ color: muted, transparent: true, opacity: 0.18 }));
        wire.scale.set(g[0], g[1], g[2]);
        wire.position.set(b[0], b[1], b[2]);
        this.overlays.add(wire);
      } else if (model.spec.norm === 'layernorm') {
        const u = new THREE.Vector3(1, -1, 0).normalize(), w = new THREE.Vector3(1, 1, -2).normalize();
        const n = 256, R = Math.sqrt(3);
        const pos: number[] = [], col: number[] = [];
        const fan: number[] = [], fanCol: number[] = [];
        const ctr = new THREE.Vector3(b[0], b[1], b[2]);
        const ptAt = (i: number) => {
          const a = (i / n) * Math.PI * 2;
          const q = u.clone().multiplyScalar(R * Math.cos(a)).addScaledVector(w, R * Math.sin(a));
          return new THREE.Vector3(q.x * g[0] + b[0], q.y * g[1] + b[1], q.z * g[2] + b[2]);
        };
        const ring = new Float64Array(n * 3);
        for (let i = 0; i < n; i++) {
          const q = ptAt(i);
          ring.set([q.x, q.y, q.z], i * 3);
        }
        const ringCol = paintAll(ring);
        for (let i = 0; i < n; i++) {
          const p0 = ptAt(i), p1 = ptAt(i + 1);
          const c = ringCol[i];
          pos.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
          col.push(c.r, c.g, c.b, c.r, c.g, c.b);
          fan.push(ctr.x, ctr.y, ctr.z, p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
          fanCol.push(c.r, c.g, c.b, c.r, c.g, c.b, c.r, c.g, c.b);
        }
        const lg = new THREE.BufferGeometry();
        lg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        lg.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        this.overlays.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true })));
        const fg = new THREE.BufferGeometry();
        fg.setAttribute('position', new THREE.Float32BufferAttribute(fan, 3));
        fg.setAttribute('color', new THREE.Float32BufferAttribute(fanCol, 3));
        this.overlays.add(new THREE.Mesh(fg, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: paint ? 0.3 : 0.07, side: THREE.DoubleSide, depthWrite: false })));
        // the removed direction
        const ax = new THREE.BufferGeometry().setFromPoints([ctr.clone().addScaledVector(new THREE.Vector3(1, 1, 1).normalize(), -R), ctr.clone().addScaledVector(new THREE.Vector3(1, 1, 1).normalize(), R)]);
        this.overlays.add(new THREE.Line(ax, new THREE.LineDashedMaterial({ color: muted, dashSize: 0.08, gapSize: 0.06 })).computeLineDistances());
      }
    }

    // logit lens sphere at any stage
    if (this.opts.lens && exact && st.key !== 'final') {
      const R = this.radius * 0.95;
      const geo = new THREE.IcosahedronGeometry(R, 6);
      const pa = geo.attributes.position as THREE.BufferAttribute;
      const cols = new Float32Array(pa.count * 3);
      const pts = new Float64Array(pa.count * 3);
      for (let i = 0; i < pa.count; i++) pts.set([pa.getX(i), pa.getY(i), pa.getZ(i)], i * 3);
      const lg = model.readout(pts, true);
      const V = model.vocab;
      for (let i = 0; i < pa.count; i++) {
        let best = 0;
        for (let v = 1; v < V; v++) if (lg[i * V + v] > lg[i * V + best]) best = v;
        const c = colorOf(catColor(best));
        cols.set([c.r, c.g, c.b], i * 3);
      }
      geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      this.overlays.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false })));
    }

    // vectors
    if (this.opts.vectors) {
      const vocab = this.session.data.vocab;
      const addArrow = (v: ArrayLike<number>, off: number, color: THREE.Color, text: string, scale = 1) => {
        const p = this.proj!.apply(v, off, new THREE.Vector3()).multiplyScalar(scale);
        const len = p.length();
        if (len < 1e-6) return;
        const arrow = new THREE.ArrowHelper(p.clone().normalize(), new THREE.Vector3(), len, color, Math.min(0.2 * len, this.radius * 0.08), Math.min(0.1 * len, this.radius * 0.04));
        this.overlays.add(arrow);
        this.stage.label(text, p.clone().multiplyScalar(1.08));
      };
      if (st.key === 'resid.embed') {
        const wte = model.p('wte');
        for (let v = 0; v < Math.min(wte.rows, 12); v++) addArrow(wte.data, v * D, colorOf(catColor(v)), vocab[v]);
        if (model.has('wpe')) {
          const wpe = model.p('wpe');
          for (let t = 0; t < wpe.rows; t++) addArrow(wpe.data, t * D, muted, `p${t}`);
        }
      } else if (st.key === this.stages[this.stages.length - 1].key) {
        // unembedding directions: column k of Wu (or row k of wte if tied)
        const V = model.vocab;
        const U = new Float64Array(V * D);
        if (model.spec.tie) U.set(model.p('wte').data);
        else {
          const Wu = model.p('Wu');
          for (let k2 = 0; k2 < V; k2++) for (let j = 0; j < D; j++) U[k2 * D + j] = Wu.data[j * V + k2];
        }
        let mx = 0;
        for (let k2 = 0; k2 < V; k2++) mx = Math.max(mx, Math.hypot(...U.subarray(k2 * D, k2 * D + D)));
        // arrows sized to this stage's point cloud, not the whole scene
        const P = this.P[Math.min(k, this.P.length - 1)];
        let rs = 0;
        for (let i = 0; i < P.length; i += 3) rs = Math.max(rs, Math.hypot(P[i], P[i + 1], P[i + 2]));
        const sc = mx > 0 ? (Math.max(rs, 0.3) * 1.3) / mx : 1;
        for (let k2 = 0; k2 < Math.min(V, 12); k2++) addArrow(U, k2 * D, colorOf(catColor(k2)), `u:${vocab[k2]}`, sc);
      }
    }
    // re-add probe labels (clearLabels would have removed them), placed per frame
    for (const pl of this.probeLabels) if (!pl.l.el.isConnected) this.stage.overlay.append(pl.l.el);
  }

  private lastOverlayStage = -1;

  private frame(dt: number): boolean {
    let anim = false;
    const target = Math.max(0, Math.min(this.stages.length - 1, ui.stage()));
    if (this.playing) {
      this.s += dt * 0.9;
      if (this.s >= this.stages.length - 1) {
        this.s = this.stages.length - 1;
        this.playing = false;
        this.hud.querySelector('button:nth-child(3)')?.classList.remove('on');
      }
      const k = Math.round(this.s);
      if (k !== ui.stage()) ui.stage(k);
      anim = true;
    } else if (ui.stage() !== target) {
      ui.stage(target);
    } else if (Math.abs(this.s - target) > 1e-3) {
      const d = target - this.s;
      this.s += Math.sign(d) * Math.min(Math.abs(d), dt * 3);
      anim = true;
    } else this.s = target;
    const k = Math.round(this.s);
    if (k !== this.lastOverlayStage) {
      this.lastOverlayStage = k;
      this.rebuildLabelsAndOverlays();
    }
    if (anim) this.updateInstances();
    return anim;
  }

  private rebuildLabelsAndOverlays() {
    const pr = probe();
    this.buildProbeLabels(pr.tokens);
    this.buildOverlays();
  }

  private updateInstances() {
    if (!this.cloud || !this.P.length) return;
    const k0 = Math.max(0, Math.min(this.P.length - 1, Math.floor(this.s)));
    const k1 = Math.min(this.P.length - 1, k0 + 1);
    const f = this.s - k0;
    const A = this.P[k0], B = this.P[k1];
    const m = new THREE.Matrix4();
    const r = this.radius * 0.018 * this.opts.size;
    m.makeScale(r, r, r);
    for (let k = 0; k < this.order.length; k++) {
      const i = this.order[k];
      m.setPosition(A[i * 3] + (B[i * 3] - A[i * 3]) * f, A[i * 3 + 1] + (B[i * 3 + 1] - A[i * 3 + 1]) * f, A[i * 3 + 2] + (B[i * 3 + 2] - A[i * 3 + 2]) * f);
      this.cloud.setMatrixAt(k, m);
    }
    this.cloud.instanceMatrix.needsUpdate = true;
    if (this.cloud.instanceColor) this.cloud.instanceColor.needsUpdate = true;
    this.cloud.computeBoundingSphere();
    if (this.probeMesh && this.probeP.length) {
      const PA = this.probeP[k0], PB = this.probeP[k1];
      const PT = PA.length / 3;
      const focus = ui.probePos();
      for (let t = 0; t < PT; t++) {
        const sc = focus === t ? r * 2.2 : r * 1.4;
        const x = PA[t * 3] + (PB[t * 3] - PA[t * 3]) * f, y = PA[t * 3 + 1] + (PB[t * 3 + 1] - PA[t * 3 + 1]) * f, z = PA[t * 3 + 2] + (PB[t * 3 + 2] - PA[t * 3 + 2]) * f;
        m.makeScale(sc, sc, sc);
        m.setPosition(x, y, z);
        this.probeMesh.setMatrixAt(t, m);
        const pl = this.probeLabels.find((p) => p.t === t);
        if (pl) pl.l.pos.set(x, y, z);
      }
      this.probeMesh.instanceMatrix.needsUpdate = true;
      if (this.probeMesh.instanceColor) this.probeMesh.instanceColor.needsUpdate = true;
      this.probeMesh.computeBoundingSphere();
    }
    this.stage.invalidate();
  }

  private pick(e: MouseEvent): { b: number; t: number } | null {
    if (!this.cloud) return null;
    const rc = new THREE.Raycaster();
    rc.setFromCamera(this.stage.ndc(e), this.stage.camera);
    const hits = rc.intersectObject(this.cloud);
    for (const h of hits) {
      if (h.instanceId === undefined) continue;
      const i = this.order[h.instanceId];
      const T = viewSet().T;
      return { b: Math.floor(i / T), t: i % T };
    }
    return null;
  }

  private hover(e: PointerEvent) {
    if (e.buttons) return;
    const p = this.pick(e);
    if (!p) return hideTip();
    const vs = viewSet();
    const d = this.session.data;
    const i = p.b * vs.T + p.t;
    const V = this.session.model.vocab;
    let best = 0;
    for (let v = 1; v < V; v++) if (vs.probs[i * V + v] > vs.probs[i * V + best]) best = v;
    const ex = vs.examples[p.b];
    const lines = [
      `example ${p.b}${vs.isVal[p.b] ? ' (val)' : ' (train)'}: ${d.fmt(ex.x)}`,
      `position ${p.t}, token ${d.vocab[ex.x[p.t]]}`,
      ex.y[p.t] >= 0 ? `target ${d.vocab[ex.y[p.t]]} · predicts ${d.vocab[best]} (${fmtNum(vs.probs[i * V + best])})` : `no target here · predicts ${d.vocab[best]}`,
      'click: make this the probe',
    ];
    showTip(lines.join('\n'), e);
  }

  private click(e: MouseEvent) {
    const p = this.pick(e);
    if (!p) return;
    ui.probeTokens(null);
    ui.probe(p.b);
    ui.probePos(p.t);
  }
}
