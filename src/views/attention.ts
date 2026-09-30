// Attention. Left: every head's attention pattern for the probe sequence
// and averaged over the dataset. Right, for the selected head and query:
//   query/key space: attention = softmax of dot products, so a key gets
//     weight by how far it lies along the query direction;
//   value space: the head's output is a weighted average of the value
//     vectors, so it always lies inside their convex hull.

import * as THREE from 'three';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Session } from '../app/session';
import { probe, viewSet } from '../app/inspect';
import { rt, ui } from '../app/state';
import { Heatmap } from '../ui/heatmap';
import { catColor, css, cssVar, token } from '../ui/palette';
import { fmtNum } from '../ui/controls';
import { pca } from '../engine/linalg';
import { Stage3D, colorOf, disposeTree } from './stage3d';
import type { View } from './view';

export function focusLayer(session: Session, kind: 'attn' | 'mlp'): number {
  const s = session.model.spec;
  const m = ui.focus().match(/^L(\d+)\./);
  const want = m ? Number(m[1]) : -1;
  if (want >= 0 && want < s.nLayer && s.layers[want][kind]) return want;
  for (let l = 0; l < s.nLayer; l++) if (s.layers[l][kind]) return l;
  return -1;
}

export class AttentionView implements View {
  readonly id = 'attention';
  readonly title = 'attention';
  readonly el = document.createElement('div');
  private session: Session;
  private left = document.createElement('div');
  private right = document.createElement('div');
  private qk: Stage3D;
  private vv: Stage3D;
  private qkGroup = new THREE.Group();
  private vGroup = new THREE.Group();
  private active = false;
  private dirty = true;
  private qkCap!: HTMLDivElement;
  private vCap!: HTMLDivElement;
  private fitted = false;

  constructor(session: Session) {
    this.session = session;
    this.el.classList.add('flexview');
    Object.assign(this.left.style, { width: '42%', minWidth: '240px', overflow: 'auto', padding: '8px', borderRight: '1px solid var(--border)' });
    Object.assign(this.right.style, { flex: '1', display: 'flex', flexDirection: 'column', minWidth: '0' });
    const top = document.createElement('div');
    const bot = document.createElement('div');
    for (const d of [top, bot]) Object.assign(d.style, { position: 'relative', flex: '1', minHeight: '0', borderBottom: '1px solid var(--border)' });
    this.right.append(top, bot);
    this.el.append(this.left, this.right);
    this.qk = new Stage3D(top);
    this.vv = new Stage3D(bot);
    this.qk.scene.add(this.qkGroup);
    this.vv.scene.add(this.vGroup);
    this.qkCap = this.caption(this.qk, 'query / key space');
    this.vCap = this.caption(this.vv, 'value space');

    effect(() => {
      rt.modelVersion();
      rt.paramsVersion();
      ui.focus();
      ui.probe();
      ui.probeTokens();
      ui.probePos();
      ui.head();
      this.dirty = true;
      if (untracked(() => this.active)) untracked(() => this.render());
    });
  }

  private caption(st: Stage3D, title: string): HTMLDivElement {
    const h = document.createElement('div');
    h.className = 'hud';
    h.style.left = '6px';
    h.style.top = '6px';
    h.style.maxWidth = '300px';
    h.style.opacity = '0.92';
    h.style.whiteSpace = 'pre-wrap';
    h.textContent = title;
    st.overlay.append(h);
    return h;
  }

  activate(): void {
    this.active = true;
    this.qk.start();
    this.vv.start();
    if (this.dirty) this.render();
  }

  deactivate(): void {
    this.active = false;
    this.qk.stop();
    this.vv.stop();
  }

  themeChanged(): void {
    this.qk.themeChanged();
    this.vv.themeChanged();
    this.dirty = true;
    if (this.active) this.render();
  }

  private render() {
    this.dirty = false;
    const model = this.session.model, s = model.spec, d = this.session.data;
    const L = focusLayer(this.session, 'attn');
    this.left.replaceChildren();
    if (L < 0) {
      this.left.textContent = 'No attention in this model (every layer has attention switched off).';
      for (const g of [this.qkGroup, this.vGroup]) for (const c of [...g.children]) disposeTree(c);
      return;
    }
    const pr = probe();
    const vs = viewSet();
    const T = pr.tokens.length;
    const rowL = Array.from(pr.tokens, (id, i) => `${i}:${d.vocab[id]}`);
    const H = s.nHead;
    const head = Math.min(ui.head(), H - 1);
    const qpos = ui.probePos() >= 0 && ui.probePos() < T ? ui.probePos() : T - 1;
    const W = Math.max(160, this.left.clientWidth - 24);

    const h = document.createElement('h3');
    h.style.margin = '0 0 4px';
    h.style.color = 'var(--accent)';
    h.textContent = `layer ${L} attention`;
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = 'Row = query position, column = key position; each row sums to 1. Click a row to pick the query, click a head to pick it. Pick a layer in the map.';
    this.left.append(h, note);

    for (let hh = 0; hh < H; hh++) {
      const box = document.createElement('div');
      box.style.margin = '8px 0';
      box.style.padding = '4px';
      box.style.border = `1px solid ${hh === head ? 'var(--sel)' : 'var(--border)'}`;
      box.style.borderRadius = '4px';
      const t = document.createElement('div');
      t.textContent = `head ${hh}${hh === head ? ' (selected)' : ''}`;
      t.style.cursor = 'pointer';
      t.onclick = () => ui.head(hh);
      box.append(t);
      const probs = pr.fwd.trace.get(`L${L}.attn.probs.h${hh}`)!;
      const scores = pr.fwd.trace.get(`L${L}.attn.scores.h${hh}`)!;
      const hm = new Heatmap({
        rows: T, cols: T, rowLabels: rowL, colLabels: rowL.map((_, i) => String(i)), editable: false, positive: true, width: W / (H > 1 ? 1 : 1),
        maxCell: 30, name: `p.h${hh}`,
        describe: (i) => `query ${Math.floor(i / T)} → key ${i % T}: score ${fmtNum(scores.data[i])}`,
        onPick: (i) => {
          ui.head(hh);
          ui.probePos(Math.floor(i / T));
        },
      });
      hm.set(probs.data, { scale: 1 });
      hm.highlight(hh === head ? qpos : -1, -1);
      const lab2 = document.createElement('div');
      lab2.className = 'note';
      lab2.textContent = `mean over ${vs.B} examples`;
      const mean = new Float64Array(T * T);
      const all = vs.fwd.trace.get(`L${L}.attn.probs.h${hh}`)!;
      const VT = vs.T;
      if (VT === T) {
        for (let b = 0; b < vs.B; b++) for (let k = 0; k < T * T; k++) mean[k] += all.data[b * T * T + k] / vs.B;
      }
      const hm2 = new Heatmap({ rows: T, cols: T, rowLabels: rowL.map((_, i) => String(i)), editable: false, positive: true, width: W, maxCell: 14, name: `mean.h${hh}` });
      hm2.set(mean, { scale: 1 });
      box.append(hm.el, lab2, hm2.el);
      this.left.append(box);
    }

    this.renderSpaces(L, head, qpos);
  }

  private renderSpaces(L: number, head: number, qpos: number) {
    const model = this.session.model, s = model.spec, d = this.session.data;
    const pr = probe();
    const T = pr.tokens.length, dh = s.dHead, H = s.nHead;
    const a = `L${L}.attn`;
    const sl = (key: string) => {
      const t = pr.fwd.trace.get(key)!;
      const out = new Float64Array(T * dh);
      for (let i = 0; i < T; i++) for (let j = 0; j < dh; j++) out[i * dh + j] = t.data[i * H * dh + head * dh + j];
      return out;
    };
    const q = sl(`${a}.q`), k = sl(`${a}.k`), v = sl(`${a}.v`);
    const z = pr.fwd.trace.get(`${a}.z.h${head}`)!.data;
    const p = pr.fwd.trace.get(`${a}.probs.h${head}`)!.data;

    // map dh dims → 3D (exact if dh <= 3)
    const mk = (pts: Float64Array[]): ((x: ArrayLike<number>, o: number) => THREE.Vector3) => {
      if (dh <= 3) return (x, o) => new THREE.Vector3(x[o], dh > 1 ? x[o + 1] : 0, dh > 2 ? x[o + 2] : 0);
      const n = pts.reduce((s2, c) => s2 + c.length, 0) / dh;
      const all = new Float64Array(n * dh);
      let off = 0;
      for (const c of pts) {
        all.set(c, off);
        off += c.length;
      }
      const { axes } = pca(all, n, dh, 3);
      return (x, o) => {
        const c = axes.map((ax) => ax.reduce((acc, w, j) => acc + w * x[o + j], 0));
        return new THREE.Vector3(c[0], c[1], c[2]);
      };
    };
    const projQK = mk([q, k]);
    const projV = mk([v]);
    const muted = new THREE.Color(cssVar('muted'));
    const sel = new THREE.Color(cssVar('sel'));
    const vocab = d.vocab;

    // ---- query/key space ----
    for (const c of [...this.qkGroup.children]) disposeTree(c);
    this.qk.clearLabels();
    const scale = s.attnScaleMult / Math.sqrt(dh);
    let rq = 0;
    const kpts: THREE.Vector3[] = [];
    for (let j = 0; j < T; j++) {
      const kp = projQK(k, j * dh);
      kpts.push(kp);
      rq = Math.max(rq, kp.length());
    }
    const qp = projQK(q, qpos * dh);
    rq = Math.max(rq, qp.length(), 0.3);
    const dot = new THREE.IcosahedronGeometry(rq * 0.035, 1);
    for (let j = 0; j < T; j++) {
      const masked = s.causal && j > qpos;
      const m = new THREE.Mesh(dot, new THREE.MeshLambertMaterial({ color: masked ? muted : colorOf(catColor(pr.tokens[j])), transparent: masked, opacity: masked ? 0.3 : 1 }));
      m.position.copy(kpts[j]);
      this.qkGroup.add(m);
      this.qk.label(`k${j}:${vocab[pr.tokens[j]]}`, kpts[j].clone().multiplyScalar(1.0).add(new THREE.Vector3(0, 0, rq * 0.07)), masked ? css(token('muted')) : undefined);
      if (!masked) {
        // line from origin to key, opacity = attention weight
        const w = p[qpos * T + j];
        const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), kpts[j]]);
        this.qkGroup.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: sel, transparent: true, opacity: 0.15 + 0.85 * w })));
      }
    }
    // query arrow
    if (qp.length() > 1e-6) {
      this.qkGroup.add(new THREE.ArrowHelper(qp.clone().normalize(), new THREE.Vector3(), qp.length(), sel, rq * 0.1, rq * 0.05));
      this.qk.label(`q${qpos}:${vocab[pr.tokens[qpos]]}`, qp.clone().multiplyScalar(1.1), css(token('sel')));
      // the plane of zero score: keys on the far side get suppressed
      if (dh === 3) {
        const n = qp.clone().normalize();
        const pg = new THREE.PlaneGeometry(rq * 2.2, rq * 2.2);
        const pm = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ color: sel, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false }));
        pm.lookAt(n);
        this.qkGroup.add(pm);
      }
    }
    this.axesInto(this.qkGroup, rq);
    this.qkCap.textContent = `query / key · head ${head} · query ${qpos} · ${dh <= 3 ? 'exact' : 'PCA projection'}\nscore = q·k × ${fmtNum(scale)}: keys far along the query arrow win (faint plane: score 0). Line brightness = attention weight.`;

    // ---- value space ----
    for (const c of [...this.vGroup.children]) disposeTree(c);
    this.vv.clearLabels();
    let rv = 0.3;
    const vpts: THREE.Vector3[] = [];
    for (let j = 0; j < T; j++) {
      const vp = projV(v, j * dh);
      vpts.push(vp);
      rv = Math.max(rv, vp.length());
    }
    const vdot = new THREE.IcosahedronGeometry(rv * 0.035, 1);
    const live: THREE.Vector3[] = [];
    for (let j = 0; j < T; j++) {
      const masked = s.causal && j > qpos;
      const w = p[qpos * T + j];
      const m = new THREE.Mesh(vdot, new THREE.MeshLambertMaterial({ color: masked ? muted : colorOf(catColor(pr.tokens[j])), transparent: masked, opacity: masked ? 0.3 : 1 }));
      m.position.copy(vpts[j]);
      m.scale.setScalar(masked ? 0.7 : 0.7 + 1.3 * Math.sqrt(w));
      this.vGroup.add(m);
      this.vv.label(`v${j}:${vocab[pr.tokens[j]]} ${masked ? '' : fmtNum(w)}`, vpts[j].clone().add(new THREE.Vector3(0, 0, rv * 0.08)), masked ? css(token('muted')) : undefined);
      if (!masked) live.push(vpts[j]);
    }
    // convex hull of the values this query can see
    try {
      if (live.length >= 4 && dh >= 3) {
        const hg = new ConvexGeometry(live);
        this.vGroup.add(new THREE.Mesh(hg, new THREE.MeshBasicMaterial({ color: sel, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false })));
        this.vGroup.add(new THREE.LineSegments(new THREE.EdgesGeometry(hg), new THREE.LineBasicMaterial({ color: sel, transparent: true, opacity: 0.5 })));
      } else if (live.length >= 2) {
        const pts = [...live, live[0]];
        this.vGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: sel, transparent: true, opacity: 0.5 })));
      }
    } catch {
      /* degenerate hull (coplanar values) */
    }
    // the output: weighted average
    const zp = projV(z, qpos * dh);
    const zm = new THREE.Mesh(new THREE.OctahedronGeometry(rv * 0.06), new THREE.MeshLambertMaterial({ color: sel }));
    zm.position.copy(zp);
    this.vGroup.add(zm);
    this.vv.label(`z${qpos} = Σ p·v`, zp.clone().add(new THREE.Vector3(0, 0, -rv * 0.1)), css(token('sel')));
    this.axesInto(this.vGroup, rv);
    this.vCap.textContent = `value · head ${head} · ${dh <= 3 ? 'exact' : 'PCA projection'}\nz (diamond) = Σⱼ pⱼ vⱼ lies inside the hull of the visible values; Wo then writes it into the stream.`;

    if (!this.fitted) {
      this.fitted = true;
      this.qk.fit(new THREE.Vector3(), rq * 1.2);
      this.vv.fit(new THREE.Vector3(), rv * 1.2);
    }
    this.qk.invalidate();
    this.vv.invalidate();
  }

  private axesInto(g: THREE.Group, r: number) {
    const pos: number[] = [];
    for (let i = 0; i < 3; i++) {
      const a = [0, 0, 0], b = [0, 0, 0];
      a[i] = -r * 1.1;
      b[i] = r * 1.1;
      pos.push(...a, ...b);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('border')) })));
  }
}
