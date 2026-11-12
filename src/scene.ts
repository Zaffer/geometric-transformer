// The 3D picture. One sheet: the plane of the focused position's point.
//
//   colour  = which cell a point is in (who it attends to / which neurons fire / what is predicted)
//   height  = the numbers that decide that cell, one sheet each; the top one is drawn solid
//   floor   = the same sheet flattened: the 2D cell diagram the solid surface projects down to
//   flow    = slides the sheet from where points start, to after attention, to after the MLP

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { type Field, sample } from './field';
import { type Config, type Layout, type Trace, at, probe } from './model';

export type Mode = 'attention' | 'mlp' | 'output';

export interface ViewState {
  mode: Mode;
  neuron: number; // the neuron whose sheet the MLP mode lifts
  flow: number; // 0 = embedded, 1 = after attention, 2 = after MLP
  lift: number; // 0 = flat diagram, 1 = full height
  creases: boolean;
  sheets: boolean;
}

export const PALETTE = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#EE8866'];
const PAPER = '#f6f4ee';
const INK = '#1c1b19';
const FLOOR = -0.62; // where the flattened copy sits, in scene units, at full lift

const G = 151; // grid points per side
const WIRE_STEP = 10;
const WINDOW = 1.3; // half-width of the visible window, in scene units

export class Scene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(36, 1, 0.05, 50);
  private controls: OrbitControls;

  private field?: Field;
  private surfPos = new Float32Array(G * G * 3);
  private floorPos = new Float32Array(G * G * 3);
  private colors = new Float32Array(G * G * 3);
  private env = new Float32Array(G * G);
  private surf: THREE.Mesh;
  private floor: THREE.Mesh;
  private wires: THREE.LineSegments[] = [];
  private creaseAll: LineSegments2;
  private creaseSel: LineSegments2;
  private frame: THREE.LineSegments;
  private links: THREE.LineSegments;
  private stems: THREE.LineSegments;
  private sprites: THREE.Sprite[] = [];
  private textures = new Map<string, THREE.Texture>();
  private clipLow = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  private clipHigh = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1.9);
  // the view is a window onto the plane: the sheet is cut off at its edges
  private window = [[1, 0], [-1, 0], [0, 1], [0, -1]].map(([x, y]) => new THREE.Plane(new THREE.Vector3(x, y, 0), WINDOW));

  // eased so the picture does not jump while weights move
  private cx = 0; private cy = 0; private R = 0; // the square of plane that is sampled
  private wx = 0; private wy = 0; private wR = 0; // where the window looks: follows the points as they flow
  private z0 = 0; private zs = 1; private zKey = '';

  constructor(private canvas: HTMLCanvasElement, private c: Config, private L: Layout) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(PAPER);
    this.renderer.localClippingEnabled = true;

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.8, -4.7, 3.2);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, -0.05);
    this.controls.enableDamping = true;
    this.controls.update();

    this.scene.add(new THREE.AmbientLight(0xffffff, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(-1.5, -1, 3);
    this.scene.add(sun);

    // the sheet and its flattened copy share topology and colours
    const index: number[] = [];
    for (let j = 0; j < G - 1; j++)
      for (let i = 0; i < G - 1; i++) {
        const a = j * G + i, b = a + 1, d = a + G, e = d + 1;
        index.push(a, b, e, a, e, d);
      }
    const colorAttr = new THREE.BufferAttribute(this.colors, 3);
    const mesh = (pos: Float32Array, material: THREE.Material) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', colorAttr);
      g.setIndex(index);
      const m = new THREE.Mesh(g, material);
      m.frustumCulled = false;
      this.scene.add(m);
      return m;
    };
    const offset = { polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 };
    const shared = { vertexColors: true, side: THREE.DoubleSide, clippingPlanes: this.window, ...offset };
    this.surf = mesh(this.surfPos, new THREE.MeshLambertMaterial(shared));
    this.floor = mesh(this.floorPos, new THREE.MeshBasicMaterial(shared));

    // one wire sheet per key / token / neuron: a coarse grid of lines over the same vertices
    const wireIndex: number[] = [];
    for (let j = 0; j < G; j += WIRE_STEP)
      for (let i = 0; i < G - 1; i++) wireIndex.push(j * G + i, j * G + i + 1, i * G + j, (i + 1) * G + j);
    for (let k = 0; k < Math.max(c.V, c.T); k++) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(G * G * 3), 3));
      g.setIndex(wireIndex);
      const w = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
        transparent: true, opacity: 0.55, clippingPlanes: [this.clipLow, this.clipHigh, ...this.window],
      }));
      w.frustumCulled = false;
      this.wires.push(w);
      this.scene.add(w);
    }

    const fat = (color: string, width: number) => {
      const l = new LineSegments2(new LineSegmentsGeometry(), new LineMaterial({ color, linewidth: width, clippingPlanes: this.window }));
      l.frustumCulled = false;
      this.scene.add(l);
      return l;
    };
    this.creaseAll = fat(INK, 1.4);
    this.creaseSel = fat(INK, 3.6);

    const lines = (n: number, material: THREE.LineBasicMaterial) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 6), 3));
      g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 6), 3));
      const l = new THREE.LineSegments(g, material);
      l.frustumCulled = false;
      this.scene.add(l);
      return l;
    };
    this.frame = lines(6, new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.35 }));
    this.links = lines(c.T, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }));
    this.stems = lines(c.V + c.T, new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.3 }));
    this.links.renderOrder = 5;

    for (let i = 0; i < c.V + c.T; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, sizeAttenuation: false }));
      s.renderOrder = 10;
      this.sprites.push(s);
      this.scene.add(s);
    }

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement!);
    this.resize();
  }

  private resize() {
    const p = this.canvas.parentElement!, w = p.clientWidth, h = p.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Recompute the whole picture for these weights. */
  update(th: Float64Array, t: Trace, tokens: number[], pos: number, v: ViewState) {
    const started = performance.now();
    const { c, L } = this, B = L.blocks, N = G * G;
    const pr = probe(c, L, th, t, pos);

    // every point in the picture: the tokens this position could hold, and the earlier positions.
    // Each has a place at the start, a place at the current flow, and a height.
    const a = Math.min(1, Math.max(0, v.flow)), b = Math.min(1, Math.max(0, v.flow - 1));
    const flowed = (s: number, r1: number, r2: number) => s + a * (r1 - s) + b * (r2 - r1);
    const height = (o: Trace) => {
      if (v.mode === 'mlp') return Math.max(0, o.pre[v.neuron]);
      const arr = v.mode === 'attention' ? o.s : o.logits, n = v.mode === 'attention' ? pos + 1 : c.V;
      let m = -Infinity;
      for (let k = 0; k < n; k++) m = Math.max(m, arr[k]);
      return m;
    };
    interface Dot { x0: number; y0: number; x: number; y: number; h: number; token: number; key: number }
    const dots: Dot[] = [];
    for (let j = 0; j < pos; j++) {
      const h = height(pr.eval(t.x[2 * j], t.x[2 * j + 1]));
      dots.push({
        x0: t.x[2 * j], y0: t.x[2 * j + 1], h, token: tokens[j], key: j,
        x: flowed(t.x[2 * j], t.r1[2 * j], t.r2[2 * j]), y: flowed(t.x[2 * j + 1], t.r1[2 * j + 1], t.r2[2 * j + 1]),
      });
    }
    for (let w = 0; w < c.V; w++) {
      const o = pr.eval(th[at(B.E, w, 0)] + th[at(B.P, pos, 0)], th[at(B.E, w, 1)] + th[at(B.P, pos, 1)]);
      dots.push({ x0: o.x[0], y0: o.x[1], h: height(o), token: w, key: -1, x: flowed(o.x[0], o.r1[0], o.r2[0]), y: flowed(o.x[1], o.r1[1], o.r2[1]) });
    }
    const fit = (xs: number[], ys: number[]) => {
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      return [(x0 + x1) / 2, (y0 + y1) / 2, Math.max(0.6, 0.72 * Math.max(x1 - x0, y1 - y0))];
    };
    const [tcx, tcy, tR] = fit(dots.map((d) => d.x0), dots.map((d) => d.y0));
    const [twx, twy, twR] = fit(dots.map((d) => d.x), dots.map((d) => d.y));
    if (!this.R) { this.cx = tcx; this.cy = tcy; this.R = tR; this.wx = twx; this.wy = twy; this.wR = twR; }
    const ease = 0.12;
    this.cx += ease * (tcx - this.cx); this.cy += ease * (tcy - this.cy); this.R += ease * (tR - this.R);
    this.wx += ease * (twx - this.wx); this.wy += ease * (twy - this.wy); this.wR += ease * (twR - this.wR);
    const { cx, cy, R, wx, wy, wR } = this;

    const f = (this.field = sample(c, L, th, t, pos, G, cx, cy, R, this.field));

    // which numbers are lifted
    const src = v.mode === 'attention' ? f.score : v.mode === 'output' ? f.logit : f.pre;
    const stride = v.mode === 'attention' ? f.nKeys : v.mode === 'output' ? c.V : c.H;
    const K = v.mode === 'mlp' ? 1 : stride, k0 = v.mode === 'mlp' ? v.neuron : 0;
    const env = this.env;
    let lo = Infinity, hi = -Infinity, amax = 1e-6;
    for (let n = 0; n < N; n++) {
      let e = v.mode === 'mlp' ? 0 : -Infinity;
      for (let k = 0; k < K; k++) e = Math.max(e, src[n * stride + k0 + k]);
      if (v.mode === 'mlp') amax = Math.max(amax, Math.abs(src[n * stride + k0]));
      env[n] = e;
      if (e < lo) lo = e;
      if (e > hi) hi = e;
    }
    const tz0 = v.mode === 'mlp' ? 0 : lo;
    const tzs = 0.95 / (v.mode === 'mlp' ? amax : Math.max(hi - lo, 1e-6));
    const key = `${v.mode}${k0}${pos}`;
    if (key !== this.zKey) { this.zKey = key; this.z0 = tz0; this.zs = tzs; }
    this.z0 += 0.12 * (tz0 - this.z0); this.zs += 0.12 * (tzs - this.zs);
    const zOf = (val: number) => (val - this.z0) * this.zs * v.lift;
    const floorZ = FLOOR * v.lift;

    // where the sheet is: slid from start (0) to after attention (1) to after the MLP (2)
    const { surfPos, floorPos, colors } = this;
    for (let j = 0, n = 0; j < G; j++) {
      const y = cy + R * ((2 * j) / (G - 1) - 1);
      for (let i = 0; i < G; i++, n++) {
        const x = cx + R * ((2 * i) / (G - 1) - 1);
        const X = (flowed(x, f.r1[2 * n], f.r2[2 * n]) - wx) / wR, Y = (flowed(y, f.r1[2 * n + 1], f.r2[2 * n + 1]) - wy) / wR;
        surfPos[3 * n] = X; surfPos[3 * n + 1] = Y; surfPos[3 * n + 2] = zOf(env[n]);
        floorPos[3 * n] = X; floorPos[3 * n + 1] = Y; floorPos[3 * n + 2] = floorZ;
        const col = v.mode === 'attention' ? cellColor(f.topKey[n]) : v.mode === 'output' ? cellColor(f.topTok[n]) : patternColor(f.pattern[n]);
        colors[3 * n] = col.r; colors[3 * n + 1] = col.g; colors[3 * n + 2] = col.b;
      }
    }
    this.surf.geometry.attributes.position.needsUpdate = true;
    this.surf.geometry.attributes.color.needsUpdate = true;
    this.surf.geometry.computeVertexNormals();
    this.floor.geometry.attributes.position.needsUpdate = true;
    const lifted = v.lift > 0.03;
    this.floor.visible = lifted;

    // the individual sheets, as wire grids
    this.clipLow.constant = -floorZ - 0.01;
    this.wires.forEach((w, k) => {
      w.visible = v.sheets && lifted && k < K;
      if (!w.visible) return;
      const p = w.geometry.attributes.position.array as Float32Array;
      for (let n = 0; n < N; n++) {
        p[3 * n] = surfPos[3 * n]; p[3 * n + 1] = surfPos[3 * n + 1]; p[3 * n + 2] = zOf(src[n * stride + k0 + k]);
      }
      w.geometry.attributes.position.needsUpdate = true;
      (w.material as THREE.LineBasicMaterial).color.set(v.mode === 'mlp' ? INK : PALETTE[k % PALETTE.length]);
    });

    // every neuron's fold line, on the sheet and on its flattened copy
    const all: number[] = [], sel: number[] = [];
    if (v.creases)
      for (let h = 0; h < c.H; h++) {
        const out = h === v.neuron && v.mode === 'mlp' ? sel : all;
        contour(f.pre, c.H, h, surfPos, out);
        if (lifted) contour(f.pre, c.H, h, floorPos, out);
      }
    setSegments(this.creaseAll, all);
    setSegments(this.creaseSel, sel);

    // the window's edge, and the plane's own axes through its origin
    const fr = this.frame.geometry.attributes.position.array as Float32Array;
    const seg = (i: number, ax: number, ay: number, bx: number, by: number) => fr.set([ax, ay, floorZ, bx, by, floorZ], i * 6);
    const E = WINDOW;
    seg(0, -E, -E, E, -E); seg(1, E, -E, E, E); seg(2, E, E, -E, E); seg(3, -E, E, -E, -E);
    const ox = Math.min(E, Math.max(-E, -wx / wR)), oy = Math.min(E, Math.max(-E, -wy / wR));
    seg(4, ox, -E, ox, E); seg(5, -E, oy, E, oy);
    this.frame.geometry.attributes.position.needsUpdate = true;

    // the points: every token this position could hold (hollow), the one it does hold, and the earlier positions.
    // In the attention picture colour means "which position", so real points take their position's colour.
    const stem = this.stems.geometry.attributes.position.array as Float32Array;
    const link = this.links.geometry.attributes.position.array as Float32Array;
    const linkCol = this.links.geometry.attributes.color.array as Float32Array;
    const att = v.mode === 'attention', hue = (k: number) => PALETTE[k % PALETTE.length];
    const where: number[][] = [];
    let here = [0, 0, 0];
    let s = 0;
    for (const d of dots) {
      const X = (d.x - wx) / wR, Y = (d.y - wy) / wR, Z = zOf(d.h);
      const earlier = d.key >= 0, actual = !earlier && d.token === tokens[pos];
      if (earlier) where.push([X, Y, Z]);
      if (actual) here = [X, Y, Z];
      const tex = earlier ? this.label(String(d.token), hue(att ? d.key : d.token), INK, '#fff', `${d.key}`)
        : actual ? this.label(String(d.token), hue(att ? pos : d.token), INK, '#fff', '', true)
        : this.label(String(d.token), '#fff', att ? '#8a8578' : hue(d.token), INK);
      const sp = this.sprites[s];
      sp.visible = Math.abs(X) <= WINDOW && Math.abs(Y) <= WINDOW;
      sp.position.set(X, Y, Z);
      sp.material.map = tex;
      sp.scale.setScalar(earlier ? 0.036 : actual ? 0.044 : 0.03);
      stem.set(sp.visible ? [X, Y, Z, X, Y, floorZ] : [0, 0, 0, 0, 0, 0], s * 6);
      s++;
    }
    for (; s < this.sprites.length; s++) { this.sprites[s].visible = false; stem.fill(0, s * 6, s * 6 + 6); }
    this.stems.visible = lifted;
    this.stems.geometry.attributes.position.needsUpdate = true;

    // attention: a line from the focused point to each earlier point, as dark as its weight
    const paper = new THREE.Color(PAPER), ink = new THREE.Color(INK), mix = new THREE.Color();
    link.fill(0);
    for (let j = 0; j < pos; j++) {
      link.set([...here, ...where[j]], j * 6);
      mix.copy(paper).lerp(ink, Math.min(1, 0.08 + t.a[pos * t.n + j]));
      linkCol.set([mix.r, mix.g, mix.b, mix.r, mix.g, mix.b], j * 6);
    }
    this.links.geometry.attributes.position.needsUpdate = true;
    this.links.geometry.attributes.color.needsUpdate = true;
    this.ms += 0.05 * (performance.now() - started - this.ms);
  }

  /** Smoothed time one update takes, in milliseconds. */
  ms = 0;

  /** A round marker with a glyph, drawn once and cached. */
  private label(text: string, fill: string, ring: string, ink: string, sub = '', bold = false): THREE.Texture {
    const key = [text, fill, ring, ink, sub, bold].join('|');
    let tex = this.textures.get(key);
    if (tex) return tex;
    const S = 128, cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const g = cv.getContext('2d')!;
    g.beginPath(); g.arc(S / 2, S / 2, S * 0.36, 0, 2 * Math.PI);
    g.fillStyle = fill; g.fill();
    g.lineWidth = bold ? 12 : 8; g.strokeStyle = ring; g.stroke();
    g.fillStyle = ink; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.font = `700 ${S * 0.42}px ui-sans-serif, system-ui, sans-serif`;
    g.fillText(text, S / 2, S / 2 + 3);
    if (sub) {
      g.font = `600 ${S * 0.2}px ui-sans-serif, system-ui, sans-serif`;
      g.fillStyle = INK;
      g.fillText(sub, S * 0.9, S * 0.88);
    }
    tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.textures.set(key, tex);
    return tex;
  }
}

const cellColors = PALETTE.map((h) => new THREE.Color(h).lerp(new THREE.Color('#ffffff'), 0.22));
const cellColor = (k: number) => cellColors[k % cellColors.length];

const patternColors = new Map<number, THREE.Color>();
/** One colour per on/off pattern of the neurons: one colour per polytope. */
export function patternColor(mask: number): THREE.Color {
  let col = patternColors.get(mask);
  if (!col) patternColors.set(mask, (col = new THREE.Color().setHSL(patternHue(mask) / 360, 0.5, 0.74)));
  return col;
}
export const patternHue = (mask: number) => (mask * 137.508 + 20) % 360;

/** Segments where value h of `vals` crosses zero, placed on the vertices `pos` (marching squares). */
function contour(vals: Float32Array, stride: number, h: number, pos: Float32Array, out: number[]) {
  const cut = (na: number, nb: number, va: number, vb: number) => {
    const u = va / (va - vb);
    out.push(
      pos[3 * na] + u * (pos[3 * nb] - pos[3 * na]),
      pos[3 * na + 1] + u * (pos[3 * nb + 1] - pos[3 * na + 1]),
      pos[3 * na + 2] + u * (pos[3 * nb + 2] - pos[3 * na + 2]) + 0.002,
    );
  };
  for (let j = 0; j < G - 1; j++)
    for (let i = 0; i < G - 1; i++) {
      const n0 = j * G + i, n1 = n0 + 1, n3 = n0 + G, n2 = n3 + 1;
      const v0 = vals[n0 * stride + h], v1 = vals[n1 * stride + h], v2 = vals[n2 * stride + h], v3 = vals[n3 * stride + h];
      const p0 = v0 > 0, p1 = v1 > 0, p2 = v2 > 0, p3 = v3 > 0;
      if (p0 === p1 && p1 === p2 && p2 === p3) continue;
      const before = out.length;
      if (p0 !== p1) cut(n0, n1, v0, v1);
      if (p1 !== p2) cut(n1, n2, v1, v2);
      if (p2 !== p3) cut(n2, n3, v2, v3);
      if (p3 !== p0) cut(n3, n0, v3, v0);
      if ((out.length - before) % 6) out.length = before; // a crossing exactly on a vertex: skip the cell
    }
}

function setSegments(line: LineSegments2, pts: number[]) {
  line.visible = pts.length > 0;
  if (!line.visible) return;
  line.geometry.dispose();
  line.geometry = new LineSegmentsGeometry().setPositions(pts);
}
