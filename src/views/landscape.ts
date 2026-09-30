// Phase 2: the loss landscape, and the cells the activations carve into it.
//
// Pick 1–3 directions through parameter space. Every point of the slice is
// a complete network; we evaluate its loss on the whole dataset and hash its
// activation pattern (which ReLUs fire, for every input). Points with the
// same hash are in the same cell: there the network computes one fixed
// piecewise-linear function of its parameters, so the loss is smooth inside
// a cell and can only kink or jump on the cell walls.
//
//   1D   a line (or the actual training path) with the walls it crosses
//   2D   a map and a surface: loss height, cells painted, walls drawn exactly
//   3D   a volume: movable slice planes, a loss level set painted by cell,
//        and the cell walls as a point cloud
//
// Click anywhere to move the model to that point; every other view follows.

import * as THREE from 'three';
import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Session, Snapshot } from '../app/session';
import { LandscapePool } from '../app/landscapePool';
import { rt, ui } from '../app/state';
import {
  EVAL_CAP,
  EVAL_STRIDE,
  coarseToFine,
  dot,
  orthogonalize,
  projectToSlice,
  randomDirection,
  tensorDirection,
  thetaAt,
  trajectoryPCA,
  unitDirection,
  unpackEval,
  type CarveKey,
  type DirKind,
  type DirSpec,
  type PointEval,
} from '../engine/landscape';
import { css, cssVar, sequential, token, type RGB } from '../ui/palette';
import { fmtNum } from '../ui/controls';
import { hideTip, showTip } from '../ui/heatmap';
import { Stage3D, disposeTree } from './stage3d';
import { isosurface } from './iso';
import type { View } from './view';

type Dims = 'path' | 1 | 2 | 3;
type Carve = CarveKey | 'all' | 'none';
type Paint = 'cells' | 'loss' | 'acc' | 'flips';

interface Slice {
  k: number;
  n: number;
  R: number;
  center: Float64Array;
  dirs: Float64Array[];
  labels: string[];
  coords: Float64Array;
  res: Float64Array;
  /** 0 = computed exactly; otherwise the stride of the coarse point standing in */
  level: Uint8Array;
  proxy: Int32Array;
  done: number;
  total: number;
  modelVersion: number;
  paramsVersion: number;
  spec: string;
  t0: number;
  elapsed: number;
}

const DIR_KINDS: [DirKind, string][] = [
  ['random', 'random (filter-normalized)'],
  ['weight', 'one weight'],
  ['tensor', 'scale a tensor'],
  ['gradient', '−gradient'],
  ['hessian', 'Hessian eigenvector'],
  ['pca', 'training-path PCA'],
  ['init', 'toward init'],
];

export function hashColor(h: number, h2: number, shade: number): RGB {
  const hue = ((h >>> 0) % 3600) / 3600;
  const sat = 0.45 + (((h2 >>> 0) % 100) / 100) * 0.25;
  const l = 0.28 + 0.42 * shade;
  const f = (n: number) => {
    const k = (n + hue * 12) % 12;
    const a = sat * Math.min(l, 1 - l);
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

export class LandscapeView implements View {
  readonly id = 'landscape';
  readonly title = 'loss landscape';
  readonly el = document.createElement('div');
  private session: Session;
  private pool: LandscapePool | null = null;
  private side = document.createElement('div');
  private main = document.createElement('div');
  private mapWrap = document.createElement('div');
  private mapCv = document.createElement('canvas');
  private stageWrap = document.createElement('div');
  private stage: Stage3D;
  private g3 = new THREE.Group();
  private status = document.createElement('div');
  private volCap = document.createElement('div');
  private eigenOut = document.createElement('div');
  private pathOut = document.createElement('div');
  private active = false;
  private slice: Slice | null = null;
  private history: Snapshot[] | null = null;
  private eigen: { values: number[]; vectors: Float64Array[]; key: string } | null = null;
  private path: { a: number[][]; off: number[] } | null = null;
  private pathEval: { steps: number[]; evals: PointEval[] } | null = null;
  private here: { a: number[]; off: number } | null = null;
  private clicked: number[] | null = null;
  private dirty = false;
  private lastDraw = 0;
  private autoRan = false;
  private cfg = {
    dims: 2 as Dims,
    axes: [
      { kind: 'random', name: 'L0.mlp.W1', index: 0, seed: 1, k: 0 },
      { kind: 'random', name: 'L0.mlp.W1', index: 1, seed: 2, k: 1 },
      { kind: 'random', name: 'L0.mlp.W1', index: 2, seed: 3, k: 2 },
    ] as DirSpec[],
    norm: 'filter' as 'filter' | 'unit' | 'none',
    ortho: true,
    ranges: { 1: 0.3, 2: 0.1, 3: 0.05 } as Record<number, number>,
    res: { 1: 241, 2: 65, 3: 21 } as Record<number, number>,
    center: 'current' as 'current' | 'init',
    split: 'train' as 'train' | 'val' | 'all',
    carve: 'examples' as Carve,
    carveK: 1,
    carveLayer: 0,
    carveNeuron: 0,
    paint: 'cells' as Paint,
    walls: true,
    contours: true,
    trajectory: true,
    iso: 0.35,
    slices: [0.5, 0.5, 0.5],
    showWalls3d: false,
    planes: true,
    height: 0.8,
  };

  constructor(session: Session) {
    this.session = session;
    this.el.classList.add('flexview');
    Object.assign(this.side.style, { width: '250px', minWidth: '250px', overflow: 'auto', padding: '8px', borderRight: '1px solid var(--border)' });
    Object.assign(this.main.style, { flex: '1', position: 'relative', minWidth: '0', display: 'flex' });
    Object.assign(this.mapWrap.style, { position: 'relative', flex: '1', minWidth: '0' });
    Object.assign(this.stageWrap.style, { position: 'relative', flex: '1', minWidth: '0' });
    this.mapCv.style.position = 'absolute';
    this.mapCv.style.cursor = 'crosshair';
    this.mapWrap.append(this.mapCv);
    this.main.append(this.mapWrap, this.stageWrap);
    this.el.append(this.side, this.main);
    this.stage = new Stage3D(this.stageWrap);
    this.stage.scene.add(this.g3);
    this.stage.camera.position.set(1.8, -2.6, 2.2);
    this.volCap.className = 'caption';
    this.stage.overlay.append(this.volCap);
    this.buildControls();
    this.mapCv.addEventListener('pointermove', (e) => this.hover(e));
    this.mapCv.addEventListener('pointerleave', () => hideTip());
    this.mapCv.addEventListener('click', (e) => this.click(e));
    this.stage.renderer.domElement.addEventListener('dblclick', (e) => this.click3d(e));
    new ResizeObserver(() => {
      if (!this.active) return;
      this.layout();
      this.draw(true);
    }).observe(this.main);

    // the live model moves: update "you are here"
    effect(() => {
      rt.paramsVersion();
      untracked(() => {
        if (!this.slice || !this.active) return;
        this.updateHere();
        this.markStale();
        this.dirty = true;
      });
    });
    effect(() => {
      rt.modelVersion();
      untracked(() => {
        this.pool?.cancel();
        this.slice = null;
        this.path = null;
        this.pathEval = null;
        this.eigen = null;
        this.history = null;
        this.autoRan = false;
        this.fillParamSelects();
        if (this.active) this.compute();
      });
    });
    // the inspector's →x / →y buttons
    effect(() => {
      const req = ui.landscapeAxis();
      if (!req) return;
      untracked(() => {
        const ax = this.cfg.axes[req.axis];
        ax.kind = 'weight';
        ax.name = req.name;
        ax.index = req.index;
        if (this.cfg.dims !== 2) this.cfg.dims = 2;
        const other = this.cfg.axes[1 - req.axis];
        if (other.kind !== 'weight') {
          other.kind = 'weight';
          other.name = req.name;
          const size = this.session.model.byName.get(req.name)?.t.size ?? 1;
          other.index = req.index + 1 < size ? req.index + 1 : Math.max(0, req.index - 1);
        }
        this.cfg.norm = 'none';
        this.cfg.ranges[2] = Math.max(0.5, this.cfg.ranges[2]);
        this.refreshControls();
        if (this.active) this.compute();
        else this.autoRan = false;
      });
    });
    const loop = () => {
      requestAnimationFrame(loop);
      if (!this.active || !this.dirty) return;
      const now = performance.now();
      if (now - this.lastDraw < 90) return;
      this.lastDraw = now;
      this.dirty = false;
      this.draw(false);
    };
    requestAnimationFrame(loop);
  }

  activate(): void {
    this.active = true;
    if (!this.pool) this.pool = new LandscapePool();
    this.stage.start();
    if (!this.autoRan) {
      this.autoRan = true;
      this.compute();
    } else this.draw(true);
  }

  deactivate(): void {
    this.active = false;
    this.stage.stop();
    hideTip();
  }

  themeChanged(): void {
    this.stage.themeChanged();
    if (this.active) this.draw(true);
  }

  // ---------------------------------------------------------------- controls

  private axisBoxes: HTMLDivElement[] = [];
  private carveRows: HTMLElement[] = [];
  private rangeIn!: HTMLInputElement;
  private resIn!: HTMLInputElement;

  private syncCarve() {
    const c = this.cfg.carve;
    this.carveRows[0].hidden = c !== 'examples';
    this.carveRows[1].hidden = c !== 'layer' && c !== 'neuron';
    this.carveRows[2].hidden = c !== 'neuron';
  }

  private carveOpts() {
    return { k: this.cfg.carveK, layer: this.cfg.carveLayer, neuron: this.cfg.carveNeuron };
  }
  private paramSelects: HTMLSelectElement[] = [];

  private buildControls() {
    const S = this.side;
    const h = document.createElement('h3');
    h.textContent = 'loss landscape';
    h.style.margin = '0 0 4px';
    h.style.color = 'var(--accent)';
    const intro = document.createElement('div');
    intro.className = 'note';
    intro.textContent = 'A slice through parameter space: θ = θ₀ + a·u + b·v (+ c·w). Each point is a whole network. Cells: points with the same activation pattern on every input; inside one, the loss is smooth; walls are where some neuron switches for some input.';
    S.append(h, intro);

    const row = (label: string, el: HTMLElement, title = '') => {
      const l = document.createElement('label');
      Object.assign(l.style, { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px', margin: '3px 0' });
      l.append(label, el);
      if (title) l.title = title;
      S.append(l);
      return l;
    };
    const select = <T extends string | number>(opts: [T, string][], get: () => T, set: (v: T) => void) => {
      const s = document.createElement('select');
      for (const [v, t] of opts) s.append(new Option(t, String(v)));
      s.value = String(get());
      s.onchange = () => {
        const v = opts.find(([o]) => String(o) === s.value)![0];
        set(v);
      };
      return s;
    };
    const num = (get: () => number, set: (v: number) => void, step = 1, w = '64px') => {
      const i = document.createElement('input');
      i.type = 'number';
      i.step = String(step);
      i.value = String(get());
      i.style.width = w;
      i.onchange = () => set(Number(i.value));
      return i;
    };
    const check = (get: () => boolean, set: (v: boolean) => void) => {
      const c = document.createElement('input');
      c.type = 'checkbox';
      c.checked = get();
      c.onchange = () => set(c.checked);
      return c;
    };

    row('dimensions', select<Dims>([[1, '1D line'], [2, '2D plane'], [3, '3D volume'], ['path', 'training path']], () => this.cfg.dims, (v) => {
      this.cfg.dims = v;
      this.syncAxes();
      this.compute();
    }), 'path: evaluate the actual checkpoints of this run, in order');

    ['x (a)', 'y (b)', 'z (c)'].forEach((name, ai) => {
      const box = document.createElement('div');
      Object.assign(box.style, { border: '1px solid var(--border)', borderRadius: '4px', padding: '3px 6px', margin: '5px 0' });
      const t = document.createElement('div');
      t.textContent = `axis ${name}`;
      t.className = 'muted';
      box.append(t);
      const ax = this.cfg.axes[ai];
      const line = (label: string, el: HTMLElement, show: () => boolean) => {
        const l = document.createElement('label');
        Object.assign(l.style, { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '4px', margin: '2px 0' });
        l.append(label, el);
        (l as HTMLElement & { _show?: () => boolean })._show = show;
        box.append(l);
      };
      const kind = select<DirKind>(DIR_KINDS, () => ax.kind, (v) => {
        ax.kind = v;
        this.syncAxes();
      });
      kind.style.maxWidth = '150px';
      line('dir', kind, () => true);
      const ps = document.createElement('select');
      ps.style.maxWidth = '150px';
      ps.onchange = () => (ax.name = ps.value);
      this.paramSelects.push(ps);
      line('tensor', ps, () => ax.kind === 'weight' || ax.kind === 'tensor');
      line('index', num(() => ax.index, (v) => (ax.index = Math.max(0, Math.round(v)))), () => ax.kind === 'weight');
      line('seed', num(() => ax.seed, (v) => (ax.seed = Math.round(v))), () => ax.kind === 'random');
      line('rank', num(() => ax.k, (v) => (ax.k = Math.max(0, Math.round(v)))), () => ax.kind === 'hessian' || ax.kind === 'pca');
      S.append(box);
      this.axisBoxes.push(box);
    });

    row('normalize', select([['filter', 'filter-wise (Li et al.)'], ['unit', 'unit length'], ['none', 'raw']], () => this.cfg.norm, (v) => (this.cfg.norm = v as 'filter')), 'filter-wise: each row of each matrix scaled to the norm of that row of θ₀ (random dirs)');
    row('orthogonalize', check(() => this.cfg.ortho, (v) => (this.cfg.ortho = v)), 'Gram-Schmidt the axes, so the slice is not skewed');
    const rangeIn = num(() => this.cfg.ranges[this.dimsK()], (v) => (this.cfg.ranges[this.dimsK()] = Math.max(1e-5, v)), 0.01);
    this.rangeIn = rangeIn;
    const zoom = document.createElement('span');
    zoom.style.display = 'flex';
    zoom.style.gap = '2px';
    for (const [t, f] of [['×½', 0.5], ['×2', 2]] as const) {
      const b = document.createElement('button');
      b.textContent = t;
      b.title = f < 1 ? 'zoom in: half the range, recompute' : 'zoom out: double the range, recompute';
      b.onclick = () => {
        const k = this.dimsK();
        this.cfg.ranges[k] = Number((this.cfg.ranges[k] * f).toPrecision(3));
        rangeIn.value = String(this.cfg.ranges[k]);
        this.compute();
      };
      zoom.append(b);
    }
    zoom.prepend(rangeIn);
    row('range ±', zoom, 'the slice spans a ∈ [−range, range] along every axis. Cells get finer the further out you look: zoom in to resolve them');
    this.resIn = num(() => this.cfg.res[this.dimsK()], (v) => (this.cfg.res[this.dimsK()] = Math.max(5, Math.min(401, Math.round(v)))));
    row('resolution', this.resIn, 'grid points per axis (cost grows as n^dims)');
    row('center θ₀', select([['current', 'current weights'], ['init', 'initial weights']], () => this.cfg.center, (v) => (this.cfg.center = v as 'current')));
    row('data', select([['train', 'train set'], ['val', 'validation set'], ['all', 'all']], () => this.cfg.split, (v) => (this.cfg.split = v as 'train')));

    const go = document.createElement('div');
    go.style.display = 'flex';
    go.style.gap = '4px';
    go.style.margin = '6px 0';
    const cb = document.createElement('button');
    cb.textContent = 'compute';
    cb.onclick = () => this.compute();
    const cc = document.createElement('button');
    cc.textContent = 'stop';
    cc.onclick = () => {
      this.pool?.cancel();
      this.setStatus('stopped');
    };
    const back = document.createElement('button');
    back.textContent = 'back to θ₀';
    back.title = 'put the model back at the slice center';
    back.onclick = () => {
      if (this.slice) {
        this.session.setAll(this.slice.center);
        this.clicked = null;
      }
    };
    go.append(cb, cc, back);
    S.append(go);
    this.status.className = 'note';
    S.append(this.status);

    const h2 = document.createElement('div');
    h2.className = 'muted';
    h2.style.marginTop = '8px';
    h2.textContent = 'display';
    S.append(h2);
    const carveSel = select<Carve>([
      ['examples', 'MLP on the first K examples'],
      ['mlp', 'MLP on every example (true cells)'],
      ['layer', 'MLP of one layer'],
      ['neuron', 'one neuron, every input'],
      ['attn', 'attention argmax'],
      ['pred', 'predicted tokens'],
      ['all', 'MLP + attention + predictions'],
      ['none', 'nothing'],
    ], () => this.cfg.carve, (v) => {
      this.cfg.carve = v;
      this.syncCarve();
      this.draw(true);
    });
    carveSel.style.maxWidth = '150px';
    row('carve by', carveSel, 'which switching defines a cell. Every example lays its own walls; the training loss\'s cells are the overlay of all of them, so they shrink as examples are added');
    this.carveRows.push(row('K examples', num(() => this.cfg.carveK, (v) => (this.cfg.carveK = Math.max(1, Math.round(v)))), 'overlay the walls of the first K examples (recompute to apply)'));
    this.carveRows.push(row('layer', num(() => this.cfg.carveLayer, (v) => (this.cfg.carveLayer = Math.max(0, Math.round(v)))), 'recompute to apply'));
    this.carveRows.push(row('neuron', num(() => this.cfg.carveNeuron, (v) => (this.cfg.carveNeuron = Math.max(0, Math.round(v)))), 'recompute to apply'));
    row('paint', select<Paint>([['cells', 'cells (shaded by loss)'], ['loss', 'loss'], ['acc', 'accuracy'], ['flips', 'switches from θ₀']], () => this.cfg.paint, (v) => {
      this.cfg.paint = v;
      this.draw(true);
    }));
    row('cell walls', check(() => this.cfg.walls, (v) => {
      this.cfg.walls = v;
      this.draw(true);
    }));
    row('loss contours', check(() => this.cfg.contours, (v) => {
      this.cfg.contours = v;
      this.draw(true);
    }));
    row('training path', check(() => this.cfg.trajectory, (v) => {
      this.cfg.trajectory = v;
      this.draw(true);
    }), 'the run\'s checkpoints projected into the slice (faded = far off the slice)');
    const hs = document.createElement('input');
    hs.type = 'range';
    hs.min = '0';
    hs.max = '2';
    hs.step = '0.05';
    hs.value = String(this.cfg.height);
    hs.oninput = () => {
      this.cfg.height = Number(hs.value);
      this.draw(true);
    };
    row('surface height', hs);
    // 3D volume controls
    const vol = document.createElement('div');
    vol.dataset.role = 'vol';
    const vr = (label: string, get: () => number, set: (v: number) => void, title: string) => {
      const r = document.createElement('input');
      r.type = 'range';
      r.min = '0';
      r.max = '1';
      r.step = '0.01';
      r.value = String(get());
      r.oninput = () => {
        set(Number(r.value));
        this.draw(true);
      };
      const l = document.createElement('label');
      Object.assign(l.style, { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px', margin: '3px 0' });
      l.append(label, r);
      l.title = title;
      vol.append(l);
    };
    vr('loss level', () => this.cfg.iso, (v) => (this.cfg.iso = v), 'the isosurface: every point where the loss equals this level (log scale between min and max)');
    vr('slice x', () => this.cfg.slices[0], (v) => (this.cfg.slices[0] = v), 'plane a = const');
    vr('slice y', () => this.cfg.slices[1], (v) => (this.cfg.slices[1] = v), 'plane b = const');
    vr('slice z', () => this.cfg.slices[2], (v) => (this.cfg.slices[2] = v), 'plane c = const');
    const wl = document.createElement('label');
    Object.assign(wl.style, { display: 'flex', justifyContent: 'space-between', margin: '3px 0' });
    wl.append('walls as points', check(() => this.cfg.showWalls3d, (v) => {
      this.cfg.showWalls3d = v;
      this.draw(true);
    }));
    wl.title = 'a dot on every grid face whose two sides lie in different cells';
    vol.append(wl);
    const pl = document.createElement('label');
    Object.assign(pl.style, { display: 'flex', justifyContent: 'space-between', margin: '3px 0' });
    pl.append('slice planes', check(() => this.cfg.planes, (v) => {
      this.cfg.planes = v;
      this.draw(true);
    }));
    vol.append(pl);
    S.append(vol);

    // Hessian + path analysis
    const h3 = document.createElement('div');
    h3.className = 'muted';
    h3.style.marginTop = '8px';
    h3.textContent = 'curvature and path';
    S.append(h3);
    const eb = document.createElement('button');
    eb.textContent = 'Hessian spectrum at θ₀';
    eb.title = 'Lanczos on finite-difference Hessian-vector products (30 steps)';
    eb.onclick = () => this.computeEigen();
    const pb = document.createElement('button');
    pb.textContent = 'walls crossed in training';
    pb.title = 'evaluate every checkpoint: how many activation-pattern changes did training go through?';
    pb.onclick = () => {
      this.cfg.dims = 'path';
      (S.querySelector('select') as HTMLSelectElement).value = 'path';
      this.syncAxes();
      this.compute();
    };
    S.append(eb, pb);
    this.eigenOut.className = 'note';
    this.pathOut.className = 'note';
    S.append(this.eigenOut, this.pathOut);
    this.syncAxes();
    this.syncCarve();
  }

  /** re-read cfg into the controls after a programmatic change */
  private refreshControls() {
    const dimsSel = this.side.querySelector('select') as HTMLSelectElement;
    dimsSel.value = String(this.cfg.dims);
    this.axisBoxes.forEach((b, i) => {
      const ax = this.cfg.axes[i];
      const sels = b.querySelectorAll('select');
      (sels[0] as HTMLSelectElement).value = ax.kind;
      (sels[1] as HTMLSelectElement).value = ax.name;
      const nums = b.querySelectorAll<HTMLInputElement>('input[type=number]');
      nums[0].value = String(ax.index);
      nums[1].value = String(ax.seed);
      nums[2].value = String(ax.k);
    });
    const normSel = [...this.side.querySelectorAll('select')].find((s) => [...s.options].some((o) => o.value === 'filter'));
    if (normSel) normSel.value = this.cfg.norm;
    this.syncAxes();
  }

  private dimsK(): number {
    return this.cfg.dims === 'path' ? 1 : this.cfg.dims;
  }

  private fillParamSelects() {
    const names = this.session.model.params.map((p) => p.name);
    this.paramSelects.forEach((s, i) => {
      s.replaceChildren(...names.map((n) => new Option(n, n)));
      const ax = this.cfg.axes[i];
      if (!names.includes(ax.name)) ax.name = names.find((n) => n.includes('mlp.W1')) ?? names[0];
      s.value = ax.name;
    });
  }

  private syncAxes() {
    const k = this.cfg.dims === 'path' ? 0 : this.cfg.dims;
    this.axisBoxes.forEach((b, i) => {
      b.hidden = i >= k;
      for (const l of b.querySelectorAll('label')) {
        const f = (l as HTMLElement & { _show?: () => boolean })._show;
        if (f) (l as HTMLElement).hidden = !f();
      }
    });
    (this.side.querySelector('[data-role=vol]') as HTMLElement).hidden = this.cfg.dims !== 3;
    this.mapWrap.hidden = this.cfg.dims === 3;
    this.stageWrap.hidden = this.cfg.dims === 1 || this.cfg.dims === 'path';
    if (this.rangeIn) this.rangeIn.value = String(this.cfg.ranges[this.dimsK()]);
    if (this.resIn) this.resIn.value = String(this.cfg.res[this.dimsK()]);
    this.layout();
    if (this.cfg.dims === 3) this.stage.fit(new THREE.Vector3(0, 0, 0), 1.9);
    else if (this.cfg.dims === 2) this.stage.fit(new THREE.Vector3(0, 0, this.cfg.height / 3), 1.5);
  }

  /** map and surface side by side when wide, stacked when tall */
  private layout() {
    const w = this.main.clientWidth, h = this.main.clientHeight;
    this.main.style.flexDirection = this.cfg.dims === 2 && w < h * 1.4 ? 'column' : 'row';
    this.stage.resize();
  }

  private setStatus(t: string) {
    this.status.textContent = t;
  }

  private markStale() {
    const s = this.slice;
    if (!s) return;
    if (s.paramsVersion !== rt.paramsVersion() && this.clicked === null) {
      const here = this.here;
      this.setStatus(`${this.progressText()} · the model has moved${here ? ` (${fmtNum(here.off)} off the slice)` : ''}: compute again to re-center`);
    }
  }

  private progressText(): string {
    const s = this.slice;
    if (!s) return '';
    const secs = ((s.elapsed || performance.now() - s.t0) / 1000).toFixed(1);
    const d = this.session.data;
    const nEx = this.cfg.split === 'train' ? d.train.length : this.cfg.split === 'val' ? d.val.length : d.train.length + d.val.length;
    const capped = nEx > EVAL_CAP ? ` · loss on the first ${EVAL_CAP} of ${nEx} examples` : ` · ${nEx} examples per point`;
    return `${s.done}/${s.total} points · ${secs}s · ${this.pool?.size ?? 0} workers${capped}`;
  }

  // ---------------------------------------------------------------- compute

  private historyKey = -1;

  private async getHistory(): Promise<Snapshot[]> {
    const last = this.history?.[this.history.length - 1]?.step ?? -1;
    if (!this.history || this.historyKey !== rt.historyVersion() || Math.abs(last - rt.step()) > 50) {
      this.historyKey = rt.historyVersion();
      this.history = await this.session.fetchHistory();
    }
    return this.history;
  }

  private async computeEigen(at?: Float64Array): Promise<void> {
    if (!this.pool) return;
    const center = at ?? this.slice?.center ?? this.session.model.flat();
    this.eigenOut.textContent = 'computing Hessian spectrum…';
    const r = await this.pool.eigen(this.session.trainer.spec, this.cfg.split, center, 30);
    this.eigen = { ...r, key: `${rt.modelVersion()}:${this.slice?.paramsVersion ?? rt.paramsVersion()}` };
    const lr = this.session.trainer.spec.optim.lr;
    const top = r.values.slice(0, 5).map(fmtNum).join(', ');
    const bot = r.values.slice(-2).map(fmtNum).join(', ');
    this.eigenOut.textContent = `Hessian at θ₀ (Ritz values): largest ${top}; smallest ${bot}. Sharpness λmax = ${fmtNum(r.values[0])}; for plain gradient descent the edge of stability is 2/lr = ${fmtNum(2 / lr)}. Negative values = saddle directions. Use "Hessian eigenvector" axes (rank 0 = sharpest) to slice along them.`;
  }

  private async buildDirs(center: Float64Array): Promise<{ dirs: Float64Array[]; labels: string[] }> {
    const model = this.session.model;
    const k = this.dimsK();
    const dirs: Float64Array[] = [];
    const labels: string[] = [];
    for (let i = 0; i < k; i++) {
      const ax = this.cfg.axes[i];
      let d: Float64Array;
      let label: string;
      switch (ax.kind) {
        case 'weight': {
          const p = model.byName.get(ax.name)!;
          const idx = Math.min(ax.index, p.t.size - 1);
          d = unitDirection(model, ax.name, idx);
          label = p.t.rows === 1 ? `${ax.name}[${idx}]` : `${ax.name}[${Math.floor(idx / p.t.cols)},${idx % p.t.cols}]`;
          break;
        }
        case 'tensor':
          d = tensorDirection(model, center, ax.name);
          label = `scale ${ax.name}`;
          break;
        case 'gradient': {
          const g = await this.pool!.gradient(this.session.trainer.spec, this.cfg.split, center);
          d = g.grad.map((v) => -v);
          const n = Math.sqrt(dot(d, d));
          if (n > 0) for (let j = 0; j < d.length; j++) d[j] /= n;
          label = '−∇L (unit)';
          break;
        }
        case 'hessian': {
          const key = `${rt.modelVersion()}:${rt.paramsVersion()}:${this.cfg.center}:${this.cfg.split}`;
          if (!this.eigen || this.eigen.key !== key) {
            this.setStatus('computing Hessian eigenvectors…');
            await this.computeEigen(center);
            this.eigen!.key = key;
          }
          d = this.eigen!.vectors[Math.min(ax.k, this.eigen!.vectors.length - 1)].slice();
          label = `Hessian v${ax.k} (λ=${fmtNum(this.eigen!.values[ax.k])})`;
          break;
        }
        case 'pca': {
          const h = await this.getHistory();
          const { dirs: pd, explained } = trajectoryPCA(h.map((c) => c.params), center, Math.max(3, ax.k + 1));
          d = pd[Math.min(ax.k, pd.length - 1)] ?? randomDirection(model, center, 1, 'unit');
          label = `path PC${ax.k + 1} (${((explained[ax.k] ?? 0) * 100).toFixed(0)}%)`;
          break;
        }
        case 'init': {
          const h = await this.getHistory();
          const init = h[0]?.params ?? center;
          d = new Float64Array(center.length);
          for (let j = 0; j < d.length; j++) d[j] = init[j] - center[j];
          label = 'toward init (a=1: init)';
          break;
        }
        default:
          d = randomDirection(model, center, ax.seed, this.cfg.norm);
          label = `random #${ax.seed}${this.cfg.norm === 'filter' ? ' (filter-norm)' : ''}`;
      }
      if (this.cfg.norm === 'unit' && ax.kind !== 'random') {
        const n = Math.sqrt(dot(d, d));
        if (n > 0) for (let j = 0; j < d.length; j++) d[j] /= n;
      }
      dirs.push(d);
      labels.push(label);
    }
    return { dirs: this.cfg.ortho && dirs.length > 1 ? orthogonalize(dirs) : dirs, labels };
  }

  async compute(): Promise<void> {
    if (!this.pool) return;
    this.pool.cancel();
    this.clicked = null;
    if (this.cfg.dims === 'path') return this.computePath();
    const model = this.session.model;
    const spec = this.session.trainer.spec;
    let center = model.flat();
    if (this.cfg.center === 'init') center = Float64Array.from((await this.getHistory())[0]?.params ?? center);
    this.setStatus('building directions…');
    const { dirs, labels } = await this.buildDirs(center);
    const k = dirs.length, n = this.cfg.res[k], R = this.cfg.ranges[k];
    const total = Math.pow(n, k);
    const coords = new Float64Array(total * k);
    for (let g = 0; g < total; g++) {
      let r = g;
      for (let d = 0; d < k; d++) {
        coords[g * k + d] = -R + (2 * R * (r % n)) / (n - 1);
        r = Math.floor(r / n);
      }
    }
    const order = coarseToFine(n, k);
    let maxStride = 1;
    while (maxStride * 2 < n) maxStride *= 2;
    const slice: Slice = {
      k, n, R, center, dirs, labels, coords,
      res: new Float64Array(total * EVAL_STRIDE).fill(NaN),
      level: new Uint8Array(total).fill(255),
      proxy: new Int32Array(total).fill(-1),
      done: 0, total,
      modelVersion: rt.modelVersion(), paramsVersion: rt.paramsVersion(),
      spec: JSON.stringify(spec.model), t0: performance.now(), elapsed: 0,
    };
    this.slice = slice;
    this.path = null; // the old slice's projection has the wrong dimension
    this.updateHere();
    if (this.cfg.trajectory) void this.projectPath();
    const strideOf = (g: number) => {
      let s = maxStride, r = g;
      for (let d = 0; d < k; d++) {
        const c = r % n;
        while (s > 1 && c % s !== 0) s /= 2;
        r = Math.floor(r / n);
      }
      return s;
    };
    this.pool.run({
      spec, split: this.cfg.split, center, dirs, coords, order, carve: this.carveOpts(),
      onChunk: (idx, out) => {
        if (this.slice !== slice) return;
        idx.forEach((g, i) => {
          slice.res.set(out.subarray(i * EVAL_STRIDE, (i + 1) * EVAL_STRIDE), g * EVAL_STRIDE);
          const s = strideOf(g);
          slice.level[g] = 0;
          slice.proxy[g] = g;
          // stand in for the not-yet-computed block this coarse point covers
          if (s > 1) {
            const cs = Array.from({ length: k }, (_, d) => Math.floor(g / Math.pow(n, d)) % n);
            const visit = (d: number, off: number) => {
              if (d < 0) {
                if (off !== g && slice.level[off] > s) {
                  slice.level[off] = s;
                  slice.proxy[off] = g;
                }
                return;
              }
              for (let q = 0; q < s && cs[d] + q < n; q++) visit(d - 1, off + q * Math.pow(n, d));
            };
            visit(k - 1, g);
          }
        });
        slice.done += idx.length;
        this.setStatus(this.progressText());
        this.dirty = true;
      },
      onDone: () => {
        slice.elapsed = performance.now() - slice.t0;
        this.setStatus(`${this.progressText()} · done. Hover to read a point; click to move the model there.`);
        this.dirty = true;
      },
    });
    this.setStatus(this.progressText());
    this.draw(true);
  }

  /** 1D along the real training path: every checkpoint, in order. */
  private async computePath(): Promise<void> {
    const h = await this.getHistory();
    if (h.length < 2) {
      this.setStatus('no training history yet: train for a while first');
      this.pathEval = null;
      this.draw(true);
      return;
    }
    this.setStatus(`evaluating ${h.length} checkpoints…`);
    const center = Float64Array.from(h[h.length - 1].params);
    const out = await this.pool!.thetas(this.session.trainer.spec, this.cfg.split, center, h.map((c) => Float64Array.from(c.params)), this.carveOpts());
    const evals: PointEval[] = [];
    for (let i = 0; i < h.length; i++) evals.push(unpackEval(out, i * EVAL_STRIDE));
    this.pathEval = { steps: h.map((c) => c.step), evals };
    let changes = 0;
    for (let i = 1; i < evals.length; i++) if (!sameHash(evals[i], evals[i - 1], this.cfg.carve)) changes++;
    const cells = new Set(evals.map((e) => hashKey(e, this.cfg.carve))).size;
    this.pathOut.textContent = `Training path: ${h.length} checkpoints (step ${h[0].step} → ${h[h.length - 1].step}) visited ${cells} distinct cells and crossed a wall between ${changes} consecutive checkpoints. Switches from the final pattern at the start: ${evals[0].flips}.`;
    this.setStatus('training path evaluated');
    this.draw(true);
  }

  private async projectPath() {
    const s = this.slice;
    if (!s) return;
    const h = await this.getHistory();
    if (this.slice !== s) return;
    const a: number[][] = [], off: number[] = [];
    for (const c of h) {
      if (c.params.length !== s.center.length) continue;
      const p = projectToSlice(c.params, s.center, s.dirs);
      a.push(p.a);
      off.push(p.off);
    }
    this.path = { a, off };
    this.dirty = true;
  }

  private updateHere() {
    const s = this.slice;
    if (!s) return;
    const th = this.session.model.flat();
    if (th.length !== s.center.length) return;
    this.here = projectToSlice(th, s.center, s.dirs);
  }

  // ---------------------------------------------------------------- reading

  private at(g: number): PointEval | null {
    const s = this.slice!;
    const p = s.proxy[g];
    if (p < 0) return null;
    return unpackEval(s.res, p * EVAL_STRIDE);
  }

  private lossRange(): [number, number] {
    const s = this.slice!;
    let lo = Infinity, hi = -Infinity;
    for (let g = 0; g < s.total; g++) {
      if (s.level[g] !== 0) continue;
      const l = Math.log10(Math.max(1e-9, s.res[g * EVAL_STRIDE]));
      lo = Math.min(lo, l);
      hi = Math.max(hi, l);
    }
    if (!Number.isFinite(lo)) return [0, 1];
    return [lo, Math.max(hi, lo + 1e-6)];
  }

  private colorOf(e: PointEval, lr: [number, number], maxFlips: number): RGB {
    const tl = (Math.log10(Math.max(1e-9, e.loss)) - lr[0]) / (lr[1] - lr[0]);
    switch (this.cfg.paint) {
      case 'loss':
        return sequential(1 - tl);
      case 'acc':
        return sequential(e.acc);
      case 'flips':
        return sequential(maxFlips > 0 ? e.flips / maxFlips : 0);
      default: {
        if (this.cfg.carve === 'none') return sequential(1 - tl);
        const [h1, h2] = hashPair(e, this.cfg.carve);
        return hashColor(h1, h2, 1 - tl);
      }
    }
  }

  // ---------------------------------------------------------------- drawing

  private draw(force: boolean) {
    void force;
    if (this.cfg.dims === 'path') return this.drawPath();
    const s = this.slice;
    if (!s) {
      this.clearCanvas();
      return;
    }
    if (s.k === 1) this.draw1D();
    else if (s.k === 2) {
      this.draw2D();
      this.drawSurface();
    } else this.drawVolume();
  }

  private clearCanvas() {
    const W = this.mapWrap.clientWidth, H = this.mapWrap.clientHeight;
    this.sizeCanvas(W, H);
    this.mapCv.getContext('2d')!.clearRect(0, 0, W, H);
  }

  private sizeCanvas(W: number, H: number) {
    const dpr = window.devicePixelRatio || 1;
    this.mapCv.width = Math.max(1, Math.round(W * dpr));
    this.mapCv.height = Math.max(1, Math.round(H * dpr));
    this.mapCv.style.width = `${W}px`;
    this.mapCv.style.height = `${H}px`;
    const ctx = this.mapCv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  /** map rectangle inside the canvas (2D) */
  private mapRect(): { x: number; y: number; w: number; h: number } {
    const W = this.mapWrap.clientWidth, H = this.mapWrap.clientHeight;
    const side = Math.max(50, Math.min(W - 70, H - 84));
    return { x: 52, y: 16, w: side, h: side };
  }

  private draw2D() {
    const s = this.slice!;
    const W = this.mapWrap.clientWidth, H = this.mapWrap.clientHeight;
    const ctx = this.sizeCanvas(W, H);
    ctx.clearRect(0, 0, W, H);
    const r = this.mapRect(), n = s.n, cw = r.w / n, ch = r.h / n;
    const lr = this.lossRange();
    let maxFlips = 0;
    for (let g = 0; g < s.total; g++) if (s.level[g] === 0) maxFlips = Math.max(maxFlips, s.res[g * EVAL_STRIDE + 3]);
    // cells
    const evals: (PointEval | null)[] = new Array(s.total);
    for (let g = 0; g < s.total; g++) evals[g] = this.at(g);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const e = evals[i + n * j];
        if (!e) continue;
        ctx.fillStyle = css(this.colorOf(e, lr, maxFlips));
        ctx.fillRect(r.x + i * cw, r.y + (n - 1 - j) * ch, cw + 0.6, ch + 0.6);
      }
    // walls between neighbouring points with different patterns
    if (this.cfg.walls && this.cfg.carve !== 'none') {
      ctx.strokeStyle = css(token('text'), 0.75);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const e = evals[i + n * j];
          if (!e) continue;
          const x = r.x + i * cw, y = r.y + (n - 1 - j) * ch;
          if (i + 1 < n && evals[i + 1 + n * j] && !sameHash(e, evals[i + 1 + n * j]!, this.cfg.carve)) {
            ctx.moveTo(x + cw, y);
            ctx.lineTo(x + cw, y + ch);
          }
          if (j + 1 < n && evals[i + n * (j + 1)] && !sameHash(e, evals[i + n * (j + 1)]!, this.cfg.carve)) {
            ctx.moveTo(x, y);
            ctx.lineTo(x + cw, y);
          }
        }
      ctx.stroke();
    }
    // loss contours (marching squares on log loss)
    if (this.cfg.contours) {
      const L = (g: number) => Math.log10(Math.max(1e-9, evals[g]?.loss ?? NaN));
      ctx.strokeStyle = css(token('panel'), 0.7);
      ctx.lineWidth = 1;
      const levels = 10;
      ctx.beginPath();
      for (let lv = 1; lv < levels; lv++) {
        const c = lr[0] + ((lr[1] - lr[0]) * lv) / levels;
        for (let j = 0; j + 1 < n; j++)
          for (let i = 0; i + 1 < n; i++) {
            const v = [L(i + n * j), L(i + 1 + n * j), L(i + 1 + n * (j + 1)), L(i + n * (j + 1))];
            if (v.some((x) => !Number.isFinite(x))) continue;
            const px = (u: number) => r.x + (i + 0.5 + u) * cw, py = (w: number) => r.y + (n - 1 - j - w + 0.5) * ch;
            const pts: [number, number][] = [];
            const ed = [[0, 1, 0, 0, 1, 0], [1, 2, 1, 0, 1, 1], [3, 2, 0, 1, 1, 1], [0, 3, 0, 0, 0, 1]];
            for (const [a, b, ax0, ay0, ax1, ay1] of ed) {
              if ((v[a] < c) !== (v[b] < c)) {
                const t = (c - v[a]) / (v[b] - v[a]);
                pts.push([px(ax0 + (ax1 - ax0) * t), py(ay0 + (ay1 - ay0) * t)]);
              }
            }
            for (let q = 0; q + 1 < pts.length; q += 2) {
              ctx.moveTo(pts[q][0], pts[q][1]);
              ctx.lineTo(pts[q + 1][0], pts[q + 1][1]);
            }
          }
      }
      ctx.stroke();
    }
    const toPx = (a: number, b: number): [number, number] => [r.x + ((a + s.R) / (2 * s.R)) * (r.w - cw) + cw / 2, r.y + (1 - (b + s.R) / (2 * s.R)) * (r.h - ch) + ch / 2];
    // training path
    if (this.cfg.trajectory && this.path && this.path.a.length > 1) {
      const maxOff = Math.max(...this.path.off, 1e-9);
      ctx.lineWidth = 2;
      for (let q = 1; q < this.path.a.length; q++) {
        const [x0, y0] = toPx(this.path.a[q - 1][0], this.path.a[q - 1][1]);
        const [x1, y1] = toPx(this.path.a[q][0], this.path.a[q][1]);
        ctx.strokeStyle = css(token('sel'), 0.25 + 0.75 * (1 - this.path.off[q] / maxOff));
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
      const [sx, sy] = toPx(this.path.a[0][0], this.path.a[0][1]);
      ctx.fillStyle = css(token('sel'));
      ctx.fillRect(sx - 3, sy - 3, 6, 6);
      ctx.fillStyle = css(token('text'));
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('init', sx + 5, sy - 4);
    }
    // center, you are here, clicked
    const [cx, cy] = toPx(0, 0);
    ctx.strokeStyle = css(token('text'));
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - 6, cy);
    ctx.lineTo(cx + 6, cy);
    ctx.moveTo(cx, cy - 6);
    ctx.lineTo(cx, cy + 6);
    ctx.stroke();
    if (this.here && (Math.abs(this.here.a[0]) > 1e-9 || Math.abs(this.here.a[1]) > 1e-9 || this.here.off > 1e-9)) {
      const [hx, hy] = toPx(this.here.a[0], this.here.a[1]);
      ctx.strokeStyle = css(token('sel'));
      ctx.beginPath();
      ctx.arc(hx, hy, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = css(token('text'));
      ctx.fillText(`model${this.here.off > 1e-6 ? ` (${fmtNum(this.here.off)} off-slice)` : ''}`, hx + 8, hy + 12);
    }
    // axes
    ctx.fillStyle = css(token('muted'));
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`a: ${s.labels[0]}`, r.x + r.w / 2, r.y + r.h + 26);
    ctx.fillText(fmtNum(-s.R), r.x, r.y + r.h + 12);
    ctx.fillText('0', r.x + r.w / 2, r.y + r.h + 12);
    ctx.fillText(fmtNum(s.R), r.x + r.w, r.y + r.h + 12);
    ctx.save();
    ctx.translate(r.x - 32, r.y + r.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`b: ${s.labels[1]}`, 0, 0);
    ctx.restore();
    ctx.textAlign = 'right';
    ctx.fillText(fmtNum(s.R), r.x - 4, r.y + 8);
    ctx.fillText(fmtNum(-s.R), r.x - 4, r.y + r.h);
    ctx.textAlign = 'left';
    this.legend(ctx, r.x, r.y + r.h + 40, lr, maxFlips);
  }

  private legend(ctx: CanvasRenderingContext2D, x: number, y: number, lr: [number, number], maxFlips: number) {
    ctx.fillStyle = css(token('muted'));
    ctx.font = '10px ui-monospace, monospace';
    const s = this.slice!;
    const cells = new Set<string>();
    for (let g = 0; g < s.total; g++) if (s.level[g] === 0) cells.add(hashKey(unpackEval(s.res, g * EVAL_STRIDE), this.cfg.carve));
    let text = '';
    let trueCells = 0;
    if (this.cfg.carve !== 'mlp') {
      const t = new Set<string>();
      for (let g = 0; g < s.total; g++) if (s.level[g] === 0) t.add(hashKey(unpackEval(s.res, g * EVAL_STRIDE), 'mlp'));
      trueCells = t.size;
    }
    const truth = this.cfg.carve !== 'mlp' ? ` · with every example overlaid: ${trueCells} cells${trueCells >= s.done * 0.9 ? ' (finer than the grid)' : ''}` : '';
    if (this.cfg.paint === 'cells') text = `${this.carveName()}: ${cells.size} seen${truth} · colors = arbitrary cell identity, darker = higher loss`;
    else if (this.cfg.paint === 'loss') text = `loss: blue = low (${fmtNum(10 ** lr[0])}), gray = high (${fmtNum(10 ** lr[1])}), log scale · ${cells.size} cells`;
    else if (this.cfg.paint === 'acc') text = 'sequence accuracy: gray = 0, blue = 1';
    else text = `MLP switches from θ₀: gray = 0, blue = ${maxFlips}`;
    // wrap to the canvas width
    const maxW = this.mapWrap.clientWidth - x - 8;
    let line = '', yy = y;
    for (const word of text.split(' ')) {
      const t = line ? `${line} ${word}` : word;
      if (ctx.measureText(t).width > maxW && line) {
        ctx.fillText(line, x, yy);
        line = word;
        yy += 13;
      } else line = t;
    }
    if (line) ctx.fillText(line, x, yy);
  }

  private draw1D() {
    const s = this.slice!;
    const W = this.mapWrap.clientWidth, H = this.mapWrap.clientHeight;
    const ctx = this.sizeCanvas(W, H);
    ctx.clearRect(0, 0, W, H);
    const pad = { l: 56, r: 20, t: 16, b: 90 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const n = s.n;
    const lr = this.lossRange();
    const X = (i: number) => pad.l + (i / (n - 1)) * pw;
    const Y = (loss: number) => pad.t + (1 - (Math.log10(Math.max(1e-9, loss)) - lr[0]) / (lr[1] - lr[0])) * ph;
    const evals = Array.from({ length: n }, (_, i) => this.at(i));
    // cell background bands
    for (let i = 0; i < n; i++) {
      const e = evals[i];
      if (!e) continue;
      const [h1, h2] = hashPair(e, this.cfg.carve === 'none' ? 'mlp' : this.cfg.carve);
      ctx.fillStyle = css(hashColor(h1, h2, 0.5), 0.22);
      ctx.fillRect(X(i) - pw / (n - 1) / 2, pad.t, pw / (n - 1) + 0.5, ph);
    }
    // walls
    if (this.cfg.walls && this.cfg.carve !== 'none') {
      ctx.strokeStyle = css(token('text'), 0.5);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < n; i++) {
        if (evals[i] && evals[i - 1] && !sameHash(evals[i]!, evals[i - 1]!, this.cfg.carve)) {
          const x = (X(i) + X(i - 1)) / 2;
          ctx.moveTo(x, pad.t);
          ctx.lineTo(x, pad.t + ph);
        }
      }
      ctx.stroke();
    }
    // loss
    ctx.strokeStyle = css(token('text'));
    ctx.lineWidth = 2;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const e = evals[i];
      if (!e) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(X(i), Y(e.loss));
      else ctx.moveTo(X(i), Y(e.loss));
      pen = true;
    }
    ctx.stroke();
    // accuracy strip + flips strip
    const strip = (y0: number, label: string, f: (e: PointEval) => number) => {
      for (let i = 0; i < n; i++) {
        const e = evals[i];
        if (!e) continue;
        ctx.fillStyle = css(sequential(f(e)));
        ctx.fillRect(X(i) - pw / (n - 1) / 2, y0, pw / (n - 1) + 0.5, 12);
      }
      ctx.fillStyle = css(token('muted'));
      ctx.textAlign = 'right';
      ctx.fillText(label, pad.l - 4, y0 + 10);
      ctx.textAlign = 'left';
    };
    ctx.font = '10px ui-monospace, monospace';
    let maxFlips = 1;
    for (const e of evals) if (e) maxFlips = Math.max(maxFlips, e.flips);
    strip(pad.t + ph + 22, 'accuracy', (e) => e.acc);
    strip(pad.t + ph + 38, 'switches', (e) => e.flips / maxFlips);
    // axes
    ctx.fillStyle = css(token('muted'));
    ctx.textAlign = 'right';
    for (let d = Math.ceil(lr[0]); d <= Math.floor(lr[1]); d++) ctx.fillText(fmtNum(10 ** d), pad.l - 4, Y(10 ** d) + 3);
    ctx.textAlign = 'center';
    ctx.fillText(`a: ${s.labels[0]}`, pad.l + pw / 2, H - 10);
    ctx.fillText(fmtNum(-s.R), pad.l, pad.t + ph + 14);
    ctx.fillText('0', pad.l + pw / 2, pad.t + ph + 14);
    ctx.fillText(fmtNum(s.R), pad.l + pw, pad.t + ph + 14);
    ctx.textAlign = 'left';
    ctx.fillText('loss (log)', pad.l + 4, pad.t + 10);
    // center + here
    ctx.strokeStyle = css(token('sel'));
    const xc = pad.l + pw / 2;
    ctx.beginPath();
    ctx.moveTo(xc, pad.t);
    ctx.lineTo(xc, pad.t + ph);
    ctx.stroke();
    let walls = 0;
    for (let i = 1; i < n; i++) if (evals[i] && evals[i - 1] && !sameHash(evals[i]!, evals[i - 1]!, this.cfg.carve)) walls++;
    ctx.fillStyle = css(token('muted'));
    ctx.fillText(`${walls} walls crossed along this line; shaded bands = cells`, pad.l, H - 26);
  }

  private drawPath() {
    const W = this.mapWrap.clientWidth, H = this.mapWrap.clientHeight;
    const ctx = this.sizeCanvas(W, H);
    ctx.clearRect(0, 0, W, H);
    const pe = this.pathEval;
    ctx.font = '11px ui-monospace, monospace';
    if (!pe) {
      ctx.fillStyle = css(token('muted'));
      ctx.fillText('train for a while, then "compute": every checkpoint is evaluated in order.', 20, 30);
      return;
    }
    const pad = { l: 56, r: 20, t: 16, b: 90 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const n = pe.evals.length;
    let lo = Infinity, hi = -Infinity;
    for (const e of pe.evals) {
      const l = Math.log10(Math.max(1e-9, e.loss));
      lo = Math.min(lo, l);
      hi = Math.max(hi, l);
    }
    if (hi - lo < 1e-6) hi = lo + 1;
    const maxStep = pe.steps[n - 1] || 1;
    const X = (i: number) => pad.l + (pe.steps[i] / maxStep) * pw;
    const Y = (loss: number) => pad.t + (1 - (Math.log10(Math.max(1e-9, loss)) - lo) / (hi - lo)) * ph;
    for (let i = 0; i < n; i++) {
      const e = pe.evals[i];
      const [h1, h2] = hashPair(e, this.cfg.carve === 'none' ? 'mlp' : this.cfg.carve);
      const x0 = i ? (X(i - 1) + X(i)) / 2 : X(0), x1 = i + 1 < n ? (X(i) + X(i + 1)) / 2 : X(n - 1);
      ctx.fillStyle = css(hashColor(h1, h2, 0.5), 0.25);
      ctx.fillRect(x0, pad.t, x1 - x0 + 0.5, ph);
    }
    ctx.strokeStyle = css(token('text'), 0.5);
    ctx.beginPath();
    for (let i = 1; i < n; i++) {
      if (!sameHash(pe.evals[i], pe.evals[i - 1], this.cfg.carve)) {
        const x = (X(i) + X(i - 1)) / 2;
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, pad.t + ph);
      }
    }
    ctx.stroke();
    ctx.strokeStyle = css(token('text'));
    ctx.lineWidth = 2;
    ctx.beginPath();
    pe.evals.forEach((e, i) => (i ? ctx.lineTo(X(i), Y(e.loss)) : ctx.moveTo(X(i), Y(e.loss))));
    ctx.stroke();
    let maxF = 1;
    for (const e of pe.evals) maxF = Math.max(maxF, e.flips);
    const strip = (y0: number, label: string, f: (e: PointEval) => number) => {
      pe.evals.forEach((e, i) => {
        const x0 = i ? (X(i - 1) + X(i)) / 2 : X(0), x1 = i + 1 < n ? (X(i) + X(i + 1)) / 2 : X(n - 1);
        ctx.fillStyle = css(sequential(f(e)));
        ctx.fillRect(x0, y0, x1 - x0 + 0.5, 12);
      });
      ctx.fillStyle = css(token('muted'));
      ctx.textAlign = 'right';
      ctx.fillText(label, pad.l - 4, y0 + 10);
      ctx.textAlign = 'left';
    };
    ctx.font = '10px ui-monospace, monospace';
    strip(pad.t + ph + 22, 'accuracy', (e) => e.acc);
    strip(pad.t + ph + 38, 'switches', (e) => e.flips / maxF);
    ctx.fillStyle = css(token('muted'));
    ctx.textAlign = 'right';
    for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) ctx.fillText(fmtNum(10 ** d), pad.l - 4, Y(10 ** d) + 3);
    ctx.textAlign = 'center';
    ctx.fillText('training step', pad.l + pw / 2, H - 10);
    ctx.fillText('0', pad.l, pad.t + ph + 14);
    ctx.fillText(String(maxStep), pad.l + pw, pad.t + ph + 14);
    ctx.textAlign = 'left';
    ctx.fillText('loss along the actual training path (log); bands = cells; lines = walls crossed; switches = MLP (neuron, input) pairs that differ from the final pattern', pad.l, H - 26);
  }

  private drawSurface() {
    const s = this.slice!;
    this.volCap.textContent = 'height = log loss; colors as in the map; lines = walls. Drag to orbit, double-click to move the model there.';
    for (const c of [...this.g3.children]) disposeTree(c);
    this.stage.clearLabels();
    const n = s.n;
    const lr = this.lossRange();
    let maxFlips = 0;
    for (let g = 0; g < s.total; g++) if (s.level[g] === 0) maxFlips = Math.max(maxFlips, s.res[g * EVAL_STRIDE + 3]);
    const H = this.cfg.height;
    const pos = new Float32Array(s.total * 3), col = new Float32Array(s.total * 3);
    const evals: (PointEval | null)[] = [];
    const hgt = (e: PointEval | null) => (e ? ((Math.log10(Math.max(1e-9, e.loss)) - lr[0]) / (lr[1] - lr[0])) * H : 0);
    const tmp = new THREE.Color();
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const g = i + n * j;
        const e = this.at(g);
        evals.push(e);
        pos.set([-1 + (2 * i) / (n - 1), -1 + (2 * j) / (n - 1), hgt(e)], g * 3);
        const c = e ? this.colorOf(e, lr, maxFlips) : token('panel');
        tmp.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
        col.set([tmp.r, tmp.g, tmp.b], g * 3);
      }
    const idx: number[] = [];
    for (let j = 0; j + 1 < n; j++)
      for (let i = 0; i + 1 < n; i++) {
        const a = i + n * j, b = a + 1, c = a + n, d = c + 1;
        if (!evals[a] || !evals[b] || !evals[c] || !evals[d]) continue;
        idx.push(a, b, d, a, d, c);
      }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    this.g3.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide })));
    // walls on the surface
    if (this.cfg.walls && this.cfg.carve !== 'none') {
      const wp: number[] = [];
      const lift = 0.004;
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const g = i + n * j, e = evals[g];
          if (!e) continue;
          const x = -1 + (2 * i) / (n - 1), y = -1 + (2 * j) / (n - 1), dx = 1 / (n - 1);
          if (i + 1 < n && evals[g + 1] && !sameHash(e, evals[g + 1]!, this.cfg.carve)) {
            const z = (hgt(e) + hgt(evals[g + 1])) / 2 + lift;
            wp.push(x + dx, y - dx, z, x + dx, y + dx, z);
          }
          if (j + 1 < n && evals[g + n] && !sameHash(e, evals[g + n]!, this.cfg.carve)) {
            const z = (hgt(e) + hgt(evals[g + n])) / 2 + lift;
            wp.push(x - dx, y + dx, z, x + dx, y + dx, z);
          }
        }
      const wg = new THREE.BufferGeometry();
      wg.setAttribute('position', new THREE.Float32BufferAttribute(wp, 3));
      this.g3.add(new THREE.LineSegments(wg, new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('text')), transparent: true, opacity: 0.6 })));
    }
    // path on the surface
    const surfZ = (a: number, b: number) => {
      const fi = ((a + s.R) / (2 * s.R)) * (n - 1), fj = ((b + s.R) / (2 * s.R)) * (n - 1);
      const i = Math.max(0, Math.min(n - 1, Math.round(fi))), j = Math.max(0, Math.min(n - 1, Math.round(fj)));
      return hgt(evals[i + n * j]);
    };
    const toXY = (a: number, b: number) => [a / s.R, b / s.R];
    if (this.cfg.trajectory && this.path && this.path.a.length > 1) {
      const pts = this.path.a.filter((a) => Math.abs(a[0]) <= s.R && Math.abs(a[1]) <= s.R).map((a) => {
        const [x, y] = toXY(a[0], a[1]);
        return new THREE.Vector3(x, y, surfZ(a[0], a[1]) + 0.02);
      });
      if (pts.length > 1) this.g3.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('sel')) })));
    }
    // center marker: a vertical pin
    const pin = (a: number, b: number, color: string, label: string) => {
      const [x, y] = toXY(a, b);
      const z = surfZ(a, b);
      this.g3.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, y, z), new THREE.Vector3(x, y, z + 0.25)]), new THREE.LineBasicMaterial({ color: new THREE.Color(color) })));
      this.stage.label(label, new THREE.Vector3(x, y, z + 0.3), color);
    };
    pin(0, 0, cssVar('text'), 'θ₀');
    if (this.here && (Math.abs(this.here.a[0]) > 1e-9 || Math.abs(this.here.a[1]) > 1e-9)) pin(this.here.a[0], this.here.a[1], cssVar('sel'), 'model');
    // frame
    const fr = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, H || 0.01)), new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('border')) }));
    fr.position.z = H / 2;
    this.g3.add(fr);
    this.stage.label(`a: ${s.labels[0]}`, new THREE.Vector3(0, -1.15, 0), css(token('muted')));
    this.stage.label(`b: ${s.labels[1]}`, new THREE.Vector3(-1.15, 0, 0), css(token('muted')));
    this.stage.label('↑ log loss', new THREE.Vector3(-1, -1, H + 0.08), css(token('muted')));
    this.stage.invalidate();
  }

  private drawVolume() {
    const s = this.slice!;
    for (const c of [...this.g3.children]) disposeTree(c);
    this.stage.clearLabels();
    const n = s.n;
    const lr = this.lossRange();
    let maxFlips = 0;
    for (let g = 0; g < s.total; g++) if (s.level[g] === 0) maxFlips = Math.max(maxFlips, s.res[g * EVAL_STRIDE + 3]);
    const evals: (PointEval | null)[] = Array.from({ length: s.total }, (_, g) => this.at(g));
    const P = (i: number) => -1 + (2 * i) / (n - 1);
    // three slice planes as textures
    const plane = (axis: number, frac: number) => {
      const q = Math.round(frac * (n - 1));
      const data = new Uint8Array(n * n * 4);
      for (let v = 0; v < n; v++)
        for (let u = 0; u < n; u++) {
          const c3 = [0, 0, 0];
          const others = [0, 1, 2].filter((a) => a !== axis);
          c3[axis] = q;
          c3[others[0]] = u;
          c3[others[1]] = v;
          const g = c3[0] + n * (c3[1] + n * c3[2]);
          const e = evals[g];
          let col: RGB = token('panel');
          if (e) {
            col = this.colorOf(e, lr, maxFlips);
            // darken walls within the plane
            if (this.cfg.walls && this.cfg.carve !== 'none') {
              const nb = (du: number, dv: number) => {
                const cc = c3.slice();
                cc[others[0]] += du;
                cc[others[1]] += dv;
                if (cc[others[0]] >= n || cc[others[1]] >= n) return null;
                return evals[cc[0] + n * (cc[1] + n * cc[2])];
              };
              const a = nb(1, 0), b = nb(0, 1);
              if ((a && !sameHash(e, a, this.cfg.carve)) || (b && !sameHash(e, b, this.cfg.carve))) col = [col[0] * 0.3, col[1] * 0.3, col[2] * 0.3];
            }
          }
          data.set([col[0] * 255, col[1] * 255, col[2] * 255, 235], (u + n * v) * 4);
        }
      const tex = new THREE.DataTexture(data, n, n, THREE.RGBAFormat);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.magFilter = THREE.NearestFilter;
      tex.needsUpdate = true;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.6, depthWrite: false }));
      const off = P(q);
      if (axis === 2) mesh.position.z = off;
      else if (axis === 1) {
        mesh.rotation.x = Math.PI / 2;
        mesh.position.y = off;
      } else {
        // local x → +y (axis b), local y → +z (axis c)
        mesh.rotation.set(Math.PI / 2, Math.PI / 2, 0);
        mesh.position.x = off;
      }
      this.g3.add(mesh);
    };
    if (this.cfg.planes) {
      plane(0, this.cfg.slices[0]);
      plane(1, this.cfg.slices[1]);
      plane(2, this.cfg.slices[2]);
    }
    // isosurface of log loss, painted by cell
    const f = new Float64Array(s.total);
    const valid = new Uint8Array(s.total);
    for (let g = 0; g < s.total; g++) {
      const e = evals[g];
      if (e) {
        f[g] = Math.log10(Math.max(1e-9, e.loss));
        valid[g] = 1;
      }
    }
    const level = lr[0] + this.cfg.iso * (lr[1] - lr[0]);
    const iso = isosurface(f, n, level, valid);
    if (iso.positions.length) {
      const pos = new Float32Array(iso.positions.length);
      for (let i = 0; i < iso.positions.length; i++) pos[i] = -1 + (2 * iso.positions[i]) / (n - 1);
      const col = new Float32Array(iso.positions.length);
      const tmp = new THREE.Color();
      for (let v = 0; v < iso.corner.length; v++) {
        const e = evals[iso.corner[v]];
        const c = e ? (this.cfg.carve === 'none' ? sequential(0.6) : hashColor(...hashPair(e, this.cfg.carve), 0.6)) : token('muted');
        tmp.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
        col.set([tmp.r, tmp.g, tmp.b], v * 3);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      geo.computeVertexNormals();
      this.g3.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.75 })));
    }
    // cell walls: a point on every grid face whose two sides differ
    if (this.cfg.showWalls3d && this.cfg.carve !== 'none') {
      const wp: number[] = [], wc: number[] = [];
      const tmp = new THREE.Color();
      for (let z = 0; z < n; z++)
        for (let y = 0; y < n; y++)
          for (let x = 0; x < n; x++) {
            const g = x + n * (y + n * z), e = evals[g];
            if (!e) continue;
            const nbs: [number, number, number, number][] = [[1, 0, 0, g + 1], [0, 1, 0, g + n], [0, 0, 1, g + n * n]];
            for (const [dx, dy, dz, h] of nbs) {
              if (x + dx >= n || y + dy >= n || z + dz >= n) continue;
              const o = evals[h];
              if (!o || sameHash(e, o, this.cfg.carve)) continue;
              wp.push(P(x + dx / 2), P(y + dy / 2), P(z + dz / 2));
              const c = hashColor(...hashPair(e, this.cfg.carve), 0.55);
              tmp.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
              wc.push(tmp.r, tmp.g, tmp.b);
            }
          }
      const wg = new THREE.BufferGeometry();
      wg.setAttribute('position', new THREE.Float32BufferAttribute(wp, 3));
      wg.setAttribute('color', new THREE.Float32BufferAttribute(wc, 3));
      this.g3.add(new THREE.Points(wg, new THREE.PointsMaterial({ size: 2.5, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false })));
    }
    // path
    if (this.cfg.trajectory && this.path && this.path.a.length > 1 && this.path.a[0].length === 3) {
      const pts = this.path.a.map((a) => new THREE.Vector3(a[0] / s.R, a[1] / s.R, a[2] / s.R));
      this.g3.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('sel')) })));
      this.stage.label('init', pts[0], css(token('sel')));
    }
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)), new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('border')) }));
    this.g3.add(box);
    this.stage.label('θ₀', new THREE.Vector3(0, 0, 0), css(token('text')));
    if (this.here && this.here.a.length === 3) this.stage.label('● model', new THREE.Vector3(this.here.a[0] / s.R, this.here.a[1] / s.R, this.here.a[2] / s.R), css(token('sel')));
    this.stage.label(`a: ${s.labels[0]}`, new THREE.Vector3(0, -1.2, -1), css(token('muted')));
    this.stage.label(`b: ${s.labels[1]}`, new THREE.Vector3(-1.2, 0, -1), css(token('muted')));
    this.stage.label(`c: ${s.labels[2]}`, new THREE.Vector3(-1.2, -1.2, 0), css(token('muted')));
    this.volCap.textContent = `level set: every point with loss = ${fmtNum(10 ** level)} (painted by cell). Slice planes: loss or cells on a = const, b = const, c = const; dark = walls. Drag to orbit.`;
    this.stage.invalidate();
  }

  // ---------------------------------------------------------------- interaction

  private gridAt(e: MouseEvent): { g: number; a: number[] } | null {
    const s = this.slice;
    if (!s) return null;
    const rc = this.mapCv.getBoundingClientRect();
    const x = e.clientX - rc.left, y = e.clientY - rc.top;
    if (s.k === 2) {
      const r = this.mapRect();
      const i = Math.floor(((x - r.x) / r.w) * s.n), j = s.n - 1 - Math.floor(((y - r.y) / r.h) * s.n);
      if (i < 0 || j < 0 || i >= s.n || j >= s.n) return null;
      const g = i + s.n * j;
      return { g, a: [s.coords[g * 2], s.coords[g * 2 + 1]] };
    }
    if (s.k === 1) {
      const pw = rc.width - 76;
      const i = Math.round(((x - 56) / pw) * (s.n - 1));
      if (i < 0 || i >= s.n) return null;
      return { g: i, a: [s.coords[i]] };
    }
    return null;
  }

  private carveName(): string {
    const c = this.cfg.carve;
    if (c === 'examples') return `cells of the first ${this.cfg.carveK} example${this.cfg.carveK > 1 ? 's' : ''}`;
    if (c === 'mlp') return 'true cells (every example)';
    if (c === 'layer') return `cells of layer ${this.cfg.carveLayer}`;
    if (c === 'neuron') return `cells of neuron ${this.cfg.carveNeuron} of layer ${this.cfg.carveLayer}`;
    return `carving: ${c}`;
  }

  private hover(e: PointerEvent) {
    if (this.cfg.dims === 'path') {
      const pe = this.pathEval;
      if (!pe) return;
      const rc = this.mapCv.getBoundingClientRect();
      const step = ((e.clientX - rc.left - 56) / (rc.width - 76)) * pe.steps[pe.steps.length - 1];
      let best = 0;
      for (let i = 0; i < pe.steps.length; i++) if (Math.abs(pe.steps[i] - step) < Math.abs(pe.steps[best] - step)) best = i;
      const ev = pe.evals[best];
      showTip(`step ${pe.steps[best]}\nloss ${fmtNum(ev.loss)} · accuracy ${(ev.acc * 100).toFixed(0)}%\ncell ${hashKey(ev, this.cfg.carve).slice(0, 8)} · ${ev.flips} switches from the final pattern\n${(ev.active * 100).toFixed(0)}% of MLP (neuron, input) pairs on`, e);
      return;
    }
    const p = this.gridAt(e);
    if (!p) return hideTip();
    const ev = this.at(p.g);
    if (!ev) return hideTip();
    const exact = this.slice!.level[p.g] === 0;
    showTip(`a = (${p.a.map((x) => fmtNum(x)).join(', ')})${exact ? '' : ' (coarse preview)'}\n${this.carveName()}\nloss ${fmtNum(ev.loss)} · accuracy ${(ev.acc * 100).toFixed(0)}% (tokens ${(ev.tokAcc * 100).toFixed(0)}%)\ncell ${hashKey(ev, this.cfg.carve).slice(0, 8)} · ${ev.flips} MLP switches from θ₀\n${(ev.active * 100).toFixed(0)}% of MLP (neuron, input) pairs on\nclick: move the model here`, e);
  }

  private click(e: MouseEvent) {
    if (this.cfg.dims === 'path') {
      const pe = this.pathEval;
      if (!pe || !this.history) return;
      const rc = this.mapCv.getBoundingClientRect();
      const step = ((e.clientX - rc.left - 56) / (rc.width - 76)) * pe.steps[pe.steps.length - 1];
      const snap = this.history.reduce((a, c) => (Math.abs(c.step - step) < Math.abs(a.step - step) ? c : a));
      this.session.scrubTo(snap);
      return;
    }
    const p = this.gridAt(e);
    if (!p || !this.slice) return;
    this.clicked = p.a;
    this.session.setAll(thetaAt(this.slice.center, this.slice.dirs, p.a));
  }

  private click3d(e: MouseEvent) {
    // double-click on a surface point moves the model there
    const s = this.slice;
    if (!s || s.k !== 2) return;
    const rc = new THREE.Raycaster();
    rc.setFromCamera(this.stage.ndc(e), this.stage.camera);
    const hit = rc.intersectObjects(this.g3.children.filter((c) => (c as THREE.Mesh).isMesh))[0];
    if (!hit) return;
    const a = [hit.point.x * s.R, hit.point.y * s.R];
    this.clicked = a;
    this.session.setAll(thetaAt(s.center, s.dirs, a));
  }
}

function hashPair(e: PointEval, carve: Carve): [number, number] {
  if (carve === 'none') return [0, 0];
  if (carve === 'all') {
    const { mlp, attn, pred } = e.h;
    return [(Math.imul(mlp[0] ^ attn[0], 16777619) ^ pred[0]) >>> 0, (Math.imul(mlp[1] ^ attn[1], 16777619) ^ pred[1]) >>> 0];
  }
  return e.h[carve];
}

function hashKey(e: PointEval, carve: Carve): string {
  const [a, b] = hashPair(e, carve);
  return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0');
}

function sameHash(a: PointEval, b: PointEval, carve: Carve = 'examples'): boolean {
  const x = hashPair(a, carve), y = hashPair(b, carve);
  return x[0] === y[0] && x[1] === y[1];
}
