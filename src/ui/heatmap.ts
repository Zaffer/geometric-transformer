// An editable heatmap of one matrix. The basic unit of "adjust every
// parameter": each cell is one scalar.
//
//   drag up/down   change the value (shift: fine)
//   wheel          nudge
//   double-click   type an exact value
//   alt-click      set to 0
//   right-click    freeze / unfreeze this one scalar (hatched = frozen)
//   hover          name, value, gradient

import { css, diverging, sequential, token } from './palette';
import { fmtNum } from './controls';

export interface HeatOpts {
  rows: number;
  cols: number;
  rowLabels?: string[];
  colLabels?: string[];
  /** column group size: draws a divider every n columns (heads) */
  colGroup?: number;
  editable: boolean;
  /** sequential [0,1] colors instead of diverging (probabilities) */
  positive?: boolean;
  /** max cell size in px */
  maxCell?: number;
  /** width to fit into */
  width?: number;
  name?: string;
  onEdit?: (index: number, value: number, final: boolean) => void;
  onMask?: (index: number) => void;
  onHover?: (index: number | null) => void;
  /** click on a read-only map */
  onPick?: (index: number) => void;
  describe?: (index: number) => string;
}

const tip = () => document.getElementById('tooltip')!;

export function showTip(text: string, e: MouseEvent): void {
  const t = tip();
  t.textContent = text;
  t.hidden = false;
  const x = Math.min(e.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}

export function hideTip(): void {
  tip().hidden = true;
}

export class Heatmap {
  readonly el: HTMLDivElement;
  private cv: HTMLCanvasElement;
  private o: HeatOpts;
  private data: ArrayLike<number> = [];
  private mask: Uint8Array | null = null;
  private scale = 1;
  private cell = 12;
  private lw = 0;
  private lh = 0;
  private hover = -1;
  private hlRow = -1;
  private hlCol = -1;
  private drag: { index: number; y0: number; v0: number; moved: boolean } | null = null;

  constructor(o: HeatOpts) {
    this.o = o;
    this.el = document.createElement('div');
    this.el.className = 'heatwrap';
    this.cv = document.createElement('canvas');
    this.cv.className = 'heat';
    if (!o.editable) this.cv.style.cursor = 'crosshair';
    this.el.append(this.cv);
    this.layout();
    this.cv.addEventListener('pointerdown', (e) => this.down(e));
    this.cv.addEventListener('pointermove', (e) => this.move(e));
    this.cv.addEventListener('pointerup', (e) => this.up(e));
    this.cv.addEventListener('pointercancel', (e) => this.up(e));
    this.cv.addEventListener('pointerleave', () => {
      if (!this.drag) {
        this.hover = -1;
        hideTip();
        this.o.onHover?.(null);
        this.draw();
      }
    });
    this.cv.addEventListener('dblclick', (e) => this.dbl(e));
    this.cv.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const i = this.hit(e);
      if (i >= 0 && this.o.editable) this.o.onMask?.(i);
    });
    this.cv.addEventListener('wheel', (e) => {
      if (!this.o.editable) return;
      const i = this.hit(e);
      if (i < 0) return;
      e.preventDefault();
      const step = this.scale * 0.02 * (e.shiftKey ? 0.1 : 1);
      this.o.onEdit?.(i, Number(this.data[i]) - Math.sign(e.deltaY) * step, true);
    }, { passive: false });
  }

  get rows(): number {
    return this.o.rows;
  }

  get cols(): number {
    return this.o.cols;
  }

  private layout() {
    const o = this.o;
    const hasRow = !!o.rowLabels?.length, hasCol = !!o.colLabels?.length;
    this.lw = hasRow ? Math.min(46, 8 + 6.5 * Math.max(...o.rowLabels!.map((s) => s.length))) : 0;
    this.lh = hasCol ? 14 : 0;
    const avail = (o.width ?? 300) - this.lw - 2;
    this.cell = Math.max(4, Math.min(o.maxCell ?? 26, Math.floor(avail / o.cols)));
    const dpr = window.devicePixelRatio || 1;
    const w = this.lw + this.cell * o.cols + 1, h = this.lh + this.cell * o.rows + 1;
    this.cv.width = Math.round(w * dpr);
    this.cv.height = Math.round(h * dpr);
    this.cv.style.width = `${w}px`;
    this.cv.style.height = `${h}px`;
  }

  /** New values. scale = the |value| that maps to full color (default: max |value|). */
  set(data: ArrayLike<number>, opts: { scale?: number; mask?: Uint8Array | null } = {}): void {
    this.data = data;
    this.mask = opts.mask ?? null;
    if (opts.scale !== undefined) this.scale = opts.scale;
    else {
      let m = 0;
      for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i])) m = Math.max(m, Math.abs(data[i]));
      this.scale = this.o.positive ? 1 : m || 1;
    }
    this.draw();
  }

  highlight(row: number, col: number): void {
    this.hlRow = row;
    this.hlCol = col;
    this.draw();
  }

  private hit(e: MouseEvent): number {
    const r = this.cv.getBoundingClientRect();
    const x = e.clientX - r.left - this.lw, y = e.clientY - r.top - this.lh;
    const j = Math.floor(x / this.cell), i = Math.floor(y / this.cell);
    if (i < 0 || j < 0 || i >= this.o.rows || j >= this.o.cols) return -1;
    return i * this.o.cols + j;
  }

  private label(index: number): string {
    const o = this.o;
    const i = Math.floor(index / o.cols), j = index % o.cols;
    const rl = o.rowLabels?.[i] ?? String(i), cl = o.colLabels?.[j] ?? String(j);
    const head = o.rows === 1 ? `${o.name ?? ''}[${cl}]` : `${o.name ?? ''}[${rl}, ${cl}]`;
    const base = `${head} = ${fmtNum(Number(this.data[index]))}`;
    const extra = o.describe ? o.describe(index) : '';
    const frozen = this.mask?.[index] ? '\nfrozen (right-click to unfreeze)' : '';
    const how = o.editable ? '\ndrag ↕ · wheel · dbl-click type · alt-click 0 · right-click freeze' : '';
    return base + (extra ? `\n${extra}` : '') + frozen + how;
  }

  private down(e: PointerEvent) {
    const i = this.hit(e);
    if (i >= 0 && !this.o.editable && e.button === 0) this.o.onPick?.(i);
    if (i < 0 || !this.o.editable || e.button !== 0) return;
    if (e.altKey) {
      this.o.onEdit?.(i, 0, true);
      return;
    }
    this.cv.setPointerCapture(e.pointerId);
    this.drag = { index: i, y0: e.clientY, v0: Number(this.data[i]), moved: false };
  }

  private move(e: PointerEvent) {
    if (this.drag) {
      const dy = this.drag.y0 - e.clientY;
      if (Math.abs(dy) > 1) this.drag.moved = true;
      const sens = Math.max(this.scale, 0.05) * 0.01 * (e.shiftKey ? 0.1 : 1);
      this.o.onEdit?.(this.drag.index, this.drag.v0 + dy * sens, false);
      showTip(this.label(this.drag.index), e);
      return;
    }
    const i = this.hit(e);
    if (i !== this.hover) {
      this.hover = i;
      this.o.onHover?.(i >= 0 ? i : null);
      this.draw();
    }
    if (i >= 0) showTip(this.label(i), e);
    else hideTip();
  }

  private up(e: PointerEvent) {
    if (!this.drag) return;
    const d = this.drag;
    this.drag = null;
    if (d.moved) this.o.onEdit?.(d.index, Number(this.data[d.index]), true);
    try {
      this.cv.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
  }

  private dbl(e: MouseEvent) {
    const i = this.hit(e);
    if (i < 0 || !this.o.editable) return;
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = 'any';
    inp.value = String(Number(Number(this.data[i]).toPrecision(6)));
    const r = this.cv.getBoundingClientRect();
    const col = i % this.o.cols, row = Math.floor(i / this.o.cols);
    Object.assign(inp.style, {
      position: 'fixed',
      left: `${r.left + this.lw + col * this.cell - 20}px`,
      top: `${r.top + this.lh + row * this.cell - 2}px`,
      width: '90px',
      zIndex: '4000',
    });
    document.body.append(inp);
    inp.focus();
    inp.select();
    const done = (commit: boolean) => {
      if (commit && inp.value.trim() !== '' && Number.isFinite(Number(inp.value))) this.o.onEdit?.(i, Number(inp.value), true);
      inp.remove();
    };
    inp.addEventListener('keydown', (k) => {
      if (k.key === 'Enter') done(true);
      if (k.key === 'Escape') done(false);
      k.stopPropagation();
    });
    inp.addEventListener('blur', () => done(true));
  }

  draw(): void {
    const o = this.o, c = this.cell;
    const ctx = this.cv.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    const text = css(token('text')), muted = css(token('muted'));
    ctx.font = `${Math.min(10, Math.max(8, c * 0.42))}px ui-monospace, monospace`;
    ctx.textBaseline = 'middle';
    // labels
    ctx.fillStyle = muted;
    if (o.rowLabels) {
      ctx.textAlign = 'right';
      for (let i = 0; i < o.rows; i++) {
        ctx.fillStyle = i === this.hlRow ? css(token('sel')) : muted;
        ctx.fillText(o.rowLabels[i] ?? '', this.lw - 4, this.lh + i * c + c / 2);
      }
    }
    if (o.colLabels) {
      ctx.textAlign = 'center';
      const every = Math.ceil(12 / c);
      for (let j = 0; j < o.cols; j += every) {
        ctx.fillStyle = j === this.hlCol ? css(token('sel')) : muted;
        ctx.fillText(o.colLabels[j] ?? '', this.lw + j * c + c / 2, 7);
      }
    }
    // cells
    const showNums = c >= 24;
    ctx.textAlign = 'center';
    for (let i = 0; i < o.rows; i++)
      for (let j = 0; j < o.cols; j++) {
        const k = i * o.cols + j;
        const v = Number(this.data[k]);
        const x = this.lw + j * c, y = this.lh + i * c;
        ctx.fillStyle = css(o.positive ? sequential(v) : diverging(v / this.scale));
        if (!Number.isFinite(v)) ctx.fillStyle = css(token('panel'));
        ctx.fillRect(x, y, c - 1, c - 1);
        if (this.mask?.[k]) {
          ctx.strokeStyle = text;
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let s = -c; s < c; s += 4) {
            ctx.moveTo(x + Math.max(0, s), y + Math.max(0, -s));
            ctx.lineTo(x + Math.min(c - 1, s + c - 1), y + Math.min(c - 1, c - 1 - s));
          }
          ctx.stroke();
        }
        if (showNums && Number.isFinite(v)) {
          ctx.fillStyle = text;
          const a = Math.abs(v);
          ctx.fillText(a >= 10 ? v.toFixed(0) : a >= 1 ? v.toFixed(1) : v.toFixed(2).replace('0.', '.'), x + c / 2, y + c / 2);
        }
      }
    // head dividers
    if (o.colGroup && o.colGroup < o.cols) {
      ctx.strokeStyle = css(token('bg'));
      ctx.lineWidth = 2;
      for (let j = o.colGroup; j < o.cols; j += o.colGroup) {
        ctx.beginPath();
        ctx.moveTo(this.lw + j * c - 0.5, this.lh);
        ctx.lineTo(this.lw + j * c - 0.5, this.lh + o.rows * c);
        ctx.stroke();
      }
    }
    // highlights
    ctx.strokeStyle = css(token('sel'));
    ctx.lineWidth = 1.5;
    if (this.hlRow >= 0 && this.hlRow < o.rows) ctx.strokeRect(this.lw + 0.5, this.lh + this.hlRow * c + 0.5, o.cols * c - 1, c - 1);
    if (this.hlCol >= 0 && this.hlCol < o.cols) ctx.strokeRect(this.lw + this.hlCol * c + 0.5, this.lh + 0.5, c - 1, o.rows * c - 1);
    if (this.hover >= 0) {
      const i = Math.floor(this.hover / o.cols), j = this.hover % o.cols;
      ctx.strokeStyle = text;
      ctx.strokeRect(this.lw + j * c + 0.5, this.lh + i * c + 0.5, c - 2, c - 2);
    }
  }
}
