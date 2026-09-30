// Bottom strip: loss (log scale) and accuracy over steps, and the timeline
// that scrubs through checkpoints. Crosshair tooltip on hover.

import { effect, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Series } from '../../vendor/plainpanel/plainpanel.js';
import type { Session, Snapshot } from '../app/session';
import { charts, rt } from '../app/state';
import { css, catColor, token } from './palette';
import { hideTip, showTip } from './heatmap';
import { fmtNum } from './controls';

interface Line {
  name: string;
  x: Series;
  y: Series;
  color: () => string;
  width: number;
  faint?: boolean;
}

class LineChart {
  readonly el: HTMLDivElement;
  private cv: HTMLCanvasElement;
  private lines: Line[];
  private log: boolean;
  private yFixed: [number, number] | null;
  private hoverX: number | null = null;
  private marker: number | null = null;

  constructor(title: string, lines: Line[], opts: { log?: boolean; y?: [number, number] } = {}) {
    this.lines = lines;
    this.log = !!opts.log;
    this.yFixed = opts.y ?? null;
    this.el = document.createElement('div');
    this.el.className = 'chart';
    const t = document.createElement('div');
    t.className = 'title';
    t.textContent = title;
    this.cv = document.createElement('canvas');
    this.el.append(this.cv, t);
    this.cv.addEventListener('pointermove', (e) => {
      const r = this.cv.getBoundingClientRect();
      this.hoverX = e.clientX - r.left;
      this.draw();
      const txt = this.tipText();
      if (txt) showTip(txt, e);
    });
    this.cv.addEventListener('pointerleave', () => {
      this.hoverX = null;
      hideTip();
      this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(this.el);
  }

  setTitle(t: string): void {
    this.el.querySelector('.title')!.textContent = t;
  }

  setMarker(step: number | null): void {
    this.marker = step;
    this.draw();
  }

  private frame() {
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    const pad = { l: 38, r: 58, t: 16, b: 16 };
    let xmax = 1, ymin = Infinity, ymax = -Infinity;
    for (const L of this.lines) {
      const xs = L.x.read(), ys = L.y.read();
      if (xs.length) xmax = Math.max(xmax, (xs[xs.length - 1] as number) ?? 1);
      for (const v of ys) {
        if (v === null || !Number.isFinite(v)) continue;
        const u = this.log ? Math.log10(Math.max(v, 1e-6)) : v;
        ymin = Math.min(ymin, u);
        ymax = Math.max(ymax, u);
      }
    }
    if (this.yFixed) [ymin, ymax] = this.yFixed;
    if (!Number.isFinite(ymin)) [ymin, ymax] = [0, 1];
    if (ymax - ymin < 1e-9) ymax = ymin + 1;
    const X = (s: number) => pad.l + (s / xmax) * (W - pad.l - pad.r);
    const Y = (v: number) => {
      const u = this.log ? Math.log10(Math.max(v, 1e-6)) : v;
      return pad.t + (1 - (u - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
    };
    return { W, H, pad, xmax, ymin, ymax, X, Y };
  }

  private tipText(): string {
    if (this.hoverX === null) return '';
    const f = this.frame();
    const step = ((this.hoverX - f.pad.l) / (f.W - f.pad.l - f.pad.r)) * f.xmax;
    if (step < 0) return '';
    const out = [`step ${Math.round(step)}`];
    for (const L of this.lines) {
      const xs = L.x.read(), ys = L.y.read();
      let best = -1;
      for (let i = 0; i < xs.length; i++) if ((xs[i] as number) <= step) best = i;
      if (best >= 0 && ys[best] !== null) out.push(`${L.name}: ${fmtNum(ys[best] as number)}`);
    }
    return out.join('\n');
  }

  draw(): void {
    const dpr = window.devicePixelRatio || 1;
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    if (!W || !H) return;
    if (this.cv.width !== Math.round(W * dpr)) this.cv.width = Math.round(W * dpr);
    if (this.cv.height !== Math.round(H * dpr)) this.cv.height = Math.round(H * dpr);
    const ctx = this.cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const f = this.frame();
    const muted = css(token('muted')), border = css(token('border'));
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = muted;
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    // y gridlines
    const ticks = this.log
      ? Array.from({ length: Math.floor(f.ymax) - Math.ceil(f.ymin) + 1 }, (_, i) => Math.pow(10, Math.ceil(f.ymin) + i))
      : [0, 0.25, 0.5, 0.75, 1].map((t) => f.ymin + t * (f.ymax - f.ymin));
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of ticks) {
      const y = f.Y(t);
      ctx.beginPath();
      ctx.moveTo(f.pad.l, y);
      ctx.lineTo(W - f.pad.r, y);
      ctx.stroke();
      ctx.fillText(fmtNum(t), f.pad.l - 4, y);
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`${Math.round(f.xmax)}`, W - f.pad.r - 20, H - 3);
    // lines
    const ends: { name: string; x: number; y: number }[] = [];
    for (const L of this.lines) {
      const xs = L.x.read(), ys = L.y.read();
      ctx.strokeStyle = L.color();
      ctx.globalAlpha = L.faint ? 0.35 : 1;
      ctx.lineWidth = L.width;
      ctx.beginPath();
      let pen = false;
      const stride = Math.max(1, Math.floor(xs.length / (W * 2)));
      let lastX = 0, lastY = 0;
      for (let i = 0; i < xs.length; i += stride) {
        const v = ys[i];
        if (v === null || !Number.isFinite(v as number)) {
          pen = false;
          continue;
        }
        lastX = f.X(xs[i] as number);
        lastY = f.Y(v as number);
        if (pen) ctx.lineTo(lastX, lastY);
        else ctx.moveTo(lastX, lastY);
        pen = true;
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (!L.faint && xs.length) ends.push({ name: L.name, x: Math.min(lastX + 4, W - f.pad.r + 4), y: lastY + 3 });
    }
    // direct labels at the line ends, pushed apart so they never collide
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 11) ends[i].y = ends[i - 1].y + 11;
    ctx.fillStyle = css(token('text'));
    for (const e of ends) ctx.fillText(e.name, e.x, Math.min(e.y, H - 2));
    // checkpoint marker
    if (this.marker !== null) {
      ctx.strokeStyle = css(token('sel'));
      ctx.lineWidth = 1.5;
      const x = f.X(this.marker);
      ctx.beginPath();
      ctx.moveTo(x, f.pad.t);
      ctx.lineTo(x, H - f.pad.b);
      ctx.stroke();
    }
    if (this.hoverX !== null) {
      ctx.strokeStyle = muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(this.hoverX, f.pad.t);
      ctx.lineTo(this.hoverX, H - f.pad.b);
      ctx.stroke();
    }
  }
}

export function buildCharts(root: HTMLElement, session: Session): { redraw(): void } {
  const c1 = () => css(catColor(0));
  const c2 = () => css(catColor(1));
  const gray = () => css(token('muted'));
  const lossChart = new LineChart('loss (log)', [
    { name: 'batch', x: charts.lossStep, y: charts.loss, color: gray, width: 1, faint: true },
    { name: 'train', x: charts.evalStep, y: charts.trainLoss, color: c1, width: 2 },
    { name: 'val', x: charts.evalStep, y: charts.valLoss, color: c2, width: 2 },
  ], { log: true });
  const accChart = new LineChart('sequence accuracy', [
    { name: 'train', x: charts.evalStep, y: charts.trainAcc, color: c1, width: 2 },
    { name: 'val', x: charts.evalStep, y: charts.valAcc, color: c2, width: 2 },
  ], { y: [0, 1] });
  const gradChart = new LineChart('gradient norm (log)', [
    { name: '|∇|', x: charts.lossStep, y: charts.gradNorm, color: gray, width: 1 },
  ], { log: true });

  // timeline
  const tl = document.createElement('div');
  tl.className = 'chart timeline';
  const head = document.createElement('div');
  head.className = 'muted';
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '0';
  slider.value = '0';
  slider.title = 'scrub through checkpoints of this run; every view shows those weights';
  const row = document.createElement('div');
  row.style.display = 'flex';
  row.style.gap = '4px';
  const live = document.createElement('button');
  live.textContent = 'back to live';
  const resume = document.createElement('button');
  resume.textContent = 'train from here';
  resume.title = 'branch: forget later checkpoints and continue training from these weights';
  const refresh = document.createElement('button');
  refresh.textContent = 'load history';
  refresh.title = 'fetch the checkpoint list from the trainer';
  row.append(refresh, live, resume);
  const info = document.createElement('div');
  info.className = 'note';
  info.textContent = 'Checkpoints thin out as the run grows, so they always span it. Editing a weight while scrubbing branches from that checkpoint.';
  tl.append(head, slider, row, info);
  root.append(lossChart.el, accChart.el, gradChart.el, tl);

  let hist: Snapshot[] = [];
  const setHist = (h: Snapshot[]) => {
    hist = h;
    slider.max = String(Math.max(0, h.length - 1));
    slider.value = String(h.length - 1);
  };
  const load = async () => setHist(await session.fetchHistory());
  refresh.onclick = load;
  slider.addEventListener('pointerdown', () => {
    if (!rt.scrubbing() && hist.length === 0) void load();
  });
  slider.addEventListener('input', async () => {
    if (!hist.length) await load();
    const s = hist[Number(slider.value)];
    if (s) session.scrubTo(s);
  });
  live.onclick = () => session.exitScrub();
  resume.onclick = () => session.resumeFromHere();

  effect(() => {
    const scrub = rt.scrubbing(), vs = rt.viewStep(), st = rt.step();
    head.textContent = scrub ? `showing checkpoint at step ${vs} (live: ${st})` : `live · step ${st}`;
    live.disabled = !scrub;
    resume.disabled = !scrub;
    lossChart.setMarker(scrub ? vs : null);
    accChart.setMarker(scrub ? vs : null);
  });
  effect(() => {
    rt.modelVersion();
    untracked(() => {
      hist = [];
      slider.max = '0';
      accChart.setTitle(session.trainer.spec.task.kind === 'text' ? 'token accuracy' : 'sequence accuracy');
    });
  });

  const redraw = () => {
    lossChart.draw();
    accChart.draw();
    gradChart.draw();
  };
  effect(() => {
    charts.loss.read();
    charts.trainLoss.read();
    redraw();
  });
  return { redraw };
}
