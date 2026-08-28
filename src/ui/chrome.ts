// The hand-written HTML chrome: top bar (menu, status, camera reset), the
// drag handles between panels, theme + no-CSS effects, and the bottom charts.
// Everything binds to the store; effects push store values into the DOM.

import { bind, computed, effect, type Series } from '../../vendor/plainpanel/plainpanel.js';
import * as s from '../state';

export interface ChromeActions {
  resetCamera(): void;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function setupChrome(actions: ChromeActions): void {
  const views = {
    themeLabel: computed(() => (s.theme() === 'dark' ? '☾ dark mode' : '☀ light mode')),
    step: s.stepCount,
    loss: computed(() => (Number.isNaN(s.lossVal()) ? '-' : s.lossVal().toFixed(4))),
    accuracy: computed(() => (Number.isNaN(s.accuracy()) ? '-' : `${(s.accuracy() * 100).toFixed(0)} %`)),
    backend: s.backendName,
  };

  // Panel resizing: pointer capture keeps move/up on the handle itself.
  let drag: { which: string; x: number; y: number; from: number } | null = null;
  const layoutSignal = (which: string) =>
    ({ left: s.leftW, right: s.rightW, bottom: s.bottomH })[which]!;
  const uiActions = {
    toggleTheme: () => s.theme(s.theme() === 'dark' ? 'light' : 'dark'),
    resetCamera: () => actions.resetCamera(),
    rzDown(e: Event) {
      const el = e.currentTarget as HTMLElement;
      const p = e as PointerEvent;
      el.setPointerCapture(p.pointerId);
      const which = el.dataset.rz!;
      drag = { which, x: p.clientX, y: p.clientY, from: layoutSignal(which)() };
      e.preventDefault();
    },
    rzMove(e: Event) {
      if (!drag) return;
      const p = e as PointerEvent;
      if (drag.which === 'left') s.leftW(clamp(drag.from + p.clientX - drag.x, 180, window.innerWidth * 0.45));
      else if (drag.which === 'right') s.rightW(clamp(drag.from - (p.clientX - drag.x), 180, window.innerWidth * 0.45));
      else s.bottomH(clamp(drag.from - (p.clientY - drag.y), 64, window.innerHeight * 0.6));
    },
    rzUp() {
      drag = null;
    },
  };

  bind(document.body, { actions: uiActions, views, ui: { noCss: s.noCss } });

  // Layout and theme: effects push store values into CSS.
  const rootStyle = document.documentElement.style;
  effect(() => rootStyle.setProperty('--left-w', `${s.leftW()}px`));
  effect(() => rootStyle.setProperty('--right-w', `${s.rightW()}px`));
  effect(() => rootStyle.setProperty('--bottom-h', `${s.bottomH()}px`));
  effect(() => {
    document.documentElement.dataset.theme = s.theme();
  });
  effect(() => {
    const off = s.noCss(); // "no CSS": the same HTML, zero styling
    for (const el of document.querySelectorAll<HTMLLinkElement | HTMLStyleElement>('link[rel=stylesheet], style')) {
      el.disabled = off;
    }
  });

  // Tutorial popover: the guide page loads on the first open, and its theme
  // follows the app theme (the page reads data-theme on its root element).
  const tutFrame = document.querySelector<HTMLIFrameElement>('#tutorialFrame')!;
  const stampTutorialTheme = () => {
    const doc = tutFrame.contentDocument;
    if (doc?.documentElement) doc.documentElement.dataset.theme = s.theme();
  };
  document.getElementById('tutorialPop')!.addEventListener('toggle', (e) => {
    if ((e as ToggleEvent).newState === 'open' && !tutFrame.src) tutFrame.src = tutFrame.dataset.src!;
  });
  tutFrame.addEventListener('load', stampTutorialTheme);
  effect(stampTutorialTheme);

  sparkline(document.querySelector<HTMLCanvasElement>('#lossChart canvas')!, s.lossSeries, {});
  sparkline(document.querySelector<HTMLCanvasElement>('#accChart canvas')!, s.accSeries, { min: 0, max: 1, percent: true });
}

interface SparkOptions {
  min?: number;
  max?: number;
  percent?: boolean;
}

// A canvas chart that fills its container, redraws on data, size, and theme.
function sparkline(canvas: HTMLCanvasElement, data: Series, opts: SparkOptions): void {
  const ctx = canvas.getContext('2d')!;
  const fmt = (v: number) => (opts.percent ? `${(v * 100).toFixed(0)} %` : v.toFixed(3));
  const ro = new ResizeObserver(() => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w && h && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
      s.chartResize(s.chartResize() + 1);
    }
  });
  ro.observe(canvas);

  effect(() => {
    s.chartResize();
    const light = s.theme() === 'light';
    const stroke = light ? '#1f6fd6' : '#8fc7ff';
    const text = light ? '#5f6a7a' : '#8a93a5';
    const values = data.read();
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px monospace';
    ctx.fillStyle = text;
    const finite = values.filter((v): v is number => v !== null);
    if (finite.length < 2) {
      ctx.fillText('no data yet', 6, 14);
      return;
    }
    const min = opts.min ?? Math.min(...finite);
    const max = opts.max ?? Math.max(...finite);
    const span = max - min || 1;
    const left = 52;
    const right = W - 6;
    const top = 6;
    const bottom = H - 6;
    ctx.fillText(fmt(max), 4, top + 9);
    ctx.fillText(fmt(min), 4, bottom);
    const last = finite[finite.length - 1];
    const lastText = `last ${fmt(last)}  n=${finite.length}`;
    ctx.fillText(lastText, right - ctx.measureText(lastText).width, top + 9);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    let pen = false;
    values.forEach((v, i) => {
      if (v === null) {
        pen = false;
        return;
      }
      const x = left + (i / Math.max(values.length - 1, 1)) * (right - left);
      const y = bottom - ((v - min) / span) * (bottom - top);
      if (pen) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      pen = true;
    });
    ctx.stroke();
  });
}
