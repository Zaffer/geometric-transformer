// Wiring: spec signals → session (mirror model + training worker) → views.

import '../vendor/plainpanel/plainpanel.css';
import './style.css';
import { bind, computed, effect, untracked } from '../vendor/plainpanel/plainpanel.js';
import { cloneSpec, normalizeSpec, structuralKey, type Spec } from './engine/spec';
import { Session } from './app/session';
import { initInspect } from './app/inspect';
import { readSpec, rt, spec, ui, writeSpec } from './app/state';
import { decodeSpec, encodeSpec, preset, saveFile, type SaveFile } from './app/presets';
import { buildLeftPanel, setTaskNote } from './ui/leftPanel';
import { buildArchMap } from './ui/archmap';
import { buildInspector } from './ui/inspector';
import { buildCharts } from './ui/charts';
import { clearPalette } from './ui/palette';
import { fmtNum } from './ui/controls';
import { HELP_HTML } from './ui/help';
import type { View } from './views/view';
import { ResidualView } from './views/residual';
import { AttentionView } from './views/attention';
import { MlpView } from './views/mlp';
import { PlaygroundView } from './views/playground';

// ---- initial spec: URL hash > default ----
const fromUrl = location.hash.startsWith('#s=') ? decodeSpec(location.hash.slice(3)) : null;
if (fromUrl) writeSpec(fromUrl);
const params = new URLSearchParams(location.search);

const session = new Session(readSpec());
initInspect(session);

// ---- spec → session ----
let current: Spec = cloneSpec(session.trainer.spec);
let pending = 0;
effect(() => {
  const s = readSpec();
  untracked(() => {
    clearTimeout(pending);
    pending = window.setTimeout(() => apply(normalizeSpec(cloneSpec(s))), 120);
  });
});

function apply(s: Spec) {
  const initChanged = JSON.stringify(s.init) !== JSON.stringify(current.init);
  if (structuralKey(s) !== structuralKey(current)) {
    session.rebuild(s, !initChanged);
    clampUi();
  } else if (JSON.stringify(s.optim) !== JSON.stringify(current.optim)) {
    session.setOptim(s.optim);
  }
  current = cloneSpec(s);
  history.replaceState(null, '', `${location.pathname}${location.search}#s=${encodeSpec(s)}`);
}

function clampUi() {
  const stages = session.model.stages();
  if (ui.stage() >= stages.length) ui.stage(stages.length - 1);
  if (ui.probe() >= session.data.val.length + session.data.train.length) ui.probe(0);
  ui.probeTokens(null);
  if (!session.model.params.some((p) => p.node === ui.focus()) && !['tokens', 'loss', 'unembed'].includes(ui.focus())) {
    const first = session.model.params.find((p) => p.node.endsWith('.mlp'))?.node ?? 'embed';
    ui.focus(first);
  }
}

effect(() => {
  session.setSpeed(rt.stepsPerSec());
});

setTaskNote(() => {
  const d = session.data;
  return `${d.about}. ${d.train.length} train / ${d.val.length} val examples, context ${d.T}.`;
});

// ---- actions (top bar, keyboard) ----
const actions = {
  toggleRun: () => session.run(!rt.running()),
  step1: () => session.steps(1),
  step100: () => session.steps(100),
  resetWeights: () => {
    session.run(false);
    session.rebuild(session.trainer.spec, false);
  },
  toggleTheme: () => ui.theme(ui.theme() === 'dark' ? 'light' : 'dark'),
  copyLink: async () => {
    await navigator.clipboard?.writeText(location.href).catch(() => {});
    flash('link copied: it reproduces this spec (not your hand edits: use save for those)');
  },
  save: () => {
    const f = saveFile(session.trainer.spec, session.model, rt.viewStep());
    const blob = new Blob([JSON.stringify(f)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `transformer-step${f.step}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  },
  load: () => (document.getElementById('fileIn') as HTMLInputElement).click(),
};

document.getElementById('fileIn')!.addEventListener('change', async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  try {
    const data = JSON.parse(await f.text()) as SaveFile;
    if (data.format !== 'geometric-transformer/2') throw new Error('not a geometric-transformer v2 file');
    loadSave(data);
    flash(`loaded ${f.name}`);
  } catch (err) {
    flash(`load failed: ${(err as Error).message}`);
  }
  (e.target as HTMLInputElement).value = '';
});

function loadSave(data: SaveFile) {
  session.run(false);
  const s = normalizeSpec(data.spec);
  writeSpec(s);
  clearTimeout(pending);
  current = cloneSpec(s);
  session.rebuild(s, false);
  for (const p of session.model.params) {
    const v = data.params[p.name];
    if (v && v.length === p.t.size) p.t.data.set(v);
    const m = data.meta?.[p.name];
    if (m) Object.assign(p, { lrMult: m.lrMult, frozen: m.frozen, mask: m.mask ? Uint8Array.from(m.mask) : null });
  }
  session.trainer.step = data.step ?? 0;
  session.rebuild(session.trainer.spec, true);
  clampUi();
}

const views = {
  runLabel: computed(() => (rt.running() ? 'pause' : 'train')),
  step: computed(() => (rt.scrubbing() ? `${rt.viewStep()} / ${rt.step()}` : String(rt.step()))),
  loss: computed(() => fmtNum(rt.train()?.loss ?? NaN)),
  trainAcc: computed(() => acc(rt.train())),
  valAcc: computed(() => acc(rt.val())),
  nParams: computed(() => {
    rt.modelVersion();
    return String(session.model.numParams());
  }),
  speed: computed(() => (rt.running() ? `${rt.stepsPerSecMeasured()} steps/s` : '')),
  themeLabel: computed(() => (ui.theme() === 'dark' ? 'light' : 'dark')),
};

function acc(e: { seqAcc: number; tokenAcc: number; n: number } | null): string {
  if (!e || !e.n) return '–';
  const v = session.trainer.spec.task.kind === 'text' ? e.tokenAcc : e.seqAcc;
  return `${(v * 100).toFixed(0)}%`;
}

bind(document.getElementById('top')!, { actions, views });

// steps / second readout
{
  let lastStep = 0, lastT = performance.now();
  setInterval(() => {
    const now = performance.now(), st = untracked(() => rt.step());
    rt.stepsPerSecMeasured(Math.round(((st - lastStep) * 1000) / (now - lastT)));
    lastStep = st;
    lastT = now;
  }, 1000);
}

// ---- panels ----
buildLeftPanel(document.getElementById('left')!, {
  reinit: actions.resetWeights,
  preset: (name) => {
    session.run(false);
    const s = preset(name);
    writeSpec(s);
    clearTimeout(pending);
    current = cloneSpec(s);
    session.rebuild(s, false);
    clampUi();
    history.replaceState(null, '', `${location.pathname}${location.search}#s=${encodeSpec(s)}`);
  },
});
buildArchMap(document.getElementById('map')!, session, (node) => {
  ui.focus(node);
  ui.focusParam('');
  // follow the focus in the node-specific views
  const stages = session.model.stages();
  const m = node.match(/^L(\d+)\.(attn|mlp)/);
  if (m) {
    const id = `L${m[1]}.resid.${m[2] === 'attn' ? 'mid' : 'out'}`;
    const k = stages.findIndex((s) => s.id === id);
    if (k >= 0) ui.stage(k);
    if (ui.tab() === 'attention' && m[2] === 'mlp') ui.tab('mlp');
    else if (ui.tab() === 'mlp' && m[2] === 'attn') ui.tab('attention');
  } else if (node === 'embed' || node === 'pos') ui.stage(0);
  else if (node === 'lnf' || node === 'unembed' || node === 'loss') ui.stage(stages.length - 1);
});
buildInspector(document.getElementById('right')!, session);
const charts = buildCharts(document.getElementById('bottom')!, session);
document.getElementById('helpPop')!.innerHTML = HELP_HTML;

// ---- tabs ----
const viewsEl = document.getElementById('views')!;
const tabsEl = document.getElementById('tabs')!;
const allViews: View[] = [
  new ResidualView(session),
  new AttentionView(session),
  new MlpView(session),
  new PlaygroundView(session),
];
for (const v of allViews) {
  v.el.classList.add('view');
  v.el.hidden = true;
  viewsEl.append(v.el);
  const b = document.createElement('button');
  b.textContent = v.title;
  b.setAttribute('role', 'tab');
  b.dataset.tab = v.id;
  b.onclick = () => ui.tab(v.id);
  tabsEl.append(b);
}
effect(() => {
  const t = ui.tab();
  untracked(() => {
    for (const v of allViews) {
      const on = v.id === t;
      v.el.hidden = !on;
      tabsEl.querySelector(`[data-tab="${v.id}"]`)!.setAttribute('aria-selected', String(on));
      if (on) v.activate();
      else v.deactivate();
    }
  });
});
if (params.get('tab')) ui.tab(params.get('tab')!);

// ---- theme ----
effect(() => {
  document.documentElement.dataset.theme = ui.theme();
  clearPalette();
  untracked(() => {
    rt.paramsVersion(rt.paramsVersion() + 1);
    charts.redraw();
    for (const v of allViews) v.themeChanged?.();
  });
});


// ---- keyboard ----
window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (e.key === ' ') {
    e.preventDefault();
    actions.toggleRun();
  } else if (e.key === 'ArrowRight' && !e.shiftKey) actions.step1();
  else if (e.key === 'r' || e.key === 'R') actions.resetWeights();
  else if (e.key === '[') ui.stage(Math.max(0, ui.stage() - 1));
  else if (e.key === ']') ui.stage(Math.min(session.model.stages().length - 1, ui.stage() + 1));
  else if (e.key >= '1' && e.key <= '9') {
    const v = allViews[Number(e.key) - 1];
    if (v) ui.tab(v.id);
  }
});

// ---- resize handles ----
{
  const root = document.documentElement;
  const vars = { left: '--wl', map: '--wm', right: '--wr', bottom: '--hb' } as const;
  const place = () => {
    const L = document.getElementById('left')!.getBoundingClientRect();
    const M = document.getElementById('map')!.getBoundingClientRect();
    const R = document.getElementById('right')!.getBoundingClientRect();
    const B = document.getElementById('bottom')!.getBoundingClientRect();
    for (const h of document.querySelectorAll<HTMLElement>('.rz')) {
      const k = h.dataset.rz as keyof typeof vars;
      if (k === 'bottom') Object.assign(h.style, { left: '0px', width: '100vw', top: `${B.top - 5}px` });
      else {
        const x = k === 'left' ? L.right : k === 'map' ? M.right : R.left - 5;
        Object.assign(h.style, { left: `${x - (k === 'right' ? 0 : 0)}px`, top: `${L.top}px`, height: `${L.height}px` });
      }
    }
  };
  for (const h of document.querySelectorAll<HTMLElement>('.rz')) {
    const k = h.dataset.rz as keyof typeof vars;
    h.addEventListener('pointerdown', (e) => {
      h.setPointerCapture(e.pointerId);
      h.classList.add('drag');
      const start = { x: e.clientX, y: e.clientY, v: parseFloat(getComputedStyle(root).getPropertyValue(vars[k])) };
      const move = (ev: PointerEvent) => {
        let v = start.v;
        if (k === 'left' || k === 'map') v += ev.clientX - start.x;
        else if (k === 'right') v -= ev.clientX - start.x;
        else v -= ev.clientY - start.y;
        root.style.setProperty(vars[k], `${Math.max(k === 'bottom' ? 80 : 120, Math.min(k === 'bottom' ? 500 : 700, v))}px`);
        place();
      };
      const up = () => {
        h.classList.remove('drag');
        h.removeEventListener('pointermove', move);
        h.removeEventListener('pointerup', up);
        window.dispatchEvent(new Event('resize'));
      };
      h.addEventListener('pointermove', move);
      h.addEventListener('pointerup', up);
    });
  }
  window.addEventListener('resize', place);
  requestAnimationFrame(place);
}

// ---- errors, flash ----
function flash(msg: string) {
  const d = document.createElement('div');
  d.className = 'banner';
  d.style.position = 'fixed';
  d.style.top = '42px';
  d.style.zIndex = '5000';
  d.style.background = 'var(--accent)';
  d.style.color = 'var(--bg)';
  d.textContent = msg;
  document.body.append(d);
  setTimeout(() => d.remove(), 2600);
}
effect(() => {
  const e = rt.error();
  if (e) untracked(() => flash(e.split('\n')[0]));
});

if (params.get('train') === '1') session.run(true);

// test / debug handle
(window as unknown as Record<string, unknown>).__gt = {
  session, rt, ui, spec, actions, views: allViews, loadSave,
  saveNow: () => saveFile(session.trainer.spec, session.model, rt.viewStep()),
  encode: () => encodeSpec(session.trainer.spec),
};
