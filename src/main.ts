// State, the frame loop, and the controls.

import { type Config, forward, init, layout } from './model';
import { Network, describe } from './network';
import { type Mode, PALETTE, Scene, type ViewState } from './scene';
import { TASKS, evaluate, trainer } from './train';

const cfg: Config = { V: 5, T: 5, H: 4 };
const L = layout(cfg);
const LR = 0.01, BATCH = 32, STEPS_PER_FRAME = 4;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const state = {
  task: 'max',
  seed: 1,
  th: init(cfg, 1),
  tr: trainer(cfg, L, TASKS.max, 1),
  running: true,
  tokens: [1, 3, 0, 2, 1],
  pos: 3,
  selected: null as number | null,
  wiggle: false,
  view: { mode: 'output', neuron: 0, flow: 0, lift: 1, creases: true, sheets: true } as ViewState,
  loss: [] as number[],
  acc: NaN,
  accDue: 0,
};

const scene = new Scene($<HTMLCanvasElement>('gl'), cfg, L);
const net = new Network(document.getElementById('net') as unknown as SVGSVGElement, cfg, L, {
  select,
  nudge(i, by) { select(i); state.th[i] += by; state.accDue = 0; },
  neuron(h) { state.view.neuron = h; state.view.mode = 'mlp'; },
  token(w) { state.tokens[state.pos] = w; },
  position(p) { state.pos = p; },
});

function select(i: number | null) {
  state.selected = i;
  if (i === null) { state.wiggle = false; return; }
  const d = describe(L, i);
  state.view.mode = d.mode;
  if (d.neuron !== undefined) state.view.neuron = d.neuron;
  if (d.flow !== undefined) state.view.flow = d.flow;
}

function restart(newSeed: boolean) {
  if (newSeed) state.seed++;
  state.th = init(cfg, state.seed);
  state.tr = trainer(cfg, L, TASKS[state.task], state.seed);
  state.loss = [];
  state.accDue = 0;
}

// ---- controls -------------------------------------------------------------------------------

$('blurb').textContent =
  `A whole transformer in ${L.size} parameters: one attention head, ${cfg.H} ReLU neurons, and a 2-dimensional residual stream, so everything it does happens on one plane.`;

const task = $<HTMLSelectElement>('task');
for (const k in TASKS) task.add(new Option(`task: ${TASKS[k].name}`, k));
task.onchange = () => { state.task = task.value; restart(false); };
$('run').onclick = () => { state.running = !state.running; };
$('reset').onclick = () => restart(true);

const slider = $<HTMLInputElement>('param-slider');
slider.oninput = () => { if (state.selected !== null) { state.th[state.selected] = Number(slider.value); state.accDue = 0; } };
$<HTMLInputElement>('wiggle').onchange = (e) => { state.wiggle = (e.target as HTMLInputElement).checked; };
$('zero').onclick = () => { if (state.selected !== null) { state.th[state.selected] = 0; state.accDue = 0; } };
$('deselect').onclick = () => select(null);

const MODES: Record<Mode, { tab: string; text: string }> = {
  attention: {
    tab: 'Attention',
    text: 'Height is the score the focused point gives each earlier point, wherever the focused point might be. The highest sheet wins: colour is who it attends to most.',
  },
  mlp: {
    tab: 'MLP',
    text: 'Height is one neuron’s input. The ReLU cuts it off at zero, which folds the sheet along a line. Each colour is one set of neurons being on: one polytope.',
  },
  output: {
    tab: 'Output',
    text: 'Height is each token’s logit. The highest sheet wins: colour is the token the model would say if the focused point were there.',
  },
};
const tabs = {} as Record<Mode, HTMLButtonElement>;
for (const m of Object.keys(MODES) as Mode[]) {
  const b = (tabs[m] = document.createElement('button'));
  b.textContent = MODES[m].tab;
  b.onclick = () => { state.view.mode = m; };
  $('modes').appendChild(b);
}

const flow = $<HTMLInputElement>('flow'), lift = $<HTMLInputElement>('lift');
flow.oninput = () => { state.view.flow = Number(flow.value); };
lift.oninput = () => { state.view.lift = Number(lift.value); };
$<HTMLInputElement>('creases').onchange = (e) => { state.view.creases = (e.target as HTMLInputElement).checked; };
$<HTMLInputElement>('sheets').onchange = (e) => { state.view.sheets = (e.target as HTMLInputElement).checked; };

$('legend').onclick = (e) => {
  const h = (e.target as HTMLElement).closest<HTMLElement>('[data-neuron]')?.dataset.neuron;
  if (h !== undefined) state.view.neuron = Number(h);
};
$('chips').onclick = (e) => {
  const p = (e.target as HTMLElement).closest<HTMLElement>('[data-pos]')?.dataset.pos;
  if (p !== undefined) state.pos = Number(p);
};
$('picker').onclick = (e) => {
  const w = (e.target as HTMLElement).closest<HTMLElement>('[data-tok]')?.dataset.tok;
  if (w !== undefined) state.tokens[state.pos] = Number(w);
};

// The left panel's edge drags. The network diagram is sized by the panel's width, so wider is bigger.
{
  const split = $('split'), root = document.documentElement, KEY = 'left-panel-width', DEFAULT = 440;
  const set = (w: number, save = true) => {
    w = Math.round(Math.min(Math.max(w, 320), Math.max(320, window.innerWidth - 380)));
    root.style.setProperty('--left', `${w}px`);
    if (save) try { localStorage.setItem(KEY, String(w)); } catch { /* private window: just don't remember */ }
    return w;
  };
  let width = DEFAULT;
  try { width = Number(localStorage.getItem(KEY)) || DEFAULT; } catch { /* as above */ }
  width = set(width, false);
  const move = (e: PointerEvent) => { width = set(e.clientX); };
  const stop = () => {
    document.body.classList.remove('resizing');
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', stop);
    window.removeEventListener('pointercancel', stop);
  };
  split.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    document.body.classList.add('resizing');
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
  });
  split.addEventListener('dblclick', () => { width = set(DEFAULT); });
  split.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') width = set(width - 24);
    else if (e.key === 'ArrowRight') width = set(width + 24);
  });
  window.addEventListener('resize', () => { width = set(width, false); });
}

/** Assign HTML only when it changed, so hover and focus survive the frame loop. */
const cache = new Map<HTMLElement, string>();
const html = (e: HTMLElement, s: string) => { if (cache.get(e) !== s) { cache.set(e, s); e.innerHTML = s; } };
const text = (e: HTMLElement, s: string) => { if (e.textContent !== s) e.textContent = s; };
const swatch = (color: string, label: string, extra = '') => `<span class="sw" ${extra}><i style="background:${color}"></i>${label}</span>`;

// ---- frame loop -----------------------------------------------------------------------------

function frame(now: number) {
  if (state.running) {
    let loss = 0;
    for (let k = 0; k < STEPS_PER_FRAME; k++) loss += state.tr.step(state.th, BATCH, LR);
    state.loss.push(loss / STEPS_PER_FRAME);
    if (state.loss.length > 4000) state.loss.splice(0, 2000);
  }

  // the weights the picture shows: the real ones, plus the wiggle on the selected parameter
  let th = state.th;
  if (state.wiggle && state.selected !== null) {
    th = state.th.slice();
    th[state.selected] += 0.7 * Math.sin(now / 320);
  }

  const { tokens, pos, view } = state;
  const t = forward(cfg, L, th, tokens);
  const targets = TASKS[state.task].targets(tokens);
  scene.update(th, t, tokens, pos, view);
  scene.render();
  net.update(th, t, tokens, pos, targets[pos], state.selected, view.neuron, view.mode);

  if (now > state.accDue) {
    state.acc = evaluate(cfg, L, state.th, TASKS[state.task]).acc;
    state.accDue = now + (state.running ? 500 : 1e12);
  }
  draw(t, targets);
  requestAnimationFrame(frame);
}

function draw(t: ReturnType<typeof forward>, targets: number[]) {
  const { tokens, pos, view } = state;
  text($('run'), state.running ? 'Pause' : 'Train');
  text($('steps'), String(state.tr.steps));
  text($('loss'), state.loss.length ? state.loss[state.loss.length - 1].toFixed(3) : '–');
  text($('acc'), Number.isFinite(state.acc) ? `${(100 * state.acc).toFixed(1)}%` : '–');
  text($('about'), TASKS[state.task].about);
  spark();

  html($('chips'), tokens.map((w, i) => {
    let say = 0;
    for (let k = 1; k < cfg.V; k++) if (t.p[i * cfg.V + k] > t.p[i * cfg.V + say]) say = k;
    const ok = say === targets[i];
    return `<div class="chip${i === pos ? ' on' : ''}${i > pos ? ' later' : ''}" data-pos="${i}" title="position ${i}">
      <span class="tok" style="color:${PALETTE[w]}">${w}</span>
      <span class="says${ok ? '' : ' bad'}">says ${say}${ok ? '' : ` ≠ ${targets[i]}`}</span></div>`;
  }).join(''));
  html($('picker'), `token at position ${pos}: ` + Array.from({ length: cfg.V }, (_, w) =>
    `<button data-tok="${w}" class="${w === tokens[pos] ? 'on' : ''}" style="color:${PALETTE[w]}">${w}</button>`).join(''));

  for (const m of Object.keys(MODES) as Mode[]) tabs[m].classList.toggle('on', m === view.mode);
  text($('mode-text'), MODES[view.mode].text);
  html($('legend'),
    view.mode === 'attention'
      ? Array.from({ length: pos + 1 }, (_, j) => swatch(PALETTE[j], j === pos ? 'itself' : `position ${j} (token ${tokens[j]})`)).join('')
      : view.mode === 'output'
        ? Array.from({ length: cfg.V }, (_, w) => swatch(PALETTE[w], `says ${w}`)).join('')
        : Array.from({ length: cfg.H }, (_, h) =>
            `<span class="sw pick${h === view.neuron ? ' on' : ''}" data-neuron="${h}"><i style="background:${h === view.neuron ? '#1c1b19' : '#fff'}"></i>neuron ${h}</span>`).join(''));

  if (flow.value !== String(view.flow)) flow.value = String(view.flow);
  if (lift.value !== String(view.lift)) lift.value = String(view.lift);
  text($('flow-text'), view.flow < 0.02 ? 'where points start' : view.flow < 0.98 ? 'attention moving them' : view.flow < 1.02 ? 'after attention' : view.flow < 1.98 ? 'MLP folding them' : 'after the MLP');
  text($('lift-text'), view.lift < 0.03 ? 'flat' : 'lifted');

  const sel = state.selected;
  $('param-empty').hidden = sel !== null;
  $('param-body').hidden = sel === null;
  if (sel !== null) {
    const d = describe(L, sel), val = state.th[sel];
    text($('param-name'), d.name);
    text($('param-what'), d.what);
    text($('param-value'), val.toFixed(2));
    if (document.activeElement !== slider) slider.value = String(val);
    $<HTMLInputElement>('wiggle').checked = state.wiggle;
  }
}

function spark() {
  const cv = $<HTMLCanvasElement>('spark'), g = cv.getContext('2d')!, n = state.loss.length;
  g.clearRect(0, 0, cv.width, cv.height);
  if (n < 2) return;
  const lo = Math.log(0.01), hi = Math.log(Math.max(2, state.loss[0]));
  g.beginPath();
  for (let px = 0; px < cv.width; px++) {
    const v = state.loss[Math.min(n - 1, Math.floor((px / (cv.width - 1)) * (n - 1)))];
    const y = cv.height - 2 - ((Math.log(Math.max(v, 0.01)) - lo) / (hi - lo)) * (cv.height - 4);
    px ? g.lineTo(px, y) : g.moveTo(px, y);
  }
  g.strokeStyle = '#1c1b19';
  g.lineWidth = 1.2;
  g.stroke();
}

// for scripts/shot.ts and the console
Object.assign(window, { app: { state, select, restart, cfg, L, scene } });

requestAnimationFrame(frame);
