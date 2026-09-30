// The training worker. Runs the optimizer loop off the main thread and
// streams weights, per-step loss and periodic exact evals back.

import type { FromWorker, ParamStat, StepPoint, ToWorker } from './protocol';
import { Trainer } from './trainer';

let tr: Trainer | null = null;
let running = false;
let version = 0;
let stepsPerSec = 0; // 0 = as fast as possible
let points: StepPoint[] = [];
let lastStats: ParamStat[] | null = null;
let lastEval = 0;
let lastPost = 0;
let budget = 0;
let lastTick = 0;

const post = (m: FromWorker, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

function doSteps(n: number) {
  if (!tr) return;
  for (let i = 0; i < n; i++) {
    const wantStats = tr.step % 20 === 0;
    const { loss, stats } = tr.trainStep(wantStats);
    points.push({ step: tr.step, loss, lr: stats.lr, gradNorm: stats.gradNorm });
    if (wantStats) lastStats = stats.perParam;
  }
}

function sendState(force = false) {
  if (!tr) return;
  const now = performance.now();
  if (!force && now - lastPost < 33) return;
  lastPost = now;
  const flat = tr.model.flat();
  post(
    { type: 'state', step: tr.step, flat, version, running, points, stats: lastStats, stepsPerSec },
    [flat.buffer],
  );
  points = [];
  lastStats = null;
  if (force || now - lastEval > 250) {
    lastEval = now;
    const e = tr.evalAll();
    post({ type: 'eval', step: tr.step, train: e.train, val: e.val });
  }
}

function tick() {
  if (!running || !tr) return;
  const now = performance.now();
  if (stepsPerSec > 0) {
    budget += ((now - lastTick) / 1000) * stepsPerSec;
    budget = Math.min(budget, stepsPerSec); // never burst more than 1 s worth
    const n = Math.floor(budget);
    budget -= n;
    doSteps(n);
  } else {
    const t0 = performance.now();
    while (performance.now() - t0 < 15) doSteps(1);
  }
  lastTick = now;
  sendState();
  setTimeout(tick, stepsPerSec > 0 ? 8 : 0);
}

self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  try {
    switch (m.type) {
      case 'load': {
        const old = tr;
        tr = new Trainer(m.spec);
        tr.model.setFlat(m.flat);
        for (const pm of m.meta) {
          const p = tr.model.byName.get(pm.name);
          if (p) Object.assign(p, { lrMult: pm.lrMult, frozen: pm.frozen, mask: pm.mask });
        }
        tr.step = m.step;
        if (!m.resetHistory && old) tr.history = old.history.filter((c) => c.step <= m.step && c.params.length === tr!.model.numParams());
        else tr.history = [];
        tr.checkpoint(NaN);
        version = m.version;
        points = [];
        sendState(true);
        break;
      }
      case 'optim':
        tr?.setOptim(m.optim);
        break;
      case 'run':
        if (m.on && !running) {
          running = true;
          lastTick = performance.now();
          budget = 0;
          tick();
        } else if (!m.on) {
          running = false;
          sendState(true);
        }
        break;
      case 'steps':
        doSteps(m.n);
        sendState(true);
        break;
      case 'speed':
        stepsPerSec = m.stepsPerSec;
        break;
      case 'params':
        tr?.model.setFlat(m.flat);
        version = m.version;
        if (!running) sendState(true);
        break;
      case 'edit': {
        const p = tr?.model.byName.get(m.name);
        if (p) p.t.data[m.index] = m.value;
        version = m.version;
        break;
      }
      case 'meta': {
        const p = tr?.model.byName.get(m.meta.name);
        if (p) Object.assign(p, { lrMult: m.meta.lrMult, frozen: m.meta.frozen, mask: m.meta.mask });
        break;
      }
      case 'history':
        if (tr) post({ type: 'history', checkpoints: tr.history.map((c) => ({ step: c.step, loss: c.loss, params: c.params.slice() })) });
        break;
    }
  } catch (e) {
    post({ type: 'error', message: String((e as Error)?.stack ?? e) });
  }
};
