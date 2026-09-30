// The main-thread side of a run: a mirror Trainer (for every view), the
// training worker, edits, and the checkpoint timeline.

import { batch, untracked } from '../../vendor/plainpanel/plainpanel.js';
import type { Model, Param } from '../engine/model';
import type { FromWorker, ParamMeta, ToWorker } from '../engine/protocol';
import { cloneSpec, type Spec } from '../engine/spec';
import { Trainer, evaluate } from '../engine/trainer';
import { charts, rt } from './state';

export interface Snapshot {
  step: number;
  loss: number;
  params: Float64Array;
}

export class Session {
  trainer: Trainer;
  private worker: Worker;
  private version = 0;
  /** full checkpoint history, fetched from the worker when you scrub */
  history: Snapshot[] = [];
  private liveFlat: Float64Array | null = null;
  private liveStep = 0;
  private evalTimer = 0;
  private historyWaiters: ((h: Snapshot[]) => void)[] = [];
  private bumpT = 0;
  private bumpTimer = 0;

  constructor(spec: Spec) {
    this.trainer = new Trainer(cloneSpec(spec));
    this.worker = new Worker(new URL('../engine/worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
    this.worker.onerror = (e) => rt.error(`worker: ${e.message}`);
    this.load(true);
    this.refreshEval();
  }

  get model(): Model {
    return this.trainer.model;
  }

  get data() {
    return this.trainer.data;
  }

  private send(m: ToWorker, transfer: Transferable[] = []) {
    this.worker.postMessage(m, transfer);
  }

  private meta(p: Param): ParamMeta {
    return { name: p.name, lrMult: p.lrMult, frozen: p.frozen, mask: p.mask };
  }

  /** Push the mirror's full state (spec + weights + meta) to the worker. */
  private load(resetHistory: boolean) {
    this.version++;
    this.send({
      type: 'load',
      spec: cloneSpec(this.trainer.spec),
      flat: this.model.flat(),
      meta: this.model.params.map((p) => this.meta(p)),
      step: this.trainer.step,
      version: this.version,
      resetHistory,
    });
  }

  /**
   * Rebuild from a new spec. keep = carry over every weight whose name and
   * shape still match (architecture/task edits); otherwise a fresh init.
   */
  rebuild(spec: Spec, keep: boolean): void {
    this.exitScrub();
    const old = this.trainer;
    this.trainer = new Trainer(cloneSpec(spec), keep ? old.model : undefined);
    this.trainer.step = keep ? old.step : 0;
    if (!keep) this.clearCharts();
    this.load(!keep);
    batch(() => {
      rt.step(this.trainer.step);
      rt.viewStep(this.trainer.step);
      rt.modelVersion(rt.modelVersion() + 1);
      rt.paramsVersion(rt.paramsVersion() + 1);
    });
    this.refreshEval();
  }

  clearCharts(): void {
    for (const s of Object.values(charts)) s.clear();
  }

  setOptim(o: Spec['optim']): void {
    this.trainer.setOptim({ ...o });
    this.send({ type: 'optim', optim: { ...o } });
  }

  run(on: boolean): void {
    if (on) this.exitScrub();
    rt.running(on);
    this.send({ type: 'run', on });
  }

  steps(n: number): void {
    this.exitScrub();
    this.send({ type: 'steps', n });
  }

  setSpeed(stepsPerSec: number): void {
    this.send({ type: 'speed', stepsPerSec });
  }

  /** Set one scalar weight. Applied to the mirror at once, then the worker. */
  edit(name: string, index: number, value: number): void {
    const p = this.model.byName.get(name);
    if (!p || !Number.isFinite(value)) return;
    this.claimScrub();
    p.t.data[index] = value;
    this.version++;
    this.send({ type: 'edit', name, index, value, version: this.version });
    this.touched();
  }

  /** Replace a whole tensor (zero, scale, re-init, paste). */
  setTensor(name: string, values: ArrayLike<number>): void {
    const p = this.model.byName.get(name);
    if (!p) return;
    this.claimScrub();
    p.t.data.set(values);
    this.pushParams();
  }

  reinitTensor(name: string): void {
    const p = this.model.byName.get(name);
    if (!p) return;
    this.claimScrub();
    this.model.initParam(p, this.trainer.spec.init);
    this.pushParams();
  }

  setMeta(name: string, m: Partial<Pick<Param, 'lrMult' | 'frozen' | 'mask'>>): void {
    const p = this.model.byName.get(name);
    if (!p) return;
    Object.assign(p, m);
    this.send({ type: 'meta', meta: this.meta(p) });
    rt.paramsVersion(rt.paramsVersion() + 1);
  }

  private pushParams() {
    this.version++;
    this.send({ type: 'params', flat: this.model.flat(), version: this.version });
    this.touched();
  }

  private touched() {
    rt.paramsVersion(rt.paramsVersion() + 1);
    if (!untracked(() => rt.running())) this.refreshEval();
  }

  /** Exact train/val metrics on the mirror (after edits while paused). */
  refreshEval(): void {
    clearTimeout(this.evalTimer);
    this.evalTimer = window.setTimeout(() => {
      const tr = this.trainer;
      batch(() => {
        rt.train(evaluate(tr.model, tr.data.train, tr.data.T));
        rt.val(evaluate(tr.model, tr.data.val, tr.data.T));
      });
    }, 60);
  }

  private onMessage(m: FromWorker) {
    switch (m.type) {
      case 'state': {
        if (m.version < this.version) return; // stale: an edit is in flight
        if (m.flat.length !== this.model.numParams()) return; // from before a rebuild
        batch(() => {
          for (const pt of m.points) {
            charts.loss.push(pt.loss);
            charts.lossStep.push(pt.step);
            charts.gradNorm.push(pt.gradNorm);
          }
          if (m.points.length) rt.loss(m.points[m.points.length - 1].loss);
          if (m.stats) rt.stats(m.stats);
          rt.step(m.step);
          if (!rt.scrubbing()) {
            this.model.setFlat(m.flat);
            this.trainer.step = m.step;
            rt.viewStep(m.step);
            this.bumpThrottled(m.running);
          } else {
            this.liveFlat = m.flat;
            this.liveStep = m.step;
          }
          rt.running(m.running);
        });
        break;
      }
      case 'eval':
        if (rt.scrubbing()) return;
        batch(() => {
          rt.train(m.train);
          rt.val(m.val);
          charts.evalStep.push(m.step);
          charts.trainLoss.push(m.train.loss);
          charts.valLoss.push(m.val.n ? m.val.loss : null);
          charts.trainAcc.push(m.train.seqAcc);
          charts.valAcc.push(m.val.n ? m.val.seqAcc : null);
        });
        break;
      case 'history': {
        this.history = m.checkpoints;
        const w = this.historyWaiters;
        this.historyWaiters = [];
        w.forEach((f) => f(this.history));
        break;
      }
      case 'error':
        rt.error(m.message);
        console.error(m.message);
        break;
    }
  }

  /** While training, views redraw at most ~15 times a second. */
  private bumpThrottled(running: boolean) {
    const now = performance.now();
    clearTimeout(this.bumpTimer);
    if (!running || now - this.bumpT > 66) {
      this.bumpT = now;
      rt.paramsVersion(rt.paramsVersion() + 1);
    } else {
      this.bumpTimer = window.setTimeout(() => {
        this.bumpT = performance.now();
        rt.paramsVersion(rt.paramsVersion() + 1);
      }, 70);
    }
  }

  fetchHistory(): Promise<Snapshot[]> {
    return new Promise((res) => {
      this.historyWaiters.push(res);
      this.send({ type: 'history' });
    });
  }

  /** Show a past checkpoint in every view (training pauses). */
  scrubTo(snap: Snapshot): void {
    if (snap.params.length !== this.model.numParams()) return;
    if (!rt.scrubbing()) {
      if (rt.running()) this.run(false);
      this.liveFlat = this.model.flat();
      this.liveStep = this.trainer.step;
      rt.scrubbing(true);
    }
    this.model.setFlat(snap.params);
    batch(() => {
      rt.viewStep(snap.step);
      rt.paramsVersion(rt.paramsVersion() + 1);
    });
    this.refreshEval();
  }

  /** Leave the timeline: back to the live weights. */
  exitScrub(): void {
    if (!rt.scrubbing()) return;
    if (this.liveFlat) this.model.setFlat(this.liveFlat);
    this.liveFlat = null;
    batch(() => {
      rt.scrubbing(false);
      rt.viewStep(this.liveStep);
      rt.paramsVersion(rt.paramsVersion() + 1);
    });
    this.refreshEval();
  }

  /** Branch: continue training from the checkpoint on screen. */
  resumeFromHere(): void {
    if (!rt.scrubbing()) return;
    const step = rt.viewStep();
    this.claimScrub();
    this.trainer.step = step;
    this.load(false);
    rt.step(step);
  }

  /** An edit while scrubbing makes the shown checkpoint the live state. */
  private claimScrub() {
    if (!rt.scrubbing()) return;
    const step = rt.viewStep();
    this.liveFlat = null;
    rt.scrubbing(false);
    this.trainer.step = step;
    this.load(false);
    rt.step(step);
  }
}
