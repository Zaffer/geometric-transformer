import { describe, expect, it } from 'vitest';
import { Trainer } from '../src/engine/trainer';
import { cloneSpec, defaultSpec } from '../src/engine/spec';
import { buildDataset } from '../src/engine/tasks';
import { Model } from '../src/engine/model';
import { lrAt } from '../src/engine/optim';

describe('training', () => {
  it('the default d=3 model learns sort', () => {
    const tr = new Trainer(defaultSpec());
    for (let i = 0; i < 1500; i++) tr.trainStep();
    const e = tr.evalAll();
    expect(e.train.seqAcc).toBeGreaterThan(0.95);
    expect(e.val.seqAcc).toBeGreaterThan(0.9);
  });

  it('is deterministic', () => {
    const a = new Trainer(defaultSpec()), b = new Trainer(defaultSpec());
    for (let i = 0; i < 50; i++) {
      a.trainStep();
      b.trainStep();
    }
    expect(Array.from(a.model.flat())).toEqual(Array.from(b.model.flat()));
  });

  for (const kind of ['sgd', 'momentum', 'adam', 'adamw'] as const) {
    it(`${kind} lowers the loss`, () => {
      const s = defaultSpec();
      s.optim.kind = kind;
      s.optim.lr = kind === 'sgd' || kind === 'momentum' ? 0.1 : 0.01;
      s.optim.weightDecay = 0.01;
      const tr = new Trainer(s);
      const l0 = tr.evalAll().train.loss;
      for (let i = 0; i < 300; i++) tr.trainStep();
      expect(tr.evalAll().train.loss).toBeLessThan(l0 * 0.8);
    });
  }

  it('frozen params, masked elements and lrMult 0 never move', () => {
    const tr = new Trainer(defaultSpec());
    const m = tr.model;
    const wte = m.byName.get('wte')!;
    wte.frozen = true;
    const w1 = m.byName.get('L0.mlp.W1')!;
    w1.mask = new Uint8Array(w1.t.size);
    w1.mask[3] = 1;
    const wo = m.byName.get('L0.attn.Wo')!;
    wo.lrMult = 0;
    const before = { wte: wte.t.data.slice(), w1: w1.t.data[3], w1b: w1.t.data[4], wo: wo.t.data.slice() };
    for (let i = 0; i < 20; i++) tr.trainStep();
    expect(Array.from(wte.t.data)).toEqual(Array.from(before.wte));
    expect(w1.t.data[3]).toBe(before.w1);
    expect(w1.t.data[4]).not.toBe(before.w1b);
    expect(Array.from(wo.t.data)).toEqual(Array.from(before.wo));
  });

  it('keeps matching weights across an architecture change', () => {
    const s = defaultSpec();
    const tr = new Trainer(s);
    for (let i = 0; i < 20; i++) tr.trainStep();
    const s2 = cloneSpec(s);
    s2.model.dMlp = 16; // MLP weights change shape; everything else is kept
    const tr2 = new Trainer(s2, tr.model);
    expect(Array.from(tr2.model.p('wte').data)).toEqual(Array.from(tr.model.p('wte').data));
    expect(tr2.model.p('L0.mlp.W1').cols).toBe(16);
  });

  it('per-tensor init streams: toggling a bias leaves other weights alone', () => {
    const s = defaultSpec();
    const d = buildDataset(s.task);
    const a = new Model(s.model, d.vocab.length, d.T, s.init);
    const s2 = cloneSpec(s);
    s2.model.biasQKV = false;
    const b = new Model(s2.model, d.vocab.length, d.T, s2.init);
    expect(Array.from(b.p('L1.attn.Wq').data)).toEqual(Array.from(a.p('L1.attn.Wq').data));
  });

  it('checkpoint history thins but spans the run', () => {
    const tr = new Trainer(defaultSpec());
    for (let i = 0; i < 1000; i++) tr.trainStep();
    expect(tr.history.length).toBeLessThanOrEqual(tr.maxCheckpoints);
    expect(tr.history[0].step).toBe(0);
    expect(tr.history[tr.history.length - 1].step).toBeGreaterThan(900);
  });

  it('schedules', () => {
    const o = defaultSpec().optim;
    o.warmup = 10;
    o.schedule = 'cosine';
    o.totalSteps = 110;
    expect(lrAt(o, 0)).toBeCloseTo(o.lr / 10);
    expect(lrAt(o, 10)).toBeCloseTo(o.lr);
    expect(lrAt(o, 110)).toBeCloseTo(0);
  });
});

describe('tasks', () => {
  for (const kind of ['sort', 'reverse', 'copy', 'modadd', 'text'] as const) {
    it(`${kind} builds consistent examples`, () => {
      const s = defaultSpec().task;
      s.kind = kind;
      const d = buildDataset(s);
      expect(d.train.length).toBeGreaterThan(0);
      for (const e of [...d.train, ...d.val]) {
        expect(e.x.length).toBe(d.T);
        expect(e.y.length).toBe(d.T);
        for (let t = 0; t < d.T - 1; t++) if (e.y[t] >= 0) expect(e.y[t]).toBe(e.x[t + 1]);
      }
    });
  }
});
